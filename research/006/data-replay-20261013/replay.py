#!/usr/bin/env python3
"""Produce actual CPU tool-transition data from pinned external task states.

This is a data producer, not a verification suite. It executes no stored shell
trajectory and no repository test suite. Official editing/response functions
are loaded unchanged; only filesystem/process transport is local to macOS.
"""
from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import types
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = ROOT / 'research/006/official-data-20261013/wheels-source'
CORE = HERE / 'official-core-source'


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def package(name, directory):
    # Avoid importing unrelated evaluator/ADK dependencies through __init__.
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module
    return module


def official_tools():
    package('adk_eval_core', CORE / 'adk_eval_core')
    package('swegemma', OFFICIAL / 'swegemma')
    package('swegemma.tools', OFFICIAL / 'swegemma/tools')
    package('swegemma.harness', OFFICIAL / 'swegemma/harness')
    verification = types.ModuleType('swegemma.harness.verification')
    verification.__dict__.update(Path=Path, Sequence=Sequence, re=re)
    # Original ASTs only: the patch-capture helper imports an entire evaluator
    # otherwise. No function body, condition, or default is rewritten.
    selections = [
        ('swegemma/models/task.py', {'_unquote_git_path'}),
        ('swegemma/harness/verification.py', {
            '_is_protected_test_or_config_path', 'strip_protected_files_from_patch'}),
    ]
    for rel, names in selections:
        tree = ast.parse((OFFICIAL / rel).read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(OFFICIAL / rel), 'exec'), verification.__dict__)
    sys.modules[verification.__name__] = verification
    workspace = importlib.import_module('swegemma.tools.workspace')
    execution = importlib.import_module('swegemma.tools.execution')
    return {name: getattr(module, name) for module, names in [
        (workspace, ('read_file', 'edit_file', 'write_file')),
        (execution, ('run_command', 'submit_patch', 'get_status')),
    ] for name in names}


class LocalTransport:
    """Confined file copies and reviewed inline commands, not a Docker replica."""
    def __init__(self, workspace):
        self.workspace = workspace.resolve()
        self.commands = []

    def mapped(self, value):
        value = str(value)
        if value == '/workspace':
            return self.workspace
        if not value.startswith('/workspace/'):
            raise ValueError('Only /workspace file transport is available')
        result = (self.workspace / value[len('/workspace/'):]).resolve()
        if not result.is_relative_to(self.workspace):
            raise ValueError('Path escapes replay workspace')
        return result

    def copy_from(self, _cid, source, dest):
        shutil.copyfile(self.mapped(source), dest)

    def copy_to(self, _cid, source, dest):
        target = self.mapped(dest)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target / Path(source).name)

    def exec(self, _cid, command, timeout=30):
        # Official submit_patch has this fixed prefix. Source trajectory shell
        # strings are never sent here. Commands below are reviewed literals.
        command = command.removeprefix('cd /workspace && ')
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL') if k in os.environ}
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '/usr/bin:/bin')
        env.update(HOME=str(self.workspace.parent), TMPDIR=str(self.workspace.parent),
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', GIT_CONFIG_NOSYSTEM='1')
        self.commands.append(command)
        try:
            run = subprocess.run(['/bin/bash', '-c', command], cwd=self.workspace,
                                 env=env, text=True, capture_output=True, timeout=timeout)
            return types.SimpleNamespace(stdout=run.stdout, stderr=run.stderr,
                                         exit_code=run.returncode, timed_out=False)
        except subprocess.TimeoutExpired as exc:
            return types.SimpleNamespace(stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''),
                                         exit_code=124, timed_out=True)


class Context:
    def __init__(self, transport, repo):
        self.docker, self.container_id, self.repo = transport, 'cpu-replay', repo
        self.harness = types.SimpleNamespace(command_timeout_seconds=30,
            max_stdout_chars=5000, max_file_lines=150, max_file_chars=10000)
        self.budget = types.SimpleNamespace(time_minutes=None, tool_calls=50, turns=None)
        self.tool_calls_used, self.submitted_patch, self.patch_submitted = 0, None, False

    def check_budget(self, count_tool_call=True):
        if count_tool_call:
            self.tool_calls_used += 1
        if self.tool_calls_used > self.budget.tool_calls:
            return json.dumps({'status': 'error', 'error_type': 'BudgetExceeded'})


def git(workspace, *args):
    return subprocess.run(['git', '-C', str(workspace), *args], check=True,
                          capture_output=True, text=True).stdout


def baseline(workspace):
    git(workspace, 'init', '-q')
    git(workspace, 'add', '.')
    git(workspace, '-c', 'user.name=CPU Replay', '-c', 'user.email=replay@localhost',
        'commit', '-q', '-m', 'Pinned buggy production source subset')
    git(workspace, 'tag', '_swegemma_baseline')


def source(name):
    row = json.loads((HERE / 'sources' / f'{name}-source-trajectory.json').read_text())
    messages = json.loads(row['messages_raw'])
    text = messages[1]['content'][0]['text']
    issue = text.split('<pr_description>', 1)[1].split('</pr_description>', 1)[0].strip()
    return row, messages, issue


def edit_args(messages, index):
    args = json.loads(messages[index]['tool_calls'][0]['function']['arguments'])
    assert args['command'] == 'str_replace'
    return {'filepath': args['path'].removeprefix('/testbed/'),
            'old_string': args['old_str'], 'new_string': args['new_str'],
            'allow_multiple': False}


class Recorder:
    def __init__(self, ctx, tools, user):
        self.ctx, self.tools, self.events = ctx, tools, []
        self.messages = [
            {'role': 'system', 'content': 'Use the supplied repository tools. Preserve unrelated behavior. Treat tool failures as observations and correct the action before retrying. Do not modify tests.'},
            {'role': 'user', 'content': user},
        ]

    def call(self, name, args, content=''):
        cid = f'cpu_{len(self.events):02d}'
        self.messages.append({'role': 'assistant', 'content': content,
            'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': args}}]})
        result = self.tools[name](self.ctx, **args)
        payload = json.loads(result)
        self.messages.append({'role': 'tool', 'tool_call_id': cid, 'name': name,
                              'status': payload['status'], 'content': result})
        self.events.append({'tool': name, 'arguments': args, 'response': payload})
        return payload


REPRODUCE = """python - <<'PY'
import io, json, sys
sys.path.insert(0, 'src')
import click
results = []
for err in (False, True):
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = None
    try:
        click.echo('Hello world', err=err)
        result = {'case': 'missing_stderr' if err else 'missing_stdout', 'no_exception': True}
    except Exception as exc:
        result = {'case': 'missing_stderr' if err else 'missing_stdout', 'no_exception': False, 'error': type(exc).__name__ + ': ' + str(exc)}
    finally:
        sys.stdout, sys.stderr = saved
    results.append(result)
explicit = io.StringIO()
click.echo('Hello world', file=explicit)
results.append({'case': 'explicit_stream', 'expected_output': explicit.getvalue() == 'Hello world\\n'})
print(json.dumps({'click_file': click.__file__.split('/workspace/')[-1], 'results': results}, sort_keys=True))
sys.exit(0 if all(x.get('no_exception', x.get('expected_output', False)) for x in results) else 1)
PY"""


def finish(name, row, rec, workspace, commit, targets, outcome, state_kind, extras=None):
    dest = HERE / 'outputs' / name
    dest.mkdir(parents=True, exist_ok=True)
    patch = git(workspace, 'diff', '--binary', '_swegemma_baseline')
    (dest / 'production.patch').write_text(patch)
    tools_schema = json.loads((ROOT / 'training/configs/harness_tools.json').read_text())
    schema_hash = digest(json.dumps(tools_schema, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    sample = {
        'sample_id': 'p006-' + name, 'transaction_id': 'p006-' + name + '-tx01',
        'origin_kind': 'cpu_replayed_expert_demonstration', 'synthetic': False,
        'group': {'issue': row['instance_id'], 'repo': row['source_repo_from_instance_id'], 'commit': commit},
        'source': {'uri': 'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/08e109b4a59eaeebf80e4675cd125d42e7ac99a4',
                   'license': 'MIT dataset; BSD-3-Clause touched source', 'sha256': row['messages_sha256']},
        'protocol_id': 'p005-harness-tools-v1', 'tool_schema_hash': schema_hash,
        'messages': copy.deepcopy(rec.messages), 'target_assistant_steps': targets,
        'observed_outcome': outcome, 'behavior': 'tool_recovery' if 'recovery' in name else 'local_edit',
        'state_kind': state_kind, 'needs_replay': False,
        'source_resolved_label': row['resolved'], 'source_label_not_revalidated': True,
        'replay_boundary': 'CPU macOS local filesystem transport; unchanged v31 official tool functions; no Docker or repository test suite',
        'continuation_policy': 'All included tool responses were executed anew; no old future observations or reasoning copied',
    }
    if extras:
        sample.update(extras)
    (dest / 'transaction.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n')
    (dest / 'events.json').write_text(json.dumps(rec.events, ensure_ascii=False, indent=2) + '\n')
    (dest / 'commands.json').write_text(json.dumps(rec.ctx.docker.commands, indent=2) + '\n')
    return sample


def run_click(tools, root):
    row, messages, issue = source('click')
    workspace = root / 'click' / 'workspace'
    shutil.copytree(HERE / 'sources/click-instance', workspace)
    baseline(workspace)
    rec = Recorder(Context(LocalTransport(workspace), 'pallets/click'), tools, issue)
    rec.call('read_file', {'filepath': 'src/click/utils.py', 'start_line': 222, 'end_line': 315})
    before = rec.call('run_command', {'command': REPRODUCE}, 'Reproduce absent standard streams without creating a test file.')
    assert before['status'] == 'error' and 'AttributeError' in json.dumps(before), before
    edit = rec.call('edit_file', edit_args(messages, 12), 'Return without writing when the selected default stream is unavailable.')
    assert edit['status'] == 'ok'
    after = rec.call('run_command', {'command': REPRODUCE}, 'Repeat the same observed cases after the edit.')
    assert after['status'] == 'ok'
    submitted = rec.call('submit_patch', {}, 'Capture the production-only change.')
    assert submitted['status'] == 'ok' and submitted['files_changed'] == 1
    sample = finish('click-echo', row, rec, workspace, '8154a324c30a33f618b7226cc5be18f54bacac99', [2, 3, 4],
        {'action_succeeded': True, 'task_resolved': None, 'issue_reproduction_before_failed': True,
         'issue_reproduction_after_passed': True, 'cases': 3, 'official_suite_ran': False},
        'complete_new_issue_transaction_from_pinned_buggy_source_subset')
    return sample


def run_pydicom(tools, root):
    row, messages, issue = source('pydicom')
    workspace = root / 'pydicom' / 'workspace'
    target = workspace / 'src/pydicom/sr/codedict.py'
    target.parent.mkdir(parents=True)
    shutil.copyfile(HERE / 'sources/pydicom-pinned-codedict.py', target)
    baseline(workspace)
    ctx = Context(LocalTransport(workspace), 'pydicom/pydicom')
    preparation = []
    for index in (14, 16, 18, 26, 30, 32):
        args = edit_args(messages, index)
        response = json.loads(tools['edit_file'](ctx, **args))
        preparation.append({'source_message_index': index, 'arguments': args, 'response': response})
        assert response['status'] == 'ok', (index, response)
    initial = target.read_text()
    state_diff = git(workspace, 'diff', '_swegemma_baseline')
    user = (issue + '\n\nLocal recovery subtask: earlier edits have introduced CodeLike. '
        'Continue the currently requested _CID_Dict.__getattr__ annotation change from Code '
        'to Union[CodeLike, Code], without altering the other __getattr__ occurrence. '
        'This is an existing in-progress source state, not a fresh full-issue attempt.\n\n'
        'Current production diff (from the materialized state, before this continuation):\n' + state_diff)
    rec = Recorder(ctx, tools, user)
    failed = rec.call('edit_file', edit_args(messages, 34))
    assert failed['status'] == 'error' and '2 times' in failed['error_message'], failed
    assert target.read_text() == initial
    rec.call('read_file', {'filepath': 'src/pydicom/sr/codedict.py', 'start_line': 70, 'end_line': 82},
             'The short match is ambiguous; inspect the requested class method before making it unique.')
    fixed = rec.call('edit_file', edit_args(messages, 38), 'Include the observed method docstring to identify one occurrence.')
    assert fixed['status'] == 'ok' and fixed['occurrences'] == 1
    rec.call('read_file', {'filepath': 'src/pydicom/sr/codedict.py', 'start_line': 70, 'end_line': 82},
             'Inspect the edited method; this confirms the local text change only.')
    ast.parse(target.read_text())
    sample = finish('pydicom-recovery', row, rec, workspace, '0f0e22c5fcc977fc211bd08854308e3b2aee459c', [1, 2, 3],
        {'action_succeeded': True, 'task_resolved': None, 'recovery_events': 1,
         'failed_edit_preserved_file': True, 'corrected_edit_occurrences': 1, 'syntax_parsed': True,
         'mypy_ran': False, 'official_suite_ran': False}, 'explicit_local_recovery_subtask',
        {'masked_invalid_assistant_steps': [0], 'behavior_scope': 'tool_match_recovery_only',
         'state_before_sha256': digest(initial), 'state_after_sha256': digest(target.read_text())})
    dest = HERE / 'outputs/pydicom-recovery'
    (dest / 'state-before.py.txt').write_text(initial)
    (dest / 'state-after.py.txt').write_text(target.read_text())
    (dest / 'state-preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2) + '\n')
    return sample


MARKUP_REPRODUCE = """python - <<'PY'
import json, sys
sys.path[:0] = ['src', '../support']
from django_functional import SimpleLazyObject
import markupsafe
from markupsafe import Markup
rows = []
for name, value in [('plain', '<value>'), ('lazy', SimpleLazyObject(lambda: '<value>'))]:
    entry = {'case': name, 'isinstance_str': isinstance(value, str), 'type_is_str': type(value) is str}
    try:
        result = Markup.escape(value)
        entry.update(result=str(result), expected_output=str(result) == '&lt;value&gt;')
    except Exception as exc:
        entry.update(error=type(exc).__name__ + ': ' + str(exc), expected_output=False)
    rows.append(entry)
print(json.dumps({'escape_backend': markupsafe._escape_inner.__module__, 'results': rows}, sort_keys=True))
sys.exit(0 if all(row['expected_output'] for row in rows) else 1)
PY"""

MARKUP_CONTRACTS = """python - <<'PY'
import json, sys
sys.path[:0] = ['src', '../support']
from django_functional import SimpleLazyObject
import markupsafe
from markupsafe import Markup
class TrustedHTML:
    def __html__(self):
        return '<em>trusted</em>'
cases = [
    ('plain', '<value>', '&lt;value&gt;'),
    ('lazy', SimpleLazyObject(lambda: '<value>'), '&lt;value&gt;'),
    ('trusted_markup', Markup('<em>trusted</em>'), '<em>trusted</em>'),
    ('html_protocol', TrustedHTML(), '<em>trusted</em>'),
    ('number', 123, '123'),
]
rows = []
for name, value, expected in cases:
    entry = {'case': name, 'isinstance_str': isinstance(value, str), 'type_is_str': type(value) is str}
    try:
        result = Markup.escape(value)
        entry.update(result=str(result), expected=expected, passed=str(result) == expected)
    except Exception as exc:
        entry.update(error=type(exc).__name__ + ': ' + str(exc), passed=False)
    rows.append(entry)
print(json.dumps({'escape_backend': markupsafe._escape_inner.__module__, 'results': rows}, sort_keys=True))
sys.exit(0 if all(row['passed'] for row in rows) else 1)
PY"""


def run_markupsafe(tools, root):
    row, messages, issue = source('markupsafe')
    workspace = root / 'markupsafe' / 'workspace'
    shutil.copytree(HERE / 'sources/markupsafe-instance', workspace)
    support = workspace.parent / 'support'
    support.mkdir()
    shutil.copyfile(HERE / 'sources/django-functional-3.2.20.py', support / 'django_functional.py')
    baseline(workspace)
    # Compile only the reviewed, fixed-commit 205-line extension. No setup.py,
    # build backend, downloaded wheel, or package installation is executed.
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    command = ['/usr/bin/clang', '-bundle', '-undefined', 'dynamic_lookup',
               '-I' + sysconfig.get_paths()['include'],
               'src/markupsafe/_speedups.c', '-o', 'src/markupsafe/_speedups' + suffix]
    build = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=60)
    if build.returncode:
        raise RuntimeError('Pinned speedups compilation failed: ' + build.stderr)
    (workspace / '.git/info/exclude').write_text('*.so\n__pycache__/\n')
    ctx = Context(LocalTransport(workspace), 'pallets/markupsafe')
    prepared_args = edit_args(messages, 22)
    prepared = json.loads(tools['edit_file'](ctx, **prepared_args))
    assert prepared['status'] == 'ok', prepared
    target = workspace / 'src/markupsafe/__init__.py'
    initial = target.read_text()
    state_diff = git(workspace, 'diff', '_swegemma_baseline')
    user = (issue + '\n\nLocal recovery subtask: the workspace already contains one earlier edit '
        'to escape(). Continue the attempted replacement, refreshing stale old text from the '
        'current file instead of reusing the original version. The native _speedups extension '
        'was compiled from this fixed buggy snapshot; SimpleLazyObject is the unchanged '
        'Django 3.2.20 functional.py implementation. A tool edit succeeding does not prove '
        'the proxy behavior is fixed.\n\nCurrent production diff before this continuation:\n' + state_diff)
    rec = Recorder(ctx, tools, user)
    before = rec.call('run_command', {'command': MARKUP_REPRODUCE}, 'Observe the current proxy behavior before another edit.')
    assert before['status'] == 'error' and 'SystemError' in json.dumps(before), before
    first = rec.call('edit_file', edit_args(messages, 30))
    assert first['status'] == 'error' and target.read_text() == initial, first
    second = rec.call('edit_file', edit_args(messages, 32))
    assert second['status'] == 'error' and target.read_text() == initial, second
    rec.call('read_file', {'filepath': 'src/markupsafe/__init__.py', 'start_line': 35, 'end_line': 45},
             'Both attempted matches used stale text. Read the current function tail.')
    fixed = rec.call('edit_file', edit_args(messages, 36), 'Use the observed s_str assignment in the old text.')
    assert fixed['status'] == 'ok' and fixed['occurrences'] == 1, fixed
    after = rec.call('run_command', {'command': MARKUP_REPRODUCE},
        'Repeat the same proxy behavior command; record its result independently of the edit result.')
    ast.parse(target.read_text())
    outcome = {'action_succeeded': True, 'task_resolved': None, 'recovery_events': 1,
        'failed_edit_attempts': 2, 'failed_edits_preserved_file': True,
        'corrected_edit_occurrences': 1, 'syntax_parsed': True,
        'behavior_reproduction_before_passed': before['status'] == 'ok',
        'behavior_reproduction_after_passed': after['status'] == 'ok',
        'official_suite_ran': False}
    sample = finish('markupsafe-recovery', row, rec, workspace,
        '000489747bba0b3234687459f883f17b53551b17', [3, 4], outcome,
        'explicit_local_recovery_subtask_with_fresh_behavior_observation',
        {'masked_invalid_assistant_steps': [1, 2], 'behavior_scope': 'stale_old_string_recovery_only',
         'state_before_sha256': digest(initial), 'state_after_sha256': digest(target.read_text()),
         'data_usage': {'role': 'candidate_negative_semantic_diagnostic',
                        'allow_parameter_update': False,
                        'exclude_from_positive_repair_sft': True}})
    dest = HERE / 'outputs/markupsafe-recovery'
    (dest / 'state-before.py.txt').write_text(initial)
    (dest / 'state-after.py.txt').write_text(target.read_text())
    (dest / 'state-preparation.json').write_text(json.dumps({
        'source_message_index': 22, 'arguments': prepared_args, 'response': prepared,
        'native_build': {'command': command, 'exit_code': build.returncode,
                         'stdout': build.stdout, 'stderr': build.stderr,
                         'python': sys.version, 'extension_suffix': suffix}}, ensure_ascii=False, indent=2) + '\n')
    (dest / 'transaction.jsonl').write_text(json.dumps(sample, ensure_ascii=False) + '\n')

    # Continue only from the new observed failure. Do not import the old source
    # trajectory's later actions, patches, observations, or success claim.
    semantic_before = rec.call('run_command', {'command': MARKUP_CONTRACTS},
        'The proxy still reaches the native string function. Check the documented __html__ behavior as well before changing the type guard.')
    assert semantic_before['status'] == 'error', semantic_before
    rec.call('read_file', {'filepath': 'src/markupsafe/__init__.py', 'start_line': 24, 'end_line': 48},
        'Read the current implementation and its promise to trust __html__ output.')
    corrected_body = '''    # Only a real str can bypass conversion before the native escape call.
    if type(s) is str:
        return Markup(_escape_inner(s))

    # __html__ explicitly supplies trusted HTML and must not be escaped again.
    if hasattr(s, "__html__"):
        return Markup(s.__html__())

    return Markup(_escape_inner(str(s)))'''
    semantic_edit = rec.call('edit_file', {
        'filepath': 'src/markupsafe/__init__.py',
        'old_string': edit_args(messages, 36)['new_string'],
        'new_string': corrected_body, 'allow_multiple': False},
        'The observed lazy proxy passes isinstance but is not a real str. Convert it, and preserve the documented trusted-HTML fast path.')
    assert semantic_edit['status'] == 'ok' and semantic_edit['occurrences'] == 1, semantic_edit
    semantic_after = rec.call('run_command', {'command': MARKUP_CONTRACTS},
        'Repeat the exact same five contract cases after the semantic change.')
    assert semantic_after['status'] == 'ok', semantic_after
    submitted = rec.call('submit_patch', {}, 'Capture only the production source change; native build files are excluded.')
    assert submitted['status'] == 'ok' and submitted['files_changed'] == 1, submitted
    before_contracts = json.loads(semantic_before['details']['stdout'])['results']
    after_contracts = json.loads(semantic_after['stdout'])['results']
    positive = finish('markupsafe-semantic-repair', row, rec, workspace,
        '000489747bba0b3234687459f883f17b53551b17', [7, 8, 9, 10],
        {'action_succeeded': True, 'task_resolved': None, 'recovery_events': 1,
         'behavior_cases_before_passed': sum(case['passed'] for case in before_contracts),
         'behavior_cases_after_passed': sum(case['passed'] for case in after_contracts),
         'behavior_cases_total': len(after_contracts), 'native_backend_compiled_from_pinned_source': True,
         'official_suite_ran': False, 'source_resolved_label_not_used_as_replay_judgment': True},
        'new_semantic_continuation_of_explicit_local_recovery_subtask',
        {'masked_invalid_assistant_steps': [1, 2], 'masked_semantically_failed_assistant_steps': [4],
         'behavior_scope': 'fresh_native_proxy_and_html_contract_cases',
         'data_usage': {'role': 'candidate_unassigned', 'allow_parameter_update': False,
                        'shares_issue_group_with': 'p006-markupsafe-recovery'},
         'positive_targets_note': 'Only the fresh semantic continuation is supervised. Earlier stale-call correction remains context, not a positive repair target.'})
    positive_dest = HERE / 'outputs/markupsafe-semantic-repair'
    (positive_dest / 'state-after.py.txt').write_text(target.read_text())
    (positive_dest / 'transaction.jsonl').write_text(json.dumps(positive, ensure_ascii=False) + '\n')
    return [sample, positive]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('initial', 'markupsafe'), default='initial')
    args = parser.parse_args()
    tools = official_tools()
    with tempfile.TemporaryDirectory(prefix='p006-cpu-replay-') as tmp:
        root = Path(tmp)
        if args.only == 'markupsafe':
            samples = run_markupsafe(tools, root)
        else:
            samples = [run_click(tools, root), run_pydicom(tools, root)]
    if args.only == 'initial':
        out = HERE / 'outputs/transactions.jsonl'
        out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in samples))
    print(json.dumps([{'sample_id': x['sample_id'], 'observed_outcome': x['observed_outcome'],
                       'messages': len(x['messages'])} for x in samples], indent=2))


if __name__ == '__main__':
    main()
