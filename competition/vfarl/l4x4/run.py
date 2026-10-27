"""S600 real-policy collection, restored focal branches and official verification.

Policy input is only the frozen public task prompt, actual generated tokens and
actual six-tool responses. Recipes, source patches and verifier metadata are
host-side assets. This entry point is also the CPU task-environment preparation
consumer; --prevalidate does not import torch or load a model.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

SIX_TOOLS = {'run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch'}


class PolicyProtocolError(RuntimeError):
    """A shared sampling/state interface failed; do not retry it on every task."""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def runtime_digest(path):
    return digest(json.dumps(json.loads(path.read_text()), ensure_ascii=False,
                             sort_keys=True, separators=(',', ':')).encode())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Unexpected training task ID")
    return value


class TaskQueue:
    """Host-only, process-safe claims in the frozen task order.

    A claim is never silently retried after a worker failure. Every started
    task and focal group remains evidence, including failures and unknowns.
    Stopping affects future claims/state reservations, never an active branch.
    """
    def __init__(self, path, task_ids, *, selection_sha256, maximum_states,
                 stop_after_mixed_groups=0, stop_after_tasks=0, execution_contract=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.contract = dict(format='vfarl-shared-task-queue-v1',
            task_ids=list(task_ids), selection_sha256=selection_sha256,
            maximum_states=maximum_states,
            execution_contract=execution_contract,
            stop_after_mixed_groups=stop_after_mixed_groups,
            stop_after_tasks=stop_after_tasks,
            unit='One task including its predeclared draws and complete reserved state groups',
            stopping='Stop new claims and new state reservations; retain every in-flight result')
        with self.locked() as state:
            if state:
                if state.get('contract') != self.contract:
                    raise ValueError('The shared queue belongs to different frozen tasks or stopping rules')
            else:
                state.update(contract=self.contract, claimed=[], completed=[], states={}, mixed_groups={})

    def bind_policy(self, identity):
        actual = {key: identity[key] for key in ('policy_id', 'reference_id')}
        with self.locked() as state:
            if 'policy' in state and state['policy'] != actual:
                raise ValueError('Dynamic workers initialized different actual policies or references')
            state['policy'] = actual

    @contextmanager
    def locked(self):
        # The separate lock inode remains stable while the JSON is atomically
        # replaced. Locking the replaced JSON itself would permit split locks.
        with self.path.with_suffix(self.path.suffix+'.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else {}
                yield state
                temporary = self.path.with_name(self.path.name+'.%d.tmp'%os.getpid())
                with temporary.open('w') as stream:
                    json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def stop_reason(self, state):
        mixed = self.contract['stop_after_mixed_groups']
        if mixed and len(state['mixed_groups']) >= mixed:
            return 'declared-mixed-group-target-reached'
        tasks = self.contract['stop_after_tasks']
        if tasks and len(state['claimed']) >= tasks:
            return 'declared-task-claim-budget-reached'
        if len(state['claimed']) >= len(self.contract['task_ids']):
            return 'frozen-task-queue-exhausted'
        return None

    def claim(self, worker):
        with self.locked() as state:
            reason = self.stop_reason(state)
            if reason:
                return None
            task_id = self.contract['task_ids'][len(state['claimed'])]
            state['claimed'].append(dict(instance_id=task_id, worker=worker, claimed_at=time.time()))
            return task_id

    def state_available(self):
        with self.locked() as state:
            # A task budget limits task claims, not the states of those tasks.
            mixed = self.contract['stop_after_mixed_groups']
            return (len(state['states']) < self.contract['maximum_states']
                    and not (mixed and len(state['mixed_groups']) >= mixed))

    def reserve_state(self, state_id, task_id, draw, worker):
        with self.locked() as state:
            if state_id in state['states']:
                raise ValueError('A focal state was already reserved in this run')
            mixed = self.contract['stop_after_mixed_groups']
            if (len(state['states']) >= self.contract['maximum_states']
                    or (mixed and len(state['mixed_groups']) >= mixed)):
                return False
            state['states'][state_id] = dict(instance_id=task_id, draw=draw, worker=worker,
                                             reserved_at=time.time(), completed=False)
            return True

    def finish_state(self, state_id, *, mixed, branches, evidence_path, evidence_sha256):
        with self.locked() as state:
            state['states'][state_id].update(completed=True, finished_at=time.time(),
                mixed_outcomes=mixed, branches=branches,
                evidence_path=str(evidence_path), evidence_sha256_at_registration=evidence_sha256)
            if mixed:
                state['mixed_groups'][state_id] = dict(branches=branches, registered_at=time.time())

    def finish_task(self, task_id, worker):
        with self.locked() as state:
            if any(row['instance_id'] == task_id for row in state['completed']):
                raise ValueError('Task completion was registered twice')
            state['completed'].append(dict(instance_id=task_id, worker=worker, completed_at=time.time()))

    def entries(self, worker, by_id):
        while True:
            task_id = self.claim(worker)
            if task_id is None:
                return
            print(json.dumps(dict(phase='task_claimed', worker=worker, instance_id=task_id,
                                  task_queue=str(self.path))), flush=True)
            yield by_id[task_id]


class NativeArguments:
    """Parse Gemma's literal argument grammar; never eval model output."""
    def __init__(self, text):
        self.text, self.pos = text, 0

    def space(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def take(self, literal):
        self.space()
        if not self.text.startswith(literal, self.pos):
            raise ValueError("Malformed native tool arguments at character %d" % self.pos)
        self.pos += len(literal)

    def value(self):
        self.space()
        if self.text.startswith('<|"|>', self.pos):
            self.pos += 5
            end = self.text.find('<|"|>', self.pos)
            if end < 0:
                raise ValueError("Unterminated native argument string")
            value = self.text[self.pos:end]
            self.pos = end + 5
            return value
        if self.text.startswith('{', self.pos):
            self.pos += 1
            result = {}
            self.space()
            while not self.text.startswith('}', self.pos):
                self.space()
                match = re.match(r'[A-Za-z_][A-Za-z_0-9]*', self.text[self.pos:])
                if match is None:
                    raise ValueError("Invalid native argument key")
                key = match.group()
                self.pos += len(key)
                if key in result:
                    raise ValueError("Duplicate native argument key")
                self.take(':')
                result[key] = self.value()
                self.space()
                if not self.text.startswith(',', self.pos):
                    break
                self.pos += 1
            self.take('}')
            return result
        if self.text.startswith('[', self.pos):
            self.pos += 1
            result = []
            self.space()
            while not self.text.startswith(']', self.pos):
                result.append(self.value())
                self.space()
                if not self.text.startswith(',', self.pos):
                    break
                self.pos += 1
            self.take(']')
            return result
        match = re.match(r'(?:true|false|null|none|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)', self.text[self.pos:])
        if match is None:
            raise ValueError("Invalid native argument literal")
        self.pos += len(match.group())
        return json.loads(match.group().replace('none', 'null'))


def parse_call(text):
    # Only a closed, real generated call is executable. A truncated JSON/string
    # must not be repaired into an invented model action.
    # Native control tokens inside a genuine thought channel are not actions.
    visible, cursor = [], 0
    while True:
        opening = text.find('<|channel>', cursor)
        if opening < 0:
            visible.append(text[cursor:])
            break
        visible.append(text[cursor:opening])
        closing = text.find('<channel|>', opening)
        if closing < 0:
            raise ValueError('Unterminated generated thought channel')
        cursor = closing + len('<channel|>')
    text = ''.join(visible)
    start = text.find('<|tool_call>')
    if start < 0:
        return None
    end = text.find('<tool_call|>', start)
    if end < 0:
        raise ValueError("Truncated model tool call")
    body = text[start + len('<|tool_call>'):end]
    match = re.match(r'call:([A-Za-z_][A-Za-z_0-9]*)', body)
    if match is None:
        raise ValueError("Malformed native tool name")
    parser = NativeArguments(body[match.end():])
    arguments = parser.value()
    parser.space()
    if parser.pos != len(parser.text) or not isinstance(arguments, dict):
        raise ValueError("Malformed native tool-call tail")
    return match.group(1), arguments


def tool_suffix(tokenizer, name, arguments, response):
    call_id = 'actual_call'
    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call_id,
         'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
        {'role': 'tool', 'tool_call_id': call_id, 'name': name,
         'content': json.dumps(response, ensure_ascii=False)},
    ]
    rendered = tokenizer.apply_chat_template(messages, tools=[], tokenize=False,
        return_dict=False, add_generation_prompt=True, enable_thinking=True)
    start = rendered.index('<|tool_response>')
    # Keep every original generated token intact, including genuine thoughts.
    # Only the actual new tool response is serialized by the pinned template.
    return tokenizer.encode(rendered[start:], add_special_tokens=False)


def focal_order(item):
    return (not item.get('has_recovery_budget', False),
            not item.get('observed_tool_error', False),
            len(item['action']['prompt_token_ids']))


def score_focal_rows(policy, args, data_path, *, stage):
    """Score only real mixed-outcome focal rows on the actual actor path.

    The original behavior probabilities remain immutable. Two current-policy
    passes before the update measure repeat drift; one after-update pass gives
    the matching policy comparison. The frozen reference is not substituted
    for current policy.
    """
    from training.common import TrainingError
    from training.train_vfarl import prepare_rows
    if stage not in ('before', 'after'):
        raise ValueError('Unknown actor scoring stage')
    if stage == 'after' and args.adapter_run is None:
        raise TrainingError('Post-update focal scoring requires an actual adapter run')
    data_path = Path(data_path)
    if not data_path.is_file():
        if stage == 'before':
            return
        raise TrainingError('The post-update focal data file does not exist')
    source_bytes = data_path.read_bytes()
    source_hash = digest(source_bytes)
    original_lines = {}
    for number, line in enumerate(source_bytes.decode().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            original_lines[row['branch_id']] = (number, digest(line.encode()))
    rows, evidence = prepare_rows(data_path, args.context)
    # Before scores consume each collector's own complete state groups. After
    # scores consume the merged file, partitioning only after group validation.
    if stage == 'after':
        rows = rows[args.worker_index::args.worker_count]
    if not rows:
        print(json.dumps({'phase': 'actor_focal_scoring', 'stage': stage,
            'scored_rows': 0, 'mixed_group_count': evidence['mixed_group_count']}), flush=True)
        return
    output = args.out/f'actor-{stage}.jsonl'
    if output.exists():
        raise TrainingError('Refusing to overwrite an actor focal scoring result')
    repeats = 2 if stage == 'before' else 1
    for row in rows:
        if row['reference_id'] != policy.identity['reference_id']:
            raise TrainingError('Actor focal scoring reference differs from sampled data')
        if stage == 'before' and row['policy_id'] != policy.identity['policy_id']:
            raise TrainingError('Actor before scoring policy differs from actual sampled policy')
        for repeat in range(repeats):
            score = policy.score_teacher_forced(row['prompt_token_ids'],
                row['completion_token_ids'], row['loss_mask'], frozen_reference=False)
            if (score.get('prompt_token_ids') != row['prompt_token_ids']
                    or score.get('completion_token_ids') != row['completion_token_ids']
                    or score.get('loss_mask') != row['loss_mask']
                    or len(score.get('logprobs', [])) != len(row['completion_token_ids'])
                    or any(not math.isfinite(value) or value > 1e-6
                           for value in score.get('logprobs', []))
                    or any(value != 0.0 for value, keep in
                           zip(score.get('logprobs', []), row['loss_mask']) if not keep)
                    or score.get('policy_id') != policy.identity['policy_id']
                    or score.get('reference_id') != policy.identity['reference_id']):
                raise TrainingError('Actor scoring changed the exact sampled tokens, mask or policy identity')
            source_line, source_line_hash = original_lines[row['branch_id']]
            append(output, dict(stage=stage, repeat_index=repeat,
                branch_id=row['branch_id'], state_id=row['state_id'],
                instance_id=row['instance_id'], reward=row['reward'],
                advantage=row['advantage'], behavior_policy_id=row['policy_id'],
                source_data_sha256=source_hash, source_line=source_line,
                source_line_sha256=source_line_hash, score=score))
            print(json.dumps({'phase': 'actor_focal_score', 'stage': stage,
                'branch_id': row['branch_id'], 'repeat_index': repeat,
                'selected_tokens': sum(row['loss_mask']),
                'seconds': score['seconds']}), flush=True)
    write(args.out/f'actor-{stage}-provenance.json', dict(
        stage=stage, source_data_sha256=source_hash,
        source_data_path=str(data_path), scored_rows=len(rows), repeats=repeats,
        mixed_group_count=evidence['mixed_group_count'],
        policy=policy.identity, execution=policy.execution,
        scoring='Current actor, teacher-forced complete sampled tokens; original behavior logprobs are unchanged'))


def final_patch(env):
    """Use the official runner's unsubmitted-working-tree fallback semantics."""
    from swegemma.harness.verification import strip_protected_files_from_patch
    started = time.monotonic()
    evidence = {'explicit_submit': env.ctx.patch_submitted, 'method': 'submit_patch'}
    patch = env.ctx.submitted_patch or ''
    if not env.ctx.patch_submitted:
        evidence['method'] = 'official_unsubmitted_working_tree_fallback'
        staged = env.manager.exec(env.sid, 'cd /workspace && git add -N .')
        diff = env.manager.exec(env.sid,
            'cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)')
        evidence.update(add_intent_exit_code=staged.exit_code, diff_exit_code=diff.exit_code)
        if diff.exit_code:
            raise RuntimeError('Cannot collect actual final working-tree patch: '+(diff.stderr or diff.stdout))
        patch = diff.stdout or ''
    evidence['raw_patch_bytes'] = len(patch.encode())
    patch = strip_protected_files_from_patch(patch, extra_protected_paths=env.task.test_files)
    evidence.update(evaluated_patch_bytes=len(patch.encode()), seconds=time.monotonic()-started)
    return patch, evidence


class TaskEnvironment:
    def __init__(self, asset, output, *, tool_calls=32, turns=32):
        from adk_submission import ModelRegistry
        from swegemma.budget import EvaluationBudget, HarnessLimits
        from swegemma.config import EvalConfig
        from swegemma.models import Task
        from swegemma.sandbox import SubprocessManager
        from competition.vfarl.replay.run_official import ExternalTrainTask
        self.asset, self.output = asset.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        payload = json.loads((asset/'task.json').read_text())
        task_cls = ExternalTrainTask if 'verification_target_files' in payload else Task
        fields = {field.name for field in dataclasses.fields(task_cls)}
        self.task = task_cls(**{key: value for key, value in payload.items() if key in fields})
        self.runtime = json.loads((asset/'runtime.json').read_text())
        historical = json.loads((asset/'run-summary.json').read_text())
        self.site_packages_mode = self.runtime.get('site_packages_mode') or historical.get('task_site_packages_mode') or 'pth'
        if self.site_packages_mode not in ('pth', 'copy'):
            raise ValueError('Unsupported task dependency placement')
        self.budget = EvaluationBudget(time_minutes=None, tool_calls=tool_calls, turns=turns)
        self.harness = HarnessLimits(command_timeout_seconds=90, verification_timeout_seconds=300)
        self.config = EvalConfig(tasks_path=asset/'task.json', snapshots_dir=output,
            results_dir=output, submission_dir=output, models=ModelRegistry(), sandbox='subprocess',
            budget=self.budget, harness=self.harness, graph_dir='', embeddings_dir='',
            enable_sandbox_testing=True)
        self.manager = SubprocessManager(base_dir=output/'sandboxes', system_site_packages=False)
        original_start, original_stop = self.manager.start, self.manager.stop
        self.phase, self.report_number = 'agent', 0

        def start():
            sid = original_start()
            paths = self.manager.sandboxes[sid]
            # Create from the actual base executable. Creating a venv from a
            # nested UV venv can retain an intermediate host-only symlink which
            # is deliberately absent from the isolated runtime.
            python = str(Path(self.runtime['python']).resolve(strict=True))
            subprocess.run([python, '-m', 'venv', '--clear', '--without-pip', str(paths['venv'])],
                           check=True, capture_output=True)
            source_root = paths['workspace']
            if self.runtime.get('source_import_alias_root'):
                # Keep the alias inside the snapshotted task root; a short path
                # preserves the original boltons import profile without an
                # untracked write outside the restored state.
                source_root = paths['root']/'source'
                source_root.symlink_to(paths['workspace'], target_is_directory=True)
            source = source_root / self.runtime.get('source_layout', '')
            dependencies = self.runtime['site_packages']
            for site in paths['venv'].glob('lib/python*/site-packages'):
                if self.site_packages_mode == 'copy':
                    # Some source tests use the real sysconfig purelib/platlib
                    # identity. Preserve the historical copy profile, without
                    # hardlinking a writable venv to the shared package cache.
                    shutil.copytree(dependencies, site, dirs_exist_ok=True, symlinks=True,
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                imports = [str(source)] + ([] if self.site_packages_mode == 'copy' else [dependencies])
                code = f"import sys; sys.path[:0] = {imports!r}"
                preload = self.runtime.get('source_preload_module')
                if preload:
                    if not all(part.isidentifier() for part in preload.split('.')):
                        raise ValueError('Invalid source preload module')
                    code += f'; import {preload}'
                (site/'_vfarl_environment.pth').write_text(code+'\n')
            # A .pth profile exposes modules but not installed console scripts.
            # Recreate the actual package entry points in this task's PATH so
            # model-issued `pytest` uses the same interpreter as `python -m pytest`.
            for distribution in importlib.metadata.distributions(path=[dependencies]):
                for entry in distribution.entry_points:
                    if entry.group != 'console_scripts' or not re.fullmatch(r'[A-Za-z0-9_.-]+', entry.name):
                        continue
                    executable = paths['venv']/'bin'/entry.name
                    if executable.exists() or executable.is_symlink():
                        continue
                    executable.write_text('#!/usr/bin/env python3\nimport sys\n'
                        'from importlib.metadata import EntryPoint\n'
                        f'sys.exit(EntryPoint(name={entry.name!r}, value={entry.value!r}, group="console_scripts").load()())\n')
                    executable.chmod(0o755)
            for name, target in self.runtime.get('source_console_scripts', {}).items():
                if (not re.fullmatch(r'[A-Za-z0-9_.-]+', name)
                        or name in ('python', 'python3', 'pytest', 'pip')
                        or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]*:[A-Za-z_][A-Za-z_0-9.]*', target)):
                    raise ValueError('Invalid source-declared console entry point')
                executable = paths['venv']/'bin'/name
                executable.write_text('#!/usr/bin/env python3\nimport sys\n'
                    'from importlib.metadata import EntryPoint\n'
                    f'sys.exit(EntryPoint(name={name!r}, value={target!r}, group="console_scripts").load()())\n')
                executable.chmod(0o755)
            return sid

        def stop(sid):
            paths = self.manager.sandboxes.get(sid, {})
            self.report_number += 1
            for report in paths.get('tmp', Path('/nonexistent')).glob('_swegemma_junit_*.xml'):
                shutil.copy2(report, self.output/f'{self.phase}-{self.report_number}-{report.name}')
            original_stop(sid)

        self.manager.start, self.manager.stop = start, stop
        self.sid = None

    def start(self):
        from swegemma.context import SwegemmaContext
        from swegemma.tools import create_tools
        self.sid = self.manager.start()
        workspace = self.manager.sandboxes[self.sid]['workspace']
        with tarfile.open(self.asset/'source-snapshot.tgz') as archive:
            archive.extractall(workspace, filter='data')
        baseline = self.manager.exec(self.sid, 'git rev-parse --verify _swegemma_baseline')
        if baseline.exit_code:
            raise RuntimeError('Original official source snapshot has no valid baseline commit')
        self.ctx = SwegemmaContext(docker_manager=self.manager, container_id=self.sid,
                                  task=self.task, budget=self.budget, harness=self.harness)
        available = create_tools(self.ctx)
        self.tools = {name: available[name] for name in sorted(SIX_TOOLS)}
        self.ctx.start_agent_session()
        return self.ctx

    async def verify(self, patch, label):
        from swegemma.harness.verification import verify_task
        self.phase = label
        result = await verify_task(self.manager, self.config, self.task,
            self.asset/'source-snapshot.tgz', agent_patch=patch, start_time=time.perf_counter())
        record = result.model_dump(mode='json')
        verification_error = record.get('error_message') or record.get('error')
        write(self.output/(label+'-official-verification.json'), record)
        # A setup/collection/timeout failure is unknown, never a zero reward.
        valid = (record.get('test_exit_code') in (0, 1)
                 and bool(record.get('test_output'))
                 and not str(verification_error or '').startswith(('Evaluation error:', 'Failed to apply test_patch:', 'Failed to apply agent patch', 'Missing test specification')))
        reports = sorted(self.output.glob(label+'-*-_swegemma_junit_*.xml'))
        observed_cases = []
        for report in reports:
            try:
                observed_cases.extend(ET.parse(report).getroot().iter('testcase'))
            except ET.ParseError:
                valid = False
        if not observed_cases or all(case.find('skipped') is not None for case in observed_cases):
            valid = False
        if record.get('test_exit_code') == 0 and not record.get('resolved'):
            valid = False  # missing/malformed/required-skipped JUnit is not a failure signal
        record['verification_complete'] = valid
        record['junit_files'] = [str(path) for path in reports]
        record['reward'] = int(bool(record['resolved'])) if valid else None
        write(self.output/(label+'-result.json'), record)
        return record

    def close(self):
        self.manager.cleanup_all()


async def prevalidate(args, entries):
    from competition.vfarl.l4x4.state import configure_isolation
    isolation = configure_isolation(args.isolation, hidden_paths=[args.assets.resolve(),
        args.out.resolve(), Path(__file__).resolve().parents[3]])
    write(args.out/'isolation.json', isolation)
    if args.isolation != 'observable' and isolation['effective_mode'] == 'observable':
        raise RuntimeError('Requested task validation isolation could not run: '+str(isolation.get('reason')))
    for entry in entries:
        key = safe_id(entry['instance_id'])
        asset = args.assets/key
        out = args.out/key
        result = {'instance_id': key, 'execution_isolation': isolation['effective_mode'],
                  'task_input_sha256': digest((asset/'task.json').read_bytes()),
                  'snapshot_sha256': digest((asset/'source-snapshot.tgz').read_bytes()),
                  'reviewed_patch_sha256': digest((asset/'submitted.patch').read_bytes()),
                  'runtime_sha256': runtime_digest(asset/'runtime.json'),
                  'runtime_hash_encoding': 'canonical-json-utf8-sort-keys-compact'}
        env = None
        try:
            env = TaskEnvironment(asset, out)
            baseline = await env.verify('', 'baseline')
            repaired = await env.verify((asset/'submitted.patch').read_text(), 'reviewed-repair')
            result.update(baseline=baseline, repaired=repaired,
                          usable=baseline['reward'] == 0 and repaired['reward'] == 1)
        except Exception as exc:
            result.update(usable=False, error=f'{type(exc).__name__}: {exc}')
            out.mkdir(parents=True, exist_ok=True)
            (out/'exception.txt').write_text(traceback.format_exc())
        finally:
            if env is not None:
                env.close()
        append(args.out/'prevalidation.jsonl', result)
        print(json.dumps({'phase': 'prevalidate', 'instance_id': key,
                          'usable': result['usable'], 'error': result.get('error'),
                          'baseline_exit': result.get('baseline', {}).get('test_exit_code'),
                          'repaired_exit': result.get('repaired', {}).get('test_exit_code'),
                          'repaired_error': result.get('repaired', {}).get('error_message') or result.get('repaired', {}).get('error'),
                          'repaired_output_tail': (result.get('repaired', {}).get('test_output') or '')[-2500:]
                          if not result['usable'] else None}), flush=True)


async def episode(env, policy, ids, args, out, *, ledger=None, first_action=None,
                  capture=False, seed=0, first_seed=None, train_focal_first_action=False):
    from competition.vfarl.l4x4.state import Snapshot
    out.mkdir(parents=True, exist_ok=True)
    ledger = ledger if ledger is not None else {'output_tokens': 0, 'turns': 0, 'context_limit': args.context}
    candidates, actions, stop = [], [], None

    def retain_candidates():
        # Full venv copies are real state. Keep only the two best pre-action
        # checkpoints while collecting, rather than multiplying a large task
        # environment by every turn. Raw actions/tool evidence remain intact.
        candidates.sort(key=focal_order)
        for item in candidates[2:]:
            shutil.rmtree(item['snapshot'])
        del candidates[2:]
    while ledger['turns'] < args.turns:
        room = min(args.context-len(ids), args.episode_tokens-ledger['output_tokens'])
        if room < 16:
            stop = 'context_or_output_budget_exhausted'
            break
        state = None
        # Focal recovery starts only after an actual failed tool response is
        # already in this prefix. Successful reads/searches alone do not
        # establish a recovery state, and future action failures cannot do so.
        recovery_budget = (room >= args.turn_tokens + 256
            and args.turns-ledger['turns'] >= 2
            and args.tool_calls-env.ctx.tool_calls_used >= 2)
        if (capture and recovery_budget and ledger.get('tool_observations', 0) > 0
                and ledger.get('tool_failures_observed', 0) > 0):
            try:
                state = Snapshot.capture(env.manager, env.sid, env.ctx, ids, ledger,
                    policy.identity, out/f'state-{ledger["turns"]:03d}')
            except Exception as exc:
                raise PolicyProtocolError('Exact pre-action state capture failed') from exc
        env.ctx.record_llm_call()
        action_seed = first_seed if first_seed is not None and not actions else seed+ledger['turns']
        print(json.dumps({'phase': 'policy_request', 'instance_id': env.task.instance_id,
            'attempt': out.name, 'turn': ledger['turns'], 'prompt_tokens': len(ids),
            'max_new_tokens': min(room, args.turn_tokens),
            'original_action_replay': first_action is not None}), flush=True)
        try:
            action = first_action if first_action is not None else policy.sample(ids,
                max_new_tokens=min(room, args.turn_tokens), thinking_budget=1024,
                seed=action_seed, stop_token_ids=policy.vfarl_stop_ids,
                compute_reference=train_focal_first_action and not actions)
        except Exception as exc:
            raise PolicyProtocolError('Current-policy sampling failed') from exc
        replayed = first_action is not None
        first_action = None
        if action['prompt_token_ids'] != ids:
            raise PolicyProtocolError('Action prompt differs from restored exact token prefix')
        actions.append(action)
        append(out/'actions.jsonl', dict(action, original_action_replay=replayed))
        print(json.dumps({'phase': 'policy_action', 'instance_id': env.task.instance_id,
            'attempt': out.name, 'turn': ledger['turns'], 'original_action_replay': replayed,
            **{key: action.get(key) for key in ('prompt_tokens', 'completion_tokens',
                'thought_tokens', 'finish_reason', 'seconds', 'peak_allocated_bytes')}}), flush=True)
        ids = ids + action['completion_token_ids']
        ledger['output_tokens'] += len(action['completion_token_ids'])
        ledger['turns'] += 1
        if state is not None:
            candidates.append({'snapshot': str(out/f'state-{ledger["turns"]-1:03d}'),
                               'manifest': state, 'action': action,
                               'checkpoint_turn': ledger['turns']-1,
                               'tool_observations': ledger['tool_observations'],
                               'tool_failures_observed': ledger['tool_failures_observed'],
                               'last_tool_feedback': dict(ledger.get('last_tool_feedback', {})),
                               'observed_tool_error': bool(ledger.get('last_tool_error', False)),
                               'remaining_token_capacity': room,
                               'has_recovery_budget': recovery_budget})
        if action['finish_reason'] not in ('tool_calls', 'stop'):
            if state is not None:
                candidates[-1]['response_truncated'] = True
            if len(ids) >= args.context:
                stop = 'context_budget_exhausted'
            elif ledger['output_tokens'] >= args.episode_tokens:
                stop = 'episode_output_budget_exhausted'
            else:
                stop = 'single_response_token_budget_exhausted'
            retain_candidates()
            break
        try:
            call = parse_call(action['raw_text'])
        except ValueError as exc:
            if state is not None:
                candidates[-1]['action_protocol_error'] = True
            retain_candidates()
            stop = 'malformed_action: ' + str(exc)
            break
        if call is None:
            retain_candidates()
            stop = 'model_text_termination'
            break
        name, arguments = call
        if name not in env.tools:
            if state is not None:
                candidates[-1]['action_protocol_error'] = True
            retain_candidates()
            stop = 'unknown_tool: ' + name
            break
        if env.ctx.patch_submitted:
            stop = 'tool_call_after_submission'
            break
        started = time.monotonic()
        try:
            response = env.tools[name](**arguments)
        except Exception as exc:
            # An actually raised tool exception is an observed failure, not a
            # fabricated success/return. Retain it and end this attempt.
            append(out/'tool-events.jsonl', {'tool': name, 'arguments': arguments,
                   'exception': f'{type(exc).__name__}: {exc}'})
            if state is not None:
                candidates[-1]['action_tool_error'] = True
            retain_candidates()
            stop = 'tool_exception'
            break
        append(out/'tool-events.jsonl', {'tool': name, 'arguments': arguments,
            'response': response, 'seconds': time.monotonic()-started,
            'tool_calls_used': env.ctx.tool_calls_used})
        response_failed = isinstance(response, dict) and response.get('status') == 'error'
        if candidates and candidates[-1]['action'] is action:
            candidates[-1]['action_tool_error'] = response_failed
        retain_candidates()
        try:
            ids += tool_suffix(policy.tokenizer, name, arguments, response)
        except Exception as exc:
            raise PolicyProtocolError('Actual tool response serialization failed') from exc
        ledger['tool_observations'] = ledger.get('tool_observations', 0) + 1
        ledger['tool_failures_observed'] = ledger.get('tool_failures_observed', 0) + int(response_failed)
        ledger['last_tool_error'] = response_failed
        ledger['last_tool_feedback'] = dict(tool=name, failed=response_failed,
            response_sha256=digest(json.dumps(response, ensure_ascii=False).encode()),
            observation_index=ledger['tool_observations'], observed_after_turn=ledger['turns'])
        if env.ctx.tool_calls_used >= args.tool_calls:
            stop = 'tool_budget_exhausted'
            break
    retain_candidates()
    env.ctx.stop_agent_session()
    patch, patch_capture = final_patch(env)
    (out/'submitted.patch').write_text(env.ctx.submitted_patch or '')
    (out/'evaluated.patch').write_text(patch)
    write(out/'patch-capture.json', patch_capture)
    result = await env.verify(patch, out.name)
    summary = {'instance_id': env.task.instance_id, 'reward': result['reward'],
        'verified': result['verification_complete'], 'termination': stop or 'submitted',
        'actions': len(actions), 'ledger': dict(ledger), 'policy_id': policy.identity['policy_id'],
        'verification': str(env.output/(out.name+'-result.json')),
        'patch_submitted': env.ctx.patch_submitted, 'patch_capture': patch_capture,
        'final_context_tokens': len(ids)}
    write(out/'episode.json', summary)
    print(json.dumps({'phase': 'episode_result', 'attempt': out.name,
        **{key: summary[key] for key in ('instance_id', 'reward', 'verified',
            'termination', 'actions', 'patch_submitted', 'final_context_tokens')}}), flush=True)
    return summary, candidates, actions


async def collect(args, entries, questions):
    if not entries and args.score_focal_data is None:
        (args.out/'collection.jsonl').touch()
        print(json.dumps({'phase': 'empty_partition', 'worker': args.worker_index,
                          'selected_tasks': 0}), flush=True)
        return
    from competition.vfarl.l4x4.state import Snapshot, configure_isolation
    from training.vfarl_policy import build_policy, read_config
    from training.common import TrainingError
    model_config, _ = read_config(args.config)
    if model_config['max_sequence_length'] != args.context:
        raise ValueError('Sampling context and training config must match; use an explicit context-specific config')
    source_root = Path(__file__).resolve().parents[3]
    isolation = configure_isolation(args.isolation,
        hidden_paths=[args.assets.resolve(), args.out.resolve(), source_root])
    write(args.out/'isolation.json', isolation)
    if isolation['effective_mode'] == 'observable':
        raise RuntimeError('Task shell isolation is unavailable; do not load a GPU model while host judge assets are readable. '+str(isolation.get('reason')))
    print(json.dumps({'phase': 'model_loading', 'context': args.context,
                      'policy_gpus': args.policy_gpus,
                      'adapter_run': str(args.adapter_run) if args.adapter_run else None}), flush=True)
    policy = build_policy(args.config, backend=args.policy_backend,
                          adapter_run=args.adapter_run,
                          inference_device_count=args.policy_gpus)
    print(json.dumps({'phase': 'model_ready', 'policy_id': policy.identity['policy_id'],
                      'reference_id': policy.identity['reference_id'],
                      'backend': args.policy_backend,
                      'adapter_sha256': policy.identity['policy_identity']['adapter_sha256'],
                      'nf4_cache_manifest_sha256': policy.identity['policy_identity']['nf4_cache_manifest_sha256'],
                      'probability_execution_sha256': policy.identity['policy_identity']['probability_execution_sha256'],
                      'inference_device_count': policy.execution['inference_device_count'],
                      'input_device': policy.execution['input_device']}), flush=True)
    write(args.out/'policy-execution.json', policy.execution)
    # Identity contains exact base/cache, initialized LoRA, tokenizer/template
    # and sampler configuration. Tool schema is bound by the collector below.
    tools = json.loads(args.tools.read_text())
    if isinstance(tools, dict):
        tools = tools.get('tools', tools.get('functions', tools))
    tools = [tool if 'function' in tool else {'type': 'function', 'function': tool} for tool in tools]
    if digest(args.tools.read_bytes()) != policy.identity['policy_identity']['tool_schema_sha256']:
        raise ValueError('Actual sampled tool schema differs from the bound policy configuration')
    policy.vfarl_stop_ids = [policy.tokenizer.convert_tokens_to_ids(x) for x in
                            ('<tool_call|>', '<turn|>', '<|end|>')]
    policy.vfarl_stop_ids = [x for x in policy.vfarl_stop_ids if isinstance(x, int) and x != policy.tokenizer.unk_token_id]
    write(args.out/'policy.json', policy.identity)
    if args.score_focal_data is not None:
        score_focal_rows(policy, args, args.score_focal_data, stage='after')
    if not entries:
        (args.out/'collection.jsonl').touch()
        print(json.dumps({'phase': 'empty_partition', 'worker': args.worker_index,
                          'selected_tasks': 0, 'focal_scoring_completed': True}), flush=True)
        return
    preparation = {row['instance_id']: row for row in
                   (json.loads(line) for line in args.prevalidation.read_text().splitlines())}
    global_indices = {row['instance_id']: i for i, row in enumerate(json.loads(args.selection.read_text())['tasks'])}
    usable = {key for key, row in preparation.items() if row.get('usable')}
    queue_entries = [row for row in entries if row['instance_id'] in usable]
    queue = (TaskQueue(args.task_queue, [row['instance_id'] for row in queue_entries],
        selection_sha256=digest(args.selection.read_bytes()), maximum_states=args.max_states,
        stop_after_mixed_groups=args.stop_after_mixed_groups,
        stop_after_tasks=args.stop_after_tasks,
        execution_contract=dict(config_sha256=digest(args.config.read_bytes()),
            policy_backend=args.policy_backend, context=args.context, draws=args.draws,
            branches=args.branches, turns=args.turns, tool_calls=args.tool_calls,
            turn_tokens=args.turn_tokens, episode_tokens=args.episode_tokens)) if args.task_queue else None)
    if queue is not None:
        queue.bind_policy(policy.identity)
        entries = queue.entries(args.worker_index, {row['instance_id']: row for row in queue_entries})
    states = 0
    def can_start_state():
        return states < args.max_states and (queue is None or queue.state_available())
    for position, entry in enumerate(entries):
        key = safe_id(entry['instance_id'])
        if key not in usable:
            append(args.out/'collection.jsonl', {'instance_id': key, 'sampled': False,
                'reason': 'Linux original baseline/reviewed-repair verification not usable'})
            if queue is not None:
                queue.finish_task(key, args.worker_index)
            continue
        if preparation[key].get('execution_isolation') != isolation['effective_mode']:
            raise ValueError('Task was not verified in the actual collection isolation environment: '+key)
        for filename, field in [('task.json', 'task_input_sha256'),
                                ('source-snapshot.tgz', 'snapshot_sha256')]:
            if digest((args.assets/key/filename).read_bytes()) != preparation[key].get(field):
                raise ValueError('Linux verification belongs to different task assets: '+key)
        for draw in range(args.draws):
            env = None
            retained_snapshot = None
            out = args.out/key/f'draw-{draw}'
            phase = 'environment_setup'
            try:
                env = TaskEnvironment(args.assets/key, out, tool_calls=args.tool_calls, turns=args.turns)
                env.start()
                from swegemma.harness.agent_runner import build_agent_prompt
                # The source issue stays fixed. Rebuild only the official
                # runtime/budget instructions so 60min/100tools/500turns from
                # historical demonstrations are not silently misrepresented.
                actual_prompt = build_agent_prompt(env.task, env.config,
                    declared_tools=sorted(env.tools), enable_sandbox_testing=True)
                actual_prompt += (f'\n\n## This training run\nThe total context limit is {args.context} tokens, '
                    f'including all history and the next response. The session output budget is '
                    f'{args.episode_tokens} tokens; each response allows at most {args.turn_tokens} tokens, '
                    'including up to 1024 thought tokens. Full histories are never truncated.\n')
                write(out/'prompt-provenance.json', {'original_prompt_sha256': digest(questions[key].encode()),
                    'actual_prompt_sha256': digest(actual_prompt.encode()),
                    'change': 'Official build_agent_prompt with actual run budgets plus explicit token limits'})
                (out/'actual-prompt.txt').write_text(actual_prompt)
                phase = 'policy_prompt'
                ids = policy.render_prompt([{'role': 'user', 'content': actual_prompt}], tools)
                base_seed = 20261026 + global_indices[key]*10000 + draw*1000
                phase = 'policy_episode'
                summary, candidates, _ = await episode(env, policy, ids, args, out/'original',
                    capture=can_start_state(), seed=base_seed)
                environment_control = None
                if (summary['verified'] and summary['reward'] in (0, 1)
                        and candidates and can_start_state()):
                    # Confirm this actual GPU worker's task environment before
                    # comparing an observed tool-failure state. The original
                    # task may later recover and succeed; its terminal reward
                    # does not determine whether this real state is eligible.
                    # The reviewed repair remains host-only.
                    control = await env.verify((env.asset/'submitted.patch').read_text(),
                                               'focal-environment-control')
                    environment_control = str(env.output/'focal-environment-control-result.json')
                    summary['environment_control'] = environment_control
                    if control['reward'] != 1:
                        summary.update(reward=None, verified=False,
                            reward_exclusion='Reviewed repair no longer verifies in the actual collection environment')
                    write(out/'original'/'episode.json', summary)
                append(args.out/'collection.jsonl', dict(summary, sampled=True, draw=draw))
                # Every candidate prefix already contains a real tool error.
                # Include naturally recovered original episodes as well as
                # terminal failures; do not choose states by branch outcomes.
                fork = (summary['verified'] and summary['reward'] in (0, 1)
                        and candidates and can_start_state())
                if fork:
                    # Prefer feedback immediately after an actual error with
                    # remaining recovery budget.
                    candidate = min(candidates, key=focal_order)
                    if queue is not None:
                        fork = queue.reserve_state(candidate['manifest']['state_id'], key,
                                                   draw, args.worker_index)
                if fork:
                    retained_snapshot = candidate['snapshot']
                    rows = []
                    for branch in range(args.branches):
                        ledger = {}
                        phase = 'state_restoration'
                        restored = Snapshot.restore(Path(candidate['snapshot']), env.manager,
                            env.sid, env.ctx, ledger=ledger, policy_identity=policy.identity,
                            prompt_token_ids=candidate['action']['prompt_token_ids'])
                        manifest = candidate['manifest']
                        restored_ids = candidate['action']['prompt_token_ids']
                        branch_dir = out/f'branch-{branch}'
                        write(branch_dir/'restore.json', restored)
                        phase = 'branch_episode'
                        branch_summary, _, actions = await episode(env, policy, restored_ids, args,
                            branch_dir, ledger=ledger,
                            first_action=candidate['action'] if branch == 0 else None,
                            seed=base_seed+500, first_seed=base_seed+500+branch*100,
                            train_focal_first_action=branch != 0)
                        if actions:
                            row = dict(actions[0], state_id=manifest['state_id'],
                                branch_id=f'{key}-{draw}-{branch}', instance_id=key,
                                kind='original_action_replay' if branch == 0 else 'sampled_focal_action',
                                on_policy=branch != 0, verified=branch_summary['verified'],
                                reward=branch_summary['reward'],
                                origin_episode_reward=summary['reward'],
                                focal_selection_basis='observed tool failure, regardless of original final result',
                                environment_control_verified=True,
                                environment_control_result=environment_control,
                                checkpoint_turn=candidate['checkpoint_turn'],
                                checkpoint_tool_observations=candidate['tool_observations'],
                                checkpoint_tool_failures_observed=candidate['tool_failures_observed'],
                                checkpoint_last_tool_feedback=candidate['last_tool_feedback'],
                                restoration_scope=manifest['restoration_scope'],
                                restored_state_evidence=str(branch_dir/'restore.json'),
                                verification=branch_summary['verification'])
                            rows.append(row)
                            append(args.out/'focal.jsonl', row)
                    states += 1
                    state_result = {'state_id': candidate['manifest']['state_id'],
                        'instance_id': key, 'branches': len(rows),
                        'rewards': [row['reward'] for row in rows],
                        'origin_episode_reward': summary['reward'],
                        'focal_selection_basis': 'observed tool failure, regardless of original final result',
                        'environment_control': environment_control,
                        'checkpoint_observed_tool_error': candidate.get('observed_tool_error', False),
                        'checkpoint_turn': candidate['checkpoint_turn'],
                        'checkpoint_tool_observations': candidate['tool_observations'],
                        'checkpoint_tool_failures_observed': candidate['tool_failures_observed'],
                        'checkpoint_last_tool_feedback': candidate['last_tool_feedback'],
                        'checkpoint_has_recovery_budget': candidate['has_recovery_budget'],
                        'checkpoint_remaining_token_capacity': candidate['remaining_token_capacity'],
                        'restoration_scope': candidate['manifest']['restoration_scope']}
                    append(args.out/'state-results.jsonl', state_result)
                    if queue is not None:
                        from training.train_vfarl import prepare_rows
                        _, prepared = prepare_rows(args.out/'focal.jsonl', args.context)
                        accepted = next((row for row in prepared['groups']
                            if row['state_id'] == state_result['state_id']), None)
                        queue.finish_state(state_result['state_id'],
                            mixed=bool(accepted and accepted['mixed_outcomes']),
                            branches=accepted['branches'] if accepted else [],
                            evidence_path=args.out/'focal.jsonl',
                            evidence_sha256=digest((args.out/'focal.jsonl').read_bytes()))
                    print(json.dumps(dict(state_result, phase='focal_group')), flush=True)
            except Exception as exc:
                out.mkdir(parents=True, exist_ok=True)
                (out/'exception.txt').write_text(traceback.format_exc())
                fatal = (isinstance(exc, (TrainingError, PolicyProtocolError))
                    or phase in ('policy_prompt', 'state_restoration')
                    or 'out of memory' in str(exc).lower())
                failure = {'instance_id': key, 'draw': draw,
                    'sampled': any(out.rglob('actions.jsonl')), 'attempt_complete': False,
                    'error': f'{type(exc).__name__}: {exc}', 'failed_phase': phase,
                    'fatal': fatal}
                if exc.__cause__ is not None:
                    failure['cause'] = f'{type(exc.__cause__).__name__}: {exc.__cause__}'
                append(args.out/'collection.jsonl', failure)
                print(json.dumps(dict(failure, phase='collection_error')), flush=True)
                if fatal:
                    raise  # Shared framework/capacity failures cannot yield valid RL data.
            finally:
                if env is not None:
                    env.close()
                # Only the chosen focal state needs a durable full environment
                # copy; keep all real action/tool/verifier records for every draw.
                for checkpoint in (out/'original').glob('state-*'):
                    if checkpoint.is_dir() and str(checkpoint) != retained_snapshot:
                        shutil.rmtree(checkpoint)
            print(json.dumps({'phase': 'collection', 'instance_id': key, 'draw': draw,
                              'focal_states': states}), flush=True)
        if queue is not None:
            queue.finish_task(key, args.worker_index)
    if args.adapter_run is None:
        score_focal_rows(policy, args, args.out/'focal.jsonl', stage='before')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'questions', 'assets', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--policy-backend', choices=('native', 'unsloth'), default='native')
    parser.add_argument('--policy-gpus', type=int, choices=(1, 2), default=1,
                        help='Visible GPUs used by each real sampling model; FSDP training is separate')
    parser.add_argument('--adapter-run', type=Path,
                        help='Completed training run for a separate post-update rollout comparison')
    parser.add_argument('--score-focal-data', type=Path,
                        help='Merged real mixed focal rows to score with the loaded updated actor; no sampling or reward is manufactured')
    parser.add_argument('--tools', type=Path, default=Path('competition/vfarl/six-tools.json'))
    parser.add_argument('--prevalidate', action='store_true')
    parser.add_argument('--prevalidation', type=Path)
    parser.add_argument('--context', type=int, choices=(4096, 8192, 16384), default=4096)
    parser.add_argument('--draws', type=int, choices=(1, 2), default=1)
    parser.add_argument('--branches', type=int, choices=(2, 3), default=3,
                        help='Total including original replay; use 3 for two independent sampled actions and a possible PPO advantage')
    parser.add_argument('--max-states', type=int, default=5)
    parser.add_argument('--turns', type=int, default=32)
    parser.add_argument('--tool-calls', type=int, default=32)
    parser.add_argument('--turn-tokens', type=int, default=1536)
    parser.add_argument('--episode-tokens', type=int, default=8192)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--task-queue', type=Path,
                        help='Shared host-only dynamic queue; all workers receive the same frozen task list')
    parser.add_argument('--stop-after-mixed-groups', type=int, default=0,
                        help='Dynamic queue: stop new task/state claims after N valid distinct-action mixed groups; retain in-flight results')
    parser.add_argument('--stop-after-tasks', type=int, default=0,
                        help='Dynamic queue: maximum globally claimed tasks, including in-flight tasks; zero means the full fixed queue')
    parser.add_argument('--only-task', action='append', default=[],
                        help='Run named members of the frozen 100-task train pool, e.g. an environment repair or separate 8K continuation')
    parser.add_argument('--task-limit', type=int,
                        help='Optional first N selected tasks before worker partition, used for held-in post-update comparison')
    parser.add_argument('--isolation', choices=('bwrap', 'chroot', 'observable'), default='chroot')
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count or args.max_states < 0:
        parser.error('Invalid worker or focal-state count')
    if min(args.stop_after_mixed_groups, args.stop_after_tasks) < 0:
        parser.error('Dynamic stopping budgets must be nonnegative')
    if (args.stop_after_mixed_groups or args.stop_after_tasks) and not args.task_queue:
        parser.error('Dynamic stopping budgets require --task-queue')
    if args.task_queue and (args.prevalidate or args.adapter_run is not None):
        parser.error('The dynamic training queue is not used for CPU preparation or fixed post-update comparisons')
    if args.task_limit is not None and args.task_limit < 1:
        parser.error('--task-limit must be positive')
    if args.score_focal_data is not None and (args.prevalidate or args.adapter_run is None):
        parser.error('--score-focal-data requires a post-update actor loaded with --adapter-run')
    if args.out.exists():
        parser.error('Output already exists; preserve the previous real run')
    args.out.mkdir(parents=True)
    selection = json.loads(args.selection.read_text())
    entries = selection['tasks']
    if len(entries) != 100 or len({x['instance_id'] for x in entries}) != 100:
        parser.error('Expected the frozen 100 unique S600 training tasks')
    if any(x['physical_split'] != 'train' for x in entries):
        parser.error('Only train tasks are permitted')
    questions = {}
    for line in args.questions.read_text().splitlines():
        row = json.loads(line)
        if digest(row['official_initial_user_prompt'].encode()) != row['official_initial_user_prompt_sha256']:
            raise ValueError('Question content hash mismatch')
        questions[row['instance_id']] = row['official_initial_user_prompt']
    for row in entries:
        if digest(questions[row['instance_id']].encode()) != row['official_initial_user_prompt_sha256']:
            raise ValueError('Question differs from the selected S600 prompt')
        asset = args.assets/safe_id(row['instance_id'])
        if digest((asset/'task.json').read_bytes()) != row['task_input_sha256']:
            raise ValueError('Task definition differs from frozen source identity')
    if args.only_task:
        selected = set(args.only_task)
        if not selected <= {row['instance_id'] for row in entries}:
            parser.error('--only-task must remain inside the frozen S600 train100 pool')
        entries = [row for row in entries if row['instance_id'] in selected]
    if args.task_limit is not None:
        entries = entries[:args.task_limit]
    if args.task_queue is None:
        entries = entries[args.worker_index::args.worker_count]
    write(args.out/'execution.json', {'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                    'selection_sha256': digest(args.selection.read_bytes())})
    if args.prevalidate:
        asyncio.run(prevalidate(args, entries))
    else:
        if not args.config or not args.prevalidation:
            parser.error('Collection requires --config and actual Linux --prevalidation')
        asyncio.run(collect(args, entries, questions))


if __name__ == '__main__':
    main()
