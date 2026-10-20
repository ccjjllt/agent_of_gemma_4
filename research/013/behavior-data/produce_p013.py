"""P013 behavior-data producer: repair the P012 false-completion handover; gate behavior claims on real evidence.

Two modes (outputs land ONLY under research/013/behavior-data/, never under frozen dirs):

  reissue  -- read-only on frozen P012 outputs (research/012/training-data/outputs/);
              re-emit the same local-edit transactions with the corrected handover
              (requirement_done:False, verification_status:not_run, needs_other_route:None)
              and supervision targets restricted to edit_file assistant ordinals only.
              The tail handover stays in the transcript (corrected, unsupervised) for
              accurate state reporting. behavior_verified stays False, task_resolved None.
              Validation-only in this session (--admit is refused here).

  behavior -- live canonical replay of batch-01 evidence dirs via the unchanged official
              v31 tool functions (read_file / run_command / edit_file / get_status):
              read -> pre-fix failure -> full edit -> same-criterion post-fix pass ->
              verified handover. A transaction is marked behavior_verified:true ONLY if
              the live session itself observes pre status=error AND post status=ok on the
              IDENTICAL command, with the edited module file under the live workspace.
              Any deviation -> attempt recorded as refused, nothing admitted.

Hard guards (both modes):
  - never reads heldout / reserved / dev materials (explicit reserved-repo set +
    task_splits.json dev/heldout/explore ids + P009/P011 exclusion lists);
  - training sources only from frozen train usage (P009 data-usage.json admitted,
    P011 data-usage-full.json selected order, P012 frozen outputs for reissue);
  - never writes outside research/013/behavior-data/.

Chat-codec note: target_assistant_steps are zero-based ASSISTANT ordinals
(see training/encoding/gemma_tool_chat.py line ~360:
"target_assistant_steps needs zero-based assistant ordinals").
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P006 = ROOT / 'research/006/data-replay-20261013/replay.py'
P009 = ROOT / 'research/009/training-data-20261018'
P011 = ROOT / 'research/011/data-scaling-20261018'
P012 = ROOT / 'research/012/training-data'

PROTOCOL_ID = 'p013-behavior-v1'
PRODUCER_REVISION = 'p013-v1'
REISSUE_PROTOCOL_ID = 'p012-repair-v1'  # reissue keeps the original protocol id; only handover/targets fixed

spec = importlib.util.spec_from_file_location('p006_replay', P006)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
_TOOLS6 = R.official_tools()
assert 'submit_patch' in _TOOLS6
TOOLS = {k: v for k, v in _TOOLS6.items() if k != 'submit_patch'}
TOOL_NAMES_5 = ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status']
assert sorted(TOOLS) == sorted(TOOL_NAMES_5), sorted(TOOLS)
TOOLS_SRC = ROOT / 'research/012/agent/repair_tools_v1.json'
TOOLS_CANONICAL_SHA = '91ec82b15fe0961b3ef9e686fac268706ce954a69cabaa0d705c003d41088e267'

# Repos that must never enter training materialization (P009+P011 exclusion union:
# dev-panel sources, external-8 candidates, heldout-adjacent). Normalized lowercase.
RESERVED_REPOS = {
    'conan-io/conan', 'cookiecutter/cookiecutter', 'encode/httpx', 'fastapi/fastapi',
    'keleshev/schema', 'kurtmckee/feedparser', 'modin-project/modin', 'pallets/click',
    'pallets/markupsafe', 'pandas-dev/pandas', 'psf/requests', 'pydicom/pydicom',
    'pytest-dev/iniconfig', 'textualize/rich', 'tkrajina/gpxpy',
}

FROZEN_WRITE_ROOTS = (P009, P011, P012, ROOT / 'research/006')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _refuse_outside_outdir(path):
    dest = (HERE / path).resolve()
    if not dest.is_relative_to(HERE.resolve()):
        raise ValueError('Refusing to write outside research/013/behavior-data: ' + str(path))
    for frozen in FROZEN_WRITE_ROOTS:
        if dest.is_relative_to(frozen.resolve()):
            raise ValueError('Refusing to write under frozen dir: ' + str(path))
    return dest


def load_task_split_denylist():
    """Instance ids from the official train/dev/heldout split that must never be trained on."""
    splits = json.loads((ROOT / 'research/task_splits.json').read_text())
    denied = {r['instance_id'] for r in splits['records'] if r['split'] != 'train'}
    denied |= set(splits.get('dev_panel_ids', []))
    return denied


def guard_train_source(repo, instance_id, denylist):
    if (repo or '').lower() in RESERVED_REPOS:
        raise ValueError('Reserved repo refused: ' + str(repo))
    if instance_id in denylist:
        raise ValueError('Non-train split instance refused: ' + str(instance_id))


# ---------------------------------------------------------------- reissue mode

def corrected_handover(old):
    h = copy.deepcopy(old)
    h['requirement_done'] = False
    h['verification_status'] = 'not_run'
    h['requirement_scope'] = ('local production edits applied and syntactically parsed; '
                              'full-task validation NOT performed in this session')
    h['verification'] = {
        'observation': old.get('verification', {}).get('observation', 'get_status'),
        'status': old.get('verification', {}).get('status'),
        'scope': ('post-edit local patch status of the same workspace; no repository test '
                  'suite or whole-issue oracle executed'),
    }
    h['unresolved'] = ['no whole-issue oracle executed; task_resolved=null',
                       'behavior recovery chain to be upgraded by a later step, not fabricated here']
    h['needs_other_route'] = None  # unknown: no whole-issue oracle executed
    return h


def reissue_one(row, denylist):
    guard_train_source(row['repo'], row['instance_id'], denylist)
    out = P012 / 'outputs' / row['instance_id']
    tx = json.loads((out / 'transaction.json').read_text())
    assert tx.get('producer_revision') == 'p012-local-2', tx.get('producer_revision')
    assert tx.get('protocol_id') == REISSUE_PROTOCOL_ID, tx.get('protocol_id')
    messages = copy.deepcopy(tx['messages'])
    assert messages and messages[-1].get('role') == 'assistant' and not messages[-1].get('tool_calls')
    handover = corrected_handover(json.loads((out / 'handover.json').read_text()))
    messages[-1] = {'role': 'assistant', 'content': json.dumps(handover, ensure_ascii=False, indent=2),
                    'tool_calls': []}
    asst_positions = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    pos_of = {msg_idx: ordinal for ordinal, msg_idx in enumerate(asst_positions)}
    targets = []
    for i in asst_positions:
        names = [(tc.get('function') or {}).get('name') for tc in (messages[i].get('tool_calls') or [])]
        # P013 fix: supervise ONLY edit_file action steps. The (corrected) tail
        # handover and get_status observations are never supervised.
        if names and all(n == 'edit_file' for n in names):
            targets.append(pos_of[i])
    if not targets:
        return dict(row, status='reissue_refused', reason='no edit_file assistant step found')
    new_tx = copy.deepcopy(tx)
    new_tx.update({
        'sample_id': tx['sample_id'], 'transaction_id': tx['transaction_id'],
        'producer_revision': PRODUCER_REVISION, 'reissued_from': 'p012-local-2',
        'protocol_id': REISSUE_PROTOCOL_ID,
        'messages': messages, 'target_assistant_steps': sorted(targets),
        'handover': handover,
        'observed_outcome': dict(tx['observed_outcome'],
                                 behavior_verified=False, task_resolved=None),
    })
    dest = _refuse_outside_outdir(Path('outputs-reissue') / row['instance_id'] / 'transaction.json')
    dump(dest, new_tx)
    dump(dest.parent / 'handover.json', handover)
    return dict(row, status='reissued_corrected', edits=len(targets),
                transaction=str(dest.relative_to(HERE)))


def run_reissue(limit, admit):
    if admit:
        raise ValueError('reissue --admit refused in this session: validation-only (no frozen-train change)')
    usage = json.loads((P012 / 'data-usage-p012.json').read_text())
    denylist = load_task_split_denylist()
    results = []
    for row in usage['selected'][:limit]:
        try:
            results.append(reissue_one(row, denylist))
        except Exception as e:  # noqa: BLE001 - record refusal, never fabricate
            results.append(dict(row, status='reissue_refused', reason=type(e).__name__ + ': ' + str(e)))
    dump(HERE / 'reissue-validation.json', {'admitted': False, 'results': results})
    return results


# --------------------------------------------------------------- behavior mode

def ensure_python_shim():
    """The frozen P009 behavior commands invoke `python` (bare) and rely on the
    ambient third-party packages. The official LocalTransport (a) builds PATH from
    dirname(sys.executable), which on this CPU box has no `python` alias, and
    (b) scrubs the environment with PYTHONNOUSERSITE=1, which hides this box's
    user site-packages (where pip installs land; no sudo here). Both break live
    replay with exit-127 / ModuleNotFoundError even though the identical command
    passes in a normal shell. Additionally the user site contains an obsolete
    `pathlib` 1.0.1 backport that would shadow the stdlib via PYTHONPATH, so the
    producer refuses to run unless that backport is absent (uninstalled 2026-10-20;
    nothing in the pinned stack imports it).
    Repair PATH/python resolution (harness setup, documented in replay_boundary)
    while keeping the behavior command byte-identical to manual verification:
    a `python` wrapper script execs the SAME interpreter (version+sys.path verified
    identical to the manual-verification `python3`) with the user site-packages
    appended AFTER the command's own PYTHONPATH, so workspace sources always win.
    """
    import subprocess as _sp
    manual = _sp.run(['python3', '-c', 'import sys; print(sys.version.split()[0])'],
                     capture_output=True, text=True)
    producer = '.'.join(str(x) for x in sys.version_info[:3])
    if manual.stdout.strip() != producer:
        raise ValueError('interpreter mismatch: shell python3=%s producer=%s; refusing live replay'
                         % (manual.stdout.strip(), producer))
    shimdir = HERE / 'shim-bin'
    shimdir.mkdir(parents=True, exist_ok=True)
    link = shimdir / 'python'
    if link.is_symlink() or link.exists():
        link.unlink()
    import site as _site
    usersite = _site.getusersitepackages()
    link.write_text('#!/bin/bash\n# P013 harness setup: same interpreter as manual verification,\n'
                    '# user site-packages appended AFTER the command PYTHONPATH so live\n'
                    '# workspace sources always win. Exception: the pydantic live workspace\n'
                    '# needs pydantic-core 2.30.0 (pinned source era) while the shared user\n'
                    '# site carries 2.46.5 (kept healthy for the deepdiff trajectory, whose\n'
                    '# pinned helper imports the installed pydantic dist). The vendored 2.30.0\n'
                    '# wheel (research/013/behavior-data/vendor/) is exposed ONLY under the\n'
                    '# pydantic live workspace, still after the command PYTHONPATH so the\n'
                    '# pinned pydantic source itself always wins. See ensure_python_shim.\n'
                    'EXTRA=":%s"\n'
                    'case "$PWD" in\n'
                    '  */research/013/behavior-data/outputs/*pydantic*/workspace*)\n'
                    '    EXTRA=":%s%s$EXTRA" ;;\n'
                    'esac\n'
                    'export PYTHONPATH="${PYTHONPATH}${EXTRA}"\nexec "%s" "$@"\n'
                    % (usersite, HERE / 'vendor' / 'pydantic-core-2.30.0',
                       os.pathsep, sys.executable))
    link.chmod(0o755)
    os.environ['PATH'] = str(shimdir) + os.pathsep + os.environ.get('PATH', '')
    return str(shimdir), producer


def run_behavior(keys, timeout_seconds=300, batch='batch-01'):
    """Live canonical replay of batch evidence (batch-01, batch-02 or batch-03). Returns (admitted, refused)."""
    shimdir, interpreter = ensure_python_shim()
    denylist = load_task_split_denylist()
    admitted, refused = [], []
    for key in keys:
        ev = json.loads((HERE / batch / key / 'evidence.json').read_text())
        try:
            tx = behavior_one(ev, denylist, timeout_seconds, batch)
            admitted.append(tx)
        except Exception as e:  # noqa: BLE001 - record refusal with raw reason
            refused.append({'key': key, 'issue': ev.get('issue'),
                            'reason': type(e).__name__ + ': ' + str(e)})
            dump(_refuse_outside_outdir(Path('outputs') / key / 'refusal.json'),
                 refused[-1])
    summary_name = {'batch-01': 'behavior-batch01-summary.json',
                    'batch-02': 'behavior-batch02-summary.json',
                    'batch-03': 'behavior-batch03-summary.json'}[batch]
    dump(HERE / summary_name,
         {'admitted': [t['sample_id'] for t in admitted], 'refused': refused,
          'behavior_verified_true_count': sum(1 for t in admitted
                                              if t['observed_outcome']['behavior_verified'] is True),
          'interpreter': interpreter, 'python_shim': shimdir})
    return admitted, refused


def _batch_of(ev, batch):
    if batch in ('batch-01', 'batch-02', 'batch-03'):
        return batch
    raise ValueError('unknown evidence batch: ' + str(batch))


def behavior_one(ev, denylist, timeout_seconds, batch='batch-01'):
    repo, issue = ev['repo'], ev['issue']
    guard_train_source(repo, issue, denylist)
    frozen_tx = json.loads((ROOT / ev['frozen_transaction']).read_text())
    frozen_edits = [tc['function']['arguments'] for m in frozen_tx['messages']
                    for tc in (m.get('tool_calls') or []) if tc['function']['name'] == 'edit_file']
    if not frozen_edits:
        raise ValueError('no production edits in frozen transaction')
    sub = ev['key'].removeprefix('p013-')
    for batch_prefix, short_prefix in (('batch-02', 'b02-'), ('batch-03', 'b03-')):
        if batch == batch_prefix and sub.startswith(short_prefix):
            sub = sub.removeprefix(short_prefix)
    command_path = HERE / _batch_of(ev, batch) / sub / 'command.txt'
    command = command_path.read_text()
    if not command.strip():
        raise ValueError('empty behavior command')
    live = _refuse_outside_outdir(Path('outputs') / ev['key'].removeprefix('p013-') / 'workspace')
    if live.exists():
        shutil.rmtree(live)
    live.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / ev['frozen_source_root'], live)
    R.baseline(live)
    transport = R.LocalTransport(live)
    ctx = R.Context(transport, repo)
    ctx.budget.tool_calls = 100
    ctx.harness.command_timeout_seconds = timeout_seconds
    rec = R.Recorder(ctx, TOOLS, ev['user_content'])

    read_args = dict(ev['read'])
    read = rec.call('read_file', read_args)
    if read['status'] != 'ok':
        raise ValueError('live read_file failed: ' + str(read.get('status')))
    anchor = frozen_edits[0]['old_string'].rstrip('\n')
    if anchor not in (read.get('content') or ''):
        raise ValueError('live read did not expose the exact edit context')

    pre = rec.call('run_command', {'command': command},
                   'Observe the stated behavior and boundary cases before editing.')
    if pre['status'] != 'error':
        raise ValueError('no failing baseline observed (pre status=%s); refusing to mark behavior'
                         % pre['status'])
    pre_text = (pre.get('error_message') or '') + (pre.get('details', {}) or {}).get('stdout', '')
    pre_text = pre_text.replace('\\"', '"')  # tool envelope double-escapes quotes; normalize
    if ev.get('pre_marker') and ev['pre_marker'] not in pre_text:
        raise ValueError('pre-failure lacks the expected genuine failure marker %r; refusing '
                         '(environment-confounded failure, not the behavior under test)'
                         % ev['pre_marker'])

    targets = [0]  # zero-based assistant ordinals: read is assistant step 0
    targets.append(1)  # pre-fix run is assistant step 1
    for edit in frozen_edits:
        args = {k: v for k, v in edit.items()}
        result = rec.call('edit_file', args,
                          'Apply the minimal production correction supported by the observation.')
        if result['status'] != 'ok':
            raise ValueError('live edit_file failed: ' + str(result))
        targets.append(len(rec.events) - 1)
    for path in sorted({e['filepath'] for e in frozen_edits}):
        try:
            ast.parse((live / path).read_text(), filename=path)
        except SyntaxError as e:
            raise ValueError('edited production file has syntax error: ' + str(e))

    post = rec.call('run_command', {'command': command},
                    'Repeat the identical behavior contract after the production edits.')
    if post['status'] != 'ok':
        raise ValueError('post-edit behavior did not pass (status=%s); attempt not admitted'
                         % post['status'])
    post_text = (post.get('stdout') or '') + (post.get('stderr') or '')
    post_text = post_text.replace('\\"', '"')
    if (post.get('exit_code') != 0 or '"pass": false' in post_text
            or 'INTERNAL ERROR' in post_text):
        raise ValueError('post-pass lacks clean same-criterion evidence (exit=%s); refusing'
                         % post.get('exit_code'))
    targets.append(len(rec.events) - 1)

    module_file = None
    for cand in sorted({e['filepath'] for e in frozen_edits}):
        # Authoritative file-identity check: the verified file must be the live
        # workspace copy. (Real-import __file__ evidence for each trajectory lives
        # in batch-01/<key>/module-path.log from the manual verification pass.)
        resolved = (live / cand).resolve()
        if resolved.is_relative_to(live.resolve()) and resolved.exists():
            module_file = str(resolved)

    dump(_refuse_outside_outdir(Path('outputs') / ev['key'].removeprefix('p013-') / 'events.json'), rec.events)
    dump(_refuse_outside_outdir(Path('outputs') / ev['key'].removeprefix('p013-') / 'commands.json'),
         transport.commands)
    patch = R.git(live, 'diff', '_swegemma_baseline')
    if not patch.strip():
        raise ValueError('resulting live patch is empty')
    (_refuse_outside_outdir(Path('outputs') / ev['key'].removeprefix('p013-') / 'actual.patch')
     .write_text(patch))
    handover = {
        'candidate_id': ev['key'],
        'files': [{'path': p} for p in sorted({e['filepath'] for e in frozen_edits})],
        'requirement_done': True,
        'verification_status': 'pass',
        'requirement_scope': ev['scope'] + '; whole-issue oracle NOT run',
        'verification': {
            'observation': 'run_command',
            'pre_command_sha256': sha(command),
            'pre_exit_code': None,  # filled from the live response envelope below
            'post_command_sha256': sha(command),
            'post_exit_code': None,  # filled from the live response envelope below
            'same_criterion': True,
            'key_output': (post.get('stdout') or '')[:2000],
            'module_file': module_file,
            'scope': 'Identical behavior command before (must fail) and after (must pass) '
                     'the production edits, executed live in this workspace.',
        },
        'unresolved': ['whole-issue oracle not run; task_resolved=null'],
        'needs_other_route': False,
    }
    # Exit codes live in different envelope slots for ok vs error responses.
    def _exit(resp):
        if not isinstance(resp, dict):
            return None
        if isinstance(resp.get('details'), dict) and 'exit_code' in resp['details']:
            return resp['details']['exit_code']
        return resp.get('exit_code')

    handover['verification']['pre_exit_code'] = _exit(pre)
    handover['verification']['post_exit_code'] = _exit(post)
    if not isinstance(handover['verification']['pre_exit_code'], int) \
            or handover['verification']['pre_exit_code'] == 0:
        raise ValueError('pre evidence lacks a real non-zero exit code; refusing')
    if handover['verification']['post_exit_code'] != 0:
        raise ValueError('post evidence lacks exit code 0; refusing')
    if not (module_file and Path(module_file).resolve().is_relative_to(live.resolve())):
        raise ValueError('module file not under live workspace')
    rec.messages.append({'role': 'assistant',
                         'content': json.dumps(handover, ensure_ascii=False, indent=2),
                         'tool_calls': []})
    asst_positions = [i for i, m in enumerate(rec.messages) if m.get('role') == 'assistant']
    pos_of = {msg_idx: ordinal for ordinal, msg_idx in enumerate(asst_positions)}
    # Supervise exactly the canonical arc steps: read, pre-run, edits, post-run, handover.
    supervised = sorted(set(targets + [pos_of[len(rec.messages) - 1]]))
    tx = {'sample_id': ev['key'], 'transaction_id': ev['key'] + '-tx01',
          'producer_revision': PRODUCER_REVISION,
          'synthetic': False, 'origin_kind': 'cpu_replayed_behavior_recovery',
          'group': {'issue': issue, 'repo': repo, 'commit': ev['commit'], 'split': 'train'},
          'source': {'uri': 'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/'
                            '08e109b4a59eaeebf80e4675cd125d42e7ac99a4',
                     'license': 'MIT dataset; ' + ev['code_license'] + ' touched code',
                     'sha256': frozen_tx['source']['sha256'],
                     'teacher': frozen_tx['source'].get('teacher'),
                     'traj_id': frozen_tx['source'].get('traj_id'),
                     'frozen_evidence': ev['frozen_transaction']},
          'protocol_id': PROTOCOL_ID,
          'tool_schema': {'path': str(TOOLS_SRC.relative_to(ROOT)),
                          'canonical_sha256': TOOLS_CANONICAL_SHA,
                          'names': TOOL_NAMES_5,
                          'repair_role_holds_submit_patch': False},
          'messages': rec.messages, 'target_assistant_steps': supervised,
          'behavior': 'behavior_recovery', 'needs_replay': False,
          'attempt_status': 'behavior_verified',
          'observed_outcome': {'action_succeeded': True, 'task_resolved': None,
                               'behavior_verified': True,
                               'issue_reproduction_before_failed': True,
                               'issue_reproduction_after_passed': True,
                               'official_suite_ran': False,
                               'validation_scope': ev['scope']},
          'handover': handover,
          'source_resolved_label': True, 'source_label_not_revalidated': True,
          'replay_boundary': ('Unchanged official v31 tool functions minus submit_patch, macOS local '
                              'transport. Harness PATH setup only: `python` wrapper execs the same '
                              '3.9.6 interpreter with user site-packages appended after the command '
                              'PYTHONPATH (workspace sources win; obsolete pathlib backport removed '
                              'from user site 2026-10-20); pydantic live workspace additionally sees '
                              'vendored pydantic-core 2.30.0 wheel after the command PYTHONPATH. '
                              'Behavior command byte-identical. '
                              'Live pre-fail/post-pass on the identical behavior command; '
                              'no repository test suite.'),
          'continuation_policy': ('All tool observations newly executed in the live workspace; '
                                  'no old assistant future or old tool observations reused.')}
    dest = _refuse_outside_outdir(Path('outputs') / ev['key'].removeprefix('p013-') / 'transaction.json')
    dump(dest, tx)
    dump(dest.parent / 'handover.json', handover)
    stale = dest.parent / 'refusal.json'
    if stale.exists():
        stale.unlink()  # this key now has a verified admission, not a refusal
    return tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['reissue', 'behavior'], required=True)
    ap.add_argument('--limit', type=int, default=2)
    ap.add_argument('--keys', nargs='*', default=None)
    ap.add_argument('--admit', action='store_true')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--batch', choices=['batch-01', 'batch-02', 'batch-03'], default='batch-01')
    args = ap.parse_args()
    if args.mode == 'reissue':
        results = run_reissue(args.limit, args.admit)
        print(json.dumps({'mode': 'reissue', 'admitted': False,
                          'corrected': sum(r['status'] == 'reissued_corrected' for r in results),
                          'refused': sum(r['status'] != 'reissued_corrected' for r in results)}))
        for r in results:
            print(json.dumps({k: r[k] for k in ('instance_id', 'repo', 'status', 'reason', 'edits')
                              if k in r}))
    else:
        default_keys = (['yamllint', 'deepdiff', 'apispec', 'apispec-pr950',
                           'dataset', 'pydantic', 'tenacity', 'mypy']
                          if args.batch == 'batch-01' else
                          ['xmltodict-345', 'result-121', 'slugify-148', 'slugify-149',
                           'boltons-375', 'cantools-701', 'apispec-892', 'xmltodict-363',
                           'pydantic-5568', 'pydantic-3646', 'yamllint-572']
                          if args.batch == 'batch-02' else
                          ['boltons-358', 'result-165', 'jl-160', 'jl-170',
                           'alive-213', 'freeze-530', 'apispec-953', 'bleach-691',
                           'bleach-720', 'bleach-722', 'pyg-2600', 'pyg-2622',
                           'pyg-2682', 'tornado-3317', 'sqlparse-746', 'pypika-707',
                           'md-181', 'md-161', 'mypy-12943'])
        keys = args.keys or default_keys
        admitted, refused = run_behavior(keys[:args.limit] if args.limit else keys,
                                         args.timeout, args.batch)
        print(json.dumps({'mode': 'behavior', 'admitted': len(admitted), 'refused': len(refused)}))
        for t in admitted:
            print(json.dumps({'sample_id': t['sample_id'], 'behavior_verified': True}))
        for r in refused:
            print(json.dumps(r))


if __name__ == '__main__':
    main()
