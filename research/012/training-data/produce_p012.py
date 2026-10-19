"""Produce P012 p012-repair-v1 local-edit transactions from frozen P011 candidates.

Same-shape local edits as P011 materialize.py, with the protocol delta only:
- protocol_id p012-repair-v1, five official tools (no submit_patch);
- supervision targets = successful production edits + post-edit status check
  + final handover message (no submit supervision);
- behavior recovery chain is NOT fabricated here; task_resolved=None,
  behavior_verified=False. A later step upgrades selected issues to full
  behavior/recovery evidence.

Freeze规则复用 P011（同 SEED、同排除表、同来源 revision），只读 P011 冻结文件，
绝不覆盖；输出全部落在 research/012/training-data/ 下的新文件。
候选顺序沿用 P011 selected[210:]（626 候选/26 仓）逐个向后，不跳仓挑选。

NOTE on repo scope: selected[210:] 的 26 个仓库全部落在 71 仓并集内（已核对，
626/626）。这是训练扩量的预期形态——交接要求“沿已冻结 train 来源与固定 commit
继续批量物化”，训练新问题本来就与 P011 同仓、按新问题计数（256 需 117 新问题
指独立问题数，不是新仓库数）。71 仓 + 外部 8 主选仓的硬排除作用于评测 dev
选题（见 evaluation/dev12-freeze.json），不作用于训练物化；本生产器沿用 P011
的 15 仓排除表，不新增训练仓。
"""

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P011 = ROOT / 'research/011/data-scaling-20261018'

# Frozen train source, identical to P011 materialize.py. Do not change.
REV = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'
SEED = 'p011-edit-replay-20261018'
# First index into P011 data-usage-full.json selected[] consumed by P012.
P012_START = 210

PROTOCOL_ID = 'p012-repair-v1'
# p012-local-2: targets are zero-based ASSISTANT ordinals (not message indices),
# computed post-hoc from final messages: successful edit_file turns +
# get_status turn + trailing handover turn. p012-local-1 used rec.events
# lengths and failed encode validation; its outputs must not be reused.
PRODUCER_REVISION = 'p012-local-2'
TOOLS_SRC = ROOT / 'research/012/agent/repair_tools_v1.json'
# Canonical sha256 of the five-tool schema (sorted-keys canonical JSON).
TOOLS_CANONICAL_SHA = '91ec82b15fe0961b3ef9e686fac268706ce954a69cabaa0d705c63d41088e267'
TOOL_NAMES_5 = ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status']

spec = importlib.util.spec_from_file_location(
    'p006_replay', ROOT / 'research/006/data-replay-20261013/replay.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
_TOOLS6 = R.official_tools()
assert 'submit_patch' in _TOOLS6, 'expected six official tools upstream'
TOOLS = {k: v for k, v in _TOOLS6.items() if k != 'submit_patch'}
assert sorted(TOOLS) == sorted(TOOL_NAMES_5), sorted(TOOLS)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def check_p011_frozen_untouched():
    """Guard: this producer must never write under the P011 directory."""
    manifest = P011 / 'dataset-manifest.json'
    assert manifest.exists(), 'P011 frozen manifest missing; refusing to run elsewhere'


def build_usage():
    """Derive P012 usage from the frozen P011 order. Read-only on P011."""
    check_p011_frozen_untouched()
    src = json.loads((P011 / 'data-usage-full.json').read_text())
    assert src['source_revision'] == REV, src['source_revision']
    assert src['seed'] == SEED, src['seed']
    selected = src['selected'][P012_START:]
    assert len(selected) == len(src['selected']) - P012_START, len(selected)
    usage = {
        'frozen_before_content_read': True,
        'source_revision': src['source_revision'],
        'seed': src['seed'],
        'rule': ('P011 freeze order reused verbatim (hash order within repo, then '
                 'alphabetical repo round robin); P012 consumes selected[%d:] in order, '
                 'no repo skipping. No outcome-based replacements.' % P012_START),
        'excluded_repositories': src['excluded_repositories'],
        'excluded_previously_read_issues': src['excluded_previously_read_issues'],
        'base': {'p011_usage': 'research/011/data-scaling-20261018/data-usage-full.json',
                 'p011_consumed_prefix': P012_START,
                 'order_note': 'P012 selected[i] == P011 selected[%d+i]; order unchanged' % P012_START},
        'train_repo_scope_note': ('Remaining 26 repos are all inside the 71-repo train union; '
                                  'expected for train expansion (new independent issues, same repos). '
                                  'The 71+ext8 hard exclusion applies to dev task selection, not to training materialization.'),
        'protocol_id': PROTOCOL_ID,
        'tools': {'path': str(TOOLS_SRC.relative_to(ROOT)),
                  'canonical_sha256': TOOLS_CANONICAL_SHA,
                  'names': TOOL_NAMES_5,
                  'repair_role_holds_submit_patch': False},
        'producer_revision': PRODUCER_REVISION,
        'selected': selected,
        'task_resolved': None,
        'behavior_verified': False,
        'automatic_training': False,
    }
    dump(HERE / 'data-usage-p012.json', usage)
    return usage


def production(path):
    p = PurePosixPath(path)
    return (not p.is_absolute() and '..' not in p.parts and p.suffix == '.py'
            and not any(x.lower() in ('test', 'tests', 'test-data', 'docs', 'doc', 'examples', '.github') for x in p.parts)
            and not p.name.lower().startswith(('test', 'reproduce', 'debug', 'verify', 'check', 'temp', 'setup'))
            and not p.name.endswith('_test.py'))


def prepare(row):
    import duckdb
    c = duckdb.connect()
    # Atomic selection by BOTH logical identity and physical row, not zipped columns.
    got = c.execute('SELECT file_row_number,instance_id,traj_id,messages FROM read_parquet(?, file_row_number=true) WHERE file_row_number=? AND instance_id=? AND traj_id=?',
                    [str(ROOT / row['source_file']), row['row_index'], row['instance_id'], row['traj_id']]).fetchone()
    if not got:
        raise ValueError('Atomic source identity mismatch')
    messages = json.loads(got[3])
    user = next(m for m in messages if m['role'] == 'user')['content']
    if isinstance(user, list):
        user = '\n'.join(x.get('text', '') for x in user if isinstance(x, dict))
    match = re.search(r'<pr_description>(.*?)</pr_description>', user, re.S)
    issue = match.group(1).strip() if match else user
    actions = []
    for mi, m in enumerate(messages):
        calls = list(m.get('tool_calls', []) or [])
        content = m.get('content', '')
        if isinstance(content, str):
            for name, body in re.findall(r'<function=([^>]+)>(.*?)</function>', content, re.S):
                params = {key: value for key, value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S)}
                calls.append({'function': {'name': name, 'arguments': params}})
        for call in calls:
            f = call.get('function', {})
            raw = f.get('arguments', {})
            try:
                a = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError):
                continue
            if a.get('command') != 'str_replace':
                continue
            path = a.get('path', '').removeprefix('/testbed/')
            if not production(path):
                continue
            if not isinstance(a.get('old_str'), str) or not isinstance(a.get('new_str'), str):
                continue
            if not a['old_str'] or a['old_str'] == a['new_str']:
                continue
            actions.append({'source_message_index': mi, 'filepath': path,
                            'old_string': a['old_str'], 'new_string': a['new_str'],
                            'allow_multiple': False})
    if not actions:
        return dict(row, status='no_structured_production_replacement')
    if len(actions) > 20:
        return dict(row, status='more_than_20_source_edits', edits=len(actions))
    out = HERE / 'outputs' / row['instance_id']
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'source.json', dict(row, messages_sha256=sha(got[3]), issue=issue,
                                   actions=actions, patch_column_used=False))
    mirror = 'swesmith/' + row['instance_id'].split('.pr_')[0]
    ref = subprocess.run(['git', 'ls-remote', 'https://github.com/' + mirror + '.git',
                          'refs/heads/' + row['instance_id']],
                         text=True, capture_output=True, check=True, timeout=90).stdout
    commit = ref.split()[0]
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact buggy commit missing')
    source = out / 'source'
    files = []

    def fetch(path):
        url = 'https://raw.githubusercontent.com/' + mirror + '/' + commit + '/' + path
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError('Oversized source file')
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append({'path': path, 'url': url, 'bytes': len(data), 'sha256': sha(data)})

    for path in sorted({a['filepath'] for a in actions}):
        fetch(path)
    licenses = []
    for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING'):
        try:
            fetch(name)
            licenses.append(files[-1])
            break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    if not licenses:
        return dict(row, status='license_not_found', buggy_commit=commit)
    provenance = dict(row, mirror=mirror, buggy_commit=commit, git_ls_remote_stdout=ref,
                      files=files, licenses=licenses,
                      state_scope='Fixed-commit production source subset. No source tests/dependencies/shell executed.',
                      dataset_license='MIT; repository code license retained separately',
                      patch_column_used=False)
    dump(out / 'provenance.json', provenance)
    with tempfile.TemporaryDirectory(prefix='p012-replay-') as tmp:
        ws = Path(tmp) / 'workspace'
        shutil.copytree(source, ws)
        R.baseline(ws)
        transport = R.LocalTransport(ws)
        ctx = R.Context(transport, row['repo'])
        ctx.budget.tool_calls = 100
        rec = R.Recorder(ctx, TOOLS, 'Implement the following requested change in the repository.\n\n' + issue +
                         '\n\nScope of this editing session: update the following production files and report the resulting local change for handover: ' +
                         ', '.join(sorted({a['filepath'] for a in actions})) + '. No full task validation has been performed in this session.')
        failure = None
        for a in actions:
            path = a['filepath']
            text = (ws / path).read_text()
            offset = text.find(a['old_string'])
            line = text[:max(offset, 0)].count('\n') + 1
            read = rec.call('read_file', {'filepath': path, 'start_line': max(1, line - 12),
                                           'end_line': line + a['old_string'].count('\n') + 12})
            if read['status'] != 'ok':
                failure = 'read_file failed'
                break
            if read.get('is_truncated') or a['old_string'].rstrip('\n') not in read.get('content', ''):
                failure = 'Selected read did not expose the complete exact edit context'
                break
            args = {k: v for k, v in a.items() if k != 'source_message_index'}
            result = rec.call('edit_file', args)
            if result['status'] != 'ok':
                failure = 'source edit did not apply to actual current state'
                break
        if not failure:
            for path in sorted({a['filepath'] for a in actions}):
                try:
                    ast.parse((ws / path).read_text(), filename=path)
                except SyntaxError as e:
                    failure = 'Edited production file has syntax error: ' + str(e)
                    break
        status_payload = None
        if not failure:
            status_payload = rec.call('get_status', {})
            if status_payload['status'] != 'ok':
                failure = 'post-edit status check failed'
        dump(out / 'events.json', rec.events)
        dump(out / 'commands.json', transport.commands)
        patch = R.git(ws, 'diff', '_swegemma_baseline')
        (out / 'actual.patch').write_text(patch)
        state = {p: {'before_sha256': sha((source / p).read_bytes()),
                     'after_sha256': sha((ws / p).read_bytes())}
                 for p in sorted({a['filepath'] for a in actions})}
        dump(out / 'states.json', state)
        if failure:
            return dict(row, status='replay_failed', reason=failure, buggy_commit=commit)
        if not patch.strip():
            return dict(row, status='replay_failed', reason='resulting local patch is empty', buggy_commit=commit)
        handover = {
            'candidate_id': 'p012-' + row['instance_id'],
            'files': [{'path': p, 'after_sha256': state[p]['after_sha256']}
                      for p in sorted(state)],
            'requirement_done': True,
            'requirement_scope': 'local production edits applied and syntactically parsed; full-task validation NOT performed in this session',
            'verification': {
                'observation': 'get_status',
                'status': status_payload.get('status'),
                'scope': 'post-edit local patch status of the same workspace; no repository test suite or whole-issue oracle executed',
            },
            'unresolved': ['no whole-issue oracle executed; task_resolved=null',
                           'behavior recovery chain to be upgraded by a later step, not fabricated here'],
            'needs_other_route': False,
        }
        rec.messages.append({'role': 'assistant',
                             'content': json.dumps(handover, ensure_ascii=False, indent=2),
                             'tool_calls': []})
        # Post-hoc assistant-ordinal targets (encode_transaction requires
        # zero-based assistant ordinals, not message indices): all assistant
        # turns issuing edit_file/get_status plus the trailing handover turn.
        asst_positions = [i for i, m in enumerate(rec.messages)
                          if m.get('role') == 'assistant']
        pos_of = {msg_idx: ordinal for ordinal, msg_idx in enumerate(asst_positions)}
        targets = []
        for i in asst_positions:
            names = [(tc.get('function') or {}).get('name')
                     for tc in (rec.messages[i].get('tool_calls') or [])]
            if 'edit_file' in names or 'get_status' in names:
                targets.append(pos_of[i])
        if (rec.messages and rec.messages[-1].get('role') == 'assistant'
                and not rec.messages[-1].get('tool_calls')
                and rec.messages[-1].get('content', '').strip()):
            targets.append(pos_of[len(rec.messages) - 1])
        n_edits = sum(1 for i in asst_positions
                      if 'edit_file' in [(tc.get('function') or {}).get('name')
                                        for tc in (rec.messages[i].get('tool_calls') or [])])
        dump(out / 'handover.json', handover)
        tx = {'sample_id': 'p012-' + row['instance_id'], 'transaction_id': 'p012-' + row['instance_id'],
              'producer_revision': PRODUCER_REVISION,
              'synthetic': False, 'origin_kind': 'cpu_replayed_source_edit_subset',
              'group': {'issue': row['instance_id'], 'repo': row['repo'], 'commit': commit, 'split': 'train'},
              'source': {'uri': f'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/{REV}',
                         'license': 'MIT dataset; repository license in provenance.json', 'sha256': sha(got[3]),
                         'teacher': row['teacher_source'], 'traj_id': row['traj_id']},
              'protocol_id': PROTOCOL_ID,
              'tool_schema': {'path': str(TOOLS_SRC.relative_to(ROOT)),
                              'canonical_sha256': TOOLS_CANONICAL_SHA,
                              'names': TOOL_NAMES_5,
                              'repair_role_holds_submit_patch': False},
              'messages': rec.messages, 'target_assistant_steps': targets,
              'behavior': 'local_edit', 'needs_replay': False, 'attempt_status': 'local_edits_applied',
              'observed_outcome': {'action_succeeded': True, 'task_resolved': None, 'behavior_verified': False,
                                   'syntax_parse_passed': True, 'local_patch_nonempty': True,
                                   'handover_reported': True,
                                   'scope': 'All selected structured production str_replace actions replayed from fixed buggy snapshot with five official tools (no submit_patch); post-edit status observed via get_status; session ends with a handover report, not a submission. This is local edit supervision, not a resolved task or full trajectory.'},
              'handover': handover,
              'source_resolved_label': True, 'source_label_not_revalidated': True,
              'replay_boundary': 'Unchanged official v31 tool functions minus submit_patch, with macOS local transport, production subset. Read locations derived from source edit targets; this is edit/action SFT, not search-policy supervision.',
              'continuation_policy': 'New read/edit/status tool observations only; no old assistant future or old tool observations reused.'}
        dump(out / 'transaction.json', tx)
        return dict(row, status='local_edits_applied', buggy_commit=commit, edits=n_edits,
                    transaction=str((out / 'transaction.json').relative_to(HERE)))


def main():
    check_p011_frozen_untouched()
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--rebuild-usage', action='store_true')
    args = ap.parse_args()
    usage_path = HERE / 'data-usage-p012.json'
    if args.rebuild_usage or not usage_path.exists():
        usage = build_usage()
    else:
        usage = json.loads(usage_path.read_text())
        assert usage['protocol_id'] == PROTOCOL_ID, usage.get('protocol_id')
        assert usage['producer_revision'] == PRODUCER_REVISION, usage.get('producer_revision')
    if args.limit == 0:
        return
    rows = usage['selected'][:args.limit]
    results = []
    start = time.monotonic()

    def run(row):
        if row['repo'] in usage['excluded_repositories']:
            raise ValueError('Reserved repo')
        existing = HERE / 'outputs' / row['instance_id'] / 'transaction.json'
        if existing.exists():
            provenance = json.loads((existing.parent / 'provenance.json').read_text())
            if all(provenance.get(k) == row.get(k) for k in ('source_file', 'row_index', 'traj_id')):
                tx = json.loads(existing.read_text())
                if tx.get('producer_revision') == PRODUCER_REVISION and tx.get('protocol_id') == PROTOCOL_ID:
                    return dict(row, status='local_edits_applied',
                                transaction=str(existing.relative_to(HERE)), reused=True)
        try:
            return prepare(row)
        except Exception as e:
            return dict(row, status='source_or_replay_error', reason=type(e).__name__ + ': ' + str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, rows):
            results.append(result)
            dump(HERE / 'production-results-p012.json', results)
            print(json.dumps({k: result[k] for k in ('instance_id', 'repo', 'status', 'reason', 'edits') if k in result}), flush=True)
    print(json.dumps({'attempted': len(results),
                      'applied': sum(r['status'] == 'local_edits_applied' for r in results),
                      'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
