"""Stage 1b: per-file scoped inverse edits with context expansion.

Reuses downloaded workspaces/upstreams from stage 1 (no redownload).
Output: acquisition/f3-new/f3-replace/stage1b-manifest.json
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'

sys.path.insert(0, str(ACQ))
from f3replace_pool import TASKS, ACTIVE_TASKS  # noqa: E402


def split_files(patch: str):
    """Split unified diff into {path: hunk_lines} preserving order."""
    chunks, cur_path, cur_lines = {}, None, []
    for line in patch.splitlines(keepends=True):
        m = re.match(r'diff --git a/(\S+) b/\S+', line)
        if m:
            if cur_path:
                chunks[cur_path] = cur_lines
            cur_path, cur_lines = m.group(1), []
            continue
        if cur_path is not None:
            cur_lines.append(line)
    if cur_path:
        chunks[cur_path] = cur_lines
    return chunks


def file_edits(hunk_lines):
    """Inverse edits (buggy-side -> fixed-side) per @@ block within one file."""
    edits, old, new = [], [], []
    for line in hunk_lines:
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('index ', '--- ', '+++ ', '\\', 'diff ')):
            continue
        elif line.startswith(' '):
            old.append(line[1:]); new.append(line[1:])
        elif line.startswith('+'):
            old.append(line[1:])
        elif line.startswith('-'):
            new.append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [e for e in edits if e[0] != e[1]]


def expand_until_unique(text: str, old: str, new: str, max_grow: int = 40):
    """Grow old/new with symmetric file context until old is unique. Returns (old,new) or None."""
    if text.count(old) == 1:
        return old, new
    # find all occurrence starts; expand using first occurrence's surroundings
    idx = text.find(old)
    if idx < 0:
        return None
    start, end = idx, idx + len(old)
    for _ in range(max_grow):
        # grow one line each side
        ls = text.rfind('\n', 0, start - 1)
        le = text.find('\n', end)
        nstart = 0 if ls < 0 else ls + 1
        nend = len(text) if le < 0 else le + 1
        if nstart == start and nend == end:
            break
        prefix, suffix = text[nstart:start], text[end:nend]
        start, end = nstart, nend
        old, new = prefix + old + suffix, prefix + new + suffix
        if text.count(old) == 1:
            return old, new
    return None


def main():
    stage1 = json.loads((OUT / 'stage1-manifest.json').read_text())
    by_key = {t['key']: t for t in stage1['tasks']}
    upstreams = stage1['upstreams']
    out_tasks = []
    for key, iid, repo, canonical, upstream_short, symbol, contract in ACTIVE_TASKS():
        m = by_key[key]
        row_edits = {}
        ws = ART / (key + '-source')
        tag = canonical.replace('/', '__') + '.' + upstream_short
        up_dest = ROOT / upstreams[tag]['dir']
        # need original patch text: reload from page
        pg = ROOT / m['page']
        d = json.load(open(pg))
        row = next(it['row'] for it in d['rows'] if it['row']['instance_id'] == iid)
        patch = row['patch']
        per_file_ok = True
        applied_total = 0
        for f, hunk_lines in split_files(patch).items():
            edits = file_edits(hunk_lines)
            cur = (ws / f).read_text()
            fixed_actions = []
            for o, n in edits:
                if cur.count(o) == 1:
                    cur = cur.replace(o, n)
                    fixed_actions.append({'old_len': len(o), 'new_len': len(n), 'expanded': False})
                    applied_total += 1
                    continue
                grown = expand_until_unique(cur, o, n) if cur.count(o) >= 1 else None
                if grown and cur.count(grown[0]) == 1:
                    cur = cur.replace(grown[0], grown[1])
                    fixed_actions.append({'old_len': len(grown[0]), 'new_len': len(grown[1]), 'expanded': True})
                    # persist expanded anchor for recipe use
                    o, n = grown
                    applied_total += 1
                else:
                    fixed_actions.append({'old_len': len(o), 'new_len': len(n), 'expanded': False,
                                          'FAILED_count': cur.count(o)})
                    per_file_ok = False
            up_text = (up_dest / f).read_text()
            row_edits[f] = {'edits': [{'old': o, 'new': n} for o, n in
                                       [e for e in (fixed_actions and [None] or [])] ]} if False else None
            # store anchors for recipe
            anchors = []
            cur2 = (ws / f).read_text()
            for o, n in edits:
                if cur2.count(o) == 1:
                    anchors.append({'old_string': o, 'new_string': n})
                    cur2 = cur2.replace(o, n)
                else:
                    grown = expand_until_unique(cur2, o, n) if cur2.count(o) >= 1 else None
                    if grown and cur2.count(grown[0]) == 1:
                        anchors.append({'old_string': grown[0], 'new_string': grown[1]})
                        cur2 = cur2.replace(grown[0], grown[1])
                    else:
                        anchors.append({'old_string': o, 'new_string': n, 'AMBIGUOUS_count': cur2.count(o)})
            matches = cur == up_text
            if not matches:
                per_file_ok = False
            row_edits[f] = {'hunk_edits': len(edits), 'matches_upstream': matches,
                            'anchors': [{'old_chars': len(a['old_string']), 'new_chars': len(a['new_string']),
                                         **({k: v for k, v in a.items() if k not in ('old_string', 'new_string')})}
                                        for a in anchors],
                            'anchor_texts': anchors}
            if not matches:
                # save diff snippet for manual review
                dl = list(difflib.unified_diff(cur.splitlines(), up_text.splitlines(), lineterm=''))[:20]
                row_edits[f]['residual_diff_head'] = dl
        m['per_file'] = row_edits
        m['reconstruction_ok'] = per_file_ok and all(v['matches_upstream'] for v in row_edits.values())
        out_tasks.append(m)
        status = 'OK' if m['reconstruction_ok'] else 'FAIL'
        print(f'{status} {key} files={list(row_edits)}', flush=True)
        for f, v in row_edits.items():
            if not v['matches_upstream']:
                print(f'   {f} hunks={v["hunk_edits"]} residual={v.get("residual_diff_head", [])[:6]}', flush=True)
    (OUT / 'stage1b-manifest.json').write_text(
        json.dumps({'tasks': out_tasks, 'upstreams': upstreams}, ensure_ascii=False, indent=2) + '\n')
    bad = [m['key'] for m in out_tasks if not m['reconstruction_ok']]
    print(f'\nSTAGE1B DONE failures={bad}', flush=True)


if __name__ == '__main__':
    main()
