"""Stage 1c: direction auto-detect (bug-introducing vs fix), per-file anchors, EOF fix.

For each task, try direction A (+ side = buggy content) and direction B
(- side = buggy content). Accept a direction iff every hunk anchor matches
exactly once (after minimal context expansion) and the fully edited tree
equals upstream byte-for-byte (modulo a recorded EOF-newline fix).

Output: acquisition/f3-new/f3-replace/stage1c-manifest.json (includes recipe anchors).
"""
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


def file_edits(hunk_lines, direction: str):
    """direction A: old=+side (buggy), new=-side. Direction B: old=-side, new=+side."""
    edits, old, new = [], [], []
    for line in hunk_lines:
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('index ', '--- ', '+++ ', '\\', 'diff ', 'old mode', 'new mode')):
            continue
        elif line.startswith(' '):
            old.append(line[1:]); new.append(line[1:])
        elif line.startswith('+'):
            (old if direction == 'A' else new).append(line[1:])
        elif line.startswith('-'):
            (new if direction == 'A' else old).append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [e for e in edits if e[0] != e[1]]


def expand_until_unique(text: str, old: str, new: str, max_grow: int = 60):
    if text.count(old) == 1:
        return old, new, False
    if text.count(old) < 1:
        return None
    idx = text.find(old)
    start, end = idx, idx + len(old)
    for _ in range(max_grow):
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
            return old, new, True
    return None


def eof_anchor(cur: str, want_trailing_nl: bool):
    """Build a dedicated EOF edit making cur's trailing-newline state match upstream.

    Returns (old, new) anchored at EOF or None.
    """
    lines = cur.splitlines(keepends=True)
    for back in range(1, min(len(lines), 60) + 1):
        old = ''.join(lines[-back:])
        if cur.count(old) != 1:
            continue
        if want_trailing_nl:
            if old.endswith('\n'):
                return None
            return old, old + '\n'
        else:
            if not old.endswith('\n'):
                return None
            return old, old[:-1]
    return None


def try_direction(ws: Path, up_dest: Path, files: dict, direction: str):
    anchors_by_file, ok = {}, True
    for f, hunk_lines in files.items():
        cur = (ws / f).read_text()
        anchors = []
        for o, n in file_edits(hunk_lines, direction):
            grown = expand_until_unique(cur, o, n)
            if grown is None and o.endswith('\n') and cur.count(o[:-1]) == 1:
                # Patch-text trailing newline vs file EOF without newline.
                grown = (o[:-1], n, False)
            if grown is None:
                ok = False
                anchors.append({'ok': False, 'count': cur.count(o)})
                continue
            go, gn, expanded = grown
            cur = cur.replace(go, gn)
            anchors.append({'ok': True, 'old_string': go, 'new_string': gn, 'expanded': expanded})
        up_text = (up_dest / f).read_text()
        if cur != up_text:
            fixed = None
            if cur + '\n' == up_text:
                fixed = eof_anchor(cur, True)
            elif cur == up_text + '\n':
                fixed = eof_anchor(cur, False)
            if fixed is not None:
                go, gn = fixed
                cur = cur.replace(go, gn)
                anchors.append({'ok': True, 'old_string': go, 'new_string': gn,
                                'expanded': False, 'eof_fix': True})
            if cur != up_text:
                ok = False
        anchors_by_file[f] = {'anchors': anchors, 'matches_upstream': cur == up_text,
                              'all_anchors_ok': all(a['ok'] for a in anchors)}
        if not anchors_by_file[f]['all_anchors_ok'] or cur != up_text:
            ok = False
    return ok, anchors_by_file


def main():
    stage1 = json.loads((OUT / 'stage1-manifest.json').read_text())
    by_key = {t['key']: t for t in stage1['tasks']}
    upstreams = stage1['upstreams']
    out_tasks = []
    for key, iid, repo, canonical, upstream_short, symbol, contract in ACTIVE_TASKS():
        m = dict(by_key[key])
        ws = ART / (key + '-source')
        tag = canonical.replace('/', '__') + '.' + upstream_short
        up_dest = ROOT / upstreams[tag]['dir']
        pg = ROOT / m['page']
        d = json.load(open(pg))
        row = next(it['row'] for it in d['rows'] if it['row']['instance_id'] == iid)
        files = split_files(row['patch'])
        chosen, detail = None, {}
        for direction in ('A', 'B'):
            ok, info = try_direction(ws, up_dest, files, direction)
            detail[direction] = {'ok': ok,
                                 'files': {f: {'matches': v['matches_upstream'],
                                               'eof_fix': any(a.get('eof_fix') for a in v['anchors']),
                                               'anchors_ok': v['all_anchors_ok']} for f, v in info.items()}}
            if ok and chosen is None:
                chosen = (direction, info)
        if chosen is None:
            print(f'FAIL {key} {iid} A={detail["A"]} B={detail["B"]}', flush=True)
            m['direction'] = None
            m['direction_detail'] = detail
            m['reconstruction_ok'] = False
        else:
            direction, info = chosen
            recipe_edits = []
            for f, v in info.items():
                file_anchors = [{'filepath': f, 'old_string': a['old_string'],
                                 'new_string': a['new_string']} for a in v['anchors']]
                recipe_edits.extend(file_anchors)
            m['direction'] = direction
            m['recipe_edits'] = recipe_edits
            m['reconstruction_ok'] = True
            m['direction_detail'] = detail
            print(f'OK dir={direction} {key} files={list(files)} edits={len(recipe_edits)}', flush=True)
        out_tasks.append(m)
    (OUT / 'stage1c-manifest.json').write_text(
        json.dumps({'tasks': out_tasks, 'upstreams': upstreams}, ensure_ascii=False, indent=2) + '\n')
    bad = [t['key'] for t in out_tasks if not t['reconstruction_ok']]
    dirs = {t['key']: t['direction'] for t in out_tasks}
    print(f'\nSTAGE1C DONE failures={bad}', flush=True)
    print(json.dumps(dirs, indent=1), flush=True)


if __name__ == '__main__':
    main()
