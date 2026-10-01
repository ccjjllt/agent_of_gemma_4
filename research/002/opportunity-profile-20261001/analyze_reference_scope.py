"""Describe reference patch scope for already exposed tasks, not a strategy score."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


def non_test(path):
    p = Path(path)
    return path != '/dev/null' and 'tests' not in p.parts and not (
        p.name.startswith('test_') or p.name.endswith('_test.py'))


def patch_scope(patch):
    # The public data uses plain unified headers, not necessarily `diff --git`.
    files, added, removed = set(), 0, 0
    current, in_hunk, old_path = None, False, None
    for line in patch.splitlines():
        if line.startswith('--- '):
            old_path = line[4:].split('\t')[0].removeprefix('a/')
            in_hunk = False
        elif line.startswith('+++ '):
            current = line[4:].split('\t')[0].removeprefix('b/')
            if current == '/dev/null':
                current = old_path
            if current and non_test(current):
                files.add(current)
        elif line.startswith('@@ '):
            in_hunk = True
        elif in_hunk and current and non_test(current):
            added += line.startswith('+')
            removed += line.startswith('-')
    return sorted(files), added, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    allowed = set()
    for name in ('panel', 'explore'):
        with (root / f'research/003/analysis_003_{name}.csv').open() as f:
            allowed.update(r['task_id'] for r in csv.DictReader(f))
    rows = []
    with args.tasks.open() as f:
        for raw in f:
            identity = re.search(r'"instance_id"\s*:\s*"([^"]+)"', raw)
            if not identity or identity.group(1) not in allowed:
                continue
            task = json.loads(raw)
            files, added, removed = patch_scope(task['patch'])
            assert files, task['instance_id']
            issue, hints = task['problem_statement'], task.get('hints_text') or ''
            rows.append(dict(task_id=task['instance_id'], repo=task['repo'],
                gold_non_test_files=files, file_count=len(files),
                added_lines=added, removed_lines=removed, changed_lines=added+removed,
                issue_chars=len(issue), has_hints=bool(hints.strip()),
                exact_gold_basename_in_issue=any(Path(p).name in issue for p in files),
                exact_gold_basename_in_issue_or_hints=any(Path(p).name in issue+hints for p in files)))
    assert {r['task_id'] for r in rows} == allowed
    out = dict(scope='27 already exposed tasks only. Reference patches describe one known solution, not all valid solutions. Non-test includes documentation/example files. Changes count added plus deleted lines. Filename cue is exact case-sensitive basename substring, not semantic localization.',
        source_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        summary=dict(tasks=len(rows), one_non_test_file=sum(r['file_count']==1 for r in rows),
            multi_non_test_files=sum(r['file_count']>1 for r in rows),
            median_changed_lines=statistics.median(r['changed_lines'] for r in rows),
            max_changed_lines=max(r['changed_lines'] for r in rows),
            at_most_20_changed_lines=sum(r['changed_lines']<=20 for r in rows),
            exact_basename_in_issue=sum(r['exact_gold_basename_in_issue'] for r in rows),
            exact_basename_in_issue_or_hints=sum(r['exact_gold_basename_in_issue_or_hints'] for r in rows),
            with_hints=sum(r['has_hints'] for r in rows)),
        tasks=sorted(rows,key=lambda r:r['task_id']))
    Path(__file__).with_name('reference-scope.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['summary'],ensure_ascii=False))


if __name__ == '__main__':
    main()
