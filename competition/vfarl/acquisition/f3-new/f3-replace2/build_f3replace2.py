"""F3-replace2: fixed observations for the 2 NEEDS_WORK tasks (dvc-07, conan-05).

Root causes (observation bugs, not source-fix bugs; stage1c recipe_edits
already reconstruct upstream OK for both):
- f3r-dvc-07: recompute step called stage.run(checkpoint_func=...) which the
  TRUE upstream code also rejects (restore() has no **kwargs; run_stage only
  forwards dry). Fix: recompute via plain stage.run().
- f3r-conan-05: recompute step called c.exported_layout() AFTER
  c.run('cache path ...'), so self.out no longer holds the create output and
  the ': Exported:' regex finds None. Fix: capture exported_layout()
  immediately after the second create, before the cache-path lookup.

Writes ONLY under acquisition/f3-new/f3-replace2/ and replay/f3r2-* /
replay/artifacts/f3-replace2/. Never touches f3-replace/.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OLD = ACQ / 'f3-replace'
NEW = ACQ / 'f3-replace2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART2 = REPLAY / 'artifacts/f3-replace2'
ART2.mkdir(parents=True, exist_ok=True)

KEYS = ['f3r-dvc-07', 'f3r-conan-05']


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def main():
    stage1c = json.loads((OLD / 'stage1c-manifest.json').read_text())
    by_key = {t['key']: t for t in stage1c['tasks']}
    review_tasks = []
    for key in KEYS:
        assert by_key[key]['reconstruction_ok'], key
        old_recipe = json.loads((REPLAY / f'f3r-{key}-reviewed-recipe.json').read_text())
        old_prep = json.loads((REPLAY / f'f3r-{key}-source-preparation.json').read_text())
        cmds = [a['arguments']['command'] for a in old_recipe['actions']
                if a['tool'] == 'run_command']
        assert len(cmds) == 2 and cmds[0] == cmds[1], key
        cmd = cmds[0]
        if key == 'f3r-dvc-07':
            old_frag = '    with repo.lock:\n        stage.run(checkpoint_func=lambda: None)'
            new_frag = '    with repo.lock:\n        stage.run()'
            assert cmd.count(old_frag) == 1, 'dvc-07 frag missing'
            cmd = cmd.replace(old_frag, new_frag)
        else:
            old_frag = ("c.run('create .')\nc.run('cache path pkg/0.1 --folder=export_source')\n"
                        "assert c.exported_layout().export_sources() == str(c.out).rstrip()")
            new_frag = ("c.run('create .')\nref_layout2 = c.exported_layout()\n"
                        "c.run('cache path pkg/0.1 --folder=export_source')\n"
                        "assert ref_layout2.export_sources() == str(c.out).rstrip()")
            assert cmd.count(old_frag) == 1, 'conan-05 frag missing'
            cmd = cmd.replace(old_frag, new_frag)
        new_sha = digest(cmd.encode())
        recipe = json.loads(json.dumps(old_recipe))
        for a in recipe['actions']:
            if a['tool'] == 'run_command':
                a['arguments']['command'] = cmd
                a['reviewed_command_sha256'] = new_sha
        recipe['reviewed_by'] = ('Codex root source contract and actual command review '
                                 '2026-10-26; f3-replace2 observation fix (no source-edit change)')
        recipe_path = REPLAY / f'f3r2-{key}-reviewed-recipe.json'
        save(recipe_path, recipe)
        save(NEW / f'{key}-reviewed-recipe.json', recipe)
        # adapted task: identical mechanism content, new path
        old_adapted = json.loads((ROOT / old_prep['adapted_task']).read_text())
        new_adapted_path = ART2 / f'{key}-task.json'
        save(new_adapted_path, old_adapted)
        prep = json.loads(json.dumps(old_prep))
        prep['adapted_task'] = str(new_adapted_path.relative_to(ROOT))
        prep['recipe'] = str(recipe_path.relative_to(ROOT))
        prep['recipe_sha256'] = sha(recipe_path)
        prep['source_review_path'] = str((NEW / f'{key}-reviewed-recipe.json').relative_to(ROOT))
        save(REPLAY / f'f3r2-{key}-source-preparation.json', prep)
        save(NEW / f'{key}-source-preparation.json', prep)
        review_tasks.append({'instance_id': old_prep['instance_id'], 'key': key,
                             'fix': 'observation-only; recipe_edits unchanged from stage1c'})
        print(f'{key}: obs fixed, recipe_sha256={prep["recipe_sha256"][:12]}', flush=True)
    save(NEW / 'f3-replace2-preparation.json', {'tasks': review_tasks})
    print(f'DONE tasks={len(review_tasks)}', flush=True)


if __name__ == '__main__':
    main()
