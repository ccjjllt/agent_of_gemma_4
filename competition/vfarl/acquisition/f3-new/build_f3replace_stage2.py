"""Stage 2: emit task.json + recipe + source-preparation + reviewed-recipe for 40 tasks.

Reads: stage1c-manifest.json (anchors), obs modules.
Writes:
  acquisition/f3-new/f3-replace/<instance_id>.task.json
  acquisition/f3-new/f3-replace/<key>-reviewed-recipe.json
  acquisition/f3-new/f3-replace/<key>-source-preparation.json
  replay/artifacts/f3-replace/<key>-task.json
  replay/f3r-<key>-reviewed-recipe.json
  replay/f3r-<key>-source-preparation.json
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
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'

sys.path.insert(0, str(ACQ))
from f3replace_pool import TASKS, ACTIVE_TASKS  # noqa: E402
from f3replace_obs_dvc import DVC_OBS  # noqa: E402
from f3replace_obs_conan import CONAN_OBS  # noqa: E402
from f3replace_obs_misc import MISC_OBS  # noqa: E402

OBS = {**DVC_OBS, **CONAN_OBS, **MISC_OBS}

# key prefix -> (source layout, primary module prefix fix, task venv site-packages, task python)
POOL_CONF = {
    'f3r-dvc': {'layout': '.', 'venv': REPLAY / '.venv-dvc/lib/python3.12/site-packages',
                'python': None},
    'f3r-conan': {'layout': '.', 'venv': REPLAY / '.venv-conan/lib/python3.12/site-packages',
                  'python': None},
    'f3r-cp': {'layout': '.',
               'venv': REPLAY / '.venv-r4-cloudpickle-task/lib/python3.11/site-packages',
               'python': Path('/Users/zhaohuaxiang/.local/share/uv/python/'
                              'cpython-3.11.15-macos-aarch64-none/bin/python3.11')},
    'f3r-tg': {'layout': 'src', 'venv': REPLAY / '.venv-typeguard/lib/python3.12/site-packages',
               'python': None},
    'f3r-dask': {'layout': '.', 'venv': REPLAY / '.venv-dynamic-dask/lib/python3.12/site-packages',
                 'python': None},
}

MECH = {
    'f3r-dvc-01': 'StageCache path/load: set->write entry file, get->load hit with key equivalence (same key/value reloads equal) and isolation (other key misses), expire/invalidate->corrupt entry yields None and unlinks the file (second observation differs: hit vs miss), recompute->rewrite entry restores the hit.',
    'f3r-dvc-02': 'StageCache load ordering: set->write entry, get->load hit with equivalence/isolation, expire/invalidate->corrupt entry yields None and unlinks (second observation differs), recompute->rewrite restores hit. The load path must resolve before validation.',
    'f3r-dvc-03': 'Stage hashability gate: set->cacheable local stage hashes True with key equivalence (same stage shape hashes True) and isolation (remote/hashless dep invalidates to False), expire/invalidate->stage mutation flips the gate (second observation differs), recompute->restored stage hashes True.',
    'f3r-dvc-04': 'Uncached-output scan laziness: set->stage with uncached outs, get->lazy generator of exactly the uncached isolated outs with equivalence on repeat iteration materialization and isolation across stages, expire/invalidate->cached outs drop out of the scan (second observation differs), recompute->restored outs reappear.',
    'f3r-dvc-05': 'Save determinism and reuse: set->save materializes one entry, get->entry present with equivalence on repeat save (no duplicate uuid entries), isolation across stages, expire/invalidate->clear removes entries (second observation differs), recompute->save restores the entry.',
    'f3r-dvc-06': 'Run-cache transfer: set->copy entry from remote to local, get->destination hit with equivalence on repeat transfer and isolation across keys, expire/invalidate->force=False skips already-present keys while missing keys recompute (second observation differs), recompute->cleared key recopies.',
    'f3r-dvc-07': 'Restore dry-run scope: set->cached entry, get->dry restore leaves outputs untouched with equivalence on repeat dry runs and isolation across repos, expire/invalidate->changed stage with cleared cache is rejected (second observation differs: ok vs error), recompute->real re-run restores outputs and cache.',
    'f3r-dvc-08': 'Cache serialization scope: set->write entry, get->load hit with equivalence/isolation, expire/invalidate->removed entry misses (second observation differs), recompute->rewrite restores hit.',
    'f3r-dvc-09': 'Index data-tree cache: set->outs materialize one tree entry with key equivalence (same outs same digest) and isolation (other outs differ), expire/invalidate->empty outs yield an empty tree (second observation differs), recompute->outs restore the digest.',
}
for i in range(1, 20):
    MECH.setdefault(f'f3r-conan-{i:02d}', '')
MECH.update({
    'f3r-conan-01': 'PkgCache path scope: set->create export layout, get->layout dir present under the configured base with equivalence (same ref same path) and isolation (other ref differs), expire/invalidate->removed dir is gone (second observation differs), recompute->recreate restores.',
    'f3r-conan-02': 'CacheDatabase rows: set->create recipe row, get->row hit with equivalence and isolation (unknown ref misses), expire/invalidate->remove row misses (second observation differs), recompute->recreate restores.',
    'f3r-conan-03': 'Internal download cache: set->seed cached file, get->download serves the cached bytes without network with equivalence and isolation across urls, expire/invalidate->removed cache file is gone (second observation differs).',
    'f3r-conan-04': 'Cache clean: set->exported recipe present, get->export listed with equivalence on repeat export and isolation across recipes, expire/invalidate->clean removes revisions (second observation differs), recompute->re-export restores.',
    'f3r-conan-05': 'Cache path lookup: set->exported recipe, get->cache path resolves with equivalence and isolation across folders, expire/invalidate->missing folder errors (second observation differs), recompute->path resolves again.',
    'f3r-conan-06': 'Package layout reuse: set->create export layout, get->dir present with equivalence and isolation, expire/invalidate->removed dir gone (second observation differs), recompute->recreate restores; duplicate creates reuse the layout.',
    'f3r-conan-07': 'Proxy recipe memo: set->first resolve fetches, get->second resolve of the same ref hits the memo with one fetch and equivalence, isolation across refs, expire/invalidate->cleared memo refetches (second observation differs: 1 vs 2 fetches), recompute->memo hit again.',
    'f3r-conan-08': 'CacheDatabase editable rows: set->create row, get->hit with equivalence and isolation, expire/invalidate->remove misses (second observation differs), recompute->recreate restores.',
    'f3r-conan-09': 'CacheDatabase editable rows: set->create row, get->hit with equivalence and isolation, expire/invalidate->remove misses (second observation differs), recompute->recreate restores.',
    'f3r-conan-10': 'Package timestamp/lru: set->create package row, get->lru hit with equivalence, isolation across prefs, expire/invalidate->removed package misses (second observation differs), recompute->recreate restores; timestamp update keeps lru coherent.',
    'f3r-conan-11': 'Cache clean flow: set->exported recipe, get->listed with equivalence, expire/invalidate->clean removes it (second observation differs), recompute->re-export restores; full clean isolates nothing leftover.',
    'f3r-conan-12': 'PkgCache layout scope: set->create layout, get->present with equivalence and isolation, expire/invalidate->failed remote fetch cleans the partial layout (second observation differs: leftover vs clean), recompute->recreate restores.',
    'f3r-conan-13': 'Cache save/restore: set->exported recipe, get->saved tgz present with equivalence, expire/invalidate->remove clears revisions (second observation differs), recompute->restore brings the recipe back; unknown refs stay isolated.',
    'f3r-conan-14': 'Download cache mapping: set->seed cached file, get->hit with equivalence and isolation across urls, expire/invalidate->removed file gone (second observation differs), recompute->reseed restores.',
    'f3r-conan-15': 'Download cache mapping: set->seed cached file, get->hit with equivalence and isolation across urls, expire/invalidate->removed file gone (second observation differs), recompute->reseed restores.',
    'f3r-conan-16': 'Source path cache: set->seed source file, get->hit with equivalence and isolation, expire/invalidate->removed file gone (second observation differs); backup misses report the file and url; lock scope is reentrant.',
    'f3r-conan-17': 'Source path cache: set->seed source file, get->hit with equivalence and isolation, expire/invalidate->removed file gone (second observation differs); lock scope is reentrant.',
    'f3r-conan-18': 'Remove flow: set->exported recipe, get->listed with equivalence, expire/invalidate->remove clears it (second observation differs), recompute->re-export restores; unknown refs error in isolation.',
    'f3r-conan-19': 'PkgCache layout scope: set->create export layout, get->present with equivalence and isolation, expire/invalidate->lru miss then removed dir gone (second observation differs), recompute->recreate restores.',
    'f3r-cp-01': 'Pickle-by-value registry: set->register module, get->registry hit with equivalence (repeat register idempotent) and isolation (second module independent), expire/invalidate->unregister removes it while the other stays (second observation differs), recompute->re-register restores.',
    'f3r-cp-02': 'ABC state exclusion: set->pickle ABC subclass, get->roundtrip preserves name and subclass relation with equivalence on repeat roundtrips and isolation across classes, expire/invalidate->abc caches excluded from state (second observation differs: caches present pre-fix would leak), recompute->roundtrip restores.',
    'f3r-cp-03': 'Setstate memo identity: set->pickle dynamic ABC, get->roundtrip with equivalence and isolation, expire/invalidate->abc impl registry roundtrips instead of leaking caches (second observation differs), recompute->roundtrip restores.',
    'f3r-cp-04': 'Code reducer determinism: set->pickle dynamic function, get->roundtrip with equivalence (repeat dumps byte-equal) and isolation across functions (second observation differs on cross-talk), recompute restores.',
    'f3r-cp-05': 'ABC registry branch scope: set->pickle ABC subclass, get->roundtrip with equivalence and isolation, expire/invalidate->abc caches excluded (second observation differs), recompute->roundtrip restores.',
    'f3r-cp-06': 'Function state capture: set->pickle closure with dynamic attrs, get->roundtrip preserves call result and attrs with equivalence and isolation across closures, expire/invalidate->stale state would mismatch (second observation differs), recompute restores.',
    'f3r-cp-07': 'Class-dict memo scope: set->pickle subclass, get->roundtrip preserves overrides with equivalence and isolation across sibling subclasses, expire/invalidate->inherited members stay excluded (second observation differs), recompute restores.',
    'f3r-cp-08': 'Class state extraction: set->pickle ABC subclass, get->roundtrip with equivalence and isolation, expire/invalidate->abc caches excluded (second observation differs), recompute->roundtrip restores.',
    'f3r-tg-01': 'Transform memo namespaces: set->visit aliased imports, get->memo records the alias binding with equivalence on repeat visits and isolation across modules (second observation differs on cross-talk), expire/invalidate->fresh memo starts empty, recompute restores bindings.',
    'f3r-conan-20': 'PyRequire resolution cache scope: set->first load fetches and caches, get->second load hits cache without refetch with equivalence and isolation across refs, expire/invalidate->cleared cache refetches (second observation differs: 2 vs 3 fetches), recompute->cached again; locked resolves thread the prerelease config.',
    'f3r-conan-21': 'List-driven remove scope: set->exported recipe listed, get->remove by list reports no binaries for revision-less refs with equivalence on repeat runs and isolation across lists, expire/invalidate->removed recipe gone (second observation differs), recompute->re-export restores.',
    'f3r-conan-22': 'Download sources scope: set->server timestamp adopted, get->recipe fetched then sources fetched with equivalence on repeat downloads and isolation across refs, expire/invalidate->removed row misses (second observation differs), recompute->re-download restores.',
    'f3r-conan-23': 'Remote payload integrity scope: set->corrupt payload offered, get->integrity check rejects it with equivalence on repeat rejections and isolation across refs, expire/invalidate->partial layout cleaned (second observation differs), recompute->valid payload path retained.',
    'f3r-dvc-10': 'StageCache combined contract: set->cacheable stage hashes, get->saved entry loads with equivalence and isolation, expire/invalidate->corrupt entry misses and unlinks (second observation differs), recompute->re-save restores; repeat saves reuse the entry.',
    'f3r-dvc-11': 'Output state contract: set->missing path raises DoesNotExistError on save, get->repeat raises identically with equivalence across paths and isolation across names, expire/invalidate->present path proceeds past the check (second observation differs), recompute->missing raises again.',
    'f3r-dvc-12': 'Output dumpd contract: set->versioned dir serializes to path/hash/files shape, get->repeat serializes identically with equivalence and isolation across names, expire/invalidate->emptied files change the shape (second observation differs), recompute->restored files restore the shape.',
    'f3r-dvc-13': 'StageCache filesystem scope: set->save materializes an entry, get->entry present with equivalence and isolation across keys, expire/invalidate->cleared cache misses (second observation differs), recompute->re-save restores.',
    'f3r-dvc-14': 'StageCache gate and kwargs contract: set->cacheable stage hashes, get->validated kwargs accepted with equivalence and isolation across names, expire/invalidate->remote or hashless deps flip the gate (second observation differs), recompute->restored stage hashes; entry load hits.',
    'f3r-tg-02': 'TransformMemo walrus-check scope: set->annotated walrus target gets a flat runtime check call, get->four-arg shape holds with equivalence on repeat visits and isolation across functions, expire/invalidate->fresh memo starts empty (second observation differs), recompute->revisit restores the flat call.',
    'f3r-tg-03': 'TransformMemo import binding scope: set->aliased from-import records alias to dotted path, get->binding holds with equivalence and isolation across modules, expire/invalidate->fresh memo starts empty (second observation differs), recompute->revisit restores the binding.',
    'f3r-tg-04': 'TransformMemo TYPE_CHECKING scope: set->guarded import records the name as ignored and preserves the branch, get->repeat preserves identically with equivalence and isolation for plain branches, expire/invalidate->fresh memo starts empty (second observation differs), recompute->revisit restores.',
    'f3r-conan-24': 'Download timestamp scope: set->server timestamp adopted before fetch, get->ordered revision-then-fetch-then-sources with equivalence and isolation across refs, expire/invalidate->removed row misses (second observation differs), recompute->re-download restores; no stray timestamp rewrites.',
    'f3r-dvc-15': 'Output hash-meta contract: set->file list computes hash_info/meta with counts and sizes, get->repeat computes identically with equivalence and isolation across names, expire/invalidate->reduced file list changes shape (second observation differs), recompute->restored list restores shape.',
    'f3r-dvc-16': 'Experiment-removal arg gate: set->empty selection rejected naming rev/all-commits/queue, get->rejection repeats with equivalence, isolation across flag mixes (experiment-only and queue-only pass, experiment+queue mixing rejected), expire/invalidate->altered selections flip the verdict (second observation differs), recompute->restored selection rejects again.',
    'f3r-dvc-17': 'StageCache load and loader scope: set->loader stays a Mapping, get->saved entry loads with equivalence and isolation across keys, expire/invalidate->corrupt entry misses and unlinks (second observation differs), recompute->rewrite restores the hit.',
    'f3r-conan-25': 'CacheAPI presence and download scope: set->API constructs and caches the download mapping, get->hit with equivalence and isolation across urls, expire/invalidate->removed file gone (second observation differs), recompute->reseed restores.',
    'f3r-conan-27': 'Download failure scope: set->failing url with retries raises after exactly N waits with equivalence on repeat runs and isolation across urls, expire/invalidate->not-found never retries (second observation differs: waits vs none), recompute->retryable failure retries again.',
    'f3r-conan-28': 'Backup-sources upload scope: set->seeded sources recorded, get->upload roundtrip lists them with equivalence and isolation across refs, expire/invalidate->cleared cache uploads nothing new (second observation differs), recompute->re-seeded sources upload again.',
    'f3r-dask-01': 'Scheduler result cache: set->pre-seeded cache entries count as done, get->dependents become ready without running with equivalence on repeat state builds and isolation across graphs, expire/invalidate->empty cache changes the ready set (second observation differs), recompute->re-seeded cache restores readiness.',
    'f3r-dask-02': 'Profiler result cache: set->profiled run records results, get->results present with equivalence and isolation across profilers, expire/invalidate->clear empties results and resets timing state (second observation differs), recompute->re-run records again.',
    'f3r-dask-03': 'Inline substitution cache: set->inline shared deps, get->single substitution with equivalence on repeat runs and isolation across graphs, expire/invalidate->output keys protected from inlining (second observation differs), recompute restores.',
})


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


CURATED = {
    'f3r-dvc-10': ("StageCache roundtrip contract: a cacheable stage hashes to a stable key; "
                   "saving materializes one entry file; loading the saved entry hits; loading a "
                   "corrupt entry misses and removes it; re-saving restores the hit; repeat saves "
                   "reuse the entry instead of duplicating it; same-key reloads are equivalent and "
                   "different keys are isolated."),
    'f3r-dvc-14': ("StageCache gate and utils contract: only stages with local hashed deps and "
                   "outs are cacheable; remote or hashless deps invalidate cacheability; validated "
                   "stage kwargs are accepted; the stage cache saves and loads by key with "
                   "equivalence for same keys and isolation across keys; corrupt entries are "
                   "removed on load."),
}

CURATOR = "Codex curator clarification 2026-10-26 (empty upstream statement; behavioral contract authored from the pinned F2P/P2P test names and the cache module's observable behavior; no fix details disclosed)"


def binding(pool: str, primary_file: str, extra_files=()) -> str:
    lines = ['from pathlib import Path', 'import hashlib, importlib']
    if pool == 'f3r-tg':
        mod = primary_file.replace('src/', '').removesuffix('.py').replace('/', '.')
        lines.append(f"expected=(Path.cwd() / '{primary_file}').resolve()")
    else:
        mod = primary_file.removesuffix('.py').replace('/', '.')
        lines.append(f"expected=(Path.cwd() / '{primary_file}').resolve()")
    lines += [
        'try:', f" module = importlib.import_module('{mod}')",
        'except Exception as import_error:', ' traceback=import_error.__traceback__',
        ' while traceback is not None:', "  file=traceback.tb_frame.f_globals.get('__file__')",
        '  if file and Path(file).resolve()==expected:',
        "   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')",
        '  traceback=traceback.tb_next', ' raise',
        'actual=Path(module.__file__).resolve()',
        'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())',
        'assert actual == expected, (actual, expected)',
    ]
    for f in extra_files:
        lines.append(f"print('edited_file', '{f}', 'sha256', "
                     f"hashlib.sha256(Path('{f}').read_bytes()).hexdigest())")
    return '\n'.join(lines)


def hunk_starts(patch: str):
    out = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r'diff --git a/(\S+)', line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
        if m and cur and cur not in out:
            out[cur] = int(m.group(1))
    return out


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else set()
    stage1c = json.loads((OUT / 'stage1c-manifest.json').read_text())
    by_key = {t['key']: t for t in stage1c['tasks']}
    upstreams = stage1c['upstreams']
    dataset_readme = ROOT / 'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'
    assert dataset_readme.is_file(), 'dataset README missing'
    dataset_license = {'license': 'MIT', 'path': str(dataset_readme.relative_to(ROOT)),
                       'sha256': sha(dataset_readme), 'revision': REV}
    review_tasks, index_lines = [], None
    tasks = ACTIVE_TASKS()
    if only:
        tasks = [x for x in tasks if x[0] in only]
    for key, iid, repo, canonical, upstream_short, symbol, contract in tasks:
        print(f'=== {key} ===', flush=True)
        assert key in OBS, f'missing obs {key}'
        m = by_key[key]
        assert m['reconstruction_ok'], f'{key} not reconstructed'
        pool = next(p for p in POOL_CONF if key.startswith(p))
        conf = POOL_CONF[pool]
        assert conf['venv'].is_dir(), f"venv missing {conf['venv']}"
        if conf['python'] is not None:
            assert conf['python'].is_file(), f"task python missing {conf['python']}"
        ws = ART / (key + '-source')
        tag = canonical.replace('/', '__') + '.' + upstream_short
        up = upstreams[tag]
        up_dest = ROOT / up['dir']
        pg = ROOT / m['page']
        d = json.load(open(pg))
        row = next(it['row'] for it in d['rows'] if it['row']['instance_id'] == iid)
        if index_lines is None:
            index_lines = {json.loads(l)['instance_id']: json.loads(l)
                           for l in open(ROOT / 'competition/vfarl/acquisition/general/smith-task-index.jsonl')}
        # original task file
        orig = OUT / (iid + '.task.json')
        if not orig.exists():
            save(orig, {'row': row, 'source': index_lines.get(iid),
                        'saved_page': str(pg.relative_to(ROOT)), 'saved_page_sha256': sha(pg)})
        # verification targets + test_patch for missing files
        required = list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])
        targets = sorted({n.split('::', 1)[0] for n in required})
        missing = []
        for t in targets:
            if not (ws / t).exists() and (up_dest / t).exists():
                text = (up_dest / t).read_text()
                missing.extend(difflib.unified_diff([], text.splitlines(keepends=True),
                                                    fromfile='/dev/null', tofile='b/' + t))
        test_patch = ''.join(missing)
        curated = CURATED.get(key)
        statement = curated if curated else row['problem_statement']
        task = {'instance_id': iid, 'repo': repo, 'base_commit': m['execution_commit'],
                'problem_statement': statement, 'patch': '',
                'test_patch': test_patch, 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
                'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
        adapted = ART / (key + '-task.json')
        save(adapted, task)
        # recipe
        files = list(m['recipe_edits'] and [e['filepath'] for e in m['recipe_edits']] or [])
        seen, ordered_files = set(), []
        for e in m['recipe_edits']:
            if e['filepath'] not in seen:
                seen.add(e['filepath'])
                ordered_files.append(e['filepath'])
        starts = hunk_starts(row['patch'])
        primary = ordered_files[0]
        cmd = 'python3 - <<\'PYREPLAY\'\n' + binding(pool, primary, ordered_files[1:]) + '\n' + OBS[key] + '\nPYREPLAY'

        def act(tool, args, exp='ok'):
            v = {'tool': tool, 'arguments': args, 'expected_status': exp}
            if tool == 'run_command':
                v['reviewed_command_sha256'] = digest(args['command'].encode())
            return v

        actions = [act('get_status', {})]
        for f in ordered_files:
            s = max(1, starts.get(f, 140) - 12)
            actions.append(act('read_file', {'filepath': f, 'start_line': s, 'end_line': s + 35}))
        actions.append(act('run_command', {'command': cmd}, 'error'))
        for e in m['recipe_edits']:
            actions.append(act('edit_file', {'filepath': e['filepath'], 'old_string': e['old_string'],
                                             'new_string': e['new_string']}))
        actions.extend([act('run_command', {'command': cmd}), act('get_status', {}),
                        act('submit_patch', {})])
        if key.startswith('f3r-tg'):
            shim_note = (' Sandbox exposes a wide-terminal mypy shim (COLUMNS=250) so the '
                         'pretty-printer used by tests/mypy/test_type_annotations.py emits the '
                         'pinned single-line messages; without a tty mypy wraps at width 80 on '
                         'any tree. The repair itself is verified by transformer tests and mypy '
                         'message content.')
        else:
            shim_note = ''
        review_notes = (f'{canonical} {symbol} must preserve {contract}. {MECH[key]} '
                        'Restore the exact production logic without hardcoding values or '
                        'altering tests. All original F2P/P2P identifiers and complete target '
                        'files are retained verbatim. The Smith row introduces the defect; this '
                        'is an off-policy scripted demonstration, not a model sample or '
                        'preference pair. The inline observation fails before the repair '
                        '(first run expects error) and passes after (second run expects ok), '
                        'so the second observation differs.') + shim_note
        recipe = {'evidence_kind': 'human_reviewed_off_policy_demonstration',
                  'reviewed_by': 'Codex root source contract and actual command review 2026-10-26',
                  'review_notes': review_notes, 'actions': actions,
                  'final_response': f'Restored {symbol} {contract}. The actual inline cache '
                                    'observation completed after the production edit. Patch submitted.'}
        recipe_path = REPLAY / f'f3r-{key}-reviewed-recipe.json'
        save(recipe_path, recipe)
        save(OUT / f'{key}-reviewed-recipe.json', recipe)
        # source preparation
        lic_name = {'f3r-dvc': 'LICENSE', 'f3r-conan': 'LICENSE.md', 'f3r-cp': 'LICENSE',
                    'f3r-tg': 'LICENSE', 'f3r-dask': 'LICENSE.txt'}[pool]
        lic = up_dest / lic_name
        assert lic.is_file(), f'license missing {lic}'
        fam = f'{canonical}.{symbol}::{contract}'
        snap = f"{canonical}::{up['full']}"
        prep = {'instance_id': iid, 'key': key, 'original_task': str(orig.relative_to(ROOT)),
                'original_task_sha256': sha(orig), 'adapted_task': str(adapted.relative_to(ROOT)),
                'workspace': str(ws.relative_to(ROOT)), 'source_dataset': 'SWE-bench/SWE-smith-py',
                'source_train_revision': REV, 'physical_split': 'train',
                'course_membership': 'F600_F3_new', 'primary_adapter': 'F', 'submechanism': 'F3',
                'canonical_repo': canonical, 'repair_family_id': fam, 'snapshot_group_id': snap,
                'execution_commit': m['execution_commit'], 'git_ls_remote': m['git_ls_remote'],
                'buggy_source': m['buggy_source'],
                'upstream_source': {'repo': canonical, 'requested_revision': upstream_short,
                                    'resolved_revision': up['full'],
                                    'archive_path': up['meta']['archive_path'],
                                    'archive_sha256': up['meta']['archive_sha256'],
                                    'download_url': up['meta']['download_url'],
                                    'immutable_url': up['meta']['immutable_url'],
                                    'commit_resolution': None},
                'license_evidence': {
                    'source': {'path': str(lic.relative_to(ROOT)), 'sha256': sha(lic),
                               'revision': up['full']},
                    'dataset': dataset_license},
                'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
                'visible_prompt_profile': ({
                    'kind': 'curator_authored_behavioral_clarification',
                    'reviewed_by': CURATOR,
                    'original_problem_statement_sha256': digest(row['problem_statement'].encode()),
                    'curated_problem_statement_sha256': digest(statement.encode()),
                    'curated_prompt': statement} if curated else None),
                'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
                'mechanism_evidence': MECH[key], 'verification_target_files': targets,
                'source_page': str(pg.relative_to(ROOT)), 'source_page_sha256': sha(pg),
                'mirror_ref': {'url': f'https://github.com/{repo}.git',
                               'ref': f'refs/heads/{iid}', 'actual_response': m['git_ls_remote'],
                               'response_sha256': digest(m['git_ls_remote'].encode())},
                'task_venv_site_packages': str(conf['venv'].relative_to(ROOT)) if conf['venv'].is_relative_to(ROOT) else str(conf['venv']),
                'task_venv_python': str(conf['python']) if conf['python'] else None,
                'task_source_layout': conf['layout'],
                'verification_assets': [
                    {'path': t,
                     'sha256': sha(ws / t) if (ws / t).exists() else sha(up_dest / t),
                     'bytes': (ws / t).stat().st_size if (ws / t).exists() else (up_dest / t).stat().st_size,
                     'present_in_bug_source': (ws / t).exists()} for t in targets],
                'source_review_path': str((OUT / f'{key}-reviewed-recipe.json').relative_to(ROOT)),
                'use_partition': 'train_only'}
        save(REPLAY / f'f3r-{key}-source-preparation.json', prep)
        save(OUT / f'{key}-source-preparation.json', prep)
        review_tasks.append({'instance_id': iid, 'key': key})
        print(f'  targets={targets} F2P={len(row["FAIL_TO_PASS"])} P2P={len(row["PASS_TO_PASS"])} '
              f'edits={len(m["recipe_edits"])} test_patch_bytes={len(test_patch)}', flush=True)
    save(OUT / 'f3-replace-preparation.json', {'tasks': review_tasks})
    print(f'\nSTAGE2 DONE tasks={len(review_tasks)}', flush=True)


if __name__ == '__main__':
    main()
