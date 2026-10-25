"""F2 f2-recall: verify zero-overlap and build reviewed-candidates + catalog (2 CLEAN keeps)."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-recall'

V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
LIC = {
    'theacodes/nox': ('nox.LICENSE', 'Apache-2.0', None),
    'aio-libs/aiohttp': ('aiohttp.LICENSE', 'Apache-2.0', None),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_forbidden():
    forb_i, forb_s, forb_f = set(), set(), set()
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    forb_i |= set(base['forbidden_instance_ids'])
    forb_s |= set(base.get('forbidden_snapshot_group_ids', []))
    forb_f |= set(base['forbidden_repair_family_ids'])
    f1 = json.loads((VFARL / 'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks', []):
        forb_i.add(t['instance_id'])
        if t.get('snapshot_group_id'):
            forb_s.add(t['snapshot_group_id'])
        if t.get('repair_family_id'):
            forb_f.add(t['repair_family_id'])
    f5 = json.loads((VFARL / 'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks', []):
        forb_i.add(t.get('instance_id', t.get('canonical_task_id', '')))
        for key in ('snapshot_group_id', 'snapshot_id'):
            if t.get(key):
                forb_s.add(t[key])
        for key in ('repair_family_id', 'family_id'):
            if t.get(key):
                forb_f.add(t[key])
    for rel in ['f2-replace/catalog.json', 'f2-replace/reviewed-candidates.json',
                'candidate-list-40.json', 'reviewed-candidates.json', 'catalog.json']:
        p = ACQ / rel
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for t in d.get('tasks', d.get('candidates', [])):
            if t.get('instance_id'):
                forb_i.add(t['instance_id'])
            for key in ('snapshot_group_id', 'snapshot_group_id_provisional'):
                if t.get(key):
                    forb_s.add(t[key])
            for key in ('repair_family_id', 'repair_family_id_provisional'):
                if t.get(key):
                    forb_f.add(t[key])
    forb_s |= {'seperman/deepdiff::ed252022', 'pallets/click::fde47b4b',
               'pydantic/pydantic::acb0f10f', 'stanfordnlp/dspy::651a4c71',
               'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9',
               'cantools/cantools::0c6a7871'}
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c']:
        for k in json.loads((ACQ / kdir / 'keep-list.json').read_text()):
            forb_i.add(k['instance_id'])
            forb_s.add(f"{k['repo']}::{k['base_commit']}")
        forb_f |= set(json.loads((ACQ / kdir / 'keep-families.json').read_text()).values())
    return forb_i, forb_s, forb_f


MECH = {
    'theacodes__nox-439': {
        'family': 'theacodes/nox::nox.sessions.SessionRunner.posargs::slice_copy_isolation',
        'contract': ('SessionRunner owns a slice copy of the global posargs list: '
                     'runner.posargs is not global_config.posargs; extending one session '
                     'leaves the global config and sibling sessions untouched; captured at '
                     'SessionRunner init timepoint.'),
        'trigger': 'Create two sessions sharing a global config, extend one posargs.',
        'base_wrong_behavior': ('SessionRunner aliases global_config.posargs; mutating one '
                                'session pollutes the global config and sibling sessions.'),
        'production_symbols': ['nox.sessions.SessionRunner.__init__', 'nox.sessions.SessionRunner.posargs'],
        'causal_change': 'self.posargs = global_config.posargs[:].',
        'observation': ('from unittest import mock\nimport nox.sessions, nox.manifest\nfrom nox import _options\n'
                        'cfg = _options.options.namespace(posargs=["--orig"])\n'
                        'runner = nox.sessions.SessionRunner(name="test", signatures=["test"], func=mock.sentinel.func, global_config=cfg, manifest=mock.create_autospec(nox.manifest.Manifest))\n'
                        "print('copy_identity', runner.posargs is not cfg.posargs)\n"
                        'assert runner.posargs is not cfg.posargs\n'
                        'runner.posargs.append("--new")\n'
                        'assert cfg.posargs == ["--orig"]\n'
                        "print('f2_posargs_isolation', True)\n"),
        'triple_is': 'runner.posargs is not global_config.posargs (slice owns a new list)',
        'triple_shared': 'extending one session leaves the global config untouched',
        'triple_time': 'SessionRunner.__init__ copy timepoint',
    },
    'aio-libs__aiohttp-10672': {
        'family': 'aio-libs/aiohttp::aiohttp.web_response.Response.headers::copy_isolation',
        'contract': ('Response owns a copy of caller headers: resp.headers is not the input '
                     'CIMultiDict; preparing the response never mutates the shared input; '
                     'captured at the Response(headers=...) call timepoint.'),
        'trigger': 'Construct web.Response with a shared CIMultiDict and prepare it.',
        'base_wrong_behavior': ('Response aliases a CIMultiDict input; preparing one response '
                                'pollutes the caller dict with Content-Length/Date.'),
        'production_symbols': ['aiohttp.web_response.Response.__init__'],
        'causal_change': 'real_headers = CIMultiDict(headers) unconditionally.',
        'observation': ('from multidict import CIMultiDict\nfrom aiohttp import web\n'
                        'h = CIMultiDict({"X-Base": "1"})\n'
                        'r = web.Response(body=b"answer", headers=h)\n'
                        "print('copy_identity', r.headers is not h)\n"
                        'assert r.headers is not h\n'
                        'r.headers["X-New"] = "2"\n'
                        'assert "X-New" not in h\n'
                        "print('f2_headers_isolation', True)\n"),
        'triple_is': 'resp.headers is not the input headers dict',
        'triple_shared': 'mutating the response leaves the caller dict untouched',
        'triple_time': 'Response(headers=...) call timepoint',
    },
}

FINDING = ('Read real-PR fix diff, publisher problem_statement verbatim, and F2P/P2P node lists. '
           'F2 triple (is + shared visibility + copy timepoint) holds; only-== would demote to S, '
           'capability-only to R. No test/matcher edits; six-tool course; protected paths untouched.')

KEEPS = ['theacodes__nox-439', 'aio-libs__aiohttp-10672']


def main():
    forb_i, forb_s, forb_f = load_forbidden()
    keeps = [k for k in json.loads((OUT / 'keep-list.json').read_text()) if k['instance_id'] in KEEPS]
    fams = json.loads((OUT / 'keep-families.json').read_text())
    assert len(keeps) == 2
    tasks = []
    for k in keeps:
        iid = k['instance_id']
        assert iid not in forb_i, f'instance overlap {iid}'
        snap = f"{k['repo']}::{k['base_commit']}"
        for s in forb_s:
            if '::' not in s:
                continue
            repo, rev = s.split('::', 1)
            if repo.lower() == k['repo'].lower():
                c, r = k['base_commit'].lower(), rev.lower()
                assert not (c.startswith(r) or r.startswith(c)), f'snapshot overlap {iid} vs {s}'
        assert fams[iid] not in forb_f, f"family overlap {iid}"
        forb_i.add(iid)
        forb_s.add(snap)
        forb_f.add(fams[iid])
        page = OUT / f'pages-v2/{iid}.json'
        row = json.loads(page.read_text())['row']
        assert row['instance_id'] == iid and row['base_commit'] == k['base_commit']
        assert list(row['FAIL_TO_PASS']) == k['F2P_list'] and list(row['PASS_TO_PASS']) == k['P2P_list']
        m = MECH[iid]
        assert m['family'] == fams[iid]
        lic_file, lic_name, _ = LIC[k['repo']]
        lic_rel = f'competition/vfarl/acquisition/f2-new/f2-recall/{lic_file}'
        lic_sha = sha(ROOT / lic_rel)
        tasks.append({
            'instance_id': iid, 'canonical_repo': k['repo'],
            'primary_adapter': 'F', 'submechanism': 'F2',
            'repair_family_id': fams[iid], 'snapshot_group_id': snap,
            'execution_commit': k['base_commit'],
            'source_dataset': 'nebius/SWE-rebench-V2', 'source_revision': V2_REV,
            'physical_split': 'train',
            'source_task_path': f'competition/vfarl/acquisition/f2-new/f2-recall/pages-v2/{iid}.json',
            'source_task_sha256': sha(page),
            'environment': {
                'buggy_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                 'resolved_revision': k['base_commit']},
                'upstream_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                    'resolved_revision': k['base_commit']}},
            'license_evidence': {
                'source': {'license': lic_name, 'path': lic_rel,
                           'sha256': lic_sha, 'revision': k['base_commit']},
                'dataset': {'license': 'CC-BY-4.0', 'revision': V2_REV}},
            'mechanism_evidence': {
                'submechanism': 'F2', 'family': fams[iid], 'contract': m['contract'],
                'trigger': m['trigger'], 'base_wrong_behavior': m['base_wrong_behavior'],
                'production_symbols': m['production_symbols'], 'causal_change': m['causal_change'],
                'discriminating_observation': {
                    'kind': 'explicitly_reviewed_source_bound_inline_execution',
                    'description': m['observation']},
                'triple': {'is_identity': m['triple_is'],
                           'shared_mutation_visibility': m['triple_shared'],
                           'copy_timepoint': m['triple_time']},
                'bug_patch_direction': 'repair_target'},
            'original_patch_semantics': 'repair_target',
            'original_FAIL_TO_PASS_count': len(k['F2P_list']),
            'original_PASS_TO_PASS_count': len(k['P2P_list']),
            'source_mechanism_review_complete': True,
            'root_second_read': {'date': '2026-10-25', 'reviewer': 'root research agent',
                                 'finding': FINDING,
                                 'classification_scope': ('F2 reference/ownership isolation, not '
                                                          'value equality or capability enumeration.')},
            'use_partition': 'train_only', 'course_membership': 'F600_F2',
            'whitelisted_prompt': True, 'junit_dual_required': True,
            'tests_unmodified': True, 'six_tools_only': True,
            'protected_paths_untouched': True})
    reviewed = {
        'schema_version': 'f2-recall-reviewed-candidates-v1',
        'date': '2026-10-25',
        'scope': ('F600 F2 reference/ownership isolation recall-large round (Gym + Rebench-V2, suite<=200): '
                  '2 CLEAN keeps (nox-439, aiohttp-10672), each with is-identity + shared-mutation visibility + copy timepoint. '
                  'Only-== demotes to S; capability-only to R. Train only; four competition repos excluded; '
                  'disjoint from common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16/f2-gymv2b-5/f2-gymv2c-2. '
                  'keep-list holds 3 triple-KEEPs (hdmf-848 triple-KEEP but skip-DROP blocked, retained for dedup, excluded from reviewed).'),
        'pools': {'nebius/SWE-rebench-V2': V2_REV},
        'tasks': tasks}
    (OUT / 'reviewed-candidates.json').write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=1) + '\n')
    rsha = sha(OUT / 'reviewed-candidates.json')
    catalog = {'schema_version': 'reviewed-f2-recall-training-catalog-v1',
               'date': '2026-10-25',
               'selection_policy': ('Explicit F600-F2 train-only; Gym + Rebench-V2 recall-large suite<=200; disjoint from '
                                    'common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16/f2-gymv2b-5/f2-gymv2c-2 '
                                    'families/snapshots/instances; skip-collision prechecked; '
                                    'normalize eligible only.'),
               'tasks': []}
    for t in tasks:
        c = {kk: t[kk] for kk in ('instance_id', 'canonical_repo', 'primary_adapter', 'submechanism',
                               'repair_family_id', 'snapshot_group_id', 'execution_commit',
                               'source_dataset', 'source_revision', 'physical_split',
                               'original_FAIL_TO_PASS_count', 'original_PASS_TO_PASS_count',
                               'mechanism_evidence', 'original_patch_semantics', 'use_partition',
                               'course_membership', 'source_task_path', 'source_task_sha256',
                               'environment', 'license_evidence', 'source_mechanism_review_complete',
                               'root_second_read')}
        c['source_review_path'] = 'competition/vfarl/acquisition/f2-new/f2-recall/reviewed-candidates.json'
        c['source_review_sha256'] = rsha
        catalog['tasks'].append(c)
    (OUT / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n')
    base = {k['instance_id']: k['base_commit'] for k in keeps}
    (OUT / 'base-resolutions.json').write_text(json.dumps(base, ensure_ascii=False, indent=1) + '\n')
    print(f'keeps=2 zero-overlap verified; reviewed + catalog written; review_sha={rsha[:12]}')


if __name__ == '__main__':
    main()
