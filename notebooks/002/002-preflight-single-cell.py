#!/usr/bin/env python3
# 002 preflight single-cell carrier (stage 0A CPU + 0B eligibility framework).
# Paste entire file into one Kaggle notebook code cell and run top-to-bottom.
# CPU-only by default: fingerprints, static param evidence, skill checks,
# eligibility skeleton. GPU reasoning probe runs only when model mount exists
# and RUN_MINIMAL_GPU_PROBE is True (separate frozen run_id, never mixed).
# No heldout content is read. No submissions. Stdlib-first; pip install only
# frozen official wheels already mounted.
from pathlib import Path
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid

RUN_MINIMAL_GPU_PROBE = False
RUN_ID = 'run-002-preflight-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
INPUT_ROOT = Path('/kaggle/input')
WORKING = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
OUT_ROOT = WORKING / 'results' / '002' / RUN_ID


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def input_snapshot(limit=80):
    try:
        if not INPUT_ROOT.is_dir():
            return ['/kaggle/input missing (local mode)']
        files = sorted(str(p.relative_to(INPUT_ROOT)) for p in INPUT_ROOT.rglob('*') if p.is_file())
        return files[:limit] + ([f'...(+{len(files)-limit} more)'] if len(files) > limit else [])
    except Exception as e:
        return [f'list-failed: {e}']


def resolve_input(candidates, marker, substr=None):
    for c in candidates:
        if c.exists():
            print(f'input resolved: {c}', flush=True)
            return c
    found = []
    if INPUT_ROOT.is_dir():
        for p in INPUT_ROOT.rglob(marker):
            if substr is None or substr in str(p):
                found.append(p.parent)
    uniq = sorted(set(found))
    if len(uniq) == 1:
        print(f'input resolved by marker {marker}: {uniq[0]}', flush=True)
        return uniq[0]
    return None


OUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f'RUN_ID={RUN_ID}', flush=True)
print(f'input_files={input_snapshot()}', flush=True)

DATA_DIR = resolve_input([INPUT_ROOT / 'competitions/gemma-4-developer-agent',
                          INPUT_ROOT / 'gemma-4-developer-agent'], 'tasks.jsonl')
WHEEL_DIR = resolve_input([INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
                           INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
                          'swegemma-0.2.7-py3-none-any.whl')
MODEL_DIR = resolve_input([INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
                           INPUT_ROOT / 'gemma-4-31b-it-qat-w4a16-ct/2'],
                          'tokenizer.json', 'gemma-4-31b')

evidence = {'run_id': RUN_ID, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'python': sys.version, 'platform': platform.platform(),
            'input_snapshot': input_snapshot(200)}
if DATA_DIR is not None:
    tp = DATA_DIR / 'tasks.jsonl'
    if tp.is_file():
        raw = tp.read_bytes()
        evidence['tasks'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        print(f"tasks sha={evidence['tasks']['sha256'][:16]} bytes={len(raw)}", flush=True)
    for extra in ['HARNESS_README.md']:
        ep = DATA_DIR / extra
        if ep.is_file():
            evidence[extra] = {'sha256': sha256_file(ep), 'bytes': ep.stat().st_size}
if WHEEL_DIR is not None:
    wheels = sorted(WHEEL_DIR.glob('*.whl'))
    evidence['wheels'] = [{'name': w.name, 'bytes': w.stat().st_size, 'sha256': sha256_file(w)} for w in wheels]
    print(f'wheels={len(wheels)}', flush=True)

# ---- static param evidence (no GPU): replicate bridge mapping from source ----
def bridge(thinking_budget, include_thoughts, thinking_level=None):
    if include_thoughts is False or (isinstance(thinking_level, str) and thinking_level.lower() == 'none'):
        return {'enable_thinking': False}
    if include_thoughts is True or (thinking_budget is not None and int(thinking_budget) > 0):
        return {'enable_thinking': True}
    return {}
probes = [
    {'in': (4096, True), 'out': bridge(4096, True)},
    {'in': (1024, True), 'out': bridge(1024, True)},
    {'in': (4096, False), 'out': bridge(4096, False)},
]
evidence['thinking_bridge_cpu'] = probes
evidence['thinking_warning'] = '4096 vs 1024 both map to enable_thinking:true; do NOT ablate as numeric budgets'
print(json.dumps(probes, ensure_ascii=False), flush=True)

# double-serialization demo (stdlib only)
harness_return = json.dumps({'status': 'ok', 'content': 'line one\nline two "q" \\ slashrouw'})
outer = json.dumps({'result': harness_return})
evidence['double_serialization_demo'] = {'inner': harness_return, 'outer': outer}
print(f'double-serialization outer bytes={len(outer)}', flush=True)

# escaping round-trip probe (synthetic, no answers)
cases = ['quote "hi"', 'backslash \\ path', 'newline\na\nb', 'unicode \u4e2d\u6587 \U0001f600', '{"k": "v\\n"}']
rt = []
for s in cases:
    blob = json.dumps({'status': 'ok', 'content': s})
    back = json.loads(blob)['content']
    rt.append({'input': s, 'roundtrip_ok': back == s, 'outer': json.dumps({'result': blob})[:120]})
evidence['escaping_roundtrip'] = rt
assert all(r['roundtrip_ok'] for r in rt), 'stdlib json roundtrip broken'
print('escaping roundtrip: all ok', flush=True)

# ---- skill package checks (works locally and on Kaggle) ----
repo = Path('.')
if (WORKING / 'agent_of_gemma_4').is_dir():
    repo = WORKING / 'agent_of_gemma_4'
skill_base = None
for cand in [repo / 'research/002/variants/t1/agent/skills/repo_navigation',
             Path('research/002/variants/t1/agent/skills/repo_navigation')]:
    if (cand / 'SKILL.md').is_file():
        skill_base = cand
        break
skill_report = {}
if skill_base is not None:
    import py_compile
    files = sorted([p for p in skill_base.rglob('*') if p.is_file()])
    total = 0
    bad = []
    for p in files:
        if p.is_symlink():
            bad.append(f'{p}: symlink')
        total += p.stat().st_size
        if p.suffix == '.py':
            try:
                py_compile.compile(str(p), doraise=True)
            except Exception as e:
                bad.append(f'{p}: compile {e}')
    skill_report = {'skill_dir': str(skill_base), 'files': [str(p.relative_to(skill_base)) for p in files],
                    'bytes': total, 'problems': bad,
                    'has_required': all((skill_base / x).is_file() for x in
                                        ['SKILL.md', 'scripts/find_symbol.py',
                                         'scripts/read_range.py', 'scripts/check_patch.py'])}
else:
    skill_report = {'error': 'skill dir not found (upload repo or run beside checkout)'}
evidence['skill'] = skill_report
print(json.dumps(skill_report, ensure_ascii=False)[:2000], flush=True)

# ---- eligibility framework (no agent runs here) ----
elig_path = None
for cand in [repo / 'research/002/verification_eligibility.json',
             Path('research/002/verification_eligibility.json')]:
    if cand.is_file():
        elig_path = cand
        break
if elig_path is not None:
    elig = json.loads(elig_path.read_text(encoding='utf-8'))
    evidence['eligibility'] = {'path': str(elig_path), 'tasks': len(elig.get('tasks', [])),
                               'rule': elig.get('eligibility_rule', '')[:300],
                               'status': elig.get('status', '')}
    print(f"eligibility framework: {evidence['eligibility']['tasks']} tasks, status={evidence['eligibility']['status']}", flush=True)
else:
    evidence['eligibility'] = {'error': 'verification_eligibility.json not found'}

(OUT_ROOT / 'preflight_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'WROTE {OUT_ROOT / "preflight_evidence.json"}', flush=True)
print('CPU preflight done. GPU minimal reasoning probe NOT run '
      f'(RUN_MINIMAL_GPU_PROBE={RUN_MINIMAL_GPU_PROBE}). '
      'Enable only in a separate frozen run after CPU evidence is reviewed.', flush=True)
