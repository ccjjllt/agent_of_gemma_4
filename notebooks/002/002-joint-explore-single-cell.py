# 002 stage-1 skill explore FROZEN (NOT RUN - awaits GPU quota refresh 2026-10-03).
# Frozen T0 vs T1 (skill only) on 16384 base (0C kept 16384).
# 6 tasks x 2 variants = 12 attempts. No result from another arm enters an agent.
# Required Kaggle inputs: competition, official wheelhouse, and model version 2.
# The harness performs the same official evaluation as the existing single-arm carriers.
from pathlib import Path
import asyncio
import concurrent.futures
import glob
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

# Change only this list when replacing still-queued arms. Completed arms remain historical.
SELECTED_VARIANTS = ('t0', 't1')
REPEAT_ID = 'r1'
RUN_ID = 'run-002-explore-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]


def input_snapshot(limit=120):
    # Distinguish "no inputs attached" from "wrong path" within the fail-fast window.
    root = Path('/kaggle/input')
    try:
        if not root.is_dir():
            return ['/kaggle/input missing']
        files = sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_file())
        return files[:limit] + ([f'...(+{len(files) - limit} more)'] if len(files) > limit else [])
    except Exception as exc:
        return [f'list-failed: {exc}']


INPUT_ROOT = Path('/kaggle/input')


def resolve_input(candidates, marker, marker_substr=None):
    # Mount layout varies by worker: nested (datasets/metric/…, competitions/…, models/…)
    # or flat (<slug>/). Prefer known candidates, else locate by unique marker file.
    # Defined before first use (module top-level executes in order).
    for cand in candidates:
        if cand.exists():
            print(f'input resolved: {cand}', flush=True)
            return cand
    found = []
    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob(marker):
            if marker_substr is None or marker_substr in str(path):
                found.append(path.parent)
    uniq = sorted(set(found))
    assert len(uniq) == 1, (f'Input {marker} not found: '
                            f'candidates={[str(c) for c in candidates]} scan={uniq[:5]} '
                            f'input_files={input_snapshot()}')
    print(f'input resolved by marker {marker}: {uniq[0]}', flush=True)
    return uniq[0]
BUNDLE = json.loads('{"t0": {"agent.yaml": "name: p002_t0_frozen_base\\nmodel: gemma-4-31b-it-qat-w4a16-ct\\ninstruction: !include prompts/system.md\\ntools:\\n  - run_command\\n  - read_file\\n  - edit_file\\n  - write_file\\n  - get_status\\n  - submit_patch\\ngenerate_content_config: !include configs/sampling.yaml\\n", "configs/sampling.yaml": "# P002-T0 frozen base sampling (inherited from P003-A0, 2026-09-30).\\n# max_output_tokens 16384 is the pre-capacity-contrast value.\\n# thinking_budget numeric value is NOT a proven token limit: current chain maps\\n# any budget>0 + include_thoughts:true to enable_thinking:true only.\\n# Do not run 4096-vs-1024 as a budget ablation.\\ntemperature: 0.2\\ntop_p: 0.95\\nmax_output_tokens: 16384\\nthinking_config:\\n  thinking_budget: 4096\\n  include_thoughts: true\\n", "eval_config.yaml": "# P002-T0 frozen eval budgets (same as P003 A0/B/C; 001 dev-start).\\n# timeout_seconds limits BOTH agent commands and final verification in swegemma 0.2.7.\\n# Any change must apply to all arms identically and is then a new research condition.\\nevaluation:\\n  timeout_seconds: 60\\n  max_tool_calls: 50\\n  max_time_minutes: 5\\n  max_turns: 80\\n", "prompts/system.md": "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\\n\\nInitial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\\n\\nOnly call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\\n\\n## Flow: locate -> read -> edit -> check -> submit\\n\\n1. Locate\\n- Extract file/function/class/error strings from the problem statement first.\\n- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).\\n- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.\\n\\n2. Read\\n- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.\\n- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\\n- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\\n\\n3. Edit\\n- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\\n- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\\n- Match required error strings, exception types, status codes, and API signatures exactly.\\n- On `FileEditError: old_string not found`: stop repeating. Re-read the exact file range, compare whitespace/escaping/line range byte-for-byte, and only then retry once with corrected text. Do not blindly retry identical parameters.\\n\\n4. Check\\n- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\\n- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\\n- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\\n- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\\n\\n5. Budget and submit\\n- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\\n- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\\n- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\\n"}, "t1": {"agent.yaml": "name: p002_t1_skill_nav\\nmodel: gemma-4-31b-it-qat-w4a16-ct\\ninstruction: !include prompts/system.md\\ntools:\\n  - run_command\\n  - read_file\\n  - edit_file\\n  - write_file\\n  - get_status\\n  - submit_patch\\nskills:\\n  - skills/repo_navigation\\ngenerate_content_config: !include configs/sampling.yaml\\n", "configs/sampling.yaml": "# P002-T1 sampling: identical to T0 during skill-effect isolation.\\n# If 0C adopts 8192 as the new common base, both T0 and T1 switch to 8192\\n# together before the phase-1 panel; that switch is a separately recorded\\n# manifest version, never mixed into one pairing.\\ntemperature: 0.2\\ntop_p: 0.95\\nmax_output_tokens: 16384\\nthinking_config:\\n  thinking_budget: 4096\\n  include_thoughts: true\\n", "eval_config.yaml": "# P002-T1 frozen eval budgets (same as T0; skill comparison isolates skill only).\\nevaluation:\\n  timeout_seconds: 60\\n  max_tool_calls: 50\\n  max_time_minutes: 5\\n  max_turns: 80\\n", "prompts/system.md": "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\\n\\nInitial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\\n\\nOnly call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\\n\\n## Flow: locate -> read -> edit -> check -> submit\\n\\n1. Locate\\n- Extract file/function/class/error strings from the problem statement first.\\n- If `run_skill_script` is registered then prefer the `repo_navigation` skill (`run_skill_script`) for symbol search, otherwise use `read_file`/`run_command` fallback (若run_skill_script已注册则优先用，否则用read_file/run_command回退). `run_skill_script` is auto-provided by SkillToolset when `skills: [skills/repo_navigation]` is declared, not via `tools:` registry list (do NOT add it to `tools:`). Fall back to scoped `run_command`/`read_file` also when the skill returns empty. The skill is deterministic text/AST search, not a graph oracle.\\n- Keep every search scoped; preserve hit locations. Do not treat same-name hits as proven call edges.\\n\\n2. Read\\n- Use the skill `read_range.py` (or `read_file(filepath, start_line, end_line)` with 1-based inclusive range) function- or local-scope first, expand only as needed.\\n- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\\n- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\\n\\n3. Edit\\n- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\\n- Before editing, re-read the exact range (skill `read_range.py` or `read_file`) and copy `old_string` byte-for-byte from the tool output. On `FileEditError: old_string not found`: stop repeating. Re-read, compare whitespace/escaping/line range, retry once corrected.\\n- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\\n- Match required error strings, exception types, status codes, and API signatures exactly.\\n\\n4. Check\\n- Before `submit_patch`, run the skill `check_patch.py` and read its report (diff files, syntax, `git diff --check`, temp/test-file mixing). It only reports; it never auto-deletes or auto-submits.\\n- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\\n- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\\n- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\\n- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\\n\\n5. Budget and submit\\n- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\\n- Skill use is voluntary for the model but measured: calling the skill does not guarantee success. If the skill returns empty or ambiguous candidates, record that and continue with scoped text search.\\n- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\\n- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\\n", "skills/repo_navigation/SKILL.md": "---\\nname: repo_navigation\\ndescription: Deterministic text/AST navigation for generic Python repos. Finds symbols (incl. async def), returns bounded source slices, and reports pre-submit patch hygiene. No network, no answers, stdlib only.\\n---\\n\\n# repo_navigation skill\\n\\nDeterministic helper. It does not fix code and does not submit. It only\\nreturns bounded evidence the agent must verify by reading the source.\\n\\nWorkspace root is `/workspace`. All paths are confined under it. Symlinks are\\nnever followed. Scripts use only the Python standard library. Default rejects\\nnon-`/workspace` `--root`; local diagnosis must pass explicit\\n`--allow-non-workspace-root`.\\n\\n## Scripts (run via `run_skill_script` in the workspace sandbox)\\n\\nAll scripts print a JSON object to stdout. Stable cap is 4000 chars per\\ncall on final stdout; caller sees `truncated: true`/`is_truncated: true` when\\ncapped. When truncated, continue with `next_start` sharding (read_range\\nreturns `next_start`; re-issue with smaller window and overlap).\\n\\n### 1. `find_symbol.py`\\n\\nSearch identifiers from the issue text (file/function/class/error strings).\\n\\n```sh\\npython3 find_symbol.py --symbol <name> [--root /workspace] [--max-hits 8] [--allow-non-workspace-root]\\npython3 find_symbol.py --symbol <name> --expand [--max-hits 8] [--allow-non-workspace-root]\\n```\\n\\n- Scans `*.py` under root, skipping hidden dirs, `venv/`, `node_modules/`,\\n  `__pycache__/`, files > 512 KiB.\\n- Matches: `def <name>`, `async def <name>`, `class <name>`, exact text hits.\\n- Default output: up to 8 candidates with `path`, `kind`\\n  (`def`/`async def`/`class`/`text`), `line`, `end_line`, short `snippet`.\\n- `--expand`: for each confirmed candidate, add static近似 reference/call\\n  candidates (`ref_hits`, max 8 total) found by bounded text search for the\\n  symbol name. Results are labelled `static_approx: true`; same-name hits are\\n  NOT proven call edges. Ambiguity is reported via `ambiguous: true`.\\n- Never reads reference patches, other attempts, or host files.\\n\\n### 2. `read_range.py`\\n\\nExact bounded read with truthful boundaries.\\n\\n```sh\\npython3 read_range.py --path <repo/relative.py> --start <1-based> --end <incl> [--max-chars 4000] [--root /workspace] [--allow-non-workspace-root]\\n```\\n\\n- Returns `start_line`, `end_line_shown`, `is_truncated`, `next_start`, and\\n  `text`. `end_line_shown` is the last line actually included, never a\\n  truncation-inflated value.\\n- Rejects absolute paths, `..` escapes, missing files, symlinks, non-`.py`\\n  outside the repo tree (only files under `/workspace`).\\n\\n### 3. `check_patch.py`\\n\\nPre-submit hygiene report. Read-only except `git diff` plumbing; it never\\ndeletes files and never calls submit.\\n\\n```sh\\npython3 check_patch.py [--root /workspace] [--allow-non-workspace-root]\\n```\\n\\nReports: `files_changed`, `production_files` (non-test source), `test_files_touched`,\\n`tmp_files` (repro/logs that would pollute the diff), `syntax_ok` per changed\\n`.py` (via `ast.parse`, incl. `async def` support), `diff_check` (runs\\n`git diff --check` when git is available), `diff_bytes`, `empty_source_diff`.\\nA non-empty report is not a pass/fail verdict; the agent decides.\\n\\n## Rules the agent must follow\\n\\n1. Copy `old_string` byte-for-byte from `read_range.py`/`read_file` output.\\n   On `old_string not found`, re-read before retrying; never blindly repeat.\\n2. Candidate count and output caps are initial values for measurement, not\\n   proven optima. Record when the skill returns empty/ambiguous.\\n3. Temporary analysis stays in `/tmp`; never write scratch files into the repo\\n   tree that would enter the final diff.\\n", "skills/repo_navigation/scripts/find_symbol.py": "#!/usr/bin/env python3\\n\\"\\"\\"Find Python symbols via text + AST. Stdlib only. Bounded output.\\n\\nUsage:\\n  python3 find_symbol.py --symbol NAME [--root /workspace] [--max-hits 8] [--expand] [--allow-non-workspace-root]\\n\\nConfined to --root (default /workspace). Never follows symlinks.\\nSkips hidden dirs, venv, node_modules, __pycache__, files > 512 KiB.\\nMatches def / async def / class / plain text hits for NAME.\\n--expand adds static-approx reference candidates (same-name text hits),\\nlabelled static_approx:true, never claimed as proven call edges.\\nNon-/workspace roots require --allow-non-workspace-root (local diagnosis only).\\n\\"\\"\\"\\nimport argparse\\nimport ast\\nimport json\\nimport os\\nfrom pathlib import Path\\n\\nSKIP_DIRS = {\\"__pycache__\\", \\"venv\\", \\".venv\\", \\"node_modules\\", \\".git\\", \\".hg\\"}\\nMAX_FILE_BYTES = 512 * 1024\\nMAX_SNIPPET_CHARS = 400\\n\\n\\ndef safe_root(root: str, allow_non_ws: bool = False) -> Path:\\n    p = Path(root).resolve()\\n    ws = Path(\\"/workspace\\").resolve()\\n    # Enforce workspace confinement: ws variable gates non-/workspace roots.\\n    # Kaggle run must use /workspace; local diagnosis needs explicit flag.\\n    if p.is_symlink():\\n        raise SystemExit(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"symlink root rejected\\"}))\\n    if p != ws and not allow_non_ws:\\n        raise SystemExit(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)\\"}))\\n    return p\\n\\n\\ndef iter_py_files(root: Path):\\n    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):\\n        # prune in place\\n        dirnames[:] = [d for d in dirnames\\n                       if not d.startswith(\\".\\") and d not in SKIP_DIRS]\\n        # never descend through symlinked dirs\\n        dirnames[:] = [d for d in dirnames\\n                       if not Path(dirpath, d).is_symlink()]\\n        for fn in filenames:\\n            if not fn.endswith(\\".py\\"):\\n                continue\\n            fp = Path(dirpath, fn)\\n            if fp.is_symlink():\\n                continue\\n            try:\\n                if fp.stat().st_size > MAX_FILE_BYTES:\\n                    continue\\n            except OSError:\\n                continue\\n            yield fp\\n\\n\\ndef snippet_of(lines, lineno, width=3):\\n    lo = max(1, lineno - width)\\n    hi = lineno + width\\n    chunk = \\"\\".join(lines[lo - 1:hi])[:MAX_SNIPPET_CHARS]\\n    return chunk\\n\\n\\ndef find_in_file(fp: Path, root: Path, symbol: str):\\n    cands = []\\n    try:\\n        text = fp.read_text(encoding=\\"utf-8\\", errors=\\"replace\\")\\n    except OSError:\\n        return cands\\n    lines = text.splitlines(keepends=True)\\n    rel = fp.relative_to(root).as_posix()\\n    # AST pass for def/async def/class (exact name match)\\n    try:\\n        tree = ast.parse(text)\\n        for node in ast.walk(tree):\\n            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \\\\\\n                    and node.name == symbol:\\n                kind = (\\"async def\\" if isinstance(node, ast.AsyncFunctionDef)\\n                        else \\"def\\" if isinstance(node, ast.FunctionDef) else \\"class\\")\\n                end = getattr(node, \\"end_lineno\\", node.lineno)\\n                cands.append({\\n                    \\"path\\": rel, \\"kind\\": kind, \\"symbol\\": symbol,\\n                    \\"line\\": node.lineno, \\"end_line\\": end,\\n                    \\"snippet\\": snippet_of(lines, node.lineno),\\n                    \\"source\\": \\"ast\\",\\n                })\\n    except SyntaxError:\\n        pass\\n    # Text fallback: exact substring hits (bounded per file)\\n    for i, ln in enumerate(lines, 1):\\n        if symbol in ln and len([c for c in cands if c[\\"path\\"] == rel]) < 4:\\n            # skip lines already covered by AST hits nearby\\n            if any(abs(c[\\"line\\"] - i) <= 1 and c[\\"path\\"] == rel for c in cands):\\n                continue\\n            cands.append({\\n                \\"path\\": rel, \\"kind\\": \\"text\\", \\"symbol\\": symbol,\\n                \\"line\\": i, \\"end_line\\": i,\\n                \\"snippet\\": ln.strip()[:MAX_SNIPPET_CHARS],\\n                \\"source\\": \\"text\\",\\n            })\\n            if len(cands) >= 12:\\n                break\\n    return cands\\n\\n\\ndef main():\\n    ap = argparse.ArgumentParser()\\n    ap.add_argument(\\"--symbol\\", required=True)\\n    ap.add_argument(\\"--root\\", default=\\"/workspace\\")\\n    ap.add_argument(\\"--max-hits\\", type=int, default=8)\\n    ap.add_argument(\\"--expand\\", action=\\"store_true\\")\\n    ap.add_argument(\\"--allow-non-workspace-root\\", action=\\"store_true\\",\\n                    help=\\"Allow non-/workspace --root for local diagnosis only.\\")\\n    args = ap.parse_args()\\n    symbol = args.symbol.strip()\\n    out = {\\"status\\": \\"ok\\", \\"symbol\\": symbol, \\"candidates\\": [],\\n           \\"truncated\\": False, \\"static_approx\\": False, \\"ambiguous\\": False}\\n    if not symbol or len(symbol) > 256 or any(c in symbol for c in \\"\\\\n\\\\r\\\\x00\\"):\\n        out.update({\\"status\\": \\"error\\", \\"error\\": \\"bad symbol\\"})\\n        print(json.dumps(out, ensure_ascii=False))\\n        return\\n    try:\\n        root = safe_root(args.root, args.allow_non_workspace_root)\\n    except SystemExit as e:\\n        print(str(e))\\n        return\\n    if not root.is_dir():\\n        out.update({\\"status\\": \\"error\\", \\"error\\": \\"root missing\\"})\\n        print(json.dumps(out, ensure_ascii=False))\\n        return\\n    max_hits = min(max(1, args.max_hits), 8)\\n    all_cands = []\\n    for fp in iter_py_files(root):\\n        all_cands.extend(find_in_file(fp, root, symbol))\\n        if len(all_cands) >= 64:\\n            break\\n    # Rank: ast hits first, then shorter paths, then line number\\n    def rank(c):\\n        return (0 if c[\\"source\\"] == \\"ast\\" else 1, len(c[\\"path\\"]), c[\\"line\\"])\\n    all_cands.sort(key=rank)\\n    picked = all_cands[:max_hits]\\n    out[\\"candidates\\"] = picked\\n    out[\\"truncated\\"] = len(all_cands) > len(picked)\\n    out[\\"total_raw\\"] = len(all_cands)\\n    # Ambiguity: same symbol in >1 file\\n    files = {c[\\"path\\"] for c in picked}\\n    out[\\"ambiguous\\"] = len(files) > 1\\n    if args.expand and picked:\\n        refs = []\\n        for fp in iter_py_files(root):\\n            try:\\n                text = fp.read_text(encoding=\\"utf-8\\", errors=\\"replace\\")\\n            except OSError:\\n                continue\\n            rel = fp.relative_to(root).as_posix()\\n            for i, ln in enumerate(text.splitlines(), 1):\\n                if symbol in ln and not any(c[\\"path\\"] == rel and c[\\"line\\"] == i for c in picked):\\n                    refs.append({\\"path\\": rel, \\"line\\": i,\\n                                 \\"snippet\\": ln.strip()[:MAX_SNIPPET_CHARS]})\\n                    if len(refs) >= max_hits:\\n                        break\\n            if len(refs) >= max_hits:\\n                break\\n        out[\\"ref_hits\\"] = refs\\n        out[\\"static_approx\\"] = True\\n    blob = json.dumps(out, ensure_ascii=False)\\n    if len(blob) > 4000:\\n        # hard cap: keep candidates count but trim snippets\\n        for c in out[\\"candidates\\"]:\\n            c[\\"snippet\\"] = c[\\"snippet\\"][:120]\\n        for r in out.get(\\"ref_hits\\", []):\\n            r[\\"snippet\\"] = r[\\"snippet\\"][:120]\\n        out[\\"truncated\\"] = True\\n        blob = json.dumps(out, ensure_ascii=False)[:4000]\\n        print(blob)\\n    else:\\n        print(blob)\\n\\n\\nif __name__ == \\"__main__\\":\\n    main()\\n", "skills/repo_navigation/scripts/read_range.py": "#!/usr/bin/env python3\\n\\"\\"\\"Bounded exact read with truthful boundaries. Stdlib only.\\n\\nUsage:\\n  python3 read_range.py --path REL --start N --end M [--root /workspace] [--max-chars 4000] [--allow-non-workspace-root]\\n\\nReturns start_line, end_line_shown (last line actually included),\\nis_truncated, next_start, text. Never inflates end_line beyond shown text.\\nRejects absolute paths, .. escapes, symlinks, files outside root.\\nNon-/workspace roots require --allow-non-workspace-root (local diagnosis only).\\n\\"\\"\\"\\nimport argparse\\nimport json\\nfrom pathlib import Path\\n\\n\\ndef main():\\n    ap = argparse.ArgumentParser()\\n    ap.add_argument(\\"--path\\", required=True)\\n    ap.add_argument(\\"--start\\", type=int, required=True)\\n    ap.add_argument(\\"--end\\", type=int, required=True)\\n    ap.add_argument(\\"--root\\", default=\\"/workspace\\")\\n    ap.add_argument(\\"--max-chars\\", type=int, default=4000)\\n    ap.add_argument(\\"--allow-non-workspace-root\\", action=\\"store_true\\",\\n                    help=\\"Allow non-/workspace --root for local diagnosis only.\\")\\n    args = ap.parse_args()\\n    out = {\\"status\\": \\"ok\\"}\\n    root = Path(args.root).resolve()\\n    ws = Path(\\"/workspace\\").resolve()\\n    if root.is_symlink():\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"symlink root rejected\\"}))\\n        return\\n    if root != ws and not args.allow_non_workspace_root:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)\\"}))\\n        return\\n    rel = args.path\\n    if rel.startswith(\\"/\\") or \\"..\\" in Path(rel).parts:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"path must be repo-relative without ..\\"}))\\n        return\\n    fp = (root / rel)\\n    try:\\n        rp = fp.resolve()\\n    except OSError:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"resolve failed\\"}))\\n        return\\n    try:\\n        rp.relative_to(root)\\n    except ValueError:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"path escapes root\\"}))\\n        return\\n    if rp.is_symlink() or not rp.is_file():\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"missing or symlink\\"}))\\n        return\\n    if args.start < 1 or args.end < args.start:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"bad range\\"}))\\n        return\\n    try:\\n        lines = rp.read_text(encoding=\\"utf-8\\", errors=\\"replace\\").splitlines(keepends=True)\\n    except OSError:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"read failed\\"}))\\n        return\\n    total = len(lines)\\n    s = min(args.start, total) if total else args.start\\n    e = min(args.end, total)\\n    # accumulate until max-chars (stable cap 4000; use next_start to continue)\\n    max_chars = min(max(args.max_chars, 256), 4000)\\n    buf = []\\n    used = 0\\n    last = s - 1\\n    for i in range(s, e + 1):\\n        ln = lines[i - 1] if 1 <= i <= total else \\"\\"\\n        if used + len(ln) > max_chars and buf:\\n            break\\n        buf.append(ln)\\n        used += len(ln)\\n        last = i\\n    text = \\"\\".join(buf)\\n    truncated = (last < e) or (e < args.end)\\n    out.update({\\n        \\"path\\": rel,\\n        \\"start_line\\": s if total else args.start,\\n        \\"end_line_shown\\": last if buf else s - 1,\\n        \\"end_line_requested\\": args.end,\\n        \\"total_lines\\": total,\\n        \\"is_truncated\\": truncated,\\n        \\"next_start\\": (last + 1) if truncated and last < total else None,\\n        \\"chars\\": len(text),\\n        \\"text\\": text[:max_chars],\\n    })\\n    blob = json.dumps(out, ensure_ascii=False)\\n    print(blob[:4000])\\n\\n\\nif __name__ == \\"__main__\\":\\n    main()\\n", "skills/repo_navigation/scripts/check_patch.py": "#!/usr/bin/env python3\\n\\"\\"\\"Pre-submit hygiene report. Read-only (except git plumbing reads). Stdlib only.\\n\\nUsage: python3 check_patch.py [--root /workspace] [--allow-non-workspace-root]\\nReports files_changed, production/test/tmp splits, per-file syntax via\\nast.parse, git diff --check when available, diff size, empty-source flag.\\nNever deletes files, never submits. Scratch files belong in /tmp, not repo.\\nNon-/workspace roots require --allow-non-workspace-root (local diagnosis only).\\n\\"\\"\\"\\nimport argparse\\nimport ast\\nimport json\\nimport subprocess\\nfrom pathlib import Path\\n\\nTEST_MARKERS = (\\"test_\\", \\"_test.py\\", \\"/tests/\\", \\"tests/\\")\\nTMP_SUFFIX = (\\".log\\", \\".tmp\\", \\".bak\\", \\".orig\\", \\".rej\\", \\".pyc\\")\\nTMP_NAMES = (\\"repro_\\", \\"reproduce\\", \\"scratch\\", \\"debug_\\")\\n\\n\\ndef sh(cmd, cwd, timeout=20):\\n    try:\\n        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)\\n        return r.returncode, (r.stdout or \\"\\")[:6000], (r.stderr or \\"\\")[:2000]\\n    except Exception as e:\\n        return -1, \\"\\", f\\"exec-failed: {e}\\"\\n\\n\\ndef main():\\n    ap = argparse.ArgumentParser()\\n    ap.add_argument(\\"--root\\", default=\\"/workspace\\")\\n    ap.add_argument(\\"--allow-non-workspace-root\\", action=\\"store_true\\",\\n                    help=\\"Allow non-/workspace --root for local diagnosis only.\\")\\n    args = ap.parse_args()\\n    root = Path(args.root).resolve()\\n    ws = Path(\\"/workspace\\").resolve()\\n    out = {\\"status\\": \\"ok\\", \\"root\\": str(root)}\\n    if root.is_symlink() or not root.is_dir():\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"bad root\\"}))\\n        return\\n    if root != ws and not args.allow_non_workspace_root:\\n        print(json.dumps({\\"status\\": \\"error\\", \\"error\\": \\"non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)\\"}))\\n        return\\n    code, diff, err = sh([\\"git\\", \\"diff\\", \\"--binary\\", \\"--name-only\\", \\"--\\", \\".\\"],\\n                         cwd=root)\\n    if code != 0:\\n        code2, diff2, err2 = sh([\\"git\\", \\"diff\\", \\"--name-only\\", \\"--\\", \\".\\"], cwd=root)\\n        code, diff, err = code2, diff2, err2\\n    files = [l.strip() for l in diff.splitlines() if l.strip()] if code == 0 else []\\n    out[\\"git_available\\"] = (code == 0)\\n    out[\\"git_stderr\\"] = err[:500] if code != 0 else \\"\\"\\n    out[\\"files_changed\\"] = files\\n    prod, tests, tmp = [], [], []\\n    for f in files:\\n        low = f.lower()\\n        is_test = low.startswith(\\"tests/\\") or \\"/tests/\\" in low or \\\\\\n            Path(f).name.startswith(\\"test_\\") or low.endswith(\\"_test.py\\")\\n        is_tmp = low.endswith(TMP_SUFFIX) or any(k in Path(f).name for k in TMP_NAMES)\\n        if is_tmp:\\n            tmp.append(f)\\n        elif is_test:\\n            tests.append(f)\\n        else:\\n            prod.append(f)\\n    out[\\"production_files\\"] = prod\\n    out[\\"test_files_touched\\"] = tests\\n    out[\\"tmp_files\\"] = tmp\\n    # syntax check on changed .py files\\n    syntax = {}\\n    for f in files:\\n        if not f.endswith(\\".py\\"):\\n            continue\\n        fp = root / f\\n        try:\\n            rp = fp.resolve()\\n            rp.relative_to(root)\\n            if rp.is_symlink() or not rp.is_file():\\n                syntax[f] = \\"missing-or-symlink\\"\\n                continue\\n            src = rp.read_text(encoding=\\"utf-8\\", errors=\\"replace\\")\\n            ast.parse(src)  # supports async def\\n            syntax[f] = \\"ok\\"\\n        except SyntaxError as e:\\n            syntax[f] = f\\"syntax-error: {e}\\"\\n        except Exception as e:\\n            syntax[f] = f\\"read-failed: {e}\\"\\n    out[\\"syntax\\"] = syntax\\n    out[\\"syntax_ok\\"] = all(v == \\"ok\\" for v in syntax.values())\\n    c2, check_out, check_err = sh([\\"git\\", \\"diff\\", \\"--check\\", \\"--\\", \\".\\"], cwd=root)\\n    out[\\"diff_check_exit\\"] = c2\\n    out[\\"diff_check\\"] = (check_out + check_err)[:2000]\\n    c3, numstat, _ = sh([\\"git\\", \\"diff\\", \\"--numstat\\", \\"--\\", \\".\\"], cwd=root)\\n    out[\\"numstat\\"] = numstat[:2000]\\n    c4, stat, _ = sh([\\"git\\", \\"diff\\", \\"--stat\\", \\"--\\", \\".\\"], cwd=root)\\n    out[\\"stat\\"] = stat[:2000]\\n    out[\\"empty_source_diff\\"] = len(prod) == 0\\n    blob = json.dumps(out, ensure_ascii=False)\\n    print(blob[:4000] if len(blob) > 4000 else blob)\\n\\n\\nif __name__ == \\"__main__\\":\\n    main()\\n"}}')
MANIFEST_TEXT = '{"study": "P002 stage-1 skill explore (frozen, NOT RUN - awaits GPU quota refresh 2026-10-03)", "capacity_base": "16384 (0C decision 2026-09-30: keep 16384; 8192 not promoted)", "common": {"model_alias": "gemma-4-31b-it-qat-w4a16-ct", "official_packages": {"adk-eval-core": "0.1.0 sha256 b1608c4471a2739eca6469735e4e118ff33a20823ce9430719238ea843aa9d1a", "adk-submission": "0.2.11 sha256 d14f7d31e724f507322c763bc7c1ed56ac603085b3709afed954e50006b494a8", "google-adk": "1.36.1 sha256 1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0", "swegemma": "0.2.7 sha256 2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98"}, "eval_budgets": {"timeout_seconds": 60, "max_tool_calls": 50, "max_time_minutes": 5.0, "max_turns": 80}, "tool_list": ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"], "graph_tools": [], "sandbox": "subprocess", "sampling": {"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 16384, "thinking_config": {"thinking_budget": 4096, "include_thoughts": true}}, "single_variable": "skill only (T1 = T0 + skills/repo_navigation + skill-aware prompt); sampling/budgets identical on 16384 base"}, "data": {"source_sha256": "e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6", "tasks_bytes": 1984455, "splits_path": "research/task_splits.json", "splits_seed": 20260928}, "explore_set": {"tasks": [{"id": "rich_3942", "reason": "context-fault case (t0 ContextWindowExceeded in 0C)"}, {"id": "rich_3130", "reason": "late-edit failure case"}, {"id": "fastapi_14873", "reason": "tool-loop case"}, {"id": "fastapi_5077", "reason": "fastapi eligible (differs from 0C fastapi_9425; avoids winner-picking)"}, {"id": "requests_6644", "reason": "stable success + requests"}, {"id": "rich_3518", "reason": "stable success control"}], "orders": ["AB", "BA", "AB", "BA", "AB", "BA"], "total_attempts": 12, "note": "role mix like 0C, separately frozen list; 4/6 eligible per 0B-official (rich_3130 gold-fail, requests_6644 already-pass unknown); dual denominators reported; at most 1 candidate advances to full 24 panel"}, "heldout_note": "heldout 28 frozen; no intersection"}'
EXPERIMENT = json.loads(MANIFEST_TEXT)
EXPECTED_CONFIG_HASHES = {'t0': {'agent.yaml': 'ae12f5f5911c855970b075a72fa2571726efe86f90130b24bd139ce47906e6d0', 'configs/sampling.yaml': '5a6acb429b95163d65f68ff483ebe83031b025eb818840b39f6d5b718f09645e', 'eval_config.yaml': '2a0205536c9d418d8502be4b13b3c063e1c0800d591e38c72d651a092de2b147', 'prompts/system.md': '6c2e8f4c90ec07fa973be9cefa9c04176f99acdef3ab29a985302e04203a17b5'}, 't1': {'agent.yaml': 'a170ab3de4ae67b66bc8e6763679ad7b867a4414d30e747f617e16146c38ebd1', 'configs/sampling.yaml': '392d051b7f8229fbb9225cc833c4605a17fec594b6f0c417d90e081a35a2b102', 'eval_config.yaml': '0c6b076d4b650c33e863e739a2b6ca4db38385699e0370645522f0c0328641d9', 'prompts/system.md': '011ea0c0b99b130b7249dc2754b3201d5160d05a4085da140f3d4cf1e06646ba', 'skills/repo_navigation/SKILL.md': '675b5fb44ea89aace205d2a20de84e32537d59cd6b0fef8a1b2d543cc8eee695', 'skills/repo_navigation/scripts/find_symbol.py': '18e1b6fb87ba5fe7f63a38519ae36ea08b0162ae6559519219cce42acc59e53e', 'skills/repo_navigation/scripts/read_range.py': 'cce9d68a77efd00b188046f00c369abe2edcb22c484ff96e2a4d409f8baf5854', 'skills/repo_navigation/scripts/check_patch.py': 'f8d0675b5fb92f67ee58feae21beac918b30fac071fd96a2a723ce2da13afc9b'}}
WORKING_DIR = Path('/kaggle/working')
RUN_ROOT = WORKING_DIR / 'results' / '002' / RUN_ID
AGENT_ROOT = WORKING_DIR / 'agent_exec' / RUN_ID
DATA_DIR = resolve_input(
    [INPUT_ROOT / 'competitions/gemma-4-developer-agent', INPUT_ROOT / 'gemma-4-developer-agent'],
    'tasks.jsonl')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
WHEELHOUSE_DIR = resolve_input(
    [INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
     INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
    'swegemma-0.2.7-py3-none-any.whl')
MODEL_PATH = resolve_input(
    [INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
     INPUT_ROOT / 'gemma-4-31b-it-qat-w4a16-ct/2'],
    'tokenizer.json', 'gemma-4-31b')
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
assert SELECTED_VARIANTS and len(set(SELECTED_VARIANTS)) == len(SELECTED_VARIANTS)
assert set(SELECTED_VARIANTS) <= {'t0', 't1'}, 'Unknown variant: stop, never substitute'
assert not RUN_ROOT.exists() and not AGENT_ROOT.exists(), 'Never overwrite a prior run'
RUN_ROOT.mkdir(parents=True)
AGENT_ROOT.mkdir(parents=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
atomic_text(FROZEN_DIR / 'experiment_manifest.json', MANIFEST_TEXT)
atomic_json(FROZEN_DIR / 'embedded_config_sha256.json', EXPECTED_CONFIG_HASHES)
for variant, file_map in BUNDLE.items():
    for relative, content in file_map.items():
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        assert digest == EXPECTED_CONFIG_HASHES[variant][relative], f'Config changed: {variant}/{relative}'
        atomic_text(FROZEN_DIR / 'variants' / variant / relative, content)
        atomic_text(AGENT_ROOT / variant / relative, content)

# 0C capacity schedule: 6 tasks x 2 repeats = 12 slots, AB/BA alternation frozen; 24 attempts.
letters = {'A': 't0', 'B': 't1'}
exploration = {'tasks': [{'id': 'rich_3942'}, {'id': 'rich_3130'}, {'id': 'fastapi_14873'}, {'id': 'fastapi_5077'}, {'id': 'requests_6644'}, {'id': 'rich_3518'}], 'orders': ['AB', 'BA', 'AB', 'BA', 'AB', 'BA']}
assert len(exploration['tasks']) == len(exploration['orders']) == 6
assert exploration['orders'] == ['AB', 'BA', 'AB', 'BA', 'AB', 'BA']
full_schedule = []
for task_slot, (item, order) in enumerate(zip(exploration['tasks'], exploration['orders']), 1):
    for within_task, letter in enumerate(order, 1):
        variant = letters[letter]
        ordinal = len(full_schedule) + 1
        attempt_id = f'{ordinal:02d}-{item["id"]}-{variant}'
        full_schedule.append({
            'ordinal': ordinal, 'task_slot': task_slot, 'within_task': within_task,
            'task_id': item['id'], 'variant': variant, 'attempt_id': attempt_id,
            'selected': variant in SELECTED_VARIANTS,
            'results_dir': str(RUN_ROOT / variant / REPEAT_ID / 'attempts' / attempt_id),
            'config_sha256': EXPECTED_CONFIG_HASHES[variant],
        })
schedule = [entry for entry in full_schedule if entry['selected']]
assert len(schedule) == 6 * len(SELECTED_VARIANTS)
state = {entry['attempt_id']: {**entry, 'status': 'planned' if entry['selected'] else 'excluded_variant'}
         for entry in full_schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'selected_schedule.json', schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
print(f'RUN_ID={RUN_ID} SELECTED_VARIANTS={SELECTED_VARIANTS} ATTEMPTS={len(schedule)}', flush=True)
for entry in schedule:
    print(f'  {entry["ordinal"]:02d}: {entry["task_id"]} {entry["variant"]}', flush=True)

# Install before importing target packages. A warm kernel is rejected, never silently reused.
loaded = [name for name in ('torch', 'swegemma', 'adk_submission', 'adk_eval_core',
                           'google.adk', 'vllm', 'litellm') if name in sys.modules]
assert not loaded, f'Restart the kernel before running: packages already loaded: {loaded}'
os.environ.update({
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for path in glob.glob(pattern):
        try:
            os.unlink(path)
        except OSError:
            pass
assert WHEELHOUSE_DIR.is_dir(), f'Missing wheelhouse: {WHEELHOUSE_DIR} input_files={input_snapshot()}'
source_wheels = [path for path in sorted(WHEELHOUSE_DIR.glob('*.whl')) if 'cutlass' not in path.name.lower()]
assert source_wheels, 'Wheelhouse is empty'
wheel_manifest = []
seen_distributions = set()
for path in source_wheels:
    distribution = path.name.split('-')[0].replace('_', '-').lower()
    assert distribution not in seen_distributions, f'Duplicate distribution: {distribution}'
    seen_distributions.add(distribution)
    wheel_manifest.append({'name': path.name, 'distribution': distribution,
                           'size': path.stat().st_size, 'sha256': sha256_file(path)})
for distribution, expected in EXPERIMENT['common']['official_packages'].items():
    version, expected_sha = expected.split(' sha256 ')
    matching = [entry for entry in wheel_manifest if entry['distribution'] == distribution]
    assert len(matching) == 1 and matching[0]['sha256'] == expected_sha, f'Official package changed: {distribution}'
atomic_json(FROZEN_DIR / 'wheel_manifest.json', wheel_manifest)
staging = Path('/tmp') / ('wheelhouse-' + RUN_ID)
staging.mkdir(exist_ok=False)
for path in source_wheels:
    name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    target = staging / name
    assert not target.exists() and not target.is_symlink(), f'Staging collision: {name}'
    target.symlink_to(path)
wheels = sorted(str(path) for path in staging.glob('*.whl'))
assert len(wheels) == len(wheel_manifest)
print(f'Installing {len(wheels)} frozen wheels', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import torch
import yaml
import litellm
from adk_submission import VllmConfig, VllmServer, discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import (ALLOWED_ADAPTER_EXTENSIONS, ALLOWED_SUBMISSION_EXTENSIONS,
                            MAX_SUBMISSION_SIZE_BYTES, EvalConfig, build_submission_limits)
from swegemma.context import SwegemmaContext
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import save_trace_artifact
from swegemma.models import load_tasks
from swegemma.models.discovery import validate_single_declared_model
from swegemma.results import append_task_result

yaml.SafeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
versions = {name: metadata.version(name) for name in
            ('swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'vllm', 'litellm', 'torch')}
for name, expected in EXPERIMENT['common']['official_packages'].items():
    assert versions[name] == expected.split(' sha256 ')[0], f'Installed version mismatch: {name}'
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'CUDA unavailable'
gpus = [{'index': i, 'name': torch.cuda.get_device_properties(i).name,
         'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
assert TASKS_PATH.is_file() and MODEL_PATH.is_dir(), \
    f'Competition data or pinned model is missing input_files={input_snapshot()}'
raw_tasks = TASKS_PATH.read_bytes()
tasks_sha = hashlib.sha256(raw_tasks).hexdigest()
assert tasks_sha == EXPERIMENT['data']['source_sha256'], 'tasks.jsonl changed; stop before inference'
assert len(raw_tasks) == EXPERIMENT['data']['tasks_bytes'], 'tasks.jsonl size mismatch'
(FROZEN_DIR / 'tasks.jsonl').write_bytes(raw_tasks)
tasks = load_tasks(TASKS_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(by_id) == len(tasks), 'Duplicate task IDs'
for entry in schedule:
    assert entry['task_id'] in by_id, f'Missing frozen task: {entry["task_id"]}'

common = EXPERIMENT['common']
common_sampling = None
common_config = None
budgets = None
for variant in ('t0', 't1'):
    agent_dir = AGENT_ROOT / variant
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    sampling = yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml'])
    budget = yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation']
    comparable = {key: value for key, value in config.items() if key != 'name'}
    comparable_base = {key: value for key, value in comparable.items() if key != 'skills'}
    if common_config is None:
        common_config, common_sampling, budgets = comparable, sampling, budget
        common_base = comparable_base
    assert comparable_base == common_base and budget == budgets
    assert sampling == common_sampling and sampling['max_output_tokens'] == 16384
    assert config['tools'] == common['tool_list'] and 'adapter' not in config
    assert (config.get('skills') or []) == ([] if variant == 't0' else ['skills/repo_navigation'])
    assert budget == common['eval_budgets']
    assert config['tools'] == common['tool_list'] and 'adapter' not in config
    assert validate_single_declared_model(agent_dir) == TARGET_MODEL_NAME
    for path in agent_dir.rglob('*'):
        assert not path.is_symlink(), f'No symlinks in submission: {path}'
assert sampling == {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 16384, 'thinking_config': {'thinking_budget': 4096, 'include_thoughts': True}}
eval_signature = inspect.signature(Evaluator.evaluate_task)
assert 'context' in eval_signature.parameters, f'Unsupported evaluate_task API: {eval_signature}'
# Check support before any call; never catch TypeError and rerun an already-started task.
adapters = discover_adapters(str(AGENT_ROOT / SELECTED_VARIANTS[0]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
assert hasattr(adapters, 'adapters') and isinstance(adapters.adapters, dict) and not adapters.adapters
litellm.drop_params = True
gpu_count = torch.cuda.device_count()
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
run_manifest = {
    'run_id': RUN_ID, 'repeat_id': REPEAT_ID, 'selected_variants': list(SELECTED_VARIANTS),
    'full_schedule': full_schedule, 'config_sha256': EXPECTED_CONFIG_HASHES,
    'experiment_manifest_sha256': hashlib.sha256(MANIFEST_TEXT.encode('utf-8')).hexdigest(),
    'tasks_sha256': tasks_sha, 'tasks_bytes': len(raw_tasks), 'versions': versions,
    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
    'gpus': gpus, 'tensor_parallel_size': tp_size, 'cuda_build': torch.version.cuda,
    'budgets': budgets, 'sampling': common_sampling, 'single_variable': 'skill only (T1 = T0 + skills/repo_navigation)',     'sandbox': 'subprocess',
    'model_path': str(MODEL_PATH), 'model_alias': TARGET_MODEL_NAME,
    'input_dirs': {'data': str(DATA_DIR), 'wheelhouse': str(WHEELHOUSE_DIR), 'model': str(MODEL_PATH)},
    'server_start_count': 0, 'model_requests': 'sequential',
    'context_cache': {'min_tokens': 2048, 'ttl_seconds': 1800, 'cache_intervals': 10},
    'events_compaction': {'compaction_interval': 15, 'overlap_size': 2,
                          'token_threshold': 14336, 'event_retention_size': 5},
    'cache_note': 'Shared server may retain prefix cache; schedule position recorded, not claimed cache-equivalent.',
    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
print(f'GPUS={gpus} versions={versions}', flush=True)

# Export each unchanged config independently; these ZIPs are outputs, not competition submissions.
for variant in SELECTED_VARIANTS:
    output_dir = RUN_ROOT / variant / REPEAT_ID
    output_dir.mkdir(parents=True)
    zip_path = output_dir / f'submission-002-explore-{variant}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE[variant]:
            path = AGENT_ROOT / variant / relative
            assert path.suffix.lower() in ALLOWED_SUBMISSION_EXTENSIONS
            archive.write(path, relative)
    with zipfile.ZipFile(zip_path) as archive:
        assert 'agent.yaml' in archive.namelist()
        assert sum(info.file_size for info in archive.infolist()) <= MAX_SUBMISSION_SIZE_BYTES
    atomic_json(output_dir / 'submission_manifest.json', {
        'zip_sha256': sha256_file(zip_path), 'config_sha256': EXPECTED_CONFIG_HASHES[variant]})


def run_sync(function, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(function(**kwargs))).result()
    return asyncio.run(function(**kwargs))


class ArchivingVllmServer(VllmServer):
    # The official start() can call stop() on failure; archive before that internal cleanup too.
    def stop(self, timeout=10.0):
        log_path = getattr(self, 'log_path', None)
        try:
            if log_path and Path(log_path).is_file():
                shutil.copyfile(log_path, RUN_ROOT / 'vllm_server.archived.log')
        except Exception:
            atomic_text(RUN_ROOT / 'server_log_archive_error.txt', traceback.format_exc())
        finally:
            super().stop(timeout=timeout)


server = ArchivingVllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
    reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=tp_size, startup_timeout=60 * 20), adapter_manifest=adapters)
results_by_variant = {variant: [] for variant in SELECTED_VARIANTS}
analysis_rows = []
batch_start = time.perf_counter()
interrupted = False
try:
    startup_start = time.perf_counter()
    run_manifest['server_start_count'] = 1
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    server.start()
    run_manifest['server_startup_seconds'] = time.perf_counter() - startup_start
    run_manifest['server_base_url'] = str(server.base_url)
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    models = server.create_model_registry(aliases=[TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')
    print(f'vLLM started once in {run_manifest["server_startup_seconds"]:.1f}s', flush=True)
    for execution_index, entry in enumerate(schedule, 1):
        task = by_id[entry['task_id']]
        variant = entry['variant']
        attempt_id = entry['attempt_id']
        attempt_dir = Path(entry['results_dir'])
        attempt_dir.mkdir(parents=True, exist_ok=False)
        variant_dir = RUN_ROOT / variant / REPEAT_ID
        attempt_state = state[attempt_id]
        attempt_state.update(status='running', execution_index=execution_index,
                             started_at=time.time(), server_uptime_seconds=time.perf_counter()-startup_start,
                             first_attempt_on_server=execution_index == 1)
        atomic_json(attempt_dir / 'attempt_manifest.json', attempt_state)
        atomic_json(RUN_ROOT / 'attempt_status.json', state)
        evaluator = None
        ctx = None
        result = None
        attempt_start = time.perf_counter()
        handler = logging.FileHandler(attempt_dir / 'harness.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        logging.getLogger().addHandler(handler)
        print(f'[{execution_index}/{len(schedule)}] {attempt_id}', flush=True)
        try:
            limits, generation_constraints = build_submission_limits()
            eval_config = EvalConfig(
                tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=attempt_dir,
                submission_dir=AGENT_ROOT / variant, models=models, sandbox='subprocess',
                timeout_seconds=int(budgets['timeout_seconds']), max_tool_calls=int(budgets['max_tool_calls']),
                max_time_minutes=float(budgets['max_time_minutes']), max_turns=int(budgets['max_turns']),
                limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                               token_threshold=14336, event_retention_size=5),
                graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
                wheels_dir=DATA_DIR / 'wheels', verbose=False)
            evaluator = Evaluator(eval_config)
            # Each Evaluator owns a new SubprocessManager; each task creates fresh agent/judge workspaces.
            assert not evaluator.docker.sandboxes, 'A new evaluator must not reuse a repository'
            ctx = SwegemmaContext(docker_manager=evaluator.docker, task=task,
                                  graph_dir=eval_config.graph_dir, embeddings_dir=eval_config.embeddings_dir,
                                  budget=eval_config.budget, harness=eval_config.harness)
            result = run_sync(evaluator.evaluate_task, task=task, task_index=execution_index,
                              total_tasks=len(schedule), context=ctx)
            # Persist raw result fields before calling the official incremental writers.
            payload = {'instance_id': result.instance_id, 'repo': result.repo, 'resolved': result.resolved,
                       'agent_patch_size': len(result.agent_patch or ''), 'test_exit_code': result.test_exit_code,
                       'duration_seconds': result.duration_seconds, 'tool_calls': result.tool_calls,
                       'total_llm_calls': result.total_llm_calls, 'error': result.error,
                       'trace_json_path': getattr(result, 'trace_json_path', None)}
            atomic_json(attempt_dir / 'result.json', payload)
            safe_id = task.instance_id.replace('/', '__')
            trace_path = attempt_dir / 'traces' / f'trace_{safe_id}.json'
            if not trace_path.is_file() and getattr(result, 'trace', None) is not None:
                save_trace_artifact(result.trace, attempt_dir, task.instance_id)
            atomic_text(attempt_dir / 'patches' / f'{safe_id}.patch', result.agent_patch or '')
            atomic_text(attempt_dir / 'test_outputs' / f'{safe_id}.log', result.test_output or '')
            append_task_result(result, attempt_dir, [result])
            results_by_variant[variant].append(result)
            append_task_result(result, variant_dir, results_by_variant[variant])
            attempt_state['trace_present'] = trace_path.is_file()
            attempt_state['status'] = 'completed' if trace_path.is_file() else 'completed_evidence_gap'
            attempt_state['resolved'] = result.resolved
            analysis_rows.append({**entry, **payload, 'execution_index': execution_index,
                                  'trace_present': trace_path.is_file()})
            atomic_json(RUN_ROOT / 'analysis_rows.json', analysis_rows)
            print(f'  resolved={result.resolved} exit={result.test_exit_code} '
                  f'patch_chars={len(result.agent_patch or "")} llm_calls={ctx.llm_calls_used}', flush=True)
        except KeyboardInterrupt:
            interrupted = True
            attempt_state['status'] = 'interrupted'
            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())
        except Exception as exc:
            attempt_state.update(status='error', error_type=type(exc).__name__, error=str(exc))
            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())
            print(f'  ERROR {type(exc).__name__}: {exc}; no automatic retry', flush=True)
        finally:
            attempt_state['outer_seconds'] = time.perf_counter() - attempt_start
            attempt_state['finished_at'] = time.time()
            if ctx is not None:
                context_fields = ('task_start_time', 'agent_start_time', 'agent_end_time',
                                  'agent_elapsed_seconds', 'task_elapsed_seconds', 'patch_submitted',
                                  'tool_calls_used', 'llm_calls_used')
                atomic_json(attempt_dir / 'context_metrics.json',
                            {field: getattr(ctx, field, None) for field in context_fields})
                if result is None and getattr(ctx, 'submitted_patch', None) is not None:
                    atomic_text(attempt_dir / 'submitted_patch_before_exception.patch', ctx.submitted_patch)
            if evaluator is not None:
                try:
                    evaluator.docker.cleanup_all()
                except Exception:
                    atomic_text(attempt_dir / 'sandbox_cleanup_error.txt', traceback.format_exc())
            logging.getLogger().removeHandler(handler)
            handler.close()
            evidence = {str(path.relative_to(attempt_dir)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                        for path in attempt_dir.rglob('*') if path.is_file()
                        and path.name not in ('artifact_manifest.json', 'attempt_status.json')}
            atomic_json(attempt_dir / 'artifact_manifest.json', evidence)
            atomic_json(attempt_dir / 'attempt_status.json', attempt_state)
            atomic_json(RUN_ROOT / 'attempt_status.json', state)
        if interrupted:
            break
except BaseException as exc:
    atomic_text(RUN_ROOT / 'batch_exception.txt', traceback.format_exc())
    if getattr(exc, 'output', None):
        atomic_text(RUN_ROOT / 'server_exception_output.log', str(exc.output))
    raise
finally:
    # stop() archives the server log before official cleanup, including startup-failure paths.
    try:
        server.stop(timeout=10.0)
    except Exception:
        atomic_text(RUN_ROOT / 'server_stop_error.txt', traceback.format_exc())
    for item in state.values():
        if item['status'] == 'planned':
            item['status'] = 'not_run'
        elif item['status'] == 'running':
            item['status'] = 'interrupted_or_batch_error'
    run_manifest['batch_wall_seconds'] = time.perf_counter() - batch_start
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    final_summary = {variant: {
        'planned': sum(entry['variant'] == variant for entry in schedule),
        'returned_results': len(results_by_variant[variant]),
        'resolved': sum(bool(result.resolved) for result in results_by_variant[variant]),
        'states': {entry['task_id']: state[entry['attempt_id']]['status']
                   for entry in schedule if entry['variant'] == variant},
    } for variant in SELECTED_VARIANTS}
    atomic_json(RUN_ROOT / 'joint_summary.json', final_summary)
    print(f'FINAL {json.dumps(final_summary)} OUTPUT={RUN_ROOT}', flush=True)
