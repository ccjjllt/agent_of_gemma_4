# P013 batch-02 behavior verification report (CPU)

Date: 2026-10-20. Box: macOS CPU, no GPU, no Kaggle. Interpreter: python3 3.9.6
(same binary the producer shim execs; `python3 --version` = 3.9.6, verified
version-identical to the manual-verification interpreter per batch-01
procedure). No `/workspace` on this box; live workspaces are temp/copy dirs
with `PYTHONPATH=src:.` or `PYTHONPATH=.` (P011 single-file subsets) and
`cwd=workspace`, exactly as the frozen plans prescribe. Deviation for the two
pinned-pydantic trajectories (vendored `pydantic-core-2.30.0` exposed ONLY
under `outputs/*pydantic*/workspace`, after the command PYTHONPATH) is the
same harness setup batch-01 documents, extended by one glob character in
`ensure_python_shim` (`outputs/pydantic/` → `outputs/*pydantic*/`).

Frozen inputs (read-only, verified untouched via `git status`: no modification
under `research/009`, `research/011`, `research/012`, `research/006`):
`research/009/training-data-20261018`, `research/009/training-data-v2-20261018`
(sources, transaction-plans, outputs),
`research/011/data-scaling-20261018` (materialized subset sources +
transactions), `research/012/training-data`,
`research/006/data-replay-20261013/replay.py` (unchanged official v31 tools).
Heldout/reserved/dev (公共12, 外部8仓 gpxpy/feedparser/pandas/schema/
cookiecutter/conan/modin/iniconfig, heldout28, 旧预留8) were never read.
Guard code: `produce_p013.py::guard_train_source` (denylist from
`research/task_splits.json` + 15-repo reserved set) — enforced live on every
batch-02 admission; all 11 issues additionally audited post-hoc: train split,
not in denylist, not in batch-01's 8.

## Per-trajectory table (manual one-shot verification, real tool execution)

All commands below ran for real; logs preserved under `batch-02/<key>/`
(`pre.log` exit≠0 buggy, `post.log` exit 0 fixed, `module-path.log` workspace
file, `recovery.log` where a separate mid-session correction occurred,
`edits.json` exact old→new strings from the frozen transaction).

| # | issue | repo | pre-fail cmd + exit | edit summary | recovery event | post-pass cmd + exit | resolved |
|---|-------|------|---------------------|--------------|----------------|----------------------|----------|
| 1 | martinblech__xmltodict.0952f382.pr_345 | martinblech/xmltodict | attr/xmlns unparse cmd, exit 1: `NameError: name '_unicode' is not defined` | 1× edit_file `xmltodict.py` (`_unicode = str` def) | fix-driven (no extra mid-session error) | same cmd, exit 0 (attr, xmlns, roundtrip) | true |
| 2 | rustedpy__result.0b855e1e.pr_121 | rustedpy/result | ok/err value cmd, exit 1: `AttributeError: 'Ok' object has no attribute 'ok_value'` | 2× edit_file `src/result/result.py` (`ok_value`/`err_value` properties) | fix-driven | same cmd, exit 0 (values + map) | true |
| 3 | un33k__python-slugify.872b3750.pr_148 | un33k/python-slugify | add_uppercase_char cmd (spec-loaded), exit 1: only `A` added, input mutated | 1× edit_file `slugify/special.py` (copy + add all) | regular-import probe silently loaded INSTALLED slugify 8.0.4 (regular package wins over workspace namespace portion) → switched to `spec_from_file_location`, command prints `spec.origin` | same cmd, exit 0 (all added, input intact) | true |
| 4 | un33k__python-slugify.872b3750.pr_149 | un33k/python-slugify | same fn, exit 1: `["A","a","b"]` (early return; mutation out of scope) | 1× edit_file `slugify/special.py` (drop early return) | spec-load from the start (lesson of #3) | same cmd, exit 0 | true |
| 5 | mahmoud__boltons.3bfcfdd0.pr_375 | mahmoud/boltons | sentinel copy cmd, exit 1: `copy.copy`/`deepcopy` return distinct objects (`"same": false`) | 1× edit_file `boltons/typeutils.py` (`__copy__`/`__deepcopy__`) | first variant used `var_name='_MISSING'` (identity already preserved via `__reduce__`, pre exit 0) → corrected to default sentinel | same cmd, exit 0 (both identical) | true |
| 6 | cantools__cantools.0c6a7871.pr_701 | cantools/cantools | parse_number_string cmd, exit 1: `ValueError: invalid literal for int() with base 0: '1e1'` | 1× edit_file `src/cantools/.../arxml/utils.py` (scientific-notation branch) | fix-driven | same cmd, exit 0 (`1e1`→10, float/hex/true unchanged) | true |
| 7 | marshmallow-code__apispec.8b421526.pr_892 | marshmallow-code/apispec | IP-field mapping cmd, exit 1: `field2type_and_format(IPv4/IPv6)` → `{}` (genuine silent drop) | 1× edit_file `src/apispec/ext/marshmallow/field_converter.py` (IP/IPv4/IPv6 mapping rows) | fix-driven; workspace subset resolves through namespace packages against installed marshmallow 4.0.1 (has IP/IPv4/IPv6 + orderedset, verified) | same cmd, exit 0 (`{'type':'string','format':'ipv4'/'ipv6'}`) | true |
| 8 | martinblech__xmltodict.0952f382.pr_363 | martinblech/xmltodict | whitespace cmd, exit 1: `strip_whitespace=True` keeps `' '` (should be None) | 5× chained edit_file `xmltodict.py` (`endElement` strip logic) | live pre-marker initially too strict for `sort_keys` order (refusal, see below) → marker corrected to the genuine output substring, re-ran | same cmd, exit 0 (preserve + strip) | true |
| 9 | pydantic__pydantic.acb0f10f.pr_5568 | pydantic/pydantic | alias-generator cmd, exit 1: `ImportError: cannot import name 'camel_case' from 'pydantic.aliases'` (workspace file) | 2× edit_file `pydantic/aliases.py` (builtin generators) | env pin (real): pinned source needs `pydantic-core==2.30.0` (vendored, shim-exposed only under pydantic workspaces; user site keeps 2.46.5) | same cmd, exit 0 (camel+pascal validate/serialize, explicit alias wins) | true |
| 10 | pydantic__pydantic.acb0f10f.pr_3646 | pydantic/pydantic | Color equality cmd, exit 1: 5/8 cases false (same_name/rgb/alpha/hash/model) | 1× edit_file `pydantic/color.py` | same vendored-core pin as #9 | same cmd, exit 0 (8/8) | true |
| 11 | adrienverge__yamllint.8513d9b9.pr_572 | adrienverge/yamllint | CLI config-discovery cmd, exit 1: `parent_config` exit 1 (parent `.yamllint` ignored) | 1× edit_file `yamllint/cli.py` (transaction's production edit; plan scope covers CLI discovery) | fix-driven | same cmd, exit 0 (4/4 discovery priorities) | true |

Counts toward the 32-target: **11 full闭环, 0 partial** (manual pass + live
canonical replay, see below). `task_resolved` is local-contract only in every
row (`task_resolved=null`, whole-issue oracle NOT run — same as batch-01
precedent). "Resolved" above = same-criterion post-pass after the production
edit, nothing more.

Recovery-classification note (no inflation): every trajectory recovered from
its concrete pre-fix error state via the production edit and same-criterion
re-verification. Three trajectories additionally show an independent
mid-session correction episode (slugify-148 installed-dist shadowing,
boltons-375 wrong-scope first variant, xmltodict-363 pre-marker correction);
each is logged in its `recovery.log` (or below for the marker case) with raw
evidence, never reconstructed. Deliberately-misspelled-module probes were not
used; the one wrong-module load (slugify-148) was detected via `__file__`,
never logged as behavior evidence.

## Live canonical replay (`produce_p013.py --mode behavior --batch batch-02`)

Manual one-shot runs above prove re-executability; the producer replays each
evidence dir through the unchanged official v31 tools (`Recorder`: read_file
→ run_command[pre, must be error + genuine pre_marker] → edit_file×n →
run_command[post, identical command, must be ok] → verified handover) and
assembles `outputs/b02-<key>/transaction.json` (behavior_verified:true) ONLY
if all gates pass, else writes `outputs/<evdir>/refusal.json` and admits
nothing. Producer changes for batch-02 (diff in `produce_p013.py`, frozen
dirs untouched): `--batch batch-01|batch-02` (default batch-01, old paths
and old `behavior-batch01-summary.json` name preserved), per-batch default
key lists and summary file, evidence/command lookup under the batch dir, and
the one-glob pydantic shim widening above. No gate logic changed.

- Run 1 (1/0): `xmltodict-345` admitted (plumbing smoke test).
- Run 2 (2/0): `result-121`, `slugify-148` admitted (`--limit` default 2 cut
  the 7-key list; re-ran the rest).
- Run 3 (4 admitted / 1 refused): `slugify-149`, `boltons-375`,
  `cantools-701`, `apispec-892` admitted; `xmltodict-363` CORRECTLY refused —
  the committed pre_marker (`"whitespace_stripped", "pass": false`) does not
  occur verbatim because the command prints `sort_keys` order (`case, got,
  pass`), so the genuine output is `"case": "whitespace_stripped", "got":
  {"root": " "}, "pass": false`. Marker corrected to the genuine substring
  `"whitespace_stripped", "got": {"root": " "}` (in `evidence.json`), re-ran:
- Run 4 (1/0): `xmltodict-363` admitted. Raw refusal preserved at
  `outputs/xmltodict-363/refusal.json` (run_behavior keys refusals by
  evidence-dir name while transactions key by `ev.key`; harmless naming
  quirk, disclosed here).
- Run 5 (3/0): `pydantic-5568`, `pydantic-3646`, `yamllint-572` admitted
  (pydantic pair via the widened shim glob; verified `pydantic_core` 2.30.0
  under the live workspace, workspace `pydantic/` first on path).

Post-audit of all 11 `outputs/b02-<key>/transaction.json` (same method as
batch-01 attempt 5): `requirement_done:true`, `verification_status:pass`,
`behavior_verified:true`, `task_resolved:null`, `pre_exit_code:1` (int,
non-zero, from the error envelope), `post_exit_code:0`, same command SHA
pre/post, module file under the live workspace, non-empty `actual.patch`,
`events.json`/`commands.json` preserved, targets = canonical arc ordinals
(read → pre-run → edit(s) → post-run → handover).
Counts toward the 32-target: **11 full闭环, 0 partial** (live canonical
replay; the manual 11/11 above was the re-executability proof, the live
11/11 is the admitted layer).

## Failures with raw logs

- `xmltodict-363` run-3 refusal (CORRECT gate behavior, then corrected and
  admitted — not a behavior failure):
  `ValueError: pre-failure lacks the expected genuine failure marker
  '"whitespace_stripped", "pass": false'; refusing (environment-confounded
  failure, not the behavior under test)` — the pre run DID fail genuinely
  (exit 1, whitespace kept under `strip_whitespace=True`); only the marker
  string was wrong for `sort_keys` key order. Raw refusal at
  `outputs/xmltodict-363/refusal.json`.
- Scope-cut dead ends (investigated, honestly excluded, NOT counted):
  tenacity/tenacity-wait/yamllint-rules/deepdiff/marshmallow-core/tinydb/
  freezegun/toml-parser P011 subsets are single-file edits whose module
  cannot be imported from the subset (missing intra-package siblings, e.g.
  `tenacity/__init__.py` needs `_utils`, `marshmallow/utils.py` needs
  `marshmallow.base` which installed 4.0.1 dropped) — behavior commands
  cannot reach the fixed code without import-machinery deviation, so no
  trajectory was built. webargs pr_832's single edit is behavior-neutral
  with default `empty_value`; gunicorn pr_3196 (`%r` → `%s % repr`) and
  boltons pr_376 / dominate pr_184 / charset pr_327 are docstring/version
  bumps with no behavior delta; textfsm pr_127 is environment-dependent.
  Each was checked, none admitted.

## Encodings / codec

`target_assistant_steps` are zero-based assistant ordinals per
`training/encoding/gemma_tool_chat.py:360`. `produce_p013.py` computes them
post-hoc from final messages (same method as batch-01). `--thinking on
--thought-loss mask` policy unchanged on encode; no encoding run in this
session (no GPU, no tokenizer run — encode deferred to the SFT step).

## Cumulative toward 32–64 and blockers

- Delivered: batch-01 **8 full** + batch-02 **11 full** = **19 full闭环, 0
  partial** admitted under `research/013/behavior-data/outputs/` (13 still
  needed for the 32-target; 45 headroom to the 64-cap).
- Batch-01 blocker 4 (marshmallow reviewed command) is CLOSED by trajectory
  #7 (apispec pr_892, marshmallow-family IP-field mapping, verified pre/post).
- Remaining needs: 13+ further distinct train issues. P011 subsets whose
  edited file imports cleanly are now largely consumed (this batch took the
  verifiable ones); next sources are P009v2 `mypy-pr12943` (slow static
  contract, minutes per run, needs ≥300 s timeout) and P011 multi-file or
  env-heavy issues requiring new reviewed commands — each needs manual
  one-shot pre/post verification before `--mode behavior` ingestion (producer
  refuses without it).
- `produce_p013.py` batch-02 diff is minimal and gate-preserving; batch-01
  replay path is byte-identical (default `--batch batch-01`, same summary
  filename, same gates). Batch-01 outputs were not re-run or modified.
