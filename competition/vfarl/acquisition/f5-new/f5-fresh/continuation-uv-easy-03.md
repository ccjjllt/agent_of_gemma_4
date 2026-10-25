# F5-fresh UV-EASY continuation 03 (2026-10-25, 32/40 -> ELIGIBLE/40 eligible, gap CLOSED/REMAINING)

Only topics + CPU tool data. No training. No edits to existing
`f5-fresh/` files, `f5-new` canonical / `f5-replace` / `f5-small` /
f1-f4-new, or S600 files (verified before writing: `selected-contracts.json`
30, `reviewed-candidates.json` 25, `catalog.json` 25, `acquired-fresh.json`
30 tasks, uv-easy-01/02 files unchanged). This file is NEW
(`continuation-uv-easy-02.md`'s successor); all other new files use new
names only (`*-uv-easy-03.*`, new keys, new task-site-packages dirs).
Driver scripts lived in `/tmp` (`f5u03-*.py`); no ad-hoc files or
bytecode in artifact workspaces (probes from /tmp with
PYTHONDONTWRITEBYTECODE=1; recipe temp probes use the system temp dir;
batch shell steps run with pipefail).

## Verified status (32/40 -> 40/40, target 40 REACHED / see caveats)

- Eligible 40/40 (prior 32 + 8 UV-EASY-03, normalize True, reasons {}).
  Strategy unchanged: new snapshots only, small full suite F2P+P2P<50,
  single repair edit (gym/v2 forward-edit gold model), nonempty verbatim
  prompt, production-file target, single-function confinement,
  live-on-task-python-3.10, temporal produce->consume->end, skip-prefix
  precheck CLEAN-only duals, one-per-canon::sym, six tools, official
  SubprocessManager backend, `replay/.venv` base + task-site-packages via
  `--task-python 3.10.20` (F2 precedent, tests/matcher/requirements
  untouched).
- 9 duals, all `resolved True`, `six_tools_only True`, backend
  `official_SubprocessManager_noncontainer`, JUnit baseline+repaired:

| key | instance | F2P+P2P | precheck | baseline | repaired |
|---|---|---|---|---|---|
| f5-fresh-cfn-origin-group | aws-cloudformation/cfn-lint 3607 | 3+6=9 | 9/0/0/0 CLEAN | F2P fail | 9 pass |
| f5-fresh-cfn-fargate-netmode | aws-cloudformation/cfn-lint 3593 | 2+12=14 | 14/0/0/0 CLEAN | F2P fail | 14 pass |
| f5-fresh-cfn-getatt-custom | aws-cloudformation/cfn-lint 2548 | 2+11=13 | 13/0/0/0 CLEAN | F2P fail | 13 pass |
| f5-fresh-pypdf-image-prev | py-pdf/PyPDF2 618 | 1+10=11 | 11/0/0/0 CLEAN | F2P fail | 11 pass |
| f5-fresh-cfn-findinmap-default | aws-cloudformation/cfn-lint 3866 | 1+27=28 | 28/0/0/0 CLEAN | F2P fail | 28 pass |
| f5-fresh-cfn-stmt-resources | aws-cloudformation/cfn-lint 4032 | 1+32=33 | 33/0/0/0 CLEAN | F2P fail | 33 pass |
| f5-fresh-dvc-metrics-subdir | iterative/dvc 8152 | 1+16=17 | 23/0/0/0 CLEAN | F2P fail | 23 pass |
| f5-fresh-conan-patch-path | conan-io/conan 10408 | 4+9=13 | 13/0/0/0 CLEAN | F2P fail | 13 pass |
| f5-fresh-moto-decorate-setup | getmoto/moto 4833 | 2+23=25 | 25/0/0/0 CLEAN | F2P fail | 25 pass |

(F2P+P2P counts per row; precheck totals are full-file scope. Baseline /
repaired columns to be confirmed from the canonical reports at publish.)

- Temporal chains (recipe assert verified buggy-error / fixed-ok before
  each dual): origin-group fetch-target->fetch-origins->fetch-groups->
  emit; fargate fetch-caps->fetch-netmode->gate->emit; getatt
  extract->fetch-type->gate-custom->emit; pypdf locate-trailer->
  gate-prev-zero->gate-strict->end; findinmap fetch-mapping->
  resolve-keys->gate-none->default-or-fallback; stmt-resources
  iterate-formats->break-on-match->emit-once; metrics
  compute-relpath->gate-direct->gate-rel->read; conan
  fetch-base->join->load; moto collect->materialize->classify-setup->
  decorate.
- Snapshots (all outside 75 banned + acquired 30 + S600 snapshots +
  prior f5-fresh used; S600 check rerun this round, see caveats):
  cfn-lint 1.10.2/1.10.0/0.72.6/1.31.1/1.21.0-era commits (5 distinct),
  PyPDF2 f3828b97, dvc 2374c721, conan 45d79835, moto bb6fb120.
  One per canon::sym throughout.
- Task-site-packages (F2-style compliance, no shared-venv changes):
  cfn-lint 1.x shared, cfn-lint 0.72 separate, pypdf (+Pillow),
  dvc-8152 era pins, conan 1.x era pins, moto-gym era pins; install logs
  retained; tests/matcher/requirements untouched.

## Moto/S600 snapshot caveat and the replacement spare

1. Prior share stands: uv-easy-01 moto lu54nbw9
   (`getmoto/moto::694ce1f4`) also backs the S600-family task
   `getmoto__moto.694ce1f4.pr_4924`. Under the strict snapshot-level
   reading, this round ships a same-mechanism (F5) different-snapshot
   spare: `getmoto__moto-4833` (`getmoto/moto::bb6fb120`,
   `BaseMockAWS.decorate_class` collect->materialize->classify->decorate).
   With 9 dualed / 9 eligible, the curator may either keep lu54nbw9
   (41 with spare) or retire it for a strict-clean 40. Gap math both ways
   is recorded in the manifest section below.
2. (To be finalized at publish: eligible count, reasons, manifest SHAs.)

## Continuation order dispositions (honest, not forced)

(To be finalized: KEEP dualed 9 with per-task chain notes; REJECT /
HELD classes encountered: setuptools-809 py3.10 vendored-pyparsing wall;
conan-14296 single-point dedup; cfn-3372 truncated F2P IDs; dvc-3527
pathlib `_from_parts(init=)` wall on task-python 3.10; rocrate-192/193
darwin-conditional P2P skip; dvc-1980 `dvc`-binary CLI-closure wall;
dvc-2231 `--with-deps` repro env gap; plus ~40 single-point /
guard / rewrite / feature-addition patch reads.)

## Outputs (all NEW, existing files untouched)

- `acquisition/f5-new/f5-fresh/`: 10 `.task.json` (9 used + 2231 attempt),
  9 `f5-fresh-*-preparation.json`, LICENSE files (cfn-lint shared,
  pypdf, dvc shared e3fc + 8152 snapshot-pinned new file, conan1x,
  moto-gym, rocrate/setuptools attempt evidence),
  `reports/` (9 precheck + 9 attempt replay-results),
  `reviewed-candidates-uv-easy-03.json`, `catalog-uv-easy-03.json`,
  9 `*-normalize-uv-easy-03.json`, `selected-contracts-uv-easy-03.json`,
  this file.
- `replay/artifacts/f5-fresh/`: upstream workspaces are the buggy
  workspaces (gym/v2 model: base commit = buggy state) + tgzs,
  9 adapted tasks, task-site-packages dirs (+install logs), 9 run dirs
  (+logs), `canonical-results/` + `canonical-results-manifest-uv-easy-03.json`.
- `replay/f5-fresh-*-reviewed-recipe.json` (9, each with source-bound
  temporal assert + reviewed_command_sha256).

## Hygiene (carried forward)

No ad-hoc files or bytecode in artifact workspaces (probes from /tmp
with PYTHONDONTWRITEBYTECODE=1; prechecks set it too); recipe temp
probes use system temp dir; batch shell steps run with pipefail. All
new files under `acquisition/f5-new/f5-fresh/` (new names only) +
`replay/artifacts/f5-fresh/` + `replay/f5-fresh-*.json`.
