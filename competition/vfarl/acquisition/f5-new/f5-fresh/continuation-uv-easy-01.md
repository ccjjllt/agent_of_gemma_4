# F5-fresh UV-EASY continuation 01 (2026-10-25, 23/40 eligible, gap 17)

Only topics + CPU tool data. No training. No edits to existing
`f5-fresh/` files, `f5-new` canonical / `f5-replace` / `f5-small` /
f1-f4-new, or S600 files. This file is NEW
(`continuation-uv-easy-01.md`); all other `f5-fresh/` files untouched.
All other new files use new names only (see Outputs).

## Verified status (22/40 -> 23/40 via moto lu54nbw9)

- Eligible 23/40 (prior 22 eligible + 1 UV-EASY moto lu54nbw9 eligible,
  normalize True, reasons []). Strategy unchanged: new snapshots only,
  small full suite F2P+P2P<50, single-file single-hunk, nonempty verbatim
  prompt, production-file target, single-function confinement,
  live-on-3.12, temporal produce->consume->end, skip-prefix precheck
  CLEAN-only duals, one-per-canon::sym, six tools, official
  SubprocessManager backend, `replay/.venv` base + task-site-packages
  via `--task-python 3.10.20` (F2 precedent, tests/matcher/requirements
  untouched).
- Prior pools exhausted under validated policy (per `continuation-audit.md`):
  base 49 + extension 62 = 111 smith hits scanned, recall 257 rows
  fully patch-read, P2P-empty 16 none viable, multi-hunk 1805 out of
  scope, Gym/V2 50 hits dispositioned (2 selected). Cached-offset audit
  510/0 — no unfetched smith train pages remain.

## UV-EASY 01 dual: moto lu54nbw9 (KEEP-confirmed, pure wheels)

- Instance `getmoto__moto.694ce1f4.func_pm_ctrl_shuffle__lu54nbw9`
  (smith-py train rev `77cab905…13558e`, row 48413, total 40 = 2 F2P +
  38 P2P, file `moto/autoscaling/models.py`, single hunk
  `@@ -1580,13 +1580,12 @@`, production-file target).
- Snapshot `getmoto/moto::694ce1f4880c784fed0553bc19b2ace6691bc109`
  (resolved via `github.com/.../commit/694ce1f4.patch` after API 403;
  `git ls-remote swesmith/getmoto__moto.694ce1f4` gives execution commit
  `baaf9f348e79dc6c1454ea5633c3fe7ed0fe0b3a`). Zero overlap: not in 75
  banned + acquired 30 + 5 used; first `getmoto/moto` canon::sym
  (`AutoScalingBackend.set_instance_protection`), one-per-canon::sym holds.
- Temporal chain (KEEP per triple log, re-verified): fetch-group
  (`group = self.autoscaling_groups[group_name]`) -> filter
  (`protected_instances = [x for x in group.instance_states ...]`) ->
  apply (`for instance in protected_instances: ...`). Buggy shuffles
  apply-before-fetch (use-before-def). Enclosing symbol at buggy lines
  1580-1588 is exactly `AutoScalingBackend.set_instance_protection`
  (1577-1588); upstream order is fetch->filter->apply. Single-function
  confinement holds both sides; live-on-3.12; not single-point /
  multi-logic / escape.
- Whitelist prompt: verbatim publisher `problem_statement` (2265 chars,
  nonempty) preserved in adapted task; `patch` blanked, `test_patch`
  hydrates 1002 upstream test assets (missing 1489 lines incl.
  `tests/test_autoscaling/test_autoscaling.py` 55KB, absent in buggy tgz
  3394 vs upstream 3424 members — handled via inverse-edit + test_patch
  path, no test/matcher edits).
- Recipe assert verified both directions before dual (task-python 3.10.20
  + task-site-packages, PYTHONDONTWRITEBYTECODE=1, probes from /tmp):
  upstream `ids 2 -> states {protected: True, other: False}` ok;
  buggy `UnboundLocalError: protected_instances referenced before
  assignment` error. Source-bound (`moto.autoscaling.models`, SHA
  pinned) + temporal `moto_temporal_fetch_filter_apply` marker.
- Task-site-packages (F2-style compliance, no shared-venv changes):
  `replay/artifacts/f5-fresh/task-site-packages-moto-lu54nbw9/` via `uv`
  (`--python task-python 3.10.20 --target`, 27 pure wheels, no build:
  boto3 1.34.69 / botocore 1.34.162 / cryptography 43.0.3 prebuilt /
  requests / xmltodict / werkzeug 2.3.8 / python-dateutil / responses /
  Jinja2 / pytest 8.3.3 + deps). Install log retained. Tests/matcher/
  requirements untouched.
- Precheck CLEAN: upstream full-file `tests/test_autoscaling/
  test_autoscaling.py` 40 passed in 2.65s (task-python 3.10.20 +
  task-site-packages, JUnit 40/0/0/0, zero skipped nodes => no
  skip-prefix collisions via official matcher, no required failures on
  fixed tree). Report `reports/f5-fresh-f5-fresh-moto-lu54nbw9-
  precheck.json`.
- Dual via `replay/.venv/bin/python replay/run_official.py` with
  `--task-site-packages <new dir> --task-site-packages-mode copy
  --task-python 3.10.20 --source-layout . --verify --verify-baseline
  --command-timeout 90 --verification-timeout 180` (copy mode, bytecode
  not copied, six_tools_only True, backend official_SubprocessManager).
  Baseline `............................F.F.........` (40 tests, 2 F2P
  failures both UnboundLocalError, 38 passed, all_original_F2P_observed_
  failed True); repaired 40 passed. JUnit dual baseline+repaired.
- Normalize eligible True, reasons [] (strict verbatim, six-tools,
  SubprocessManager backend). Supplemental catalog =
  `f5-fresh/catalog-uv-easy-01.json` (1 task); reports =
  `f5-fresh/reports/f5-fresh-f5-fresh-moto-lu54nbw9-attempt001-
  replay-result.json`; manifest =
  `replay/artifacts/f5-fresh/canonical-results-manifest-uv-easy-01.json`
  (1 eligible). Canonical copy in `canonical-results/`.

## Outputs (all NEW, existing files untouched)

- `acquisition/f5-new/f5-fresh/`: `getmoto__moto.694ce1f4.func_pm_ctrl_
  shuffle__lu54nbw9.task.json` (from `pages-fresh/train-48400.json`
  row 48413), `f5-fresh-moto-lu54nbw9-preparation.json`,
  `getmoto__moto.LICENSE` (upstream, 694ce1f4), `reports/f5-fresh-
  f5-fresh-moto-lu54nbw9-precheck.json`, `reports/f5-fresh-f5-fresh-
  moto-lu54nbw9-attempt001-replay-result.json`,
  `reviewed-candidates-uv-easy-01.json`, `catalog-uv-easy-01.json`,
  `moto-lu54nbw9-normalize.json`, this file.
- `replay/artifacts/f5-fresh/`: `getmoto__moto-upstream` (+tgz),
  `f5-fresh-moto-lu54nbw9-source` (+tgz), `f5-fresh-moto-lu54nbw9-
  task.json`, `task-site-packages-moto-lu54nbw9/` (+install.log),
  `f5-fresh-moto-lu54nbw9-run-001` (+log), `canonical-results/f5-fresh-
  f5-fresh-moto-lu54nbw9-attempt001-replay-result.json`,
  `canonical-results-manifest-uv-easy-01.json`.
- `replay/f5-fresh-f5-fresh-moto-lu54nbw9-reviewed-recipe.json` (temporal
  assert + reviewed_command_sha256, 7 tool events: get_status, read_file,
  run error, edit, run ok, get_status, submit).

## Next UV-EASY (one-by-one, same gates; BUILD/POLICY/HEAVY/Docker held)

1. KEEP-confirmed first: drf-nested-routers `758felx3`/`mnv2bb2s`
   (statement shuffles, KEEP; djangorestframework pure wheels; one per
   canon::sym; locate rows — not in `pages-fresh/` 226 (checked), likely
   recall 257 pool — resolve via pinned revision receipts, then same
   task-site-packages + precheck-CLEAN-only dual path).
2. Then UV-EASY needing fresh semantic review (single-function +
   temporal chain + single-point/multi-logic/escape rejection; many are
   single-gate REJECT risks — do not force):
   moto-5885/6317 (boto3, totals 3/7), dvc-1980/10218 (dulwich, 5/21),
   k8s-2100 (oauthlib, 16; patch is bytearray rewrite — multi-logic
   risk), ably-474 (methoddispatch, 9; guard insertion — single-point
   risk), copier-1785 (pathspec, 47; hidden-filter insertion —
   single-point REJECT per triple log), sqlfluff (tblib, 34 tasks, one
   per family; need single-hunk temporal, not value flips), pipenv-976
   (plette/tomlkit, 6), sdk-3091 (simplejson, 21; visited-tuple —
   isolation, not obviously temporal), xgcm-451 (xarray, 40),
   flake8-1648 (pyflakes/pycodestyle/mccabe, 24; single-gate risk).
   Each needs patch-read + /tmp probe + precheck-CLEAN-only dual via
   new task-site-packages (F2 precedent), new names only.
3. Keep BUILD (pandas x2 C ext, kernprof x2 C ext + ubelt, nibabel
   distutils, sunpy astropy + _pyana C ext) + POLICY (result ccm3de29
   yml nodes, hub-2683 P2P-empty) + HEAVY (litellm, bokeh, dask,
   pip-tools) + flit-234 REJECT held for Docker-equivalent host or
   authorized build profile; do not edit tests/matcher/venv to force.
4. Gap: 17 to 40 (plus f5-small's 1 counted separately). No training
   until gap closes. Old f5-new/f5-replace/f5-small, f1-f4, S600
   untouched.

## Hygiene (carried forward)

No ad-hoc files or bytecode in artifact workspaces (probes from /tmp
with PYTHONDONTWRITEBYTECODE=1; prechecks set it too); recipe temp
probes use system temp dir; batch shell steps run with pipefail. All
new files under `acquisition/f5-new/f5-fresh/` (new names only) +
`replay/artifacts/f5-fresh/` + `replay/f5-fresh-*.json`.
