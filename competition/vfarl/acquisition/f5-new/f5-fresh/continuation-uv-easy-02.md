# F5-fresh UV-EASY continuation 02 (2026-10-25, 23/40 -> 32/40 eligible, gap 8)

Only topics + CPU tool data. No training. No edits to existing
`f5-fresh/` files, `f5-new` canonical / `f5-replace` / `f5-small` /
f1-f4-new, or S600 files (verified: `selected-contracts.json` 30,
`reviewed-candidates.json` 25, `catalog.json` 25, `acquired-fresh.json`
30, uv-easy-01 files unchanged). This file is NEW
(`continuation-uv-easy-01.md`'s successor); all other new files use new
names only (`*-uv-easy-02.*`, new keys, new task-site-packages dirs).
Driver scripts lived in `/tmp` (`uveasy02-*.py`); no ad-hoc files or
bytecode in artifact workspaces (probes from /tmp with
PYTHONDONTWRITEBYTECODE=1; recipe temp probes use the system temp dir;
batch shell steps run with pipefail).

## Verified status (23/40 -> 32/40, target 33 NOT reached)

- Eligible 32/40 (prior 23 + 9 UV-EASY-02, normalize True, reasons {}).
  Strategy unchanged: new snapshots only, small full suite F2P+P2P<50,
  single-file single-hunk smith / single repair edit gym/v2, nonempty
  verbatim prompt, production-file target, single-function confinement,
  live-on-task-python-3.10, temporal produce->consume->end, skip-prefix
  precheck CLEAN-only duals, one-per-canon::sym, six tools, official
  SubprocessManager backend, `replay/.venv` base + task-site-packages via
  `--task-python 3.10.20` (F2 precedent, tests/matcher/requirements
  untouched).
- 9 duals, all `resolved True`, `six_tools_only True`, backend
  `official_SubprocessManager_noncontainer`, JUnit baseline+repaired:

| key | instance | F2P+P2P | precheck | baseline | repaired |
|---|---|---|---|---|---|
| f5-fresh-drf-force-mutable | drf-nested-routers 758felx3 | 3+20=23 | 23/0/0/0 CLEAN | F2P fail | 23 pass |
| f5-fresh-drf-init-request | drf-nested-routers mnv2bb2s | 4+19=23 | 23/0/0/0 CLEAN | F2P fail | 23 pass |
| f5-fresh-sqlfluff-plugin-manager | sqlfluff w5jogbs0 | 1+1=2 | 5/0/0/0 CLEAN (run-002 env) | F2P fail | 5 pass |
| f5-fresh-sqlfluff-noqa-directive | sqlfluff o7hykee3 | 5+33=38 | 47/0/0/0 CLEAN | F2P fail | 47 pass |
| f5-fresh-sqlfluff-noqa-refmap | sqlfluff v1qmhxd0 | 17+26=43 | 43/0/0/0 CLEAN | F2P fail | 43 pass |
| f5-fresh-sdk-schema-refs | meltano sdk-3091 | 1+20=21 | 21/0/0/0 CLEAN | F2P fail | 21 pass |
| f5-fresh-flake8-fresh-parser | pycqa flake8-1648 | 1+23=24 | 24/0/0/0 CLEAN | F2P fail | 24 pass |
| f5-fresh-k8s-utf8-lines | kubernetes-client 2100 | 2+14=16 | 16/0/0/0 CLEAN | F2P fail | 16 pass |
| f5-fresh-bokeh-rollover-zero | bokeh 13239 | 1+31=32 | 32/0/0/0 CLEAN | F2P fail | 32 pass |

- Temporal chains (recipe assert verified buggy-error / fixed-ok before
  each dual): drf capture->force->yield->restore
  (`UnboundLocalError`); drf build->gate->inject->return
  (`UnboundLocalError`); sqlfluff fetch-cache->create->register->
  publish->load->mark (`UnboundLocalError`); sqlfluff trim->strip->
  locate->parse->attach (`UnboundLocalError`); sqlfluff gate->seed->
  expand->intersect (`UnboundLocalError`); sdk resolve-first->
  resolve-second with per-stack visits (`[{'type':'number'}, {}]`
  buggy); flake8 fresh-parse per candidate across two dirs (buggy
  returns nonexistent `tox.ini`); k8s buffer->split->decode->yield
  (buggy `UnicodeDecodeError`); bokeh stream->trim->notify on list
  then array branches (buggy keeps all).
- Snapshots (all outside 75 banned + 5 used + acquired 30 + S600 600 +
  prior f5-fresh used): drf-nested-routers 6144169d (x2 tasks, distinct
  symbols `_force_mutable` vs `NestedViewSetMixin.initialize_request`),
  sqlfluff 50a1c4b6 (x3 tasks, distinct symbols `get_plugin_manager` /
  `IgnoreMask._extract_ignore_from_comment` /
  `Linter.allowed_rule_ref_map`), sdk ae326fe3, flake8 70c0b3d2, k8s
  76ec6f5c, bokeh 06a6bf5d. One per canon::sym throughout.
- Drf receipts resolved via pinned smith train page
  (`f1-new/pages/train-43700.json` + `train-43700.receipt.json`,
  rev 77cab905): 758felx3 -> 788567ac, mnv2bb2s -> 38db59c4
  (checked against live `git ls-remote`
  `swesmith/alanjds__drf-nested-routers.6144169d`, new refs file).
  Sqlfluff rows from `pages-fresh/train-46400.json` (+receipt);
  execution commits via ls-remote (05758a0b/d2f2bc4a/168b4b9d).
  Gym/v2 rows from local pinned parquet
  (`SWE-Gym` bb94ed9e row-group offsets; `SWE-rebench-V2` 10483de0
  global row_idx 20359/23004);
  `pycqa__flake8-1648.task.json` written from parquet row (new file);
  other gym/v2 task.json files reused read-only.
- Task-site-packages (F2-style compliance, no shared-venv changes;
  see `task-site-packages-*-install.log`): drf (Django 4.2.13 /
  DRF 3.15.1, tox lane), sqlfluff (3.2.5-era pins + repo-local
  `plugins/sqlfluff-plugin-example` via `--no-deps` as upstream tox
  does + version `dist-info` metadata only, no PyPI code), sdk
  (referencing/simplejson/pendulum/SQLAlchemy/jsonschema chain),
  flake8 (mccabe/pycodestyle/pyflakes 5.x lane), k8s
  (oauthlib/google-auth/websocket-client), bokeh (numpy/pandas/
  tornado/contourpy + pytest-timeout per pytest config + version
  `dist-info` metadata only).

## Continuation order dispositions (honest, not forced)

KEEP dualed (9, above): drf 758felx3/mnv2bb2s, sqlfluff
w5jogbs0/o7hykee3/v1qmhxd0, sdk-3091, flake8-1648, k8s-2100,
bokeh-13239.

REJECT (patch-read, single-gate / value / rewrite class, not dualed):
moto-5885 (feature addition block), moto-6317 (dedup guard),
dvc-10218 (one-line `ignore()` insertion), ably-474 (single
membership gate), copier-1785 (single hidden-filter insertion),
pipenv-976 (single section-scan guard), xgcm-451 (single
`exclude_dims` kwarg), dask-6682 (single chunks guard), sqlfluff
nq32whwu/oev6z6zy/cy7z3c59/066drwlc/3myluc4x/cydrgbe7/bplhmm6k/
v6jhmocs (value flips / rewrites / corruptions; only the 3 statement
shuffles kept).

HELD (no dual): dvc-1980 (lifecycle chain plausible but full dvc CLI
closure for func tests is HEAVY, dulwich-era wall stands);
pipenv/pip-tools-723 (piptools 3.3.1 needs pip-19 `_internal` API;
pip 19.0.3 satisfies the API surface but its vendored html5lib uses
`collections.Mapping`, removed on task-python 3.10, so DEV_PKGS
resolves None — no compliant route without editing third-party
vendored code; dir retained as attempt evidence);
nibabel/sunpy (BUILD), hub-2683 (POLICY P2P-empty), litellm/dask
(HEAVY), flit-234 (REJECT stands).

## Caveats and corrections recorded (no old files touched)

1. S600 snapshot-share (shipped uv-easy-01 moto lu54nbw9):
   `getmoto/moto::694ce1f4` also backs S600 common task
   `getmoto__moto.694ce1f4.pr_4924` (elbv2 exceptions, different
   file/symbol/instance/family). Task-level disjointness holds;
   snapshot-level overlap with S600 was missed by the uv-easy-01
   zero-overlap check (banned+acquired+used only). Flagged for
   curator call; uv-easy-02 candidates were all checked against
   S600 snapshots and are clear.
2. sqlfluff-plugin-manager attempt001 resolved False: 3 non-required
   example-plugin tests failed (plugin not installed) while required
   F2P/P2P behaved correctly. Upstream tox installs the repo-local
   example plugin, so it was added to the task profile
   (`--no-deps`, documented) and attempt002 dualed 5/5 green.
   Attempt001 retained as evidence. Same-class lesson applied
   pre-emptively to bokeh (pytest-timeout per pytest config).
3. Precheck-guard fix: an early bokeh precheck run recorded CLEAN on
   a collection INTERNALERROR (1 test / 1 error). The gymv2 precheck
   now forces BLOCKED when collected tests < required count; the
   stale report was overwritten by the green 32/32 re-run. Smith
   prechecks were unaffected.
4. Borderline KEEP reasoning (curator-visible): sdk-3091 is
   visit-state isolation rather than statement order, flake8-1648 is
   a one-line fresh-parser move, bokeh-13239 is a zero-guard fix;
   each is admitted only because a two-probe ordered chain
   distinguishes buggy from fixed (second-property resolution,
   cross-candidate pollution, both-branch trim) inside one
   function, with full-file green duals. ably/copier/pipenv/xgcm
   single gates did not meet this bar and were not forced.
5. Gap to 40 is 8, not 7: the 33/40 target was missed by one honest
   KEEP. Remaining avenues: remaining V2/Gym small-suite hits need
   individual patch reads; BUILD/HEAVY/POLICY holds need a
   Docker-equivalent host or authorized build profile.

## Outputs (all NEW, existing files untouched)

- `acquisition/f5-new/f5-fresh/`: 2 drf + 3 sqlfluff `.task.json`,
  `pycqa__flake8-1648.task.json`, 2 refs txt (drf, sqlfluff),
  9 `f5-fresh-*-preparation.json`, `alanjds__drf-nested-routers.LICENSE`,
  `sqlfluff__sqlfluff.LICENSE`, `reports/` (9 precheck + 9
  attempt replay-results; plugin-manager attempt is 002),
  `reviewed-candidates-uv-easy-02.json` (9),
  `catalog-uv-easy-02.json` (9), 9 `*-normalize-uv-easy-02.json`,
  this file. `meltano__sdk.LICENSE`/`pycqa__flake8.LICENSE`/
  bokeh/k8s LICENSE pins reuse-or-new per same-canon rule (no
  overwrites of differing pins).
- `replay/artifacts/f5-fresh/`: 2 drf + 1 sqlfluff upstreams (+tgzs),
  9 `-source` workspaces (+tgzs), 9 adapted tasks, 7
  `task-site-packages-*` dirs (+install logs), 9 run dirs (+logs),
  `canonical-results/` + `canonical-results-manifest-uv-easy-02.json`
  (9 eligible, reasons {}).
- `replay/f5-fresh-*-reviewed-recipe.json` (9, each with source-bound
  temporal assert + reviewed_command_sha256).

## Hygiene (carried forward)

No ad-hoc files or bytecode in artifact workspaces (probes from /tmp
with PYTHONDONTWRITEBYTECODE=1; prechecks set it too); recipe temp
probes use system temp dir; batch shell steps run with pipefail. All
new files under `acquisition/f5-new/f5-fresh/` (new names only) +
`replay/artifacts/f5-fresh/` + `replay/f5-fresh-*.json`.
