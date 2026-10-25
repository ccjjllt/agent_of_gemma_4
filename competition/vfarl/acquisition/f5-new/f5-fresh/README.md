# F5-fresh batch (new-snapshot small-full-suite redo, 2026-10-25, 22/40 eligible)

Only topics + CPU tool data. No training. Old `f5-new` canonical / `f5-replace` /
`f5-small` / f1-f4-new / S600 files untouched (all new files under
`acquisition/f5-new/f5-fresh/` + `replay/artifacts/f5-fresh/` +
`replay/f5-fresh-*-reviewed-recipe.json`).

## Strategy (per new steering, unchanged from 1/40)

- Trio full-file structurally infeasible on macOS (44 env P2P fails persist even
  when 4/4 F2P fixed; 27 trio full-suite matcher-BLOCKED); freezegun prefix
  collision (`test_ticking_time` vs skipped `test_ticking_time_clock`); 32
  held-BLOCKED need Docker-equivalent host or new snapshots. So F5-fresh sweeps
  **new snapshots only** for small full suites.
- Only train; Smith-py `77cab905…13558e` + Gym `bb94ed9e…` + V2 `10483de0…` python.
  Exclude `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx`.
- Snapshot zero overlap: 75 banned reused (`banned-snapshots.json`) + 5
  f5-replace/small used (trio cfbbe2c1, freezegun 5f171db0, channels a144b4b8,
  daphne 32ac73e1, tornado d5ac65c1) + every acquired f5-fresh snapshot;
  3 f5-new snapshots already inside 75.
- Small full suite: full F2P+P2P<50 + single-file + single-hunk + nonempty
  verbatim publisher prompt + production-file target. Base scan
  (`fresh-strict-scan.json`): smith 49 + gym 14 + v2 36 = 99. Extension scan
  (`fresh-extended-scan.json`, identical policy over 226 newly fetched
  `pages-fresh/` train pages): +62. Recall pass (same hard filters, F5-signal
  gate lifted): 257-row pool, fully patch-read. P2P-empty smith recall (16 rows):
  none viable. Multi-hunk smith (1805 rows): out of scope (validated strategy
  fixes single-hunk), recorded as considered-and-declined.
- Triple review (`triple-review-log.md`): single-function confinement +
  live-on-3.12 + temporal produce->consume->end + single-point/multi-logic/
  escape rejection. One per canon::sym, no quota filling.
- Skip-prefix precheck first (full-suite upstream/fixed run, real official matcher
  `_node_matches_requirement`): only CLEAN dualed (`reports/*-precheck.json`).
- Whitelist prompt (verbatim publisher, nonempty), JUnit dual conditions
  (baseline all-F2P-failed + repaired official-matcher-True), six tools only,
  official SubprocessManager backend, file-scope, protected paths, train-only.
- Dual via `replay/.venv` (base profile, copy mode), baseline + repaired wheels.
- `normalize eligible` only counts (supplemental catalog = `f5-fresh/catalog.json`;
  reports = `f5-fresh/reports/`; manifest =
  `replay/artifacts/f5-fresh/canonical-results-manifest.json`).
- Workspace hygiene (enforced mid-batch): no ad-hoc files or bytecode in artifact
  workspaces (probes run from /tmp with PYTHONDONTWRITEBYTECODE=1; prechecks set
  it too); recipe temp probes use the system temp dir so the source-binding
  assert cannot be masked; batch shell steps run with pipefail.

## Count (honest)

- **eligible: 22 / 40, gap 18** (+ f5-small's 1 eligible counted separately).
- 30 selected (28 smith + 2 gym/v2), 25 dualed, 22 eligible, 1 near-miss, 2 held
  pre-dual (pandas C ext), 2 ineligible duals (kernprof C ext), 3 abandoned
  early-return probes (sqlite/starocks/trino, no runs).
- Eligible (normalize, all strict-verbatim, six-tools, official backend):
  scrapy from_body 08esupl2 (7), sqlglot hive datediff 25pli4wm (11), sqlglot
  redshift approx 9srpoanc (12), iniconfig iter 17ckkdzc (42), k8s-asyncio str-log
  330 (14), wtforms option-validators 615 (34), stackprinter frame-walk 2oy6fbsg /
  frame-format b105lcro / hide-callable z0up9gl4 / pick-colors er60tmxe (7 each),
  strsimpy overlap 18ll59ds / normlev 1pnopti2 / lcs 7bxevwv5 / cosine-dot
  jhaaq9av / qgram llhbww8d (18 each), sqlglot helper merge-ranges bc00zosy (3) /
  table-sql d991myym (4) / literal-sql puj4eya1 (4) / scan-var gwjdkzcb (7) /
  hive-struct 5bovf34w (11) / hive-adddate 64dgmblz (11) / hive-param c5lval0j (11).
  Each with a temporal-chain recipe assert
  verified both directions (buggy error / fixed ok) before the dual.
- Eligible (normalize, all strict-verbatim, six-tools, official backend):
  scrapy from_body 08esupl2 (7), sqlglot hive datediff 25pli4wm (11), sqlglot
  redshift approx 9srpoanc (12), iniconfig iter 17ckkdzc (42), k8s-asyncio str-log
  330 (14), wtforms option-validators 615 (34), stackprinter frame-walk 2oy6fbsg /
  frame-format b105lcro / hide-callable z0up9gl4 / pick-colors er60tmxe (7 each),
  strsimpy overlap 18ll59ds / normlev 1pnopti2 / lcs 7bxevwv5 / cosine-dot
  jhaaq9av / qgram llhbww8d (18 each). Each with a temporal-chain recipe assert
  verified both directions (buggy error / fixed ok) before the dual.
- Near-miss: colorlog reset-append 1oe4qn2p (31) — perfect repair (F2P fixed, 27
  others pass) but 3 unrelated P2P fail identically at baseline in the sandbox
  (capsys capture; all 31 pass upstream). Ineligible, retained.
- Duals completed 18 via replay/.venv (official SubprocessManager backend,
  six-tools True, JUnit dual baseline+repaired).
- Held (dependency wall, documented in triple-review-log §HELD, no duals):
  pandas x2, kernprof x2 (duals exist, ineligible), dask, bokeh, k8s-2100, ably,
  flit, piptools, nibabel, copier, sdk-3091, hub-2683, moto x2, dvc x2,
  pipenv, litellm, flake8, xgcm, sqlfluff x34, sunpy, result-ccm3de29,
  drf-nested-routers x2. replay/.venv lacks their third-party/C-extension
  dependencies and the batch order mandates replay/.venv duals.

## Outputs

- `acquisition/f5-new/f5-fresh/`: `banned-snapshots.json` (75),
  `fresh-strict-scan.json` (99), `fresh-extended-scan.json` (62),
  `pages-fresh/` (226 pages + receipts), `scan_fresh.py`, `scan_extend.py`,
  `fetch_extend.py`, `triple-review-log.md`, `selected-contracts.json` (30),
  `*.task.json` (40, incl. dep-held review rows), `acquired-fresh.json` (30),
  `acquire_fresh.py`, `acquire_gymv2.py`, `collect_fresh.py`,
  `collect_fresh_gymv2.py`, `precheck_fresh.py`, `precheck_fresh_gymv2.py`,
  `finish_fresh.py`, `*-preparation.json` (30), `*.LICENSE` (10 canons),
  `reviewed-candidates.json` (25) + `catalog.json` (25) + `reports/` (25 dual +
  28 precheck) + `*-normalize.json` (25), `artifacts-refs/` (9 refs txt).
- `replay/artifacts/f5-fresh/`: upstreams (8 smith canons + full-SHA tgzs;
  gym/v2 workspaces pin the buggy base directly), buggy sources (+tgzs), adapted tasks, runs + logs, `canonical-results/` (22) +
  `canonical-results-manifest.json` (25 reports, eligible 22).
- `replay/f5-fresh-*-reviewed-recipe.json` (27, each with source-bound temporal
  assert + reviewed_command_sha256).

## Next to close the gap (no training until then)

1. Continue fresh sweep from remaining pools only under the same policy
   (single-hunk, new snapshots, precheck-CLEAN-only duals, one-per-family).
   Audited remaining avenues: P2P-empty smith (16 rows, none viable),
   multi-hunk smith (1805 rows, out of scope), full V2/Gym coverage (done).
2. 20+ held-BLOCKED (dep wall above + trio 27 + freezegun 7 from prior batches)
   stay held for a Docker-equivalent host (or an authorized task-deps profile);
   do not force eligibility by editing tests/matcher or the shared venv.
3. Tornado Doctest-P2P structural block needs root verifier decision; out of scope.
