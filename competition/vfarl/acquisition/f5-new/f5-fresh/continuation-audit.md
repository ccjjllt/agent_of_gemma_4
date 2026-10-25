# F5-fresh continuation audit (2026-10-25, 22/40 eligible, gap 18 — pools exhausted under validated policy)

Only topics + CPU tool data. No training. No edits to existing
`f5-fresh/` files, `f5-new` canonical / `f5-replace` / `f5-small` /
f1-f4-new, or S600 files. This file is NEW (`continuation-audit.md`);
all other `f5-fresh/` files untouched.

## Verified status (unchanged from batch 22/40)

- Eligible 22/40 (25 dualed: 22 eligible + 1 near-miss colorlog + 2
  ineligible kernprof; 2 pandas held pre-dual; 3 abandoned sqlite/starocks/
  trino probes). Strategy unchanged: new snapshots only, small full suite
  F2P+P2P<50, single-file single-hunk, nonempty verbatim prompt,
  production-file target, single-function confinement, live-on-3.12,
  temporal produce->consume->end, skip-prefix precheck CLEAN-only duals,
  one-per-canon::sym, six tools, official SubprocessManager backend,
  `replay/.venv` base profile.
- Smith pools fully scanned under pinned revision
  `77cab905…13558e`: base (`f1/f2/f5` cached pages) 49 hits + extension
  (`pages-fresh/` 226 pages) 62 hits = 111; recall pass (F5-gate lifted)
  257 rows fully patch-read; P2P-empty smith 16 rows none viable;
  multi-hunk smith 1805 rows out of scope (validated single-hunk strategy).
  Cached-offset audit 2026-10-25: 510 cached / 0 missing across
  `f1/f2/f5` + `pages-fresh` — no unfetched smith train pages remain.
- Gym (`bb94ed9e…`, 2438) + V2 (`10483de0…`, python 7243) small-full-suite
  coverage done: strict 14 + 36 = 50 hits, 2 selected (k8s-330, wtforms-615).
  Remaining 48 are HELD (dependency wall below) or REJECT (patch-evidence
  classes in `triple-review-log.md` §REJECTS, spot-checked again this round:
  e.g. flit-234 is a single-gate FileNotFoundError→ConfigError message flip,
  not a temporal chain — REJECT stands).

## Why 22→40 cannot close under base-`.venv`-only duals

Remaining avenues audited this round (same hard filters, no test/matcher/
requirement edits, no shared-venv changes):

1. P2P-empty smith (16 rows): none viable (dep-dead/heavy repos).
2. Multi-hunk smith (1805 rows): out of scope — validated strategy fixes
   single-hunk; recorded as considered-and-declined.
3. Full V2/Gym small-full-suite coverage: done (50 hits dispositioned).
4. New smith snapshots: none — all 509 offsets cached and scanned.
5. 20+ HELD stay held for a Docker-equivalent host or an authorized
   task-deps profile (per-batch order: do not force eligibility by editing
   tests/matcher or the shared venv).

## Held unblock matrix (task-site-packages without touching shared venv)

F2 batches (`f2-push` P1/P4, `f2-push2` Q2) prove the pattern: parent
`.venv` base + `--task-python 3.10.20` + `--task-site-packages <new dir>`
via `uv` (period-correct wheels), `--source-layout src` where needed,
tests/matcher/requirements untouched. The same pattern is the authorized
route for F5 HELD. Classification (dep type → route):

- UV-EASY (pure-python wheels, no build; needs semantic prep + precheck +
  dual): drf-nested-routers 758felx3/mnv2bb2s (djangorestframework;
  KEEP on semantics per triple log) · moto lu54nbw9 (boto3; KEEP on
  semantics) · moto-5885/6317 (boto3) · dvc-1980/10218 (dulwich) ·
  flake8-1648 (pyflakes/pycodestyle/mccabe; semantics must be re-proved —
  single-gate risk) · k8s-2100 (oauthlib) · ably-474 (methoddispatch) ·
  copier-1785 (pathspec) · hub-2683 (tqdm; note P2P-empty — fails fresh
  full-suite filters, POLICY) · flit-234 (pytoml; single-gate REJECT,
  do not unblock) · sqlfluff x34 (tblib; 34 tasks, one-per-family) ·
  pipenv-976 (plette/tomlkit) · sdk-3091 (simplejson) · xgcm-451 (xarray).
- BUILD (workspace C ext must compile; PyPI wheel would shadow the wrong
  version — Docker or full build host): pandas x2 (pandas C ext;
  preparations + recipes exist, KEEP) · kernprof x2 (line_profiler C ext +
  ubelt; duals exist, ineligible) · nibabel-750 (distutils) · sunpy
  vtbs6n1u (astropy + _pyana C ext; KEEP on semantics but C wall).
- POLICY (not a dep problem): result ccm3de29 (F2P carries uncollectable
  `.yml` type-check nodes) · hub-2683 P2P-empty (fresh requires P2P).
- HEAVY (large stacks; Docker preferred): litellm-11006 (openai),
  bokeh-13239, dask-6682/10765, pip-tools-723.

## Recommended next F5 batch (no training until gap closes)

1. Authorize task-deps profiles for UV-EASY HELD with confirmed KEEP
   semantics first: moto lu54nbw9, drf-nested-routers 758felx3/mnv2bb2s
   (one per canon::sym; new snapshots; precheck-CLEAN-only duals via
   `replay/.venv` base + task-site-packages, same JUnit dual + normalize
   gate, supplemental catalog = `f5-fresh/catalog.json`).
2. Then UV-EASY HELD needing fresh semantic review (single-function +
   temporal chain + single-point/multi-logic/escape rejection):
   moto-5885/6317, dvc-1980/10218, k8s-2100, ably-474, copier-1785,
   sqlfluff (one per family), pipenv-976, sdk-3091, xgcm-451.
3. Keep BUILD + POLICY + HEAVY held for a Docker-equivalent host (or
   authorized build profile); do not edit tests/matcher/venv to force them.
4. flit-234 stays REJECT (single-gate message flip, re-verified this round).

## Hygiene (carried forward)

No ad-hoc files or bytecode in artifact workspaces (probes from /tmp with
PYTHONDONTWRITEBYTECODE=1); recipe temp probes use the system temp dir;
batch shell steps run with pipefail. All new files under
`acquisition/f5-new/f5-fresh/` (new names only) +
`replay/artifacts/f5-fresh/` + `replay/f5-fresh-*.json`.
