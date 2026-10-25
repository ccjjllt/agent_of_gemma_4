# F5-fresh triple review log (2026-10-25, batch 1/40 -> 22/40 eligible)

Scope: Smith-py train `77cab905…13558e` + Gym `bb94ed9e…` + V2 `10483de0…` python.
Only train; exclude `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx`.
Snapshot zero overlap: 75 banned reused (`banned-snapshots.json`) + 5 f5-replace/small used
(trio cfbbe2c1, freezegun 5f171db0, channels a144b4b8, daphne 32ac73e1, tornado d5ac65c1)
+ f5-fresh used (scrapy 35212ec5 and every acquired snapshot below); 3 f5-new snapshots
already inside 75 banned. All F5-fresh snapshots outside.
Filter: full F2P+P2P<50 + single-file + single-hunk + nonempty verbatim prompt +
production-file target (test-file mutations rejected).
Base scan (`fresh-strict-scan.json`): smith 49 + gym 14 + v2 36 = 99.
Extension scan (`fresh-extended-scan.json`, identical policy, 226 newly fetched
`pages-fresh/` train pages): +62. Recall pass (same hard filters, F5-signal gate
lifted, documented in §5): 257-row pool, fully patch-read.
Each item: single-function confinement + live-on-3.12 + temporal sequence
(produce->consume->end) + single-point/multi-logic/escape rejection. No quota filling.
Precheck-first (full-suite upstream/fixed run, real official matcher
`_node_matches_requirement`): only CLEAN dualed (`reports/*-precheck.json`).
Duals via replay/.venv, baseline+repaired wheels, six tools only.

## KEEP (22 eligible / 30 selected; duals complete)

All 22: temporal-chain recipe assert verified both directions (buggy error /
fixed ok) before the dual; precheck CLEAN; normalize eligible
(strict verbatim, six-tools, SubprocessManager backend).

Smith (13):
- KEEP `func_pm_ctrl_shuffle__08esupl2` scrapy/scrapy::ResponseTypes.from_body
  (total 7): slice->encode->gate->lower->dispatch; UnboundLocalError on buggy.
- KEEP `func_pm_ctrl_shuffle__25pli4wm` tobymao/sqlglot::_date_diff_sql
  (total 11): unit->factor-gate->months-gate->format->truncate->return;
  UnboundLocalError on buggy; 4-probe millisecond/second/minute/hour sequence.
- KEEP `func_pm_ctrl_shuffle__9srpoanc` tobymao/sqlglot::Redshift.Parser._parse_approximate_count
  (total 12): parse->gate(Count+Distinct)->build/retreat; early-return escape on buggy.
- KEEP `func_basic__17ckkdzc` pytest-dev/iniconfig::IniConfig.__iter__
  (total 28): parse-in-file-order->iterate->consume; reversed on buggy.
- KEEP `func_pm_ctrl_shuffle__1oe4qn2p` borntyping/python-colorlog::ColoredFormatter._append_reset
  (total 31): format->gate(reset)->append->return; early-return escape on buggy.
  Repaired F2P passes; 3 unrelated P2P fail in the sandbox only (capsys capture,
  fail identically at baseline; full 31 pass upstream) -> recorded INELIGIBLE
  near-miss (same class as the old freezegun-stepwidth near-miss).
- KEEP `func_pm_ctrl_shuffle__2oy6fbsg` cknd/stackprinter::format_stack_from_frame
  (total 7): init-list->walk-f_back->reverse->format; NameError on buggy.
- KEEP `func_pm_ctrl_shuffle__b105lcro` cknd/stackprinter::ColorfulFrameFormatter._format_frame
  (total 7): split-path->headline->scope->colors->append; use-before-def on buggy.
- KEEP `func_pm_ctrl_shuffle__z0up9gl4` cknd/stackprinter::FrameFormatter.select_scope.hide
  (total 7, nested helper): fetch-value->gate-callable->classify; UnboundLocalError
  on callable vars on buggy. Nested-function symbols resolve recursively; the delta
  is exactly the reviewed hunk inside `hide`.
- KEEP `func_pm_ctrl_shuffle__er60tmxe` cknd/stackprinter::ColorfulFrameFormatter._pick_colors
  (total 7): init-map->walk-lines->return; return-before-compute on buggy.
- KEEP `func_pm_ctrl_shuffle__18ll59ds` strsimpy::OverlapCoefficient.similarity
  (total 18): validate->shortcut->profile->score; compute-before-validate on buggy.
- KEEP `func_pm_ctrl_shuffle__1pnopti2` strsimpy::NormalizedLevenshtein.distance
  (total 18): validate->measure->divide; use-before-def on buggy.
- KEEP `func_pm_ctrl_shuffle__7bxevwv5` strsimpy::LongestCommonSubsequence.length
  (total 18): validate->measure->allocate->fill; use-before-def on buggy.
- KEEP `func_pm_ctrl_shuffle__jhaaq9av` strsimpy::Cosine._dot_product
  (total 18): pick-small/large->accumulate->return; return-before-compute on buggy.
- KEEP `func_pm_ctrl_shuffle__llhbww8d` strsimpy::QGram.distance
  (total 18): validate->shortcut->profile->distance; return-before-def on buggy.
- KEEP `func_pm_ctrl_shuffle__bc00zosy` tobymao/sqlglot::merge_ranges
  (total 3): gate-empty->sort->seed->fold; return-before-seed on buggy.
- KEEP `func_pm_ctrl_shuffle__d991myym` tobymao/sqlglot::Generator.table_sql
  (total 4): parts->options->format->return; return-before-parts on buggy.
- KEEP `func_pm_ctrl_shuffle__puj4eya1` tobymao/sqlglot::Generator.literal_sql
  (total 4): read-text->gate-string->return; return-before-read on buggy.
- KEEP `func_pm_ctrl_shuffle__gwjdkzcb` tobymao/sqlglot::Tokenizer._scan_var
  (total 7): scan-chars->add-VAR; add-before-scan on buggy (wrong token text).
- KEEP `func_pm_ctrl_shuffle__5bovf34w` tobymao/sqlglot::Hive.Generator.struct_sql
  (total 11): collect-values->build-STRUCT; build-before-collect on buggy.
- KEEP `func_pm_ctrl_shuffle__64dgmblz` tobymao/sqlglot::_add_date_sql
  (total 11): read-unit->gate->build; build-before-read on buggy.
- KEEP `func_pm_ctrl_shuffle__c5lval0j` tobymao/sqlglot::Hive.Generator.parameter_sql
  (total 11): read-parts->gate-parent->format; format-before-read on buggy.

Gym/V2 (2, forward-edit gold-patch model after the f2-gymv2c precedent):
- KEEP tomplus__kubernetes_asyncio-330 Watch.next (total 14): decode->gate-empty->
  dispatch(str fast-path vs event parse)->end; ended str stream reconnect-loops on
  buggy (sentinel AssertionError), stops with ordered lines on fixed.
- KEEP wtforms__wtforms-615 SelectFieldBase.__iter__ (total 34): build-option
  (validators)->process->render-required; validators dropped on buggy.

## HELD, dependency-blocked (no dual; evidence cited, same class as trio/macOS holds)

replay/.venv lacks: boto3/botocore, dulwich/git, pyflakes/pycodestyle/mccabe,
openai/litellm stack, xarray, oauthlib, methoddispatch, pytoml, pip/setuptools/
distutils, tqdm, simplejson, pathspec, tblib, tablib, ubelt, C toolchains for
pandas/line_profiler/nibabel C extensions. The batch order mandates replay/.venv
duals, so these are held, not forced:
- Smith: kernprof `7859sogj`/`q2i2jnut` (C ext + ubelt; run-001 duals exist and both
  are ineligible with collection errors — retained as evidence, no re-run),
  pandas `7ci62fyb`/`fj3pyzuo` (pandas C ext; precheck import error, no JUnit), result `ccm3de29` (F2P carries uncollectable
  `.yml` type-check nodes), drf-nested-routers `758felx3`/`mnv2bb2s` (statement
  shuffles, KEEP on semantics; djangorestframework absent).
- Gym/V2: moto-5885/6317 (boto3), dvc-1980/10218 (dulwich), flake8-1648 (pyflakes),
  pipenv-976 (plette/tomlkit), litellm-11006 (openai), xgcm-451 (xarray),
  k8s-2100 (oauthlib via kubernetes/__init__), ably-474 (methoddispatch),
  flit-234 (pytoml), pip-tools-723 (pip), nibabel-750 (distutils), copier-1785
  (pathspec), sdk-3091 (simplejson), hub-2683 (tqdm via package init).
- Extension pool: sqlfluff x34 (tblib; import chain probed via /tmp checkout,
  repo untouched), moto `lu54nbw9` (KEEP on semantics: fetch-group->filter->
  apply shuffled to use-before-def; boto3-held), sunpy `vtbs6n1u` (KEEP on
  semantics: validate->gate->read->wrap shuffled; astropy absent + _pyana C ext).

## REJECTS (patch-evidence classes; no quota filling)

- Value-only single-point flips: sqlglot format_time, amaranth brace-escape x2,
  meltano default=str, mypy plugin/fallback x2 + pr_10430, cfn-lint branch,
  pls msg prefix, bump-my-version coercion, holoviews isinstance guard,
  pyomo x4 (char-class/API-swap/loops), wtforms validators (kept instead via
  __iter__ ordering), modin rename_axis/add-attrs, pyxform messages, flit guard
  (kept instead? no — flit dep-held), walinuxagent decode removal, pybamm
  __contains__/__iter__ addition, opencensus __enter__ return + span addition,
  dynaconf new-file module, xgcm exclude_dims (single insertion, no ordering),
  add-trailing-comma x2 (single gates), jupysql refactor (multi-function +
  Exception->ValueError), copier hidden-filter (single filter insertion),
  hub normalize-then-parse (kept? no — value-handling, dep-held anyway),
  ably release tolerance (dep-held), pip-tools close/unlink (dep-held),
  k8s-asyncio kept; pyfiglet x7 (soups + single-point newline/gate flips),
  mypy pr_5617/15184 (gate/message), json-logger x16 (all single-point:
  isoformat slice, popitem sabotage, partition swap, arg swaps, pattern edits,
  get-defaults, type-check flips, guard inversion), result x40 (all func_basic
  one-line flips; sampled y8qhirr9 DoException.err=None carry-break and
  ifc7e1k7 __eq__ or/and flip), flashtext x22 (protocol flip, removal,
  loop deletions, gate inversions; sampled), records x19+5 (tablib import +
  value flips), async-timeout x11 (single-point deadline/state guards,
  dead-code), drf-nested-routers routers x2 (multi-soup), cknd 3 (multi-soup),
  dvc stage/cache x2 (rewrites), sqlglot hive-rewrite + executor rewrite,
  scrapy mail/contracts rewrites (hook-order flip buried in full rewrite),
  iniconfig swap + init-rewrite, line_profiler removal/type-corruption/rewrite,
  stackprinter inverts x3 (single-gate flips), stackprinter combines x2
  (multi-function soups), sqlparse CLI rewrite, readability x18 (soups/values),
  usaddress x4 (feature-dict soup/rewrites), conan x8 (regex simplification,
  template deletion, removals/rewrites), dask-10765 (validation rewrite),
  dvc gym x5 (precision/arg/guard values), moto-5885 (feature addition),
  dvc-1980 (lifecycle, dep-held too), moto-6317 (dedup guard, dep-held too),
  dvc-10218 (inserted ignore step, dep-held too), bokeh rollover (dep-held too),
  dask-6682 chunks (dep-held too), nibabel pipeline (dep-held too),
  sdk-3091 visited-tuple (dep-held too), flake8 fresh-parser (dep-held too),
  pipenv section-gate (dep-held too), litellm id-propagation (dep-held too),
  k8s-2100 iter-lines (dep-held too), strsimpy test-file mutations x2
  (380c1d6z/ni1hji13: production policy forbids test-file targets),
  strsimpy combine (multi-soup), stackprinter test-adjacent n/a.
- Abandoned early-return probes (preparations only, no runs): sqlite
  _parse_unique aisalu57, starrocks create_sql 2yibin5y, trino
  _parse_json_query_quote xx1erkbc — CLEAN precheck but buggy ok on minimal
  probes (undistinguishing); replaced by the 7 use-before-def keeps above.
- Same-symbol duplicates excluded (scrapy fitumlqy, result qv9c1arl).
- Cross-batch duplicates excluded (dvc ktgde2mw already in f4-new).
- P2P-empty smith recall (16 rows): all in dep-dead/heavy repos, none viable.
- Multi-hunk smith (1805 rows): out of scope — the validated strategy fixes
  single-hunk; recorded as considered-and-declined, not reviewed item by item.

## Family/snapshot status (22/40 eligible)

- Snapshots (all outside 75 banned + 5 used + prior f5-fresh used): scrapy
  35212ec5, sqlglot 036601ba (x9 tasks grouped), iniconfig 16793ead,
  kubernetes_asyncio 7222c989, wtforms 9cc7c7f4, colorlog dfa10f59,
  stackprinter 219fcc52 (x4 grouped), strsimpy 115acaac (x5 grouped).
  One per canon::sym throughout; same-snapshot grouping retained.
- 30 selected (28 smith + 2 gym/v2), 25 dualed, 22 eligible, 1 near-miss
  (colorlog), 2 sibling-dualed ineligible (kernprof), 2 held pre-dual (pandas),
  3 abandoned early-return (sqlite/starocks/trino, no runs).
- Gap: 18 to 40 (plus f5-small's 1 eligible counted separately). No training
  until the gap closes. Old f5-new/f5-replace/f5-small, f1-f4, S600 untouched;
  all new files under `acquisition/f5-new/f5-fresh/` + `replay/artifacts/f5-fresh/`
  + `replay/f5-fresh-*.json`.
