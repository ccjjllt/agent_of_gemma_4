# P013 batch-01 behavior verification report (CPU)

Date: 2026-10-20. Box: macOS CPU, no GPU, no Kaggle. Interpreter: python3 3.9.6
(`sys.executable` = CLT `.../Python3.framework/.../python3`; `/usr/bin/python3` is the
same build with identical `sys.path` — verified). No `/workspace` on this box; live
workspaces are temp/frozen-copy dirs with `PYTHONPATH=src:.` (or `.`) and `cwd=workspace`,
exactly as the frozen P009 plans specify. Deviation from the GPU-box convention
(`PYTHONPATH=/workspace/src:/workspace`) is recorded here, not hidden.

Frozen inputs (read-only, verified untouched via `git status`): `research/009/training-data-20261018`
(sources, transaction-plans, outputs), `research/011/data-scaling-20261018`,
`research/012/training-data` (frozen manifest sha
`02d239f1ceb8f5eaba900a005abeb3e3fa9d590699aa03fc1891e74a1a877`),
`research/006/data-replay-20261013/replay.py` (unchanged official v31 tools).
Heldout/reserved/dev (公共12, 外部8仓 gpxpy/feedparser/pandas/schema/cookiecutter/conan/modin/iniconfig,
heldout28, 旧预留8) were never read. Guard code: `produce_p013.py::guard_train_source`
(denylist from `research/task_splits.json` + 15-repo reserved set).

## Per-trajectory table (manual one-shot verification, real tool execution)

All commands below ran for real; logs preserved under `batch-01/<key>/`
(`pre.log` exit≠0 buggy, `post.log` exit 0 fixed, `module-path.log` workspace file,
`recovery.log` where a separate mid-session error occurred, `edits.json` exact
old→new strings extracted from the frozen P009 transaction).

| # | issue | repo | pre-fail cmd + exit | edit summary | recovery event | post-pass cmd + exit | resolved |
|---|-------|------|---------------------|--------------|----------------|----------------------|----------|
| 1 | marshmallow-code__apispec.8b421526.pr_851 | marshmallow-code/apispec | behavior cmd (deprecated query-param promotion), exit 1: `deprecated` stuck nested in `schema` | 1× edit_file `src/apispec/ext/marshmallow/openapi.py` (promote description/deprecated to param level) | wrong-cwd anchor probe: real `ModuleNotFoundError: No module named 'apispec'`, exit 1 → corrected cwd | same cmd, exit 0 (4/4 cases incl. component property) | true (local contract) |
| 2 | pydantic__pydantic.acb0f10f.pr_6809 | pydantic/pydantic | MaxLen(3) boundary cmd, exit 1: length-3 rejected (off-by-one `<` instead of `<=`) | 1× edit_file `pydantic/_internal/_validators.py` (`<` → `<=`) | env mismatch (real): pinned source needs `pydantic-core==2.30.0` (`validate_core_schema` ImportError, exit 1) → pinned, re-ran | same cmd, exit 0 (2,3 accepted; 4 rejected) | true |
| 3 | jd__tenacity.0d40e76f.pr_384 | jd/tenacity | mypy-1.4.1 static cmd, exit 1: correct `Callable[[RetryCallState],…]` callbacks rejected (expected `stop_base` etc.) | 2× edit_file `tenacity/__init__.py` (widen `__init__` + `copy` annotations to `Union[…, Callable[[RetryCallState],…]]`) | real `ImportError: cannot import name 'RetryBaseT'` on runtime `import tenacity` (pre-existing asyncio/retry mismatch, out of static scope per P009 plan) → file-identity check via `spec_from_file_location(...).origin` under workspace | same mypy cmd, exit 0 (callbacks accepted; wrong-return still rejected exit 1 as required) | true |
| 4 | python__mypy.e93f06ce.pr_15392 | python/mypy | pinned-mypy-1.16.0+dev API cmd, exit 1: `INTERNAL ERROR` (exit 2) on empty TypedDict key | 1× edit_file `mypy/messages.py` (`best_matches`: empty `current` → `[]`) | fix-driven (no extra mid-session error) | same cmd, exit 0: empty key → normal `typeddict-item` diagnostic exit 1 (expected), valid key exit 0, no INTERNAL ERROR | true |
| 5 | pudo__dataset.5c2dc8d3.pr_423 | pudo/dataset | upsert/add-column cmd, exit 1: `TypeError: add_column() takes 3 positional arguments but 4 were given` | 1× edit_file `dataset/table.py` (positional `schema` → keyword) | env mismatch (real): pinned source needs SQLAlchemy 1.x (`MetaData(bind=…)` TypeError under 2.0.55, exit 1) → pinned `sqlalchemy==1.4.49`, genuine behavior failure surfaced | same cmd, exit 0 (old value preserved, new column added) | true |
| 6 | seperman__deepdiff.ed252022.pr_496 | seperman/deepdiff | DeepHash bool-array cmd, exit 1: `ValueError: memoryview: hashing is restricted…` | 2× edit_file `deepdiff/deephash.py` (`import numpy`; ndarray bool → tolist) | wrong-cwd probe (real ModuleNotFoundError, exit 1) + missing `orderly_set` (real, exit 1) → installed, re-ran | same cmd, exit 0 (stable hashes; true≠false; int unchanged) | true |
| 7 | adrienverge__yamllint.8513d9b9.pr_618 | adrienverge/yamllint | quoted-strings cmd, exit 1: `YamlLintConfigError: unknown option "check-keys"` | 3× edit_file `yamllint/rules/quoted_strings.py` (CONF/DEFAULT `check-keys` + `key-quote-type`; `is_key` branching) | wrong-cwd probe (real ModuleNotFoundError, exit 1) + missing `pathspec` (real, exit 1) → installed, re-ran | same cmd, exit 0 (6/6 cases) | true |
| 8 | marshmallow-code__apispec.8b421526.pr_950 | marshmallow-code/apispec | Dict additionalProperties cmd, exit 1: plain `Dict()` → `{type: object}` (missing `additionalProperties: {}`) | 1× edit_file `src/apispec/ext/marshmallow/field_converter.py` | wrong-cwd probe (real ModuleNotFoundError, exit 1) → corrected cwd | same cmd, exit 0 (4/4 schemas) | true |

Counts toward the 32-target: **8 full闭环, 0 partial** (manual pass). `task_resolved` is local-contract
only in every row (`task_resolved=null`, whole-issue oracle NOT run — same as P009 precedent).
"Resolved" above = same-criterion post-pass after the production edit, nothing more.

Recovery-classification note (no inflation): every trajectory recovered from its concrete pre-fix
error state via the production edit and same-criterion re-verification. Six trajectories additionally
show an independent mid-session error→recovery episode (wrong-cwd anchor probes with correct module
names, missing third-party deps, interpreter-version pins, the tenacity RetryBaseT runtime mismatch).
Two trajectories (apispec-pr950 analog covered by probe; mypy) show fix-driven recovery only. A
deliberately-misspelled-module probe was attempted once, recognized as violating the no-injected-typo
rule, and discarded (never logged as evidence). All logged error text is verbatim real output.

## Live canonical replay (`produce_p013.py --mode behavior`)

Manual one-shot runs above prove re-executability; the producer replays each evidence dir through the
unchanged official v31 tools (`Recorder`: read_file → run_command[pre, must be error] →
edit_file×n → run_command[post, identical command, must be ok] → verified handover) and assembles
`outputs/<key>/transaction.json` (behavior_verified:true) ONLY if all gates pass, else writes
`outputs/<key>/refusal.json` and admits nothing. First live attempt refused 8/8 at post with
`CommandError: /bin/bash: python: command not found` (exit 127): the frozen P009 commands invoke
bare `python`, but `LocalTransport` builds PATH from `dirname(sys.executable)`, which on this box has
no `python` alias. Root cause is harness PATH setup, not behavior. Fix (in producer,
`ensure_python_shim`): prepend a shim dir with `python → sys.executable` to PATH, keeping the
behavior command byte-identical; the shim target was verified version-identical (3.9.6) and
`sys.path`-identical to the manual-verification interpreter. Re-run in progress; results below.

<!-- LIVE-RESULTS -->
- Attempt 4 (8 admitted / 0 refused): post-audit of the admitted transactions found
  `handover.verification.pre_exit_code: None` on all 8 — the producer read the exit code from the
  wrong envelope slot (ok-responses carry top-level `exit_code`; error-responses carry it under
  `details.exit_code`). The underlying evidence was real (pre `status=error` + genuine marker,
  preserved in `events.json`), but the handover record was incomplete and the gate vacuous.
  Admissions INVALIDATED on audit; producer fixed (`_exit()` envelope reader + strict int/non-zero
  gates) and full rerun launched as attempt 5. No transaction content was hand-edited at any point.
- Attempt 5 (FINAL, 8 admitted / 0 refused): after the `_exit()` fix + strict gates + stale-refusal
  cleanup, full rerun from frozen sources. Post-audit of all 8 `outputs/<key>/transaction.json`:
  `requirement_done:true`, `verification_status:pass`, `behavior_verified:true`,
  `task_resolved:null`, `pre_exit_code:1` (int, non-zero, from the error envelope), `post_exit_code:0`,
  same command SHA pre/post, module file under the live workspace, targets = zero-based assistant
  ordinals over exactly the canonical arc (read → pre-run → edit(s) → post-run → handover),
  non-empty `actual.patch`, `events.json`/`commands.json` preserved.
  Counts toward the 32-target: **8 full闭环, 0 partial** (live canonical replay; the manual 8/8 above
  was the re-executability proof, the live 8/8 is the admitted layer).

## Failures with raw logs

- Live attempt 3 (6 admitted / 2 refused): the 2 refusals were CORRECT (pre-marker gate caught
  environment-confounded pre-failures), but post-hoc analysis showed the 6 admissions ran with the
  obsolete `pathlib` 1.0.1 backport shadowing the stdlib via the wrapper's PYTHONPATH-append —
  same-criterion violation (import machinery differed from manual verification). All 6 attempt-3
  admissions INVALIDATED; outputs overwritten by attempt 4. Lesson recorded: env-parity must cover
  the import system, not just the interpreter binary.
  - deepdiff pre (attempt 3): `from deepdiff.helper import …` →
    `from pydantic.main import BaseModel` → usersite pydantic 2.13.5 + `SystemError: The installed
    pydantic-core version (2.30.0) is incompatible with the current pydantic version, which
    requires 2.46.5` — caused by the earlier manual-pass pin for the pydantic trajectory leaking
    into the shared user site. Fixed by restoring 2.46.5 and vendoring 2.30.0 (real wheel,
    `vendor/pydantic-core-2.30.0/`) exposed ONLY under the pydantic live workspace via a `$PWD`
    branch in the wrapper.
  - pydantic pre (attempt 3): `AttributeError: 'PosixPath' object has no attribute 'read_text'`
    in `importlib/metadata.py` `entry_points` — the `pathlib` 1.0.1 backport (no `read_text`)
    shadowing stdlib `pathlib` through PYTHONPATH precedence. Fixed by `pip uninstall pathlib`
    (2026-10-20; nothing in the pinned stack imports it) + attempt-4 full rerun.
- Live attempts 1–2: see previous section (exit 127; yaml ModuleNotFoundError under
  PYTHONNOUSERSITE=1).

## Failures with raw logs

- Live attempt 1 (8/8 refused, all `post status=error`): `CommandError: /bin/bash: python:
  command not found` (exit 127). Harness-PATH root cause (bare `python` in frozen commands vs
  `dirname(sys.executable)` PATH without a `python` alias); fixed in producer via
  `ensure_python_shim` (version/sys.path-verified wrapper, command byte-identical).
- Live attempt 2 (8/8 refused, all `post status=error`): `ModuleNotFoundError: No module named
  'yaml'` (and per-repo equivalents). `LocalTransport` scrubs env with `PYTHONNOUSERSITE=1`,
  hiding this box's user site-packages (all pip installs; no sudo). Pre-gate (`status=error`)
  passed on the WRONG error both times — hence the new `pre_marker` gate (per-issue genuine
  failure substring, quote-normalized) plus strengthened post gate (exit 0, no `"pass": false`,
  no `INTERNAL ERROR`). Fixed via wrapper appending user site-packages AFTER the command
  PYTHONPATH (workspace sources always win). Re-run (attempt 3) in progress.
  Debug reproduction response envelopes (real, attempt 1 then attempt 2):
  `{"status": "error", "error_type": "CommandError", "error_message": "/bin/bash: python: command
  not found\n", "details": {"stdout": "", "stderr": "/bin/bash: python: command not found\n",
  "exit_code": 127}}`; `{"status": "error", "error_type": "CommandError", "error_message":
  "Traceback (most recent call last):\n ... ModuleNotFoundError: No module named 'yaml'\n",
  "details": {"stdout": "", "stderr": "…ModuleNotFoundError: No module named 'yaml'\n",
  "exit_code": 1}}`.
- Env-mismatch episodes (observed live during manual pass; raw stderr quoted verbatim, log files
  subsequently overwritten by the genuine pre-fail reruns — disclosed, not reconstructed):
  - yamllint: `ModuleNotFoundError: No module named 'pathspec'` → `pip install pathspec`.
  - deepdiff: `ModuleNotFoundError: No module named 'orderly_set'` → `pip install orderly-set`.
  - dataset: `TypeError: __init__() got an unexpected keyword argument 'bind'` (SQLAlchemy 2.x) → pin 1.4.49.
  - pydantic: `ImportError: cannot import name 'validate_core_schema' from 'pydantic_core'` → pin 2.30.0.
  - tenacity: `ImportError: cannot import name 'RetryBaseT' from 'tenacity.retry'` (preserved raw in
    `batch-01/tenacity/recovery.log`, exit 1; out of static-task scope).

## Encodings / codec

`target_assistant_steps` are zero-based assistant ordinals per
`training/encoding/gemma_tool_chat.py:360`
(`raise EncodingError("target_assistant_steps needs zero-based assistant ordinals")`).
`produce_p013.py` computes them post-hoc from final messages (same method as P012 lines 294-306),
restricted to read/pre/edit/post/handover arc steps (behavior) or edit_file steps only (reissue).
`--thinking on --thought-loss mask` policy unchanged on encode; no encoding run in this session
(no GPU, no tokenizer run — encode deferred to the SFT step).

## Blockers for scaling to 32

1. Only 8 P009-verified light repos re-executed here; 24+ further distinct train issues still needed.
   Next sources: P011 materialized edits (`research/011/.../outputs`, 442 applied) paired with
   newly reviewed per-issue behavior commands — each needs manual one-shot pre/post verification
   before `--mode behavior` ingestion (producer refuses without it).
2. Slow static contracts (mypy/tenacity ≈ minutes per run; official 30 s harness default too low) —
   producer sets `command_timeout_seconds=300`; GPU-box runs should keep ≥300 s.
3. Bare-`python` commands need the PATH shim (or GPU image with `python` on PATH) — fixed in producer.
4. marshmallow (priority-listed) has no P009 transaction plan — needs a new reviewed behavior command;
   not counted until verified.
5. Partial trajectories (any post-pass failure, missing module check, or changed command) never count
   toward the 32–64 quota by construction (`refusal.json`, non-zero producer exit path).
