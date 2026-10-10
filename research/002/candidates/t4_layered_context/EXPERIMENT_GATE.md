# T4 online-experiment gate: still missing evidence vs already have it but not using it

Source: `docs/plan/002-research-opportunities-20261001.md` §8 (T4 paragraph),
`research/002/mechanism-study-20261001/README.md` S3 plus M05 (compact
transaction that submits) / M06 (compact attribute fix without broad search).

Status: T4 prototype/spec only (CPU). No online run is authorized by this
document. Whether to start a T4 online comparison is decided AFTER the frozen
T0/T1 batch (§9 step 1-2), from trajectories — not from enthusiasm for
layering.

## 1. Decision rule

Run the frozen T0/T1 batch first (six tasks, 12 attempts, pre-registered
list), then classify each T0/T1 failure offline (researcher side; gold-weak
labels allowed for diagnosis only, never as candidate input):

- Class A — still missing relevant evidence (`outline`/`locate` shaped):
  no successful read of the relevant file, or no anchor-equivalent region,
  or the skill returns empty/ambiguous and scoped fallback reads also miss,
  or the first relevant read arrives too late to leave edit budget
  (guide: >120 s or >2/3 of elapsed budget in the 5-minute control).
- Class B — evidence obtained but not converted (T2/T3 shaped, cf. §8 and
  M01/M05): relevant region read early (guide: <=60 s) yet no successful
  production edit, or prolonged re-reads after a decisive output (e.g. M01
  handler-not-triggered at 235.7 s followed by header re-reads to 294.9 s),
  or paired positive/negative witnesses present (cf. M05 B) with no submit.

Start a T4 online pilot ONLY if Class A dominates the T1 failures
(guide: at least one third of T1 failures are Class A AND the CPU prototype
returns the missing region within budget on local checkouts of the same
repos — see §3). If Class B dominates, keep priority on T2/T3 and leave T4
as a CPU prototype; do not spend online budget re-proving navigation.

Additional stop conditions (any one defers T4 online):

- T1 skill-call rate is near zero ("insufficient intervention"): report why
  the entry was not used, revise the entry, and re-measure before any T4 run
  (same logic as §6 for T2).
- The missing evidence is issue-hint-shaped (e.g. M07: the issue already
  names the function/relation): the fix is prompt discipline, not a new
  retrieval layer.
- The failure is a verification-eligibility case (cf. M08/M09 caveats):
  diagnose the process, do not count it as T4-addressable gain.

## 2. If the gate opens: minimal online comparison

- Compare T1 vs T1+layered only: same navigation strategy, same read-range
  caps, same model budget. The delta is the layered combination (outline +
  sourced locate + complete-only inspect), not "better retrieval" in general.
- Pre-register the task list (mechanical-edit, semantic, and normal-control
  mix) before running; do not cherry-pick after the fact.
- Report: skill invocations, empty/ambiguous rate, time-to-first-relevant
  read, read-to-edit conversion, final resolved, time/output volume.
- One comparison at a time: T4 is never bundled with T2/T3 in a single arm.
- The 28 heldout tasks stay untouched; new development-pool tasks are added
  only after a development-panel gain, with exposure logged (§9).

## 3. CPU prototype acceptance (this directory)

- `python3 -m py_compile` on `scripts/layered.py`; `--help` for all three
  subcommands; manual `outline`/`locate`/`inspect` runs on a synthetic `/tmp`
  fixture (local diagnosis flag), each stdout <= 4000 chars and valid JSON.
- No heldout content, no large-corpus download, no network, no new model.
- Compact-transaction lesson from M05/M06 is preserved as a negative
  constraint: T4 must not inflate retrieval when a one-function inspect plus
  a small edit already closes the loop. `outline` breadth is capped (<=8
  files) for exactly this reason.

## 4. Alignment with §8

- `outline` returns directories + definition signatures (no bodies).
- `locate` allows 2-4 sourced queries, <=8 candidates total.
- `inspect` expands a small number of complete functions by AST boundaries.
- Shared output budget (4000 chars/call) is enforced with valid JSON.
- First-400-chars-as-function and truncated-display-as-edit-input are both
  explicitly forbidden in `SKILL.md` and the script docstring.
- No neural vector model; stdlib only; path safety mirrors T1
  (`/workspace` confinement, no symlink following, explicit local flag).
