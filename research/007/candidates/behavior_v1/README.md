# P007 `behavior_v1` — Single-Variable Behavioral Verification Candidate

Derived on 2026-10-17 from [`public022_v10`](../public022_v10/README.md) (`flexonafft` V10, `scriptVersionId=357955943`, Apache 2.0).

`agent/` contains four files:
- `agent.yaml`, `configs/sampling.yaml`, and `eval_config.yaml` are byte-for-byte identical to `public022_v10` (31B CT v2, 8-minute / 28-call / 80-turn budget, 240s command timeout, temperature 0.2, top-p 0.95, max output 8192, thinking budget 4096).
- `prompts/system.md` modifies only the behavioral verification instructions (the inline reproduction bullet at the end of `## 2. Locate` and the three bullets in `## 4. Verify (at most 2 calls)`):
  1. Confirm the real API and imports from an existing caller or test, and assert the concrete observable return value, resolved type, output, or exception that distinguishes buggy from expected behaviour (merely not raising or returning an unresolved `ForwardRef` is not a pass).
  2. Once the cause and target location are supported by the read code, make the minimal production edit instead of writing further reproduction variations.
  3. Distinguish target code failure from witness script failure (wrong arguments, entry point, fixture, or import scope): fix the witness command rather than reverting a valid production edit, and never carry over an earlier pass if the check criterion changed.

This candidate does not inherit the 0.22 public score of `public022_v10`. Provenance and hashes are stored in `source-manifest.json` outside `agent/`.
