# P013 step-1 31B diag: run status (2026-10-20 UTC)

## Quota (same-command verified)
- ccjjllt: GPU used 20.83h / remaining 9.17h / 30h (refresh 2026-10-24). Identity: `kernels list --mine` author ccjjllt.
- qingyu1232: GPU used 15.29h / remaining 14.71h. Identity verified. Reserved for later (12-task pairing or behavior SFT).
- jiyujia (active env default): 2.12h remaining — NOT used for P013.

## Pushes
1. CPU prepare `ccjjllt/p013-31b-diag-pair-cpu-prepare` v1 — pushed 2026-10-20 ~04:55Z, status RUNNING (quota-before in `quota-before-cpu-prepare.json`, receipt in `cpu-prepare-push-receipt.txt`). Free (CPU session).
2. GPU full `ccjjllt/p013-31b-diag` (8 attempts: 4 tasks x 2 arms) — pending CPU prepare COMPLETE.

## Next
- CPU prepare COMPLETE → push GPU full kernel (ccjjllt, est 0.5-0.9h quota).
- GPU full COMPLETE → pull outputs, 4-metric comparison (wins/regressions, first-edit Call#, 124/timeout + patch_size==0 counts, patch hygiene) → decide Sequential vs single-agent → freeze candidate.

## 2026-10-20 ~05:40Z: CPU prepare v2 COMPLETE (2/2 prepared)
- v1 ERROR根因：worker `agent.tools` 在 SequentialAgent 上抛 AttributeError（carrier bug，非agent-yaml问题；compile_submission本身已成功）。修：`notebooks/build_006_reference_pair.py:145-148` CPU_COMPILE模板改hasattr兼容（sub_agents上报）；重建carrier（coordinator `13377c62…`，worker `33ed9f5b…`）。
- v1重推同id报409，改推新kernel `ccjjllt/p013-31b-diag-cpu-prepare-v2` v1（免费CPU）。
- v2验证：public022_v10 prepared=1/1 (LlmAgent swe_agent, 9 tools)；p013_candidate prepared=1/1 (SequentialAgent, sub-agents LlmAgent:p013_repair + LlmAgent:p013_review)。原件 /tmp/p013-cpu-v2（zip已解包run-v2）。

## 2026-10-20 ~05:45Z: GPU full已推
- kernel `ccjjllt/p013-31b-diag` v1，8 attempts（4 tasks x 2 arms, AB/BA），4xL4，est 15-25min wall (~0.5-0.9h quota)。推前配额 `quota-before-gpu-full.json`：剩9.17h。
