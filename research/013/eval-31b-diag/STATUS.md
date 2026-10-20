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

## 2026-10-20 ~06:05Z: GPU v1 ERROR = wrong GPU shape (T4x2), not agent bug
- `ccjjllt/p013-31b-diag-pair` v1 (pushed WITHOUT --accelerator): session landed on non-4xL4 → coordinator assert `Requires four L4 GPUs` at 122s, 0 attempts run, cost ~0.03h (9.17→9.14h). Same P012 precedent ("缺 L4 machine_shape，上了T4x2") and P011 V1 precedent.
- Fix: re-push with `--accelerator NvidiaL4` (+ competition binding already in metadata). Same-id versioned push → 409 Conflict twice (also seen on CPU kernel); new-id works. New kernel `ccjjllt/p013-31b-diag-pair-v2` v1 pushed ~06:10Z with --accelerator NvidiaL4, no title warning. Dir `kernel-v2/` (copy of kernel/, id+title only).

## 2026-10-20 ~06:30Z: 诊断完成 0/4 vs 0/4，修正一次后复测
- 报告 `diag-report.md`：B-4006过程占优（edit@9/124/交接/review）；B-3180零编辑负证据；B-7205同修复但交接晚致review预算尽；B-6757 review连交2次0/0违规。绑定约束是墙钟（工具调用中位~2-4分钟），非call计数。
- 修正（CPU）：repair.md加墙钟交接（120s读/200s改/300s交，<180s立即交，call12未改则下一调用必edit，handoff附time_remaining）；review.md禁无新edit二次提交（终身2次，第二次须真实edit+git diff）；single.md同步。
- 重建carrier（新prompt sha），CPU预检v3 `ccjjllt/p013-31b-diag-cpu-prepare-v3` v1已推（免费）。待COMPLETE→推GPU full v3（kernel-v3, --accelerator NvidiaL4）。

## 2026-10-20 ~06:45Z: CPU预检v3通过，GPU复测v3已推
- `ccjjllt/p013-31b-diag-cpu-prepare-v3` COMPLETE：2/2 prepared（SequentialAgent repair+review，修正后prompt）。
- `ccjjllt/p013-31b-diag-pair-v3` v1已推（kernel-v3/, --accelerator NvidiaL4）。推前配额8.18h。判据：handoff步骤提前、review有提交动作、B-3180类零编辑消失。
