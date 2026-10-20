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
