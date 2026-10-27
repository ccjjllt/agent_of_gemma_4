# 新任务小轮实际执行

所有题来自S600 train；四道研究验证题不参加focal采集或更新。这些小样本结果不构成比赛heldout成绩。

2026-10-27T07:24:34.908434+00:00：Authorized short GPU trial: exact recovered untrained batch,2 mixed groups/4 actions;8K complete context, Unsloth model-parallel backward on2 GPUs, one optimizer step only. No new collection or independent rollouts.

2026-10-27T07:25:43.343192+00:00：Actual private GPU notebook URL verified from original submission receipt: jiyujia/vfarl-unsloth-recovered-reward-update-r1/1; no duplicate submission.

2026-10-27T07:43:19.169227+00:00：Actual completed Unsloth short update: {"kernel": "jiyujia/vfarl-unsloth-recovered-reward-update-r1/1", "platform_status": "COMPLETE", "optimizer_steps": 1, "source_data_sha256": "c1e7dd56322899e95773a972a1326be9a453b63ed7fac3244ccae9d05cec7c07", "grad_norm": 0.8113248348236084, "parameter_delta_l2": 0.05792091503185881, "actual_max_sequence_tokens": 6495, "wall_seconds_to_commit": 559.883736877, "update_seconds_to_commit": 322.57317009300004, "total_experiment_wall_seconds": 815.597628233, "post_update_diagnostics_completed": true, "mean_advantage_weighted_logprob_change": -0.00010989460001216945, "memory_by_device": [{"device": "cuda:0", "peak_allocated_bytes": 11769253376, "peak_reserved_bytes": 12977176576}, {"device": "cuda:1", "peak_allocated_bytes": 15232603136, "peak_reserved_bytes": 16852713472}], "importance_weights": {"actions": 4, "limitation": "Describes this selected focal batch; does not establish full-trajectory coverage or an unbiased task-level objective.", "raw_sequence_ess": 3.2977636049747283, "raw_sequence_ess_fraction": 0.8244409012436821, "sequence_log_weight_max": 0.6565549295714703, "sequence_log_weight_min": -0.7186607709394686, "unit": "One complete independently sampled focal action, not one token", "upper_capped_actions": 0, "used_sequence_ess": 3.29776360966802, "used_sequence_ess_fraction": 0.824440902417005, "used_weight_max": 1.928138256072998, "used_weight_min": 0.48740455508232117}, "remaining_gpu_h": 17.4334929825}



## Completed short GPU result

- Platform COMPLETE; real optimizer steps: 1. Recovered2 mixed groups/4 actions, no resampling.
- Grad norm 0.81132483; LoRA delta L2 0.05792092.
- Update computation/export 5.38min; model initialization237.31s; complete notebook 13.59min.
- Actual longest backward sequence 6495tokens;2 GPUs. Peak allocated:11.0/14.2GiB.
- Actual GPU quota consumed 0.45776h; remaining 17.43349h.
- Direction metric is slightly negative: advantage-weighted -0.000109895, importance-weighted -0.000272025. Do not claim learning improvement.
- No independent task comparison or16K backward was run. This exact batch SHA is committed; do not train it again.
