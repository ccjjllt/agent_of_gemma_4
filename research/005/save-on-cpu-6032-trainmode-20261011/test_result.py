"""Tiny JSON check for the train-mode 6032 result and long-seq block. No GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT = HERE / "result.json"
RESULTS_MD = HERE / "results.md"
BLOCKED = HERE / "blocked-longseq.json"
BASELINE = HERE / "kaggle-output/save-on-cpu-6032-tm/baseline/rank-000-report.json"
OFFLOAD = HERE / "kaggle-output/save-on-cpu-6032-tm/save_on_cpu/rank-000-report.json"
HISTORICAL = ROOT / "research/005/save-on-cpu-6032-20261010/result.json"
DDP_YAML = ROOT / "training/configs/sft_qat_nf4_ddp.yaml"
FAKE_CARRIER = ROOT / "notebooks/005-base-adapter-pair-single-cell.py"


class TrainmodeResultTests(unittest.TestCase):
    def test_result_matches_rank_reports_and_gates(self):
        data = json.loads(RESULT.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        offload = json.loads(OFFLOAD.read_text(encoding="utf-8"))
        historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
        blocked = json.loads(BLOCKED.read_text(encoding="utf-8"))

        self.assertEqual(data["schema"], "p005-save-on-cpu-6032-tm-result")
        self.assertIs(data["comparison_valid_for_checkpointed_training"], True)
        self.assertIs(data["first_adam_step_completed"], False)
        self.assertIs(data["success"], False)
        self.assertIs(data["wrapper_success"], True)
        self.assertIn("not that training succeeded", data["wrapper_success_means"])
        self.assertEqual(data["kernel_live_slug"], "ccjjllt/p005-save-on-cpu-6032-train-mode")
        self.assertEqual(data["kernel_metadata_id"], "ccjjllt/p005-save-on-cpu-6032-tm")
        self.assertEqual(data["transaction_id"], "fastapi_11194-3f51fb716edd-tx01")
        self.assertEqual(data["sequence_length"], 6032)
        self.assertIs(data["did_not_truncate"], True)
        self.assertIs(data["did_not_retry_16k"], True)
        self.assertIs(data["adapter_discarded"], True)
        self.assertIs(data["loss_is_not_repair_gain"], True)
        self.assertIs(data["competition_submit"], False)
        self.assertIs(data["hypothesis_3_62_gib"]["used_as_claim"], False)

        b = data["arms"]["baseline"]
        s = data["arms"]["save_on_cpu"]
        self.assertIs(b["save_on_cpu"], False)
        self.assertIs(s["save_on_cpu"], True)
        self.assertIs(s["save_on_cpu_installed"], True)
        self.assertEqual(b["failure_phase"], "forward")
        self.assertEqual(b["last_decoder_layer"], 47)
        self.assertEqual(s["failure_phase"], "backward")
        self.assertEqual(s["last_decoder_layer"], 53)
        self.assertIs(b["optimizer_reached"], False)
        self.assertIs(s["optimizer_reached"], False)
        self.assertIs(b["first_adam_step_completed"], False)
        self.assertIs(s["first_adam_step_completed"], False)
        self.assertIsNone(b["adam_state"])
        self.assertIsNone(s["adam_state"])
        self.assertEqual(b["decoder_checkpoint_calls_sum"], 48)
        self.assertEqual(s["decoder_checkpoint_calls_sum"], 60)
        self.assertEqual(sum(baseline["decoder_checkpoint_calls"]), 48)
        self.assertEqual(sum(offload["decoder_checkpoint_calls"]), 60)
        self.assertTrue(all(baseline["training_mode"]["decoder_checkpointing_active"]))
        self.assertTrue(all(offload["training_mode"]["decoder_checkpointing_active"]))
        self.assertIn("sdpa_attention_forward", b["traceback"])
        self.assertIn("scaled_dot_product_attention", b["traceback"])
        self.assertNotIn("gate_proj", b["traceback"])
        self.assertIn("torch.autograd.backward", s["traceback"])
        self.assertIn("_engine_run_backward", s["traceback"])
        self.assertEqual(b["cuda_oom"]["tried_to_allocate_mib"], 378.0)
        self.assertEqual(b["cuda_oom"]["process_memory_in_use_gib"], 21.69)
        self.assertEqual(s["cuda_oom"]["tried_to_allocate_mib"], 124.0)
        self.assertEqual(s["cuda_oom"]["process_memory_in_use_gib"], 21.91)
        self.assertEqual(s["host_memory_at_report"]["oom_kill"], 0)
        self.assertEqual(s["host_memory_at_report"]["rss_gib"], 6.292)
        self.assertEqual(s["host_memory_at_report"]["mem_available_gib"], 166.2)
        self.assertEqual(s["forward"]["peak_allocated_gib"], 18.988)
        self.assertIs(s["forward_completed"], True)
        self.assertIs(b["forward_completed"], False)
        self.assertIs(data["measured_difference"]["promised_gib_savings_for_9841_or_12832"], False)
        self.assertIs(data["withdrawn_v1_claims_not_revived"]["live_mlp_dequant_established_root_cause"], False)
        self.assertIs(data["withdrawn_v1_claims_not_revived"]["host_pin_memory_caused_sigkill"], False)

        self.assertEqual(data["quota"]["before_s"], 27356.648)
        self.assertEqual(data["quota"]["after_s"], 27969.592)
        self.assertEqual(data["quota"]["delta_s"], 612.944)
        self.assertEqual(data["quota"]["delta_hours"], 0.170262)

        self.assertEqual(historical["status"], "invalid_checkpointed_training_comparison")
        self.assertEqual(data["historical_v1"]["status_unchanged"], "invalid_checkpointed_training_comparison")

        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIs(blocked["gpu_launched"], False)
        self.assertIs(blocked["did_not_push_9841"], True)
        self.assertIs(blocked["did_not_push_12832"], True)
        self.assertIs(blocked["did_not_retry_16k"], True)
        self.assertIs(blocked["did_not_rewrite_fsdp"], True)
        self.assertEqual(blocked["traceback_class"]["baseline"]["oom_site"], "sdpa_attention_forward / scaled_dot_product_attention")
        self.assertEqual(blocked["traceback_class"]["save_on_cpu"]["failure_phase"], "backward")

        text = RESULTS_MD.read_text(encoding="utf-8")
        self.assertIn("p005-save-on-cpu-6032-train-mode", text)
        self.assertIn("p005-save-on-cpu-6032-tm", text)
        self.assertIn("valid for checkpointed training", text)
        self.assertIn("**not** that training succeeded", text)
        self.assertIn("invalid_checkpointed_training_comparison", text)
        self.assertIn("remain withdrawn", text)
        self.assertIn("Host pin_memory SIGKILL is **not** established", text)
        yaml_text = DDP_YAML.read_text(encoding="utf-8")
        self.assertIn("max_sequence_length: 1024", yaml_text)
        self.assertFalse(FAKE_CARRIER.exists())


if __name__ == "__main__":
    unittest.main()
