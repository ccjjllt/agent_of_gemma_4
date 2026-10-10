"""Local carrier simulations only: never launches torchrun or accesses Kaggle."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(ROOT / "ddp-probe" / "worker.py", "ddp_worker_test")
builder = load(ROOT / "build_ddp_probe.py", "ddp_builder_test")


def fake_adapter(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
    header = json.dumps({
        "layer.lora_A.weight": {"dtype": "F32", "shape": [8, 2], "data_offsets": [0, 64]},
        "layer.lora_B.weight": {"dtype": "F32", "shape": [2, 8], "data_offsets": [64, 128]},
    }).encode()
    header += b" " * (-len(header) % 8)
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(128))
    (path / "README.md").write_text("Temporary PEFT generated card")


class CarrierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        dest = self.root / "built"
        self.build = builder.build(dest)
        self.wrapper = load(dest / "probe.py", "ddp_wrapper_test")

    def tearDown(self):
        self.temporary.cleanup()

    def test_bundle_is_exact_and_does_not_mount_original_model(self):
        self.wrapper.BUNDLE = self.root / "bundle"
        self.wrapper.OUT = self.root / "output"
        loaded = self.wrapper.unpack_bundle()
        self.assertEqual(loaded.LENGTHS, (256, 256, 1024, 2048))
        for name, item in self.wrapper.SOURCE_MANIFEST["files"].items():
            data = (self.wrapper.BUNDLE / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
        metadata = json.loads((self.root / "built" / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["model_sources"], [])
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["is_private"])
        self.assertEqual(len(metadata["kernel_sources"]), 2)
        self.assertFalse(self.build["launch_performed"])

    def test_inverse_shuffle_and_ineligible_diagnostic_rows(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3, 4, 5]
        rows = worker.diagnostic_rows(Tokenizer(), torch)
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
        seen = [rows[index] for index in order]
        self.assertEqual([len(row["input_ids"]) for row in seen], [n for n in worker.LENGTHS for _ in range(4)])
        for offset, row in enumerate(seen):
            self.assertEqual(row["meta"]["planned_step"], offset // 4 + 1)
            self.assertEqual(row["meta"]["planned_rank"], offset % 4)
            self.assertTrue(row["meta"]["synthetic"])
            self.assertFalse(row["meta"]["training_eligible"])
            self.assertEqual(sum(value != -100 for value in row["labels"][1:]), 64)

    def test_cache_completion_and_dict_json_fail_closed(self):
        cache = self.root / "cache" / "qat-nf4"
        cache.mkdir(parents=True)
        worker.write_json(cache / "cache-manifest.json", {"source": worker.SOURCE})
        worker.write_json(cache.parent / "cache-report.json", {"success": False})
        worker.write_json(cache.parent / "cache-child-report.json", {"stage": "complete", "success": True})
        with self.assertRaisesRegex(RuntimeError, "producer"):
            worker.verify_cache(cache)
        worker.write_json(cache.parent / "cache-report.json", {"success": True})
        self.assertTrue(worker.verify_cache(cache)["cpu_cache_verified_complete"])
        with self.assertRaises(TypeError):
            worker.write_json(self.root / "invalid.json", [])

    def test_observer_calls_real_cpu_adamw_and_restores_original(self):
        model = torch.nn.Linear(2, 1)
        class FakeDist:
            @staticmethod
            def get_rank(): return 0
            @staticmethod
            def get_world_size(): return 4
            @staticmethod
            def all_gather_object(out, value):
                for rank in range(4):
                    out[rank] = dict(copy.deepcopy(value), rank=rank)
        def digest(module):
            return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in module.parameters())).hexdigest()
        initial = digest(model)
        original = torch.optim.AdamW.step
        records, events = [], []
        with mock.patch.object(torch.cuda, "synchronize"), mock.patch.object(worker, "cuda_memory", return_value={}):
            with worker.observe_adamw(torch, model, FakeDist(), torch.device("cpu"), initial,
                                      digest, lambda stage, **values: events.append(stage), records):
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
                for _ in range(2):
                    optimizer.zero_grad()
                    model(torch.tensor([[1., 2.]])).square().sum().backward()
                    optimizer.step()
                    scheduler.step()
        self.assertIs(torch.optim.AdamW.step, original)
        self.assertNotEqual(initial, digest(model))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(entry["all_rank_hashes_equal"] and entry["all_rank_adapters_changed"] for entry in records))
        self.assertTrue(all(entry["diagnostic_timing"]["adapter_hash_seconds"] >= 0 for entry in records))

    def test_later_failure_retains_latest_complete_adapter_and_no_optimizer(self):
        output = self.root / "run"
        training = output / "training"
        for step in (1, 2, 3):
            directory = training / f"checkpoint-{step:06d}"
            fake_adapter(directory / "adapter")
            (directory / "training_state.pt").write_bytes(b"large optimizer placeholder")
            if step < 3:
                worker.write_json(directory / "progress.json", {"global_step": step})
        retention = self.wrapper.retain_latest_adapter(output)
        self.assertEqual(retention["retained"]["step"], 2)
        self.assertFalse(list(output.rglob("*.pt")))
        self.assertFalse(list(training.glob("checkpoint-*")))
        self.assertEqual({p.name for p in (output / "adapter").iterdir()},
                         {"adapter_config.json", "adapter_model.safetensors"})
        self.assertGreater(retention["removed_optimizer_and_rng_bytes"], 0)

    def test_partial_reports_and_diagnostic_time_are_separate(self):
        output = self.root / "run"
        (output / "training").mkdir(parents=True)
        records = []
        for step in (1, 2):
            records.append({"step": step, "all_rank_hashes_equal": True, "all_rank_adapters_changed": True,
                "diagnostic_timing": {"total_seconds": 2.0},
                "all_rank_gradients": [{"finite": True, "nonzero_gradient_tensors": 4}] * 4,
                "all_rank_updates": [{"trainable_sha256": str(step) * 64}] * 4})
        for rank in range(4):
            worker.write_json(output / f"rank-{rank:03d}-report.json", {"rank": rank, "status": "failed", "steps": records})
        metrics = [dict(step=step, seconds=5.0, supervised_tokens=256, loss=2., grad_norm=1.) for step in (1, 2)]
        (output / "training" / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in metrics) + '\n{"truncated":')
        result = self.wrapper.collect_results(output)
        self.assertTrue(result["first_two_steps_verified"])
        self.assertEqual(result["all_rank_verified_steps"], [1, 2])
        self.assertEqual(len(result["metric_read_errors"]), 1)
        self.assertTrue(all(x["estimated_seconds_excluding_probe_validation"] == 3.0 for x in result["step_results"]))


if __name__ == "__main__":
    unittest.main()
