"""Local checks for the v2 capacity/SFT carrier. Never launches Kaggle or torchrun."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from training.common import TrainingError, load_config, load_encoded_jsonl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESERVED = {
    "fastapi_13713", "requests_6592", "requests_6644", "requests_7309",
    "requests_7505", "rich_3130", "rich_3296", "rich_3469",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncodedDataTests(unittest.TestCase):
    def test_train_16384_is_four_complete_real_rows(self):
        path = HERE / "encoded" / "encoded-train-16384.jsonl"
        rows, data = load_encoded_jsonl(path, 16384, require_real=True)
        self.assertEqual(data["rows"], 4)
        self.assertEqual(data["sha256"], "ea0791aa106112afbb8183423794382f66c268e79b73ee0fb22bac3618f46dff")
        ids = [row["meta"]["transaction_id"] for row in rows]
        self.assertEqual(ids, [
            "rich_3518-459640e4dc35-tx01",
            "rich_3942-27200bd450f3-tx01",
            "rich_4075-5e8118f292cf-tx03",
            "rich_4075-5e8118f292cf-tx05",
        ])
        for row in rows:
            self.assertFalse(row["meta"]["synthetic"])
            self.assertTrue(row["meta"]["training_eligible"])
            self.assertFalse(row["meta"].get("truncated"))
            self.assertEqual(row["meta"]["group"]["split"], "train")
            self.assertNotIn(row["meta"]["group"]["issue"], RESERVED)
            self.assertLessEqual(len(row["input_ids"]), 16384)
            self.assertGreater(len(row["input_ids"]), 8192)
        report = json.loads((HERE / "encoded" / "encoded-train-16384.jsonl.report.json").read_text())
        self.assertEqual(report["written"], 4)
        self.assertEqual(len(report["skipped_oversize"]), 7)
        self.assertTrue(all(item["truncated"] is False and item["clipped_json"] is False
                            for item in report["skipped_oversize"]))
        self.assertFalse(report["synthetic"])

    def test_select_8k_row_is_capacity_only(self):
        path = HERE / "encoded" / "encoded-capacity-8192.jsonl"
        with self.assertRaises(TrainingError) as caught:
            load_encoded_jsonl(path, 8192, require_real=True)
        self.assertIn("selection/heldout", str(caught.exception))
        rows, data = load_encoded_jsonl(path, 8192, require_real=False)
        self.assertEqual(data["rows"], 1)
        self.assertEqual(rows[0]["meta"]["transaction_id"], "fastapi_11194-3f51fb716edd-tx01")
        self.assertEqual(rows[0]["meta"]["group"]["split"], "select")
        self.assertEqual(len(rows[0]["input_ids"]), 6032)
        self.assertFalse(rows[0]["meta"]["synthetic"])

    def test_base_ddp_yaml_still_1024(self):
        cfg = load_config(REPO / "training" / "configs" / "sft_qat_nf4_ddp.yaml")
        self.assertEqual(cfg["max_sequence_length"], 1024)
        cfg16 = load_config(REPO / "training" / "configs" / "sft_qat_nf4_ddp_16384.yaml")
        self.assertEqual(cfg16["max_sequence_length"], 16384)
        self.assertEqual(cfg16["rank"], 8)
        self.assertEqual(cfg16["alpha"], 16)
        self.assertEqual(cfg16["kbit_prepare"], "norms_fp32")
        self.assertEqual(cfg16["loss_backend"], "selected_chunked")
        self.assertEqual(cfg16["learning_rate"], 0.00005)


class CarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load(HERE / "worker.py", "capacity_sft_worker_test")
        cls.builder = load(HERE / "build.py", "capacity_sft_builder_test")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.build = self.builder.build(self.root / "built")
        self.wrapper = load(self.root / "built" / "probe.py", "capacity_sft_wrapper_test")

    def tearDown(self):
        self.temporary.cleanup()

    def test_bundle_metadata_and_no_launch(self):
        self.wrapper.BUNDLE = self.root / "bundle"
        self.wrapper.OUT = self.root / "output"
        loaded = self.wrapper.unpack_bundle()
        self.assertFalse(self.build["launch_performed"])
        metadata = json.loads((self.root / "built" / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "ccjjllt/p005-capacity-sft-v2")
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["model_sources"], [])
        self.assertEqual(metadata["kernel_sources"], [
            "ccjjllt/p005-cpu-nf4-cache", "ccjjllt/p005-training-cpu-stage"])
        train = (self.wrapper.BUNDLE / "data" / "encoded-train-16384.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(train).hexdigest(),
                         "ea0791aa106112afbb8183423794382f66c268e79b73ee0fb22bac3618f46dff")
        self.assertNotIn("diagnostic_2k", (self.root / "built" / "probe.py").read_text())
        self.assertEqual(loaded.WORLD, 4)
        self.assertEqual(loaded.SCHEMA, "p005-capacity-sft-v2")

    def test_oom_helper_and_select_not_sft(self):
        self.assertTrue(self.worker.is_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(self.worker.is_oom(RuntimeError("syntax")))
        path = HERE / "encoded" / "encoded-capacity-8192.jsonl"
        with self.assertRaises(TrainingError):
            self.worker.load_rows(path, 8192, True)
        rows, _data = self.worker.load_rows(path, 8192, False)
        self.assertEqual(rows[0]["meta"]["group"]["split"], "select")


if __name__ == "__main__":
    unittest.main()
