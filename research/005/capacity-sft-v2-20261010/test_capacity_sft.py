"""Local checks for the v2 capacity/SFT carrier. Never launches Kaggle or torchrun."""
import ast
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

    def test_formal_sft_failure_is_not_an_expected_capacity_skip(self):
        output = self.root / "failed-sft"
        output.mkdir()
        report_path = output / "rank-000-report.json"
        rank = {"rank": 0, "status": "failed", "gpu_sft": False,
                "capacity": [{"window": 8192, "backprop_ok": True},
                             {"window": 16384, "backprop_ok": True}]}
        report_path.write_text(json.dumps(rank))
        results = self.wrapper.collect_results(output)
        self.assertEqual(results["rank_zero_status"], "failed")
        # Execute the production final-status expressions without starting a worker.
        tree = ast.parse((self.root / "built" / "probe.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        assignments = {node.targets[0].id: node for node in ast.walk(main)
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in {"capacity_recorded", "sft_ok", "skipped_ok", "completed"}}
        code = compile(ast.Module(body=[assignments[name] for name in (
            "capacity_recorded", "sft_ok", "skipped_ok", "completed")], type_ignores=[]),
            "wrapper-final-status", "exec")
        for status, expected in (("failed", False), ("capacity_oom", True), ("capacity_only", True)):
            with self.subTest(status=status):
                rank["status"] = status
                report_path.write_text(json.dumps(rank))
                namespace = {"results": self.wrapper.collect_results(output),
                             "retention": {"retained": None}, "exit_code": 0}
                exec(code, namespace)
                self.assertEqual(namespace["completed"], expected)

    def test_capacity_oom_keeps_traceback_and_outer_failures_return_nonzero(self):
        tree = ast.parse((HERE / "worker.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        outer = next(node for node in main.body if isinstance(node, ast.Try))
        self.assertEqual(ast.literal_eval(outer.handlers[0].body[-1].value), 2)
        capacity_handlers = [node.handlers[0] for node in outer.body
                             if isinstance(node, ast.Try) and node.handlers]
        self.assertEqual(len(capacity_handlers), 2)
        for handler in capacity_handlers:
            try:
                raise RuntimeError("CUDA out of memory: capacity review fixture")
            except RuntimeError:
                namespace = {"entry": {}, "traceback": __import__("traceback")}
                save_trace = next(node for node in handler.body if isinstance(node, ast.Assign)
                                  and isinstance(node.targets[0], ast.Subscript)
                                  and ast.literal_eval(node.targets[0].slice) == "traceback")
                exec(compile(ast.Module(body=[save_trace], type_ignores=[]),
                             "capacity-traceback", "exec"), namespace)
            self.assertIn("capacity review fixture", namespace["entry"]["traceback"])
            self.assertIn("Traceback", namespace["entry"]["traceback"])

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
