"""Local checks for the 6032-token save_on_cpu carrier. Never launches Kaggle or torchrun."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from training.common import TrainingError, load_encoded_jsonl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TX = "fastapi_11194-3f51fb716edd-tx01"
ENCODED_SHA256 = "38cd361b7c1f15d78492963eddc226c17699ae11ec1d0c850071a7d02b585ce5"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncodedAndMemoryTests(unittest.TestCase):
    def test_capacity_row_is_the_6032_select_transaction(self):
        path = HERE / "encoded" / "encoded-capacity-8192.jsonl"
        with self.assertRaises(TrainingError) as caught:
            load_encoded_jsonl(path, 8192, require_real=True)
        self.assertIn("selection/heldout", str(caught.exception))
        rows, data = load_encoded_jsonl(path, 8192, require_real=False)
        self.assertEqual(data["sha256"], ENCODED_SHA256)
        self.assertEqual(rows[0]["meta"]["transaction_id"], TX)
        self.assertEqual(rows[0]["meta"]["group"]["split"], "select")
        self.assertEqual(len(rows[0]["input_ids"]), 6032)
        self.assertFalse(rows[0]["meta"]["synthetic"])
        self.assertFalse(rows[0]["meta"].get("truncated"))

    def test_memory_module_default_is_off(self):
        import inspect
        from training import memory
        source = inspect.getsource(memory.install_selected_token_loss)
        self.assertIn("save_on_cpu=False", source)
        self.assertIn("pin_memory=True", inspect.getsource(memory._backbone_save_context))


class CarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load(HERE / "worker.py", "save_on_cpu_worker_test")
        cls.builder = load(HERE / "build.py", "save_on_cpu_builder_test")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.build = self.builder.build(self.root / "built")
        self.wrapper = load(self.root / "built" / "probe.py", "save_on_cpu_wrapper_test")

    def tearDown(self):
        self.temporary.cleanup()

    def test_bundle_metadata_two_arms_and_no_launch(self):
        self.wrapper.BUNDLE = self.root / "bundle"
        self.wrapper.OUT = self.root / "output"
        loaded = self.wrapper.unpack_bundle()
        self.assertFalse(self.build["launch_performed"])
        metadata = json.loads((self.root / "built" / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "ccjjllt/p005-save-on-cpu-6032")
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["model_sources"], [])
        self.assertEqual(metadata["kernel_sources"], [
            "ccjjllt/p005-cpu-nf4-cache", "ccjjllt/p005-training-cpu-stage"])
        encoded = (self.wrapper.BUNDLE / "data" / "encoded-capacity-8192.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), ENCODED_SHA256)
        probe = (self.root / "built" / "probe.py").read_text()
        self.assertNotIn("diagnostic_2k", probe)
        self.assertNotIn("encoded-train-16384", probe)
        self.assertEqual(loaded.WORLD, 4)
        self.assertEqual(loaded.TX, TX)
        self.assertEqual(loaded.SEQ, 6032)
        launch = json.loads((self.root / "built" / "launch-plan.json").read_text())
        self.assertEqual(launch["command_for_operator_only"][-1], "1080")
        self.assertIn(str(self.root / "built"), launch["command_for_operator_only"])
        self.assertNotIn("agent_of_gemma_4-p005-sampling-sft", json.dumps(launch))

    def test_wrapper_runs_two_fresh_torchrun_jobs_and_kills_each(self):
        tree = ast.parse((self.root / "built" / "probe.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        text = ast.unparse(main)
        self.assertIn("for arm, flag in ARMS", text)
        self.assertIn("kill_process_group(proc)", text)
        self.assertIn("'baseline', 0", ast.unparse(tree))
        self.assertIn("'save_on_cpu', 1", ast.unparse(tree))
        self.assertIn("fresh_process", text)
        self.assertIn("discard_adapters", text)
        self.assertNotIn("run_ddp_loop", (HERE / "worker.py").read_text())

    def test_oom_exits_without_teardown_and_records_traceback(self):
        source = (HERE / "worker.py").read_text()
        self.assertIn("os._exit(0)", source)
        self.assertIn("failure_phase", source)
        self.assertIn("last_decoder_layer", source)
        self.assertIn("traceback.format_exc()", source)
        self.assertTrue(self.worker.is_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(self.worker.is_oom(RuntimeError("syntax")))
        rows, _data = self.worker.load_rows(HERE / "encoded" / "encoded-capacity-8192.jsonl")
        self.assertEqual(rows[0]["meta"]["transaction_id"], TX)
        self.assertEqual(len(rows[0]["input_ids"]), 6032)


if __name__ == "__main__":
    unittest.main()
