"""Local offline checks for the CT adapter probe. Does not call Kaggle or load 31B."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build as builder  # noqa: E402
import check_metadata  # noqa: E402
import constants  # noqa: E402
import probe  # noqa: E402


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.built = builder.build(self.root / "kernel")

    def tearDown(self):
        self.tmp.cleanup()

    def test_generated_metadata_is_pinned_gpu_v30(self):
        errors = check_metadata.check_metadata(self.root / "kernel" / "kernel-metadata.json")
        self.assertEqual(errors, [])
        data = json.loads((self.root / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual(data["docker_image"], constants.STARTER_DOCKER_IMAGE)
        self.assertIs(data["enable_gpu"], True)
        self.assertIs(data["enable_tpu"], False)
        self.assertIs(data["enable_internet"], False)
        self.assertIs(data["is_private"], True)
        self.assertEqual(data["model_sources"], [constants.CT_MODEL])
        self.assertEqual(data["kernel_sources"], [constants.DDP_PROBE_KERNEL])
        self.assertIn(constants.WHEELHOUSE_DATASET, data["dataset_sources"])
        self.assertIn(constants.COMPETITION, data["competition_sources"])
        self.assertEqual(
            sorted(path.name for path in (self.root / "kernel").iterdir()),
            ["kernel-metadata.json", "probe.py"],
        )

    def test_wrong_docker_image_and_cpu_fail(self):
        path = self.root / "kernel" / "kernel-metadata.json"
        data = json.loads(path.read_text())
        data["docker_image"] = constants.FORBIDDEN_DOCKER_IMAGE
        data["enable_gpu"] = False
        data["model_sources"] = []
        data["kernel_sources"] = []
        path.write_text(json.dumps(data))
        errors = check_metadata.check_metadata(path)
        joined = " ".join(errors)
        self.assertIn("docker_image", joined)
        self.assertIn("enable_gpu", joined)
        self.assertIn("model_sources", joined)
        self.assertIn("kernel_sources", joined)

    def test_invalid_json_fails(self):
        path = self.root / "bad.json"
        path.write_text("{not json")
        errors = check_metadata.check_metadata(path)
        self.assertTrue(any("valid JSON" in item for item in errors))


class T0AndAdapterTests(unittest.TestCase):
    def test_embedded_t0_matches_freeze(self):
        self.assertEqual(probe.verify_t0_bundle(), {})
        freeze = HERE.parents[1] / "002" / "variants" / "t0" / "agent"
        for relative, expected in constants.T0_SHA256.items():
            text = (freeze / relative).read_text(encoding="utf-8")
            self.assertEqual(text, probe.T0_BUNDLE[relative], relative)
            self.assertEqual(probe.sha256_text(text), expected, relative)
        self.assertNotIn("adapter:", probe.T0_BUNDLE["agent.yaml"])
        self.assertNotIn("skills:", probe.T0_BUNDLE["agent.yaml"])

    def test_inject_adapter_keeps_six_tools(self):
        injected = probe.inject_adapter(probe.T0_BUNDLE["agent.yaml"], constants.ADAPTER_NAME)
        self.assertIn(f"adapter: {constants.ADAPTER_NAME}", injected)
        self.assertTrue(injected.splitlines()[1].startswith("model:"))
        self.assertTrue(injected.splitlines()[2].startswith("adapter:"))
        tmp = tempfile.TemporaryDirectory()
        layout = probe.write_t0_agent(Path(tmp.name) / "agent", constants.ADAPTER_NAME)
        self.assertEqual(layout["tools"], list(probe.T0_TOOLS))
        yaml_text = (Path(tmp.name) / "agent" / "agent.yaml").read_text(encoding="utf-8")
        self.assertIn(f"adapter: {constants.ADAPTER_NAME}", yaml_text)
        self.assertNotIn("skills:", yaml_text)
        tmp.cleanup()

    def test_find_adapter_matches_fingerprints_and_skips_size_mismatch(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        other = root / "sample" / "adapters" / "main_lora"
        other.mkdir(parents=True)
        (other / "adapter_config.json").write_text("{}", encoding="utf-8")
        (other / "adapter_model.safetensors").write_bytes(b"tiny")
        good = root / "notebooks" / "ccjjllt" / "p005-training-ddp-probe" / "ddp-probe" / "adapter"
        good.mkdir(parents=True)
        config = b'{"peft_type":"LORA","r":8}'
        weights = b"x" * 128
        (good / "adapter_config.json").write_bytes(config)
        (good / "adapter_model.safetensors").write_bytes(weights)
        matched, candidates = probe.find_diagnostic_adapter(
            root,
            expected_config_sha=probe.sha256_file(good / "adapter_config.json"),
            expected_weight_sha=probe.sha256_file(good / "adapter_model.safetensors"),
            expected_weight_bytes=len(weights),
        )
        self.assertEqual(matched, good)
        self.assertGreaterEqual(len(candidates), 2)
        skipped = [item for item in candidates if item.get("skipped_weight_hash") == "size_mismatch"]
        self.assertTrue(skipped)
        tmp.cleanup()

    def test_fingerprint_mismatch_is_not_loaded(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        bad = root / "adapter"
        bad.mkdir()
        (bad / "adapter_config.json").write_text("{}", encoding="utf-8")
        (bad / "adapter_model.safetensors").write_bytes(b"nope")
        matched, candidates = probe.find_diagnostic_adapter(root)
        self.assertIsNone(matched)
        self.assertEqual(len(candidates), 1)
        tmp.cleanup()

    def test_extract_tool_names_and_classify(self):
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"name": "get_status", "arguments": "{}"},
                    }]
                }
            }]
        }
        self.assertEqual(probe.extract_tool_names(payload), ["get_status"])
        ok = {
            "adapter_files_ok": True,
            "adapter_in_manifest": True,
            "adapter_in_vllm": True,
            "vllm_started": True,
            "tool_call_ok": True,
            "repair_success_claimed": False,
        }
        self.assertEqual(probe.classify(ok), "tool_call_ok_no_repair_claim")
        missing = dict(ok, adapter_files_ok=False, tool_call_ok=False)
        self.assertEqual(probe.classify(missing), "adapter_not_loaded")
        not_in_vllm = dict(ok, adapter_in_vllm=False, tool_call_ok=False)
        self.assertEqual(probe.classify(not_in_vllm), "adapter_not_loaded")
        infer = dict(ok, tool_call_ok=False)
        self.assertEqual(probe.classify(infer), "inference_error")
        with self.assertRaises(ValueError):
            probe.classify(dict(ok, repair_success_claimed=True))

    def test_python_313_is_pin_failed_and_forbids_pip(self):
        status, allow = probe.evaluate_python((3, 13, 15))
        self.assertEqual(status, "pin_failed")
        self.assertFalse(allow)
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        out = root / "working"
        out.mkdir()
        calls = []

        def fake_pip(executable, wheels):
            calls.append((executable, list(wheels)))
            raise AssertionError("pip must not run on pin_failed")

        report = probe.run(
            out_dir=out,
            input_root=root / "input",
            version_info=(3, 13, 15),
            executable="/usr/bin/python",
            pip=fake_pip,
        )
        self.assertEqual(report["abi_status"], "pin_failed")
        self.assertFalse(report["pip_attempted"])
        self.assertEqual(report["classification"], "inference_error")
        self.assertEqual(calls, [])
        self.assertFalse(report["repair_success_claimed"])
        self.assertTrue(report["not_repair_sft"])
        tmp.cleanup()

    def test_v30_does_not_import_removed_swegemma_discovery(self):
        source = (HERE / "probe.py").read_text(encoding="utf-8")
        self.assertNotIn("swegemma.models.discovery", source)
        self.assertIn("discover_declared_models", source)

    def test_stage_skips_cutlass_and_rewrites_cu128(self):
        self.assertIsNone(probe.stage_wheel_name("nvidia_cutlass-1.0-py3-none-any.whl"))
        self.assertEqual(
            probe.stage_wheel_name("torch-2.8.0cu128-cp312-cp312-linux_x86_64.whl"),
            "torch-2.8.0+cu128-cp312-cp312-linux_x86_64.whl",
        )

    def test_missing_adapter_classifies_before_server(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        out = root / "working"
        out.mkdir()
        wheelhouse = root / "input" / "datasets" / "metric" / "gemma-4-developer-agent-wheelhouse"
        data = root / "input" / "competitions" / "gemma-4-developer-agent"
        wheelhouse.mkdir(parents=True)
        data.mkdir(parents=True)
        (data / "tasks.jsonl").write_text("{}\n")
        (wheelhouse / "swegemma-0.2.11-py3-none-any.whl").write_bytes(b"w" * 8)
        calls = []

        def fake_pip(executable, wheels):
            calls.append(list(wheels))
            return subprocess.CompletedProcess(args=["pip"], returncode=0, stdout="ok\n", stderr="")

        report = probe.run(
            out_dir=out,
            input_root=root / "input",
            version_info=(3, 12, 13),
            executable="/usr/bin/python",
            pip=fake_pip,
        )
        self.assertEqual(report["classification"], "adapter_not_loaded")
        self.assertFalse(report["adapter_files_ok"])
        self.assertFalse(report["vllm_started"])
        self.assertEqual(calls, [])
        self.assertFalse(report["repair_success_claimed"])
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
