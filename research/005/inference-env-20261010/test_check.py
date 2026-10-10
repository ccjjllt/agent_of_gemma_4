"""Local offline checks for the CPU inference-env pin. Does not call Kaggle."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_generated_metadata_is_pinned_cpu_v30(self):
        errors = check_metadata.check_metadata(self.root / "kernel" / "kernel-metadata.json")
        self.assertEqual(errors, [])
        data = json.loads((self.root / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual(data["docker_image"], constants.STARTER_DOCKER_IMAGE)
        self.assertIs(data["enable_gpu"], False)
        self.assertIs(data["enable_tpu"], False)
        self.assertIs(data["enable_internet"], False)
        self.assertIs(data["is_private"], True)
        self.assertEqual(data["model_sources"], [])
        self.assertIn(constants.WHEELHOUSE_DATASET, data["dataset_sources"])
        self.assertIn(constants.COMPETITION, data["competition_sources"])
        self.assertEqual(sorted(path.name for path in (self.root / "kernel").iterdir()),
                         ["kernel-metadata.json", "probe.py"])

    def test_wrong_docker_image_and_gpu_fail(self):
        path = self.root / "kernel" / "kernel-metadata.json"
        data = json.loads(path.read_text())
        data["docker_image"] = "gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"
        data["enable_gpu"] = True
        data["model_sources"] = ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"]
        path.write_text(json.dumps(data))
        errors = check_metadata.check_metadata(path)
        joined = " ".join(errors)
        self.assertIn("docker_image", joined)
        self.assertIn("enable_gpu", joined)
        self.assertIn("model_sources", joined)

    def test_invalid_json_fails(self):
        path = self.root / "bad.json"
        path.write_text("{not json")
        errors = check_metadata.check_metadata(path)
        self.assertTrue(any("valid JSON" in item for item in errors))


class AbiTests(unittest.TestCase):
    def test_env_hints_omit_tokens(self):
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "working"
        out.mkdir()
        env = {
            "KAGGLE_DOCKER_IMAGE": constants.STARTER_DOCKER_IMAGE,
            "KAGGLE_USER_SECRETS_TOKEN": "secret-value",
            "KAGGLE_DATA_PROXY_TOKEN": "proxy-value",
            "PATH": "/usr/bin",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            report = probe.run(
                out_dir=out,
                input_root=Path(tmp.name) / "input",
                version_info=(3, 13, 0),
                executable="/usr/bin/python",
                pip=lambda *_: None,
            )
        self.assertEqual(report["env_hints"].get("KAGGLE_DOCKER_IMAGE"), constants.STARTER_DOCKER_IMAGE)
        self.assertNotIn("KAGGLE_USER_SECRETS_TOKEN", report["env_hints"])
        self.assertNotIn("KAGGLE_DATA_PROXY_TOKEN", report["env_hints"])
        tmp.cleanup()

    def test_python_313_is_pin_failed_and_forbids_pip(self):
        status, allow = probe.evaluate_python((3, 13, 15))
        self.assertEqual(status, "pin_failed")
        self.assertFalse(allow)

    def test_python_312_allows_install(self):
        status, allow = probe.evaluate_python((3, 12, 0))
        self.assertIsNone(status)
        self.assertTrue(allow)

    def test_stage_skips_cutlass_and_rewrites_cu128(self):
        self.assertIsNone(probe.stage_wheel_name("nvidia_cutlass-1.0-py3-none-any.whl"))
        self.assertEqual(
            probe.stage_wheel_name("torch-2.8.0cu128-cp312-cp312-linux_x86_64.whl"),
            "torch-2.8.0+cu128-cp312-cp312-linux_x86_64.whl",
        )
        self.assertEqual(
            probe.stage_wheel_name("swegemma-0.2.11-py3-none-any.whl"),
            "swegemma-0.2.11-py3-none-any.whl",
        )

    def test_pin_failed_run_does_not_pip_install(self):
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
        self.assertFalse(report["installed"])
        self.assertEqual(calls, [])
        saved = json.loads((out / "result.json").read_text())
        self.assertEqual(saved["abi_status"], "pin_failed")
        self.assertEqual(saved["declared_docker_image"], constants.STARTER_DOCKER_IMAGE)
        tmp.cleanup()

    def test_python312_installs_staged_wheels_and_records_cpu_limits(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        out = root / "working"
        out.mkdir()
        wheelhouse = root / "input" / "datasets" / "metric" / "gemma-4-developer-agent-wheelhouse"
        data = root / "input" / "competitions" / "gemma-4-developer-agent"
        wheelhouse.mkdir(parents=True)
        data.mkdir(parents=True)
        (data / "tasks.jsonl").write_text("{}\n")
        tiny = {
            "swegemma-0.2.11-py3-none-any.whl": 8,
            "adk_submission-0.2.13-py3-none-any.whl": 8,
            "google_adk-1.36.1-py3-none-any.whl": 8,
            "adk_eval_core-0.1.0-py3-none-any.whl": 8,
        }
        for name, size in tiny.items():
            (wheelhouse / name).write_bytes(b"w" * size)
        (wheelhouse / "nvidia_cutlass-9.0-py3-none-any.whl").write_bytes(b"skip")
        (wheelhouse / "demo-1.0cu128-py3-none-any.whl").write_bytes(b"cuda")
        calls = []

        def fake_pip(executable, wheels):
            calls.append(list(wheels))
            return subprocess.CompletedProcess(args=["pip"], returncode=0, stdout="ok\n", stderr="")

        versions = {item.split("-")[0].replace("_", "-").lower(): "0" for item in tiny}
        versions.update({name: spec["version"] for name, spec in probe.OFFICIAL_PACKAGES.items()})

        def fake_version(name):
            if name in versions:
                return versions[name]
            raise probe.md.PackageNotFoundError(name)

        def fake_import(name):
            if name in ("vllm",):
                raise ModuleNotFoundError("vllm")
            module = type(sys)("m")
            module.__file__ = f"/tmp/{name}.py"
            return module

        with mock.patch.object(probe, "EXPECTED_WHEELS", tiny):
            with mock.patch.object(probe.md, "version", side_effect=fake_version):
                with mock.patch.object(probe.importlib, "import_module", side_effect=fake_import):
                    report = probe.run(
                        out_dir=out,
                        input_root=root / "input",
                        version_info=(3, 12, 11),
                        executable="/usr/bin/python3.12",
                        pip=fake_pip,
                        staging_dir=root / "staging",
                    )
        self.assertTrue(report["pip_attempted"])
        self.assertEqual(len(calls), 1)
        staged = [Path(item).name for item in calls[0]]
        self.assertNotIn("nvidia_cutlass-9.0-py3-none-any.whl", staged)
        self.assertIn("demo-1.0+cu128-py3-none-any.whl", staged)
        self.assertTrue(all(item["ok"] for item in report["imports"].values()))
        self.assertEqual(report["env_limits"]["vllm"]["ok"], False)
        self.assertIn("CPU environment limit", report["env_limits"]["vllm"]["note"])
        self.assertEqual(report["abi_status"], "ok_python312_v30_installed")
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
