"""CPU-only v30 inference-env pin. Does not load the 31B model. Internet off."""
from __future__ import annotations

import glob
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

DECLARED_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
REQUIRED_IMPORTS = ("swegemma", "adk_submission", "google.adk")
OFFICIAL_PACKAGES = {
    "adk-eval-core": {
        "version": "0.1.0",
        "sha256": "ca261146828a3d8ab0894b4766bc61a0905e3ef94f8eede17561a4a058404e1f",
        "wheel": "adk_eval_core-0.1.0-py3-none-any.whl",
    },
    "adk-submission": {
        "version": "0.2.13",
        "sha256": "69642a13145151040fb96b483170f96cc7cff1d509851d21d08bc7a0ad3ef371",
        "wheel": "adk_submission-0.2.13-py3-none-any.whl",
    },
    "google-adk": {
        "version": "1.36.1",
        "sha256": "1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0",
        "wheel": "google_adk-1.36.1-py3-none-any.whl",
    },
    "swegemma": {
        "version": "0.2.11",
        "sha256": "63f054a3d42eb6c582e2b42195f96b22f4b9ee69d6e42ae62cedeb4dd6cf5480",
        "wheel": "swegemma-0.2.11-py3-none-any.whl",
    },
}
EXPECTED_WHEELS = {
    "adk_eval_core-0.1.0-py3-none-any.whl": 90060,
    "adk_submission-0.2.13-py3-none-any.whl": 66528,
    "anthropic-1.4.0-py3-none-any.whl": 1300390,
    "apache_tvm_ffi-0.1.13.post3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl": 2955490,
    "astor-0.8.1-py2.py3-none-any.whl": 27488,
    "bitsandbytes-0.46.1-py3-none-manylinux_2_24_x86_64.whl": 72865511,
    "cbor2-6.1.4-cp312-cp312-manylinux_2_28_x86_64.whl": 461193,
    "compressed_tensors-0.15.0.1-py3-none-any.whl": 194260,
    "depyf-0.20.0-py3-none-any.whl": 39381,
    "diskcache-5.6.3-py3-none-any.whl": 45550,
    "flashinfer_cubin-0.6.6-py3-none-any.whl": 267661457,
    "flashinfer_python-0.6.6-py3-none-any.whl": 7817047,
    "gguf-0.19.0-py3-none-any.whl": 118475,
    "google_adk-1.36.1-py3-none-any.whl": 2877731,
    "google_genai-2.11.0-py3-none-any.whl": 984162,
    "ijson-3.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": 149848,
    "interegular-0.3.3-py37-none-any.whl": 23635,
    "llguidance-1.3.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": 3033133,
    "lm_format_enforcer-0.11.3-py3-none-any.whl": 45418,
    "loguru-0.7.3-py3-none-any.whl": 61595,
    "mistral_common-1.11.7-py3-none-any.whl": 6553583,
    "model_hosting_container_standards-0.1.16-py3-none-any.whl": 125666,
    "msgspec-0.21.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": 225025,
    "nvidia_cudnn_frontend-1.18.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": 2159245,
    "openai_harmony-0.0.8-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": 2953260,
    "opentelemetry_exporter_otlp-1.44.0-py3-none-any.whl": 6727,
    "partial_json_parser-0.2.1.1.post7-py3-none-any.whl": 10877,
    "prometheus_fastapi_instrumentator-8.1.0-py3-none-any.whl": 19649,
    "pybase64-1.5.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl": 94681,
    "pycountry-26.2.16-py3-none-any.whl": 8044600,
    "pydantic_extra_types-2.11.2-py3-none-any.whl": 82851,
    "quack_kernels-0.6.4-py3-none-any.whl": 728458,
    "opentelemetry_semantic_conventions_ai-0.5.1-py3-none-any.whl": 11250,
    "outlines_core-0.2.11-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": 2287677,
    "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": 506502,
    "setproctitle-1.3.7-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl": 32932,
    "swegemma-0.2.11-py3-none-any.whl": 119520,
    "tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": 3274982,
    "transformers-5.13.1-py3-none-any.whl": 11503977,
    "vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl": 433132506,
    "xgrammar-0.2.5.post1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": 48149933,
}


SAFE_ENV_KEYS = (
    "KAGGLE_DOCKER_IMAGE",
    "COLAB_IMAGE_TYPE",
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_CONTAINER_NAME",
)


def evaluate_python(version_info):
    if tuple(version_info[:2]) != (3, 12):
        return "pin_failed", False
    return None, True


def collect_env_hints(environ=None):
    env = os.environ if environ is None else environ
    hints = {}
    for key in SAFE_ENV_KEYS:
        if key not in env:
            continue
        lowered = key.lower()
        if any(part in lowered for part in ("token", "secret", "password", "credential")):
            continue
        hints[key] = env[key]
    return hints


def decide_install_status(*, pip_attempted, failed_wheels, official_ok, required_ok):
    if pip_attempted and not failed_wheels and official_ok and required_ok:
        return "ok_python312_v30_installed", True
    return "python312_install_incomplete", False


def stage_wheel_name(name: str):
    if "cutlass" in name.lower():
        return None
    if "cu128" in name and "+" not in name:
        return name.replace("cu128", "+cu128")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(input_root: Path, candidates, marker: str):
    for candidate in candidates:
        if candidate.is_dir() and (candidate / marker).exists():
            return candidate
    found = []
    if input_root.is_dir():
        for path in input_root.rglob(marker):
            found.append(path.parent)
    unique = sorted(set(found))
    if len(unique) == 1:
        return unique[0]
    return None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def try_import(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "file": getattr(module, "__file__", None)}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
        }


def pip_install(executable: str, wheels: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-index", *wheels],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def run(*, out_dir=None, input_root=None, version_info=None, executable=None, pip=None, staging_dir=None) -> dict:
    out = Path(out_dir) if out_dir is not None else Path("/kaggle/working")
    root = Path(input_root) if input_root is not None else Path("/kaggle/input")
    info = version_info if version_info is not None else sys.version_info
    exe = executable if executable is not None else sys.executable
    installer = pip if pip is not None else pip_install
    staging = Path(staging_dir) if staging_dir is not None else Path("/tmp") / "wheelhouse-p005-infer-env-cpu"
    started = time.monotonic()
    report = {
        "purpose": "CPU pin official Starter docker_image and install v30; no GPU, no CT load",
        "declared_docker_image": DECLARED_DOCKER_IMAGE,
        "docker_self_evidence": None,
        "python": sys.version,
        "python_version": sys.version.split()[0],
        "python_version_info": list(info[:3]),
        "executable": exe,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        "pip_attempted": False,
        "installed": False,
        "abi_status": None,
        "failed_wheels": [],
        "wheels": [],
        "installed_versions": {},
        "imports": {},
        "env_limits": {},
    }

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(out / "result.json", report)
        print(
            json.dumps(
                {
                    "abi_status": report["abi_status"],
                    "python_version": report["python_version"],
                    "pip_attempted": report["pip_attempted"],
                    "installed": report["installed"],
                }
            ),
            flush=True,
        )

    print(f"sys.version: {sys.version}", flush=True)
    print(f"sys.executable: {exe}", flush=True)
    print(f"platform: {platform.platform()}", flush=True)
    print(f"declared_docker_image: {DECLARED_DOCKER_IMAGE}", flush=True)

    report["env_hints"] = collect_env_hints()
    report["ct_model_mounted"] = any(
        (root / rel).exists()
        for rel in (
            "models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
            "gemma-4-31b-it-qat-w4a16-ct/2",
        )
    )

    abi, allow_pip = evaluate_python(info)
    if not allow_pip:
        report["abi_status"] = abi
        report["install_skipped_reason"] = (
            f"Python is {'.'.join(str(part) for part in info[:3])}; "
            "v30 has cp312-only wheels. No packages installed."
        )
        save()
        return report

    data_dir = resolve_input(
        root,
        [root / "competitions/gemma-4-developer-agent", root / "gemma-4-developer-agent"],
        "tasks.jsonl",
    )
    wheel_dir = resolve_input(
        root,
        [
            root / "datasets/metric/gemma-4-developer-agent-wheelhouse",
            root / "gemma-4-developer-agent-wheelhouse",
        ],
        "swegemma-0.2.11-py3-none-any.whl",
    )
    report["data_dir"] = str(data_dir) if data_dir else None
    report["wheelhouse_dir"] = str(wheel_dir) if wheel_dir else None
    if data_dir is not None:
        tasks = data_dir / "tasks.jsonl"
        report["tasks_jsonl"] = {
            "present": tasks.is_file(),
            "bytes": tasks.stat().st_size if tasks.is_file() else None,
        }

    if wheel_dir is None or not wheel_dir.is_dir():
        report["abi_status"] = "pin_ok_wheelhouse_missing"
        report["error"] = "wheelhouse input missing"
        save()
        return report

    for pattern in (
        "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
        "/usr/local/lib/python*/site-packages/*cutlass*.pth",
    ):
        for path in glob.glob(pattern):
            try:
                os.unlink(path)
            except OSError:
                pass

    source_wheels = [
        path for path in sorted(wheel_dir.glob("*.whl")) if "cutlass" not in path.name.lower()
    ]
    actual = {path.name: path.stat().st_size for path in source_wheels}
    report["wheelhouse_inventory"] = {
        "expected_count": len(EXPECTED_WHEELS),
        "actual_count": len(actual),
        "missing": sorted(set(EXPECTED_WHEELS) - set(actual)),
        "extra": sorted(set(actual) - set(EXPECTED_WHEELS)),
        "size_changed": [
            name
            for name in actual.keys() & EXPECTED_WHEELS.keys()
            if actual[name] != EXPECTED_WHEELS[name]
        ],
    }
    inventory_ok = actual == EXPECTED_WHEELS
    report["wheelhouse_matches_v30"] = inventory_ok

    wheels_meta = []
    for path in source_wheels:
        staged = stage_wheel_name(path.name)
        item = {
            "name": path.name,
            "staged_name": staged,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "distribution": path.name.split("-")[0].replace("_", "-").lower(),
        }
        official = next(
            (name for name, spec in OFFICIAL_PACKAGES.items() if spec["wheel"] == path.name),
            None,
        )
        if official is not None:
            item["official"] = official
            item["sha256_ok"] = item["sha256"] == OFFICIAL_PACKAGES[official]["sha256"]
        wheels_meta.append(item)
    report["wheels"] = wheels_meta

    if staging.exists():
        for child in staging.iterdir():
            child.unlink()
    else:
        staging.mkdir(parents=True)
    staged_paths = []
    for path in source_wheels:
        name = stage_wheel_name(path.name)
        if name is None:
            continue
        target = staging / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path)
        staged_paths.append(str(target))

    report["pip_attempted"] = True
    batch = installer(exe, staged_paths)
    report["pip_batch_returncode"] = batch.returncode
    report["pip_batch_tail"] = (batch.stdout or "")[-4000:]
    failed = []
    if batch.returncode != 0:
        per_wheel = []
        for wheel in staged_paths:
            one = installer(exe, [wheel])
            entry = {
                "wheel": Path(wheel).name,
                "returncode": one.returncode,
                "tail": (one.stdout or "")[-500:],
            }
            per_wheel.append(entry)
            if one.returncode != 0:
                failed.append(Path(wheel).name)
        report["pip_per_wheel"] = per_wheel
    report["failed_wheels"] = failed
    importlib.invalidate_caches()

    versions = {}
    for item in wheels_meta:
        dist = item["distribution"]
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    for spec_name in OFFICIAL_PACKAGES:
        try:
            versions[spec_name] = md.version(spec_name)
        except md.PackageNotFoundError:
            versions.setdefault(spec_name, None)
    report["installed_versions"] = versions

    imports = {name: try_import(name) for name in REQUIRED_IMPORTS}
    report["imports"] = imports
    required_ok = all(item.get("ok") for item in imports.values())
    official_ok = all(
        versions.get(name) == spec["version"] for name, spec in OFFICIAL_PACKAGES.items()
    )
    report["official_versions_ok"] = official_ok
    status, installed = decide_install_status(
        pip_attempted=True,
        failed_wheels=failed,
        official_ok=official_ok,
        required_ok=required_ok,
    )
    report["installed"] = installed
    report["abi_status"] = status
    if not inventory_ok:
        report["inventory_note"] = "wheelhouse inventory differed from recorded v30 files"
    if status != "ok_python312_v30_installed":
        report["error"] = (
            "partial v30 install: "
            f"pip_attempted=True failed_wheels={failed!r} "
            f"official_ok={official_ok} required_imports_ok={required_ok}"
        )
    # Persist before optional CUDA imports; vllm may abort the CPU process.
    save()

    env_limits = {}
    try:
        torch = importlib.import_module("torch")
        cuda_ok = bool(torch.cuda.is_available())
        env_limits["torch"] = {
            "ok": True,
            "version": torch.__version__,
            "cuda_available": cuda_ok,
            "device_count": int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else None,
            "note": None if cuda_ok else "CPU environment limit; not claiming GPU",
        }
    except Exception as exc:
        env_limits["torch"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
            "note": "CPU environment limit",
        }
    try:
        vllm = try_import("vllm")
    except Exception as exc:
        vllm = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:800]}
    if not vllm["ok"]:
        vllm["note"] = "CPU environment limit; not claiming GPU / vLLM"
    env_limits["vllm"] = vllm
    report["env_limits"] = env_limits
    save()
    return report


def main() -> int:
    try:
        report = run()
    except Exception:
        out = Path("/kaggle/working")
        payload = {
            "abi_status": "probe_exception",
            "error": traceback.format_exc(),
            "python": sys.version,
            "executable": sys.executable,
            "declared_docker_image": DECLARED_DOCKER_IMAGE,
            "pip_attempted": False,
        }
        write_json(out / "result.json", payload)
        print(traceback.format_exc(), flush=True)
        return 0
    # pin_failed is a valid negative result (exit 0). Incomplete 3.12 install still archives.
    return 0 if report.get("abi_status") else 1


if __name__ == "__main__":
    raise SystemExit(main())
