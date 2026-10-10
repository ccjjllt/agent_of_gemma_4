"""Offline check: kernel-metadata.json is valid, GPU, and pinned to the Starter 3.12 image."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from constants import (  # noqa: E402
    CODE_FILE,
    COMPETITION,
    CT_MODEL,
    DDP_PROBE_KERNEL,
    FORBIDDEN_DOCKER_IMAGE,
    KERNEL_ID,
    KERNEL_TITLE,
    STARTER_DOCKER_IMAGE,
    WHEELHOUSE_DATASET,
)

DEFAULT_METADATA = HERE / "kernel" / "kernel-metadata.json"


def check_metadata(path: Path | None = None) -> list[str]:
    metadata_path = Path(path) if path is not None else DEFAULT_METADATA
    errors: list[str] = []
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"missing metadata: {metadata_path}"]
    except json.JSONDecodeError as exc:
        return [f"metadata is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["metadata root must be a JSON object"]

    expected = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "docker_image": STARTER_DOCKER_IMAGE,
    }
    for key, value in expected.items():
        if key not in data:
            errors.append(f"missing field {key}")
        elif data[key] != value:
            errors.append(f"{key}={data[key]!r} != {value!r}")

    if data.get("docker_image") == FORBIDDEN_DOCKER_IMAGE:
        errors.append("docker_image is the default 3.13 image")
    if data.get("docker_image") != STARTER_DOCKER_IMAGE:
        errors.append("docker_image is not the official Starter 3.12 sha")
    if data.get("enable_gpu") is not True:
        errors.append("enable_gpu must be boolean true")
    if data.get("enable_internet") is not False:
        errors.append("enable_internet must be boolean false")
    if data.get("is_private") is not True:
        errors.append("is_private must be boolean true")

    datasets = data.get("dataset_sources")
    if not isinstance(datasets, list) or WHEELHOUSE_DATASET not in datasets:
        errors.append(f"dataset_sources must include {WHEELHOUSE_DATASET}")
    competitions = data.get("competition_sources")
    if not isinstance(competitions, list) or COMPETITION not in competitions:
        errors.append(f"competition_sources must include {COMPETITION}")
    models = data.get("model_sources")
    if not isinstance(models, list) or CT_MODEL not in models:
        errors.append(f"model_sources must include {CT_MODEL}")
    kernels = data.get("kernel_sources")
    if not isinstance(kernels, list) or DDP_PROBE_KERNEL not in kernels:
        errors.append(f"kernel_sources must include {DDP_PROBE_KERNEL}")

    pinning = data.get("docker_image_pinning_type")
    if pinning is not None and pinning != "original":
        errors.append(f"docker_image_pinning_type={pinning!r} would float off the pin")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_METADATA
    errors = check_metadata(path)
    if errors:
        print("FAIL")
        for item in errors:
            print(item)
        return 1
    print("OK")
    print(f"docker_image={STARTER_DOCKER_IMAGE}")
    print("enable_gpu=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
