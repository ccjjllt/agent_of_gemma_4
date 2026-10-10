"""Offline check: kernel-metadata.json is valid and pinned to the Starter 3.12 image."""
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
    KERNEL_ID,
    KERNEL_TITLE,
    STARTER_DOCKER_IMAGE,
    WHEELHOUSE_DATASET,
)

DEFAULT_METADATA = HERE / "kernel" / "kernel-metadata.json"

CT_MARKERS = (
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-31b-it-qat-q4_0-unquantized",
)


def check_metadata(path: Path | None = None) -> list[str]:
    metadata_path = Path(path) if path is not None else DEFAULT_METADATA
    errors: list[str] = []
    try:
        raw = metadata_path.read_text(encoding="utf-8")
        data = json.loads(raw)
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
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "docker_image": STARTER_DOCKER_IMAGE,
    }
    for key, value in expected.items():
        if key not in data:
            errors.append(f"missing field {key}")
        elif data[key] != value:
            errors.append(f"{key}={data[key]!r} != {value!r}")

    if data.get("docker_image") != STARTER_DOCKER_IMAGE:
        errors.append("docker_image is not the official Starter 3.12 sha")
    if data.get("enable_gpu") is not False:
        errors.append("enable_gpu must be boolean false")

    datasets = data.get("dataset_sources")
    if not isinstance(datasets, list) or WHEELHOUSE_DATASET not in datasets:
        errors.append(f"dataset_sources must include {WHEELHOUSE_DATASET}")
    competitions = data.get("competition_sources")
    if not isinstance(competitions, list) or COMPETITION not in competitions:
        errors.append(f"competition_sources must include {COMPETITION}")

    models = data.get("model_sources") or []
    if models:
        errors.append(f"model_sources must be empty (no CT weights); got {models}")
    joined = " ".join(str(item) for item in models)
    if any(marker.lower() in joined.lower() for marker in CT_MARKERS):
        errors.append("CT model source is mounted")

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
    print("enable_gpu=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
