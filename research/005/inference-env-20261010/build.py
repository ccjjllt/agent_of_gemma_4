"""Write the isolated Kaggle push directory (probe.py + kernel-metadata.json only)."""
from __future__ import annotations

import argparse
import ast
import json
import shutil
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
from probe import EXPECTED_WHEELS  # noqa: E402

REPO = HERE.parents[2]
WHEEL_FILES = REPO / "research" / "005" / "official-20261010" / "wheel-files.json"


def wheelhouse_inventory():
    payload = json.loads(WHEEL_FILES.read_text(encoding="utf-8"))
    return {
        item["name"]: int(item["total_bytes"])
        for item in payload["dataset_files"]
        if item["name"].endswith(".whl") and "cutlass" not in item["name"].lower()
    }


def metadata():
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [WHEELHOUSE_DATASET],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image": STARTER_DOCKER_IMAGE,
        "docker_image_pinning_type": "original",
    }


def build(destination: Path | None = None) -> dict:
    dest = Path(destination) if destination is not None else HERE / "kernel"
    inventory = wheelhouse_inventory()
    if inventory != EXPECTED_WHEELS:
        missing = sorted(set(inventory) - set(EXPECTED_WHEELS))
        extra = sorted(set(EXPECTED_WHEELS) - set(inventory))
        raise RuntimeError(f"probe EXPECTED_WHEELS drifted from v30 inventory: missing={missing} extra={extra}")
    source = HERE / "probe.py"
    ast.parse(source.read_text(encoding="utf-8"), filename="probe.py")
    if dest.exists():
        for child in dest.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest / CODE_FILE)
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    names = sorted(path.name for path in dest.iterdir())
    if names != ["kernel-metadata.json", CODE_FILE]:
        raise RuntimeError(f"isolated kernel dir must contain only script+metadata, got {names}")
    return {
        "built": str(dest),
        "kernel_id": KERNEL_ID,
        "docker_image": STARTER_DOCKER_IMAGE,
        "enable_gpu": False,
        "wheel_count": len(EXPECTED_WHEELS),
        "files": names,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "kernel")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
