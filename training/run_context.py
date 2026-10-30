"""CPU-only source binding for starting or resuming an unchanged training run.

This reads the mounted public index, model files and participant code in place.
It neither copies task assets nor imports Torch, Transformers or the runtime.
Hashes prove equality with the run's first binding, not upstream authenticity.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


_TRAINING_ROOT = Path(__file__).resolve().parent
_FORBIDDEN = re.compile(r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", re.I)
_IGNORED_DIRECTORIES = {".git", ".cache", "__pycache__"}
_PACKAGES = (
    "torch", "torchvision", "torchaudio", "torchao", "triton", "xformers",
    "unsloth", "unsloth-zoo", "transformers", "trl", "peft", "accelerate",
    "bitsandbytes", "cut-cross-entropy", "flash-attn", "safetensors", "numpy",
    "tokenizers", "sentencepiece", "huggingface-hub", "datasets", "jinja2",
    "swegemma", "adk-eval-core", "adk-submission", "google-adk", "litellm",
    "openai", "google-genai", "pydantic", "pydantic-core", "httpx", "httpcore",
    "aiohttp", "anyio", "PyYAML", "protobuf", "packaging",
)


def _public_path(path):
    path = Path(path).expanduser().absolute()
    if _FORBIDDEN.search(str(path)) or _FORBIDDEN.search(str(path.resolve(strict=True))):
        raise ValueError("Run binding cannot read heldout or reserved assets")
    return path


def _fingerprint(path, *, name=None):
    path = _public_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Source binding requires a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source binding requires a regular file: {path}")
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
        after = os.fstat(source.fileno())
    current = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    signature = lambda value: tuple(getattr(value, key) for key in fields)
    if signature(before) != signature(after) or signature(after) != signature(current):
        raise RuntimeError(f"Source changed while computing its fingerprint: {path}")
    return {"name": name or path.name, "size": before.st_size, "sha256": digest.hexdigest()}


def _files(root, predicate, *, ignored=()):
    """Walk only the explicitly supplied model/code directory, never data_root."""
    result = []
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = sorted(set(children) - _IGNORED_DIRECTORIES - set(ignored))
        for child in children:
            path = Path(directory) / child
            if path.is_symlink():
                raise ValueError(f"A binding cannot silently omit a linked source directory: {path}")
            _public_path(path)
        for name in sorted(names):
            path = Path(directory) / name
            if predicate(path):
                result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _inventory(root, predicate, *, ignored=()):
    paths = _files(root, predicate, ignored=ignored)
    records = [_fingerprint(path, name=path.relative_to(root).as_posix()) for path in paths]
    if paths != _files(root, predicate, ignored=ignored):
        raise RuntimeError(f"Source file inventory changed during binding: {root}")
    return records


def _model_binding(root, policy):
    if not (root / "config.json").is_file():
        raise ValueError("The local model must contain config.json")
    is_weight = lambda path: path.suffix in {".safetensors", ".bin"}
    weight_files = _inventory(root, is_weight)
    weights = {record["name"]: record for record in weight_files}
    if not weights or any(record["size"] <= 0 for record in weight_files):
        raise ValueError("The local model must contain nonempty actual weight files")
    indexes, referenced = [], set()
    for path in _files(root, lambda item: item.name.endswith((".safetensors.index.json", ".bin.index.json"))):
        record = _fingerprint(path, name=path.relative_to(root).as_posix())
        document = json.loads(path.read_text(encoding="utf-8"))
        if _fingerprint(path, name=record["name"]) != record:
            raise RuntimeError("Model shard index changed during binding")
        mapping = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing nonempty weight_map in {record['name']}")
        shards = set()
        for tensor, value in mapping.items():
            if not isinstance(tensor, str) or not tensor or not isinstance(value, str) or not value:
                raise ValueError(f"Malformed weight_map in {record['name']}")
            relative = PurePosixPath(value)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in value
                    or relative.suffix not in {".safetensors", ".bin"}):
                raise ValueError(f"Unsafe model shard path in {record['name']}")
            if not path.name.endswith(relative.suffix + ".index.json"):
                raise ValueError(f"Shard format does not match its index: {record['name']}")
            name = (path.parent / Path(*relative.parts)).relative_to(root).as_posix()
            if name not in weights:
                raise ValueError(f"Model shard referenced by {record['name']} is missing: {name}")
            shards.add(name)
        indexes.append({**record, "shards": sorted(shards)})
        referenced.update(shards)
    groups = {}
    for name in weights:
        match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.(safetensors|bin)", name)
        if match:
            prefix, position, count, extension = match.groups()
            position, count = int(position), int(count)
            if not 1 <= position <= count <= 100000 or name not in referenced:
                raise ValueError(f"Sharded model needs a complete, corresponding weight index: {name}")
            groups.setdefault((prefix, count, extension), set()).add(position)
    if any(len(positions) != count for (_, count, _), positions in groups.items()):
        raise ValueError("A numbered model shard set is incomplete")
    monolithic = any(re.fullmatch(r"(?:model(?:\.[^/]+)?\.safetensors|pytorch_model(?:\.[^/]+)?\.bin)", name)
                     for name in weights)
    if not indexes and not monolithic:
        raise ValueError("No loadable model.safetensors/pytorch_model.bin or weight index was found")
    asset_suffixes = {".json", ".jinja", ".jinja2", ".model", ".vocab", ".tiktoken", ".py"}
    assets = _inventory(root, lambda path: path.suffix in asset_suffixes or path.name in {"vocab.txt", "merges.txt"})
    if any(record["name"] == "config.json" and record["size"] == 0 for record in assets):
        raise ValueError("Model config.json is empty")
    response_template = None
    if policy.get("response_template_path"):
        path = _public_path(policy["response_template_path"])
        response_template = {"path": str(path), **_fingerprint(path)}
    return {"root": str(root), "weights": weight_files, "indexes": indexes,
            "assets": assets, "response_template": response_template}


def _environment():
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # CUDA component wheels affect kernels even when the top-level Torch
    # version is unchanged. Reading metadata does not initialize CUDA.
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name.startswith(("nvidia-", "cuda-")):
            packages[name] = distribution.version
    return {"python": sys.version, "implementation": sys.implementation.name,
            "executable": sys.executable, "platform": sys.platform,
            "packages": dict(sorted(packages.items()))}


def _runtime_environment(python):
    # Keep the venv entry point: resolving its executable symlink would execute
    # the base interpreter and fingerprint the wrong installed environment.
    python = Path(python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f"The separately prepared Python interpreter is missing: {python}")
    script = (
        "import importlib.metadata as m,json,sys\n"
        f"names={_PACKAGES!r}\n"
        "packages={}\n"
        "for name in names:\n"
        " try: packages[name]=m.version(name)\n"
        " except m.PackageNotFoundError: packages[name]=None\n"
        "for d in m.distributions():\n"
        " name=d.metadata.get('Name','').lower().replace('_','-')\n"
        " if name.startswith(('nvidia-','cuda-')): packages[name]=d.version\n"
        "print(json.dumps({'python':sys.version,'implementation':sys.implementation.name,"
        "'executable':sys.executable,'platform':sys.platform,'packages':dict(sorted(packages.items()))}))\n"
    )
    completed = subprocess.run([str(python), "-I", "-c", script], check=True,
                               capture_output=True, text=True, timeout=30)
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or not isinstance(result.get("packages"), dict):
        raise ValueError("Official runtime returned invalid package metadata")
    return result


def _runtime_timing_binding(python):
    """Check the existing timing hooks before a costly GPU model load."""
    script = (
        "import contextlib,json,os,sys\n"
        "os.environ.update(CUDA_VISIBLE_DEVICES='',LITELLM_LOCAL_MODEL_COST_MAP='True',"
        "LITELLM_TELEMETRY='False',OTEL_SDK_DISABLED='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')\n"
        f"sys.path.insert(0,{str(_TRAINING_ROOT)!r})\n"
        "with contextlib.redirect_stdout(sys.stderr):\n"
        " from session_clock import validate_official_timing_sources\n"
        " evidence=validate_official_timing_sources()\n"
        "print(json.dumps(evidence))\n"
    )
    completed = subprocess.run([str(python), "-I", "-c", script], check=True,
                               capture_output=True, text=True, timeout=30)
    result = json.loads(completed.stdout)
    if (not isinstance(result, dict) or result.get("protocol") != "gemma.training_session_clock.v1"
            or not result.get("source_sha256")):
        raise ValueError("The official runtime did not confirm its session timing implementation")
    return result


def source_binding(cfg, *, data_root, model_path, submission, runtime_python, tasks):
    """Return JSON-safe inputs whose equality is required when resuming.

    Call after applying all CLI overrides and normalizing the effective config.
    The task argument must be the metadata-only output of load_tasks; only its
    id/repo/base_commit fields are retained, in sampler order. No judge/task
    asset directory is enumerated. Missing packages are recorded as null, so
    changing an optional dependency also changes this binding.
    """
    config = json.loads(json.dumps(cfg, sort_keys=True, allow_nan=False))
    if not isinstance(config, dict):
        raise TypeError("Effective run configuration must be a JSON-safe mapping")
    data_root = _public_path(data_root).resolve(strict=True)
    model_path = _public_path(model_path).resolve(strict=True)
    submission = _public_path(submission).resolve(strict=True)
    runtime_python = Path(runtime_python).expanduser().absolute()
    if not all(path.is_dir() for path in (data_root, model_path, submission)):
        raise ValueError("Data, model and submission paths must be existing directories")
    if submission == data_root or (submission / "tasks.jsonl").exists():
        raise ValueError("Submission must be the participant code directory, not a task asset directory")
    if not runtime_python.is_file():
        raise ValueError("The separately prepared official runtime interpreter is missing")
    timing = (_runtime_timing_binding(runtime_python)
              if config.get("runtime", {}).get("session_clock") in {"queue_excluded", "wall"} else None)
    configured_model = config.get("policy", {}).get("model_path")
    if configured_model and Path(configured_model).expanduser().resolve(strict=True) != model_path:
        raise ValueError("The effective policy model_path differs from the model being fingerprinted")
    selected = []
    for task in tasks:
        if not isinstance(task, dict) or any(not isinstance(task.get(key), str) or not task[key]
                                             for key in ("id", "repo", "base_commit")):
            raise ValueError("Selected tasks need id/repo/base_commit metadata")
        selected.append({key: task[key] for key in ("id", "repo", "base_commit")})
    if not selected or len({task["id"] for task in selected}) != len(selected):
        raise ValueError("Selected public task metadata must be nonempty and unique")
    public_tasks = data_root / "tasks.jsonl"
    binding = {
        "schema_version": 1, "config": config, "selected_tasks": selected,
        "session_timing_source": timing,
        "public_tasks": {"path": str(public_tasks), **_fingerprint(public_tasks)},
        "model": _model_binding(model_path, config.get("policy", {})),
        "code": {
            "training": {"root": str(_TRAINING_ROOT),
                         "files": _inventory(_TRAINING_ROOT, lambda path: path.suffix == ".py")},
            "submission": {"root": str(submission),
                           "files": _inventory(submission, lambda path: path.suffix in
                                               {".py", ".yaml", ".yml", ".json", ".jinja", ".jinja2"},
                                               ignored={"adapters"})},
        },
        "environments": {"learner": _environment(), "runtime": _runtime_environment(runtime_python)},
    }
    runtime = config.get("runtime", {})
    if runtime.get("sandbox") == "subprocess":
        binding["environments"]["sandbox_baseline"] = _runtime_environment(
            runtime.get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"))
    # Validate the whole contract before a caller writes or compares it.
    json.dumps(binding, sort_keys=True, allow_nan=False)
    return binding
