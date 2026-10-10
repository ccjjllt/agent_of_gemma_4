"""Official CT + diagnostic LoRA compatibility probe. Not a repair-SFT result."""
from __future__ import annotations

import glob
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

DECLARED_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
TARGET_MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"
ADAPTER_NAME = "diag_ddp_step4"
EXPECTED_ADAPTER_CONFIG_SHA256 = "6e25de4c3846d89643f85b5607fffe9a436eb3668bdf148d806ce5ee16298b6a"
EXPECTED_ADAPTER_WEIGHTS_SHA256 = "a870230ea9ac96abf90e998203c8b9c423e494cccb7f0faffda33b319fe78cf0"
EXPECTED_ADAPTER_WEIGHTS_BYTES = 212261592
T0_SHA256 = {
    "agent.yaml": "ae12f5f5911c855970b075a72fa2571726efe86f90130b24bd139ce47906e6d0",
    "configs/sampling.yaml": "5a6acb429b95163d65f68ff483ebe83031b025eb818840b39f6d5b718f09645e",
    "eval_config.yaml": "2a0205536c9d418d8502be4b13b3c063e1c0800d591e38c72d651a092de2b147",
    "prompts/system.md": "6c2e8f4c90ec07fa973be9cefa9c04176f99acdef3ab29a985302e04203a17b5",
}
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
T0_TOOLS = (
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
)
GET_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_status",
        "description": "Return the remaining task budget and patch status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}
T0_BUNDLE = {
    "agent.yaml": (
        "name: p002_t0_frozen_base\n"
        "model: gemma-4-31b-it-qat-w4a16-ct\n"
        "instruction: !include prompts/system.md\n"
        "tools:\n"
        "  - run_command\n"
        "  - read_file\n"
        "  - edit_file\n"
        "  - write_file\n"
        "  - get_status\n"
        "  - submit_patch\n"
        "generate_content_config: !include configs/sampling.yaml\n"
    ),
    "configs/sampling.yaml": (
        "# P002-T0 frozen base sampling (inherited from P003-A0, 2026-09-30).\n"
        "# max_output_tokens 16384 is the pre-capacity-contrast value.\n"
        "# thinking_budget numeric value is NOT a proven token limit: current chain maps\n"
        "# any budget>0 + include_thoughts:true to enable_thinking:true only.\n"
        "# Do not run 4096-vs-1024 as a budget ablation.\n"
        "temperature: 0.2\n"
        "top_p: 0.95\n"
        "max_output_tokens: 16384\n"
        "thinking_config:\n"
        "  thinking_budget: 4096\n"
        "  include_thoughts: true\n"
    ),
    "eval_config.yaml": (
        "# P002-T0 frozen eval budgets (same as P003 A0/B/C; 001 dev-start).\n"
        "# timeout_seconds limits BOTH agent commands and final verification in swegemma 0.2.7.\n"
        "# Any change must apply to all arms identically and is then a new research condition.\n"
        "evaluation:\n"
        "  timeout_seconds: 60\n"
        "  max_tool_calls: 50\n"
        "  max_time_minutes: 5\n"
        "  max_turns: 80\n"
    ),
    "prompts/system.md": (
        "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\n"
        "\n"
        "Initial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\n"
        "\n"
        "Only call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\n"
        "\n"
        "## Flow: locate -> read -> edit -> check -> submit\n"
        "\n"
        "1. Locate\n"
        "- Extract file/function/class/error strings from the problem statement first.\n"
        "- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).\n"
        "- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.\n"
        "\n"
        "2. Read\n"
        "- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.\n"
        "- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\n"
        "- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\n"
        "\n"
        "3. Edit\n"
        "- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\n"
        "- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\n"
        "- Match required error strings, exception types, status codes, and API signatures exactly.\n"
        "- On `FileEditError: old_string not found`: stop repeating. Re-read the exact file range, compare whitespace/escaping/line range byte-for-byte, and only then retry once with corrected text. Do not blindly retry identical parameters.\n"
        "\n"
        "4. Check\n"
        "- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\n"
        "- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\n"
        "- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\n"
        "- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\n"
        "\n"
        "5. Budget and submit\n"
        "- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\n"
        "- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\n"
        "- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\n"
    ),
}


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


def stage_wheel_name(name: str):
    if "cutlass" in name.lower():
        return None
    if "cu128" in name and "+" not in name:
        return name.replace("cu128", "+cu128")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_t0_bundle(bundle=None):
    files = T0_BUNDLE if bundle is None else bundle
    mismatches = {}
    for relative, expected in T0_SHA256.items():
        actual = sha256_text(files[relative])
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    return mismatches


def inject_adapter(agent_yaml: str, name: str) -> str:
    if any(line.startswith("adapter:") for line in agent_yaml.splitlines()):
        raise ValueError("T0 freeze already has adapter")
    needle = "model: gemma-4-31b-it-qat-w4a16-ct\n"
    if needle not in agent_yaml:
        raise ValueError("T0 model line missing")
    return agent_yaml.replace(needle, needle + f"adapter: {name}\n", 1)


def write_t0_agent(agent_dir: Path, adapter_name: str, bundle=None) -> dict:
    files = dict(T0_BUNDLE if bundle is None else bundle)
    mismatches = verify_t0_bundle(files)
    if mismatches:
        raise ValueError(f"T0 freeze hash mismatch: {mismatches}")
    files["agent.yaml"] = inject_adapter(files["agent.yaml"], adapter_name)
    agent_dir = Path(agent_dir)
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    for relative, content in files.items():
        path = agent_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    tools = [
        line.strip()[2:]
        for line in files["agent.yaml"].splitlines()
        if line.startswith("  - ")
    ]
    if tools != list(T0_TOOLS):
        raise ValueError(f"T0 tools drifted: {tools}")
    if f"adapter: {adapter_name}" not in files["agent.yaml"]:
        raise ValueError("adapter line missing after inject")
    if "skills:" in files["agent.yaml"]:
        raise ValueError("T0 copy must not declare skills")
    return {
        "agent_dir": str(agent_dir),
        "adapter_name": adapter_name,
        "freeze_sha256": dict(T0_SHA256),
        "copied_agent_yaml_sha256": sha256_text(files["agent.yaml"]),
        "tools": tools,
    }


def find_diagnostic_adapter(
    input_root: Path,
    expected_config_sha=EXPECTED_ADAPTER_CONFIG_SHA256,
    expected_weight_sha=EXPECTED_ADAPTER_WEIGHTS_SHA256,
    expected_weight_bytes=EXPECTED_ADAPTER_WEIGHTS_BYTES,
):
    root = Path(input_root)
    candidates = []
    matched = None
    if not root.is_dir():
        return None, candidates
    for weights in root.rglob("adapter_model.safetensors"):
        config = weights.parent / "adapter_config.json"
        if not config.is_file():
            continue
        size = weights.stat().st_size
        item = {
            "dir": str(weights.parent),
            "config_bytes": config.stat().st_size,
            "weight_bytes": size,
            "config_sha256": sha256_file(config),
        }
        if size == expected_weight_bytes:
            item["weight_sha256"] = sha256_file(weights)
        else:
            item["weight_sha256"] = None
            item["skipped_weight_hash"] = "size_mismatch"
        candidates.append(item)
        if (
            item["config_sha256"] == expected_config_sha
            and item.get("weight_sha256") == expected_weight_sha
            and size == expected_weight_bytes
        ):
            matched = weights.parent
            break
    return matched, candidates


def adapter_summary(config_path: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    targets = cfg.get("target_modules") or []
    return {
        "peft_type": cfg.get("peft_type"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
        "target_module_count": len(targets) if isinstance(targets, list) else None,
        "bias": cfg.get("bias"),
        "task_type": cfg.get("task_type"),
    }


def adapter_manifest_names(adapters) -> list:
    inner = getattr(adapters, "adapters", None)
    if isinstance(inner, dict):
        return sorted(str(key) for key in inner)
    if isinstance(inner, (list, tuple)):
        names = []
        for item in inner:
            names.append(str(getattr(item, "name", item)))
        return names
    if adapters is None:
        return []
    dumped = getattr(adapters, "model_dump", None)
    if callable(dumped):
        payload = dumped()
        if isinstance(payload, dict) and "adapters" in payload:
            inner = payload["adapters"]
            if isinstance(inner, dict):
                return sorted(str(key) for key in inner)
    return []


def extract_tool_names(payload) -> list:
    names = []
    if not isinstance(payload, dict):
        return names
    choices = payload.get("choices") or []
    for choice in choices:
        message = (choice or {}).get("message") or {}
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
            elif isinstance(call, dict) and call.get("name"):
                names.append(call["name"])
        content = message.get("content")
        if isinstance(content, str):
            for marker in ("get_status", "read_file"):
                if f"call {marker}" in content.lower() or f'"name": "{marker}"' in content:
                    if marker not in names:
                        names.append(f"text_mention:{marker}")
    return names


def valid_get_status_call(payload) -> bool:
    """Check the one registered, forced tool call and its empty-object schema."""
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return False
    call = calls[0]
    fn = call.get("function")
    if call.get("type") != "function" or not isinstance(fn, dict) or fn.get("name") != "get_status":
        return False
    arguments = fn.get("arguments")
    if not isinstance(arguments, str):
        return False
    try:
        return json.loads(arguments) == {}
    except (TypeError, ValueError):
        return False


def classify(report: dict) -> str:
    if report.get("repair_success_claimed"):
        raise ValueError("must not claim repair success")
    files_ok = bool(report.get("adapter_files_ok"))
    in_manifest = bool(report.get("adapter_in_manifest"))
    in_vllm = bool(report.get("adapter_in_vllm"))
    started = bool(report.get("vllm_started"))
    tool_ok = bool(report.get("tool_call_ok"))
    if not files_ok or not in_manifest:
        return "adapter_not_loaded"
    if started and not in_vllm:
        return "adapter_not_loaded"
    if started and in_vllm and tool_ok:
        return "tool_call_ok_no_repair_claim"
    return "inference_error"


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def try_import(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "file": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:800]}


def pip_install(executable: str, wheels: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-index", *wheels],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def resolve_input(input_root: Path, candidates, marker: str, marker_substr=None):
    for candidate in candidates:
        if candidate.is_dir() and (candidate / marker).exists():
            return candidate
    found = []
    if input_root.is_dir():
        for path in input_root.rglob(marker):
            if marker_substr is None or marker_substr in str(path):
                found.append(path.parent)
    unique = sorted(set(found))
    if len(unique) == 1:
        return unique[0]
    return None


def http_json(url: str, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("Authorization", "Bearer EMPTY")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"_raw": raw[:4000]}
            return {"ok": True, "status": getattr(resp, "status", None), "body": parsed}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")[:4000]
        return {"ok": False, "status": exc.code, "error": text}
    except Exception as exc:
        return {"ok": False, "status": None, "error_type": type(exc).__name__, "error": str(exc)[:800]}


def v1_urls(base_url: str) -> list[str]:
    base = str(base_url).rstrip("/")
    if base.endswith("/v1"):
        return [base]
    return [base + "/v1", base]


def log_excerpt(path, needles=("lora", "adapter", ADAPTER_NAME), limit=80):
    if path is None:
        return []
    log_path = Path(path)
    if not log_path.is_file():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"log-read-failed: {exc}"]
    lowered = tuple(n.lower() for n in needles)
    hits = [line for line in lines if any(token in line.lower() for token in lowered)]
    return hits[-limit:]


def copy_adapter(src: Path, dest: Path) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    config_src = src / "adapter_config.json"
    weight_src = src / "adapter_model.safetensors"
    config_dst = dest / "adapter_config.json"
    weight_dst = dest / "adapter_model.safetensors"
    shutil.copy2(config_src, config_dst)
    shutil.copy2(weight_src, weight_dst)
    fingerprints = {
        "config_sha256": sha256_file(config_dst),
        "config_bytes": config_dst.stat().st_size,
        "weight_sha256": sha256_file(weight_dst),
        "weight_bytes": weight_dst.stat().st_size,
        "dest": str(dest),
        "src": str(src),
    }
    ok = (
        fingerprints["config_sha256"] == EXPECTED_ADAPTER_CONFIG_SHA256
        and fingerprints["weight_sha256"] == EXPECTED_ADAPTER_WEIGHTS_SHA256
        and fingerprints["weight_bytes"] == EXPECTED_ADAPTER_WEIGHTS_BYTES
    )
    fingerprints["ok"] = ok
    return fingerprints


def decide_install_status(*, pip_attempted, failed_wheels, official_ok, required_ok,
                          inventory_ok, hashes_ok):
    if (pip_attempted and not failed_wheels and official_ok and required_ok
            and inventory_ok and hashes_ok):
        return "ok_python312_v30_installed", True
    return "python312_install_incomplete", False


def run(*, out_dir=None, input_root=None, version_info=None, executable=None, pip=None,
        staging_dir=None, agent_dir=None, start_server=None, http=None) -> dict:
    out = Path(out_dir) if out_dir is not None else Path("/kaggle/working")
    root = Path(input_root) if input_root is not None else Path("/kaggle/input")
    info = version_info if version_info is not None else sys.version_info
    exe = executable if executable is not None else sys.executable
    installer = pip if pip is not None else pip_install
    staging = Path(staging_dir) if staging_dir is not None else Path("/tmp") / "wheelhouse-p005-ct-adapter-probe"
    agent_root = Path(agent_dir) if agent_dir is not None else Path("/tmp") / "p005-ct-adapter-agent"
    started_at = time.monotonic()
    report = {
        "purpose": "Load diagnostic DDP adapter on official CT + v30; one tool call. Not repair SFT.",
        "not_repair_sft": True,
        "repair_success_claimed": False,
        "classification": None,
        "declared_docker_image": DECLARED_DOCKER_IMAGE,
        "adapter_name": ADAPTER_NAME,
        "python": sys.version,
        "python_version": sys.version.split()[0],
        "python_version_info": list(info[:3]),
        "executable": exe,
        "platform": platform.platform(),
        "pip_attempted": False,
        "installed": False,
        "abi_status": None,
        "adapter_files_ok": False,
        "adapter_in_manifest": False,
        "adapter_in_vllm": False,
        "vllm_started": False,
        "tool_call_ok": False,
        "scored_panel_run": False,
        "smoke_task_run": False,
    }

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        if report.get("classification") is None:
            try:
                report["classification"] = classify(report)
            except Exception as exc:
                report["classification"] = "inference_error"
                report["classify_error"] = str(exc)[:400]
        write_json(out / "result.json", report)
        write_json(out / "run_manifest.json", {
            "purpose": report["purpose"],
            "not_repair_sft": True,
            "repair_success_claimed": False,
            "classification": report.get("classification"),
            "adapter_name": ADAPTER_NAME,
            "adapter_fingerprints": report.get("adapter_fingerprints"),
            "abi_status": report.get("abi_status"),
            "declared_docker_image": DECLARED_DOCKER_IMAGE,
            "vllm_started": report.get("vllm_started"),
            "tool_call_ok": report.get("tool_call_ok"),
            "scored_panel_run": False,
        })
        print(
            json.dumps(
                {
                    "classification": report.get("classification"),
                    "abi_status": report.get("abi_status"),
                    "adapter_files_ok": report.get("adapter_files_ok"),
                    "vllm_started": report.get("vllm_started"),
                    "tool_call_ok": report.get("tool_call_ok"),
                    "not_repair_sft": True,
                }
            ),
            flush=True,
        )

    print(f"sys.version: {sys.version}", flush=True)
    print(f"declared_docker_image: {DECLARED_DOCKER_IMAGE}", flush=True)
    report["env_hints"] = collect_env_hints()
    mismatches = verify_t0_bundle()
    report["t0_bundle_ok"] = not mismatches
    if mismatches:
        report["t0_bundle_mismatches"] = mismatches
        report["classification"] = "inference_error"
        report["error"] = "embedded T0 freeze does not match expected hashes"
        save()
        return report

    abi, allow_pip = evaluate_python(info)
    if not allow_pip:
        report["abi_status"] = abi
        report["install_skipped_reason"] = (
            f"Python is {'.'.join(str(part) for part in info[:3])}; "
            "v30 has cp312-only wheels. No packages installed."
        )
        report["classification"] = "inference_error"
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
    model_dir = resolve_input(
        root,
        [
            root / "models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
            root / "gemma-4-31b-it-qat-w4a16-ct/2",
        ],
        "tokenizer.json",
        "gemma-4-31b",
    )
    report["data_dir"] = str(data_dir) if data_dir else None
    report["wheelhouse_dir"] = str(wheel_dir) if wheel_dir else None
    report["model_dir"] = str(model_dir) if model_dir else None

    src_adapter, candidates = find_diagnostic_adapter(root)
    report["adapter_search"] = {
        "matched": str(src_adapter) if src_adapter is not None else None,
        "candidate_count": len(candidates),
        "candidates": candidates[:12],
    }
    if src_adapter is None:
        report["adapter_files_ok"] = False
        report["classification"] = "adapter_not_loaded"
        report["error"] = "diagnostic adapter fingerprints not found under /kaggle/input"
        save()
        return report

    if wheel_dir is None or not wheel_dir.is_dir():
        report["abi_status"] = "pin_ok_wheelhouse_missing"
        report["classification"] = "inference_error"
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
    report["wheelhouse_matches_v30"] = actual == EXPECTED_WHEELS
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
    official_hashes = {
        item["official"]: item["sha256_ok"]
        for item in wheels_meta if "official" in item
    }
    report["official_hashes_ok"] = official_hashes == {
        name: True for name in OFFICIAL_PACKAGES
    }

    if staging.exists():
        for child in staging.iterdir():
            if child.is_symlink() or child.is_file():
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
        failed = [Path(path).name for path in staged_paths]
    report["failed_wheels"] = failed
    importlib.invalidate_caches()

    versions = {}
    for spec_name in OFFICIAL_PACKAGES:
        try:
            versions[spec_name] = md.version(spec_name)
        except md.PackageNotFoundError:
            versions[spec_name] = None
    report["installed_versions"] = versions
    imports = {name: try_import(name) for name in REQUIRED_IMPORTS}
    report["imports"] = imports
    required_ok = all(item.get("ok") for item in imports.values())
    official_ok = all(versions.get(name) == spec["version"] for name, spec in OFFICIAL_PACKAGES.items())
    report["official_versions_ok"] = official_ok
    status, installed = decide_install_status(
        pip_attempted=True,
        failed_wheels=failed,
        official_ok=official_ok,
        required_ok=required_ok,
        inventory_ok=report["wheelhouse_matches_v30"],
        hashes_ok=report["official_hashes_ok"],
    )
    report["installed"] = installed
    report["abi_status"] = status
    if status != "ok_python312_v30_installed":
        report["classification"] = "inference_error"
        report["error"] = f"v30 install incomplete: {status}"
        save()
        return report

    try:
        layout = write_t0_agent(agent_root, ADAPTER_NAME)
        fingerprints = copy_adapter(src_adapter, agent_root / "adapters" / ADAPTER_NAME)
    except Exception as exc:
        report["adapter_files_ok"] = False
        report["classification"] = "adapter_not_loaded"
        report["error"] = f"adapter layout failed: {type(exc).__name__}: {exc}"[:800]
        save()
        return report
    report["agent_layout"] = layout
    report["adapter_fingerprints"] = fingerprints
    report["adapter_config_summary"] = adapter_summary(Path(fingerprints["dest"]) / "adapter_config.json")
    report["adapter_files_ok"] = bool(fingerprints.get("ok"))
    if not report["adapter_files_ok"]:
        report["classification"] = "adapter_not_loaded"
        report["error"] = "copied adapter fingerprints mismatch expected diagnostic adapter"
        save()
        return report

    os.environ.update({
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TRANSFORMERS_NO_TF": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    try:
        import litellm
        import torch
        from adk_submission import (
            VllmConfig,
            VllmServer,
            discover_adapters,
            discover_declared_models,
        )
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    except Exception as exc:
        report["classification"] = "inference_error"
        report["error"] = f"import after install failed: {type(exc).__name__}: {exc}"[:800]
        save()
        return report

    litellm.drop_params = True
    cuda_ok = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_ok else 0
    report["cuda"] = {
        "available": cuda_ok,
        "device_count": gpu_count,
        "torch": torch.__version__,
        "bf16": bool(cuda_ok and torch.cuda.is_bf16_supported()),
    }
    if not cuda_ok or gpu_count < 1:
        report["classification"] = "inference_error"
        report["error"] = "CUDA unavailable after v30 install; not loading CT"
        save()
        return report
    if model_dir is None:
        report["classification"] = "inference_error"
        report["error"] = "CT model v2 not mounted"
        save()
        return report

    try:
        declared_models = sorted(discover_declared_models(agent_root))
        adapters = discover_adapters(str(agent_root), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    except Exception as exc:
        report["classification"] = "adapter_not_loaded"
        report["error"] = f"discover_adapters failed: {type(exc).__name__}: {exc}"[:800]
        save()
        return report
    names = adapter_manifest_names(adapters)
    report["declared_models"] = declared_models
    if TARGET_MODEL_NAME not in declared_models:
        report["classification"] = "inference_error"
        report["error"] = f"declared models {declared_models} missing {TARGET_MODEL_NAME}"
        save()
        return report
    declared_model = TARGET_MODEL_NAME
    report["declared_model"] = declared_model
    report["adapter_manifest_type"] = type(adapters).__name__
    report["adapter_manifest_names"] = names
    report["n_adapters"] = len(names)
    report["adapter_in_manifest"] = ADAPTER_NAME in names
    if not report["adapter_in_manifest"]:
        report["classification"] = "adapter_not_loaded"
        report["error"] = f"adapter {ADAPTER_NAME!r} not in discover_adapters names {names}"
        save()
        return report

    tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
    report["tp_size"] = tp_size
    vllm_cfg = VllmConfig(
        model=str(model_dir),
        port=8000,
        host="127.0.0.1",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=32768,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto",
        gpu_memory_utilization=0.90,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=tp_size,
        startup_timeout=60 * 15,
    )
    server = VllmServer(vllm_cfg, adapter_manifest=adapters)
    t_start = time.monotonic()
    starter = start_server if start_server is not None else (lambda inst: inst.start())
    try:
        starter(server)
        report["vllm_started"] = True
        report["vllm_startup_seconds"] = round(time.monotonic() - t_start, 3)
        report["vllm_base_url"] = str(getattr(server, "base_url", ""))
        report["vllm_log_path"] = str(getattr(server, "log_path", "") or "")
    except Exception as exc:
        report["vllm_started"] = False
        report["vllm_startup_seconds"] = round(time.monotonic() - t_start, 3)
        report["vllm_log_path"] = str(getattr(server, "log_path", "") or "")
        report["vllm_log_excerpt"] = log_excerpt(getattr(server, "log_path", None))
        report["classification"] = "inference_error"
        report["error"] = f"vLLM start failed: {type(exc).__name__}: {exc}"[:1200]
        save()
        return report

    try:
        models = server.create_model_registry(
            aliases=[declared_model, TARGET_MODEL_NAME, ADAPTER_NAME],
            model_prefix="openai/",
            api_key="EMPTY",
        )
        if hasattr(models, "keys"):
            report["model_registry_keys"] = [str(key) for key in list(models.keys())[:32]]
        else:
            report["model_registry_repr"] = repr(models)[:1500]
    except Exception as exc:
        report["model_registry_error"] = f"{type(exc).__name__}: {exc}"[:800]

    http_fn = http if http is not None else http_json
    model_ids = []
    models_probe = None
    for root_url in v1_urls(getattr(server, "base_url", "http://127.0.0.1:8000")):
        models_probe = http_fn(root_url + "/models", timeout=30)
        report["vllm_models_url"] = root_url + "/models"
        report["vllm_models_probe"] = {
            "ok": models_probe.get("ok"),
            "status": models_probe.get("status"),
            "error": models_probe.get("error"),
            "error_type": models_probe.get("error_type"),
        }
        body = models_probe.get("body") if models_probe.get("ok") else None
        if isinstance(body, dict):
            model_ids = [str(item.get("id")) for item in (body.get("data") or []) if isinstance(item, dict)]
            report["vllm_model_ids"] = model_ids
            if model_ids:
                break
    report["vllm_log_excerpt"] = log_excerpt(getattr(server, "log_path", None))
    adapter_ids = [item for item in model_ids if ADAPTER_NAME in item]
    report["adapter_in_vllm"] = bool(adapter_ids) or any(
        ADAPTER_NAME.lower() in line.lower() for line in report.get("vllm_log_excerpt") or []
    )
    if adapter_ids:
        report["adapter_in_vllm"] = True
        chat_model = adapter_ids[0]
    elif ADAPTER_NAME in (report.get("model_registry_keys") or []):
        report["adapter_in_vllm"] = True
        chat_model = ADAPTER_NAME
    else:
        report["classification"] = "adapter_not_loaded"
        report["error"] = f"adapter {ADAPTER_NAME!r} missing from vLLM /v1/models ids={model_ids}"
        save()
        try:
            server.stop()
        except Exception:
            pass
        return report

    user_msg = (
        "Call get_status now. Do not edit files. Do not submit a patch. "
        "Do not run tests. One get_status tool call only."
    )
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": T0_BUNDLE["prompts/system.md"]},
            {"role": "user", "content": user_msg},
        ],
        "tools": [GET_STATUS_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_status"}},
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 1024,
    }
    chat = None
    for root_url in v1_urls(getattr(server, "base_url", "http://127.0.0.1:8000")):
        chat = http_fn(root_url + "/chat/completions", body=payload, timeout=180)
        report["tool_call_url"] = root_url + "/chat/completions"
        if chat.get("ok") or chat.get("status") not in (None, 404):
            break
    report["tool_call_model"] = chat_model
    report["tool_call_http_ok"] = bool(chat and chat.get("ok"))
    report["tool_call_http_status"] = None if chat is None else chat.get("status")
    if chat and chat.get("ok"):
        body = chat.get("body")
        report["tool_call_response_keys"] = list(body) if isinstance(body, dict) else [type(body).__name__]
        names = extract_tool_names(body if isinstance(body, dict) else {})
        report["tool_call_names"] = names
        report["tool_call_ok"] = valid_get_status_call(body)
        if isinstance(body, dict):
            choice0 = (body.get("choices") or [{}])[0]
            message = (choice0 or {}).get("message") or {}
            report["tool_call_message"] = {
                "role": message.get("role"),
                "tool_calls": message.get("tool_calls"),
                "content_head": (message.get("content") or "")[:500],
                "finish_reason": (choice0 or {}).get("finish_reason"),
            }
    else:
        report["tool_call_ok"] = False
        report["tool_call_error"] = None if chat is None else {
            "status": chat.get("status"),
            "error": chat.get("error"),
            "error_type": chat.get("error_type"),
        }
    report["classification"] = classify(report)
    report["not_repair_sft"] = True
    report["repair_success_claimed"] = False
    save()
    try:
        server.stop()
    except Exception as exc:
        report["server_stop_error"] = f"{type(exc).__name__}: {exc}"[:400]
        save()
    return report


def main() -> int:
    try:
        run()
    except Exception:
        out = Path("/kaggle/working")
        payload = {
            "classification": "inference_error",
            "not_repair_sft": True,
            "repair_success_claimed": False,
            "error": traceback.format_exc(),
            "python": sys.version,
            "executable": sys.executable,
            "declared_docker_image": DECLARED_DOCKER_IMAGE,
            "pip_attempted": False,
        }
        write_json(out / "result.json", payload)
        print(traceback.format_exc(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
