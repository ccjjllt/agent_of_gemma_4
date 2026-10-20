"""12B routing component for the shared P014 inference carrier.

No model is loaded and no task is run on import. The carrier embeds this source,
the frozen agent text files and both JSON pins in its single uploaded code_file.
"""

from pathlib import Path
import hashlib
import json
import shutil


MODEL_ALIAS = "gemma-4-12b-it-qat-w4a16-ct"
LORA_NAME = "p013_behavior_lora"
ARMS = ("base", "adapter")
MAX_CONCURRENT = 4
VLLM_EXTRA_ARGS = [
    "--max-num-seqs", "4", "--hf-overrides",
    '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_adapter(input_root, agent_root, adapter_source):
    """Stage only the independently exported final adapter, never checkpoints.

    The SFT kernel output contains several adapters. Restrict to its final
    training/adapter directory, then verify both exact frozen file hashes.
    """
    input_root, agent_root = Path(input_root), Path(agent_root)
    source_slug = adapter_source["source_kernel"].split("/")[1]
    candidates = [
        path for path in input_root.rglob("adapter_model.safetensors")
        if source_slug in path.parts and path.parent.name == "adapter"
        and path.parent.parent.name == "training"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one final P013 training/adapter output; found "
            + str([str(path) for path in candidates])
        )
    src = candidates[0].parent
    dest = agent_root / "adapter" / "adapters" / LORA_NAME
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, pin in adapter_source["files"].items():
        source = src / name
        actual = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
        if actual != pin:
            raise RuntimeError("P013 adapter differs from frozen source: " + name)
        shutil.copyfile(source, dest / name)
        files[name] = actual
    config = json.loads((dest / "adapter_config.json").read_text())
    if config["r"] != 8 or config["lora_alpha"] != 16:
        raise RuntimeError("Unexpected trained LoRA rank/alpha")
    if (agent_root / "base" / "adapters").exists():
        raise RuntimeError("Base arm must not contain an adapter")
    return {"source": str(src), "staged": str(dest), "files": files,
            "lora_name": LORA_NAME, "repair_only": True}


def prove_role_routing(agent_root):
    """Return the role mapping after checking the actual staged YAML files."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
    root = Path(agent_root)
    repairs = {}
    for arm in ARMS:
        agent = yaml.load((root / arm / "agent.yaml").read_text(), Loader=Loader)
        repair = yaml.load((root / arm / "repair.yaml").read_text(), Loader=Loader)
        review = yaml.load((root / arm / "review.yaml").read_text(), Loader=Loader)
        if agent["agent_class"] != "SequentialAgent" or agent.get("adapter"):
            raise RuntimeError("Root must be a base sequential workflow")
        expected = LORA_NAME if arm == "adapter" else None
        if repair.get("adapter") != expected or review.get("adapter") is not None:
            raise RuntimeError("LoRA must route only to the adapter arm repair role")
        if repair["model"] != MODEL_ALIAS or review["model"] != MODEL_ALIAS:
            raise RuntimeError("12B model identity mismatch")
        if repair["tools"] != [
            "run_command", "read_file", "edit_file", "write_file", "get_status"
        ]:
            raise RuntimeError("Repair must retain the five-tool training protocol")
        if review["include_contents"] != "none":
            raise RuntimeError("Review context separation changed")
        prompt = (root / arm / "prompts/review.md").read_text()
        if "{problem_description}" not in prompt or "{repair_handoff}" not in prompt:
            raise RuntimeError("Review is missing explicit issue or handoff")
        repairs[arm] = repair
    normalized_adapter = dict(repairs["adapter"])
    normalized_adapter.pop("adapter")
    if normalized_adapter != repairs["base"]:
        raise RuntimeError("Repair arms differ beyond the intended LoRA reference")
    for rel in ("agent.yaml", "review.yaml", "prompts/repair.md",
                "prompts/review.md", "configs/sampling.yaml", "eval_config.yaml"):
        if (root / "base" / rel).read_bytes() != (root / "adapter" / rel).read_bytes():
            raise RuntimeError("Unexpected arm difference: " + rel)
    return {"base": {"repair": MODEL_ALIAS, "review": MODEL_ALIAS},
            "adapter": {"repair": LORA_NAME, "review": MODEL_ALIAS}}


def register_worker_aliases(models, agent_dir, served_model, base_url):
    """Use the same adapter aliases as official VllmServer/ModelRegistry."""
    from adk_submission import discover_adapters
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    models.register(MODEL_ALIAS, LiteLlm(
        model="openai/" + served_model, api_base=base_url, api_key="EMPTY"))
    adapters = discover_adapters(
        agent_dir, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    for name in adapters.adapters:
        for alias in (name, "adapter:" + name):
            models.register(alias, LiteLlm(
                model="openai/" + name, api_base=base_url, api_key="EMPTY"))
    return adapters
