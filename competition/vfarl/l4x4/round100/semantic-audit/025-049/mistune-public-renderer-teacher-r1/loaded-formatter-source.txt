"""Render verified human demonstrations with the pinned official Gemma template.

Raw tool events stay unchanged. This produces off-policy SFT material, never
behavior logprobs, model reasoning, or counterfactual preference pairs.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema_for_template(value):
    if isinstance(value, list):
        return [schema_for_template(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "type" and isinstance(item, str):
            item = item.lower()
        result["anyOf" if key == "any_of" else key] = schema_for_template(item)
    return result


def render_episode(result_path, tokenizer, tools, out, official_prompts):
    result = json.loads(result_path.read_text())
    if result.get("schema_version") != "canonical-official-replay-v1" or result.get("eligible_for_human_off_policy_SFT") is not True:
        raise ValueError("Render admitted canonical replay records with actual source and baseline evidence")
    if not result.get("resolved_by_local_official_verify_task"):
        raise ValueError("Only completed locally verified repairs enter this SFT export")
    if result.get("evidence_kind") != "human_reviewed_off_policy_demonstration":
        raise ValueError("Unexpected demonstration origin")
    recipe_path = Path(result["recipe"])
    recipe = json.loads(recipe_path.read_text())
    if not recipe.get("reviewed_by") or not recipe.get("review_notes"):
        raise ValueError("Demonstration actions require actual review attribution")
    report_path = Path(result["reports"]["repaired"]["path"])
    messages_path = report_path.parent / "messages.json"
    messages = copy.deepcopy(json.loads(messages_path.read_text()))
    prompt = official_prompts[result["instance_id"]]
    if prompt["replay_result_sha256"] != sha(result_path):
        raise ValueError("Official prompt export belongs to a different replay result version")
    if messages[0]["role"] != "user":
        raise ValueError("Unexpected initial replay message")
    original_user_text = messages[0]["content"]
    messages[0]["content"] = prompt["prompt"]
    calls, results = {}, {}
    for mi, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            call_id = call["id"]
            if call_id in calls:
                raise ValueError("Duplicate actual tool call ID")
            args = call["function"]["arguments"]
            call["function"]["arguments"] = json.loads(args) if isinstance(args, str) else args
            if not isinstance(call["function"]["arguments"], dict):
                raise ValueError("Gemma tool arguments must be a mapping")
            calls[call_id] = mi
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in calls or calls[call_id] >= mi or call_id in results:
                raise ValueError("Unpaired or out-of-order actual tool response")
            results[call_id] = mi
    if calls.keys() != results.keys() or len(calls) != len(recipe["actions"]):
        raise ValueError("Actual replay/recipe/tool response counts disagree")

    def render(ms, declarations=tools):
        return tokenizer.apply_chat_template(ms, tools=declarations, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)

    rendered = render(messages)
    spans = []
    for call_id, mi in calls.items():
        message = messages[mi]
        # The producer currently emits one actual tool action per assistant message.
        if len(message["tool_calls"]) != 1:
            raise ValueError("Multi-call turns require explicitly reviewed loss spans")
        isolated = render([message], [])
        start = isolated.index("<|tool_call>")
        stop = isolated.rindex("<tool_call|>") + len("<tool_call|>")
        fragment = isolated[start:stop]
        prefix = render(messages[:mi + 1])
        start = prefix.rfind(fragment)
        stop = start + len(fragment)
        if start < 0 or prefix[:stop] != rendered[:stop]:
            raise ValueError("Future messages altered an action prefix; cannot mask safely")
        action = recipe["actions"][len(spans)]
        purpose = ("expected_baseline_reproduction" if action["tool"] == "run_command"
                   and action.get("expected_status") == "error" else "reviewed_diagnostic_or_repair")
        spans.append({"start_char": start, "end_char": stop, "tool_call_id": call_id,
                      "source_message_index": mi, "tool": message["tool_calls"][0]["function"]["name"],
                      "purpose": purpose,
                      "loss_reason": "Reviewed diagnostic/repair action in a locally verified human demonstration"})
    final_text_available = bool(messages[-1]["role"] == "assistant" and messages[-1].get("content")
                                and not messages[-1].get("tool_calls"))
    if final_text_available:
        if messages[-1]["content"] != recipe.get("final_response"):
            raise ValueError("Final text was not part of the reviewed human recipe")
        before_final = render(messages[:-1])
        if not rendered.startswith(before_final):
            raise ValueError("Final response changes the preceding observed prefix")
        spans.append({"start_char": len(before_final), "end_char": len(rendered),
                      "tool_call_id": None, "source_message_index": len(messages)-1,
                      "tool": None, "purpose": "observed_human_text_only_completion",
                      "loss_reason": "Reviewed human final response actually recorded after submit_patch; not model-generated"})
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tokenized["input_ids"], tokenized["offset_mapping"]
    if len(ids) > 32768:
        raise ValueError("Full replay exceeds official context; do not truncate and call it complete")
    mask = []
    for start, stop in offsets:
        matches = [s for s in spans if stop > s["start_char"] and start < s["end_char"]]
        if matches and not all(start >= s["start_char"] and stop <= s["end_char"] for s in matches):
            raise ValueError("Token overlaps action/observation boundary")
        mask.append(int(bool(matches)))
    if not sum(mask):
        raise ValueError("Empty actual assistant supervision")
    name = result_path.name.removesuffix("-replay-result.json")
    text_path = out / (name + ".gemma.txt")
    text_path.write_text(rendered)
    return {
        "schema_version": "gemma-human-replay-sft-v1", "instance_id": result["instance_id"],
        "primary_adapter": result["primary_adapter"], "submechanism": result["submechanism"],
        "canonical_repo": result["canonical_repo"], "repair_family_id": result["repair_family_id"],
        "snapshot_group_id": result["snapshot_group_id"],
        "source_dataset": result["source_dataset"], "source_revision": result["source_revision"],
        "physical_split": result["physical_split"], "source_task_sha256": result["original_task_sha256"],
        "mechanism_labels_are_metadata_not_in_policy_prefix": True,
        "evidence_kind": result["evidence_kind"], "official_verify_task_result": True,
        "demonstration_authorship": result["demonstration_authorship"],
        "source_requirement_profile": result["source_requirement_profile"],
        "strict_verbatim_required_ids_passed": result["strict_verbatim_required_ids_passed"],
        "audited_source_id_mapping": result.get("audited_source_id_mapping"),
        "baseline_evidence_profile": result["baseline_evidence_profile"],
        "baseline_label_review": result.get("baseline_label_review"),
        "visible_prompt_profile": result["visible_prompt_profile"],
        "backend": result["backend"], "competition_Docker_environment_equivalent": False,
        "messages_source": str(messages_path), "messages_source_sha256": sha(messages_path),
        "canonical_initial_user_prompt": prompt,
        "raw_initial_user_prompt_sha256": hashlib.sha256(original_user_text.encode()).hexdigest(),
        "initial_prompt_augmentation": prompt["prompt_change_kind"],
        "replay_result": str(result_path), "replay_result_sha256": sha(result_path),
        "rendered_text": str(text_path), "rendered_text_sha256": sha(text_path),
        "tool_arguments_conversion": "Actual JSON strings decoded to mappings for official Gemma template; raw messages preserved",
        "tool_observations_rewritten": False, "system_prompt": "None added; official template injects actual six-tool declarations",
        "thinking_generated": False, "on_policy": False, "old_logprobs": None,
        "input_ids": ids, "assistant_loss_mask": mask, "action_spans": spans,
        "context_tokens": len(ids), "assistant_loss_tokens": sum(mask),
        "training_ingestion": "Use per-action inference-prefix examples; full conversation rendering is an audit archive",
        "text_only_termination_supervision_available": final_text_available,
        "task_mechanism": result.get("mechanism"),
    }


def action_examples(record, tokenizer, tools):
    """Preserve the exact official generation prefix for each recorded action.

    In particular, enable_thinking=False inserts an empty thought-control block
    in a new generation prompt. It is prompt conditioning, not generated thought
    and not an assistant loss target. Complete-history rendering omits that block.
    """
    messages = json.loads(Path(record["messages_source"]).read_text())
    messages[0]["content"] = record["canonical_initial_user_prompt"]["prompt"]
    for message in messages:
        for call in message.get("tool_calls") or []:
            args = call["function"]["arguments"]
            call["function"]["arguments"] = json.loads(args) if isinstance(args, str) else args
    rendered_bytes = Path(record["rendered_text"]).read_bytes()
    if hashlib.sha256(rendered_bytes).hexdigest() != record["rendered_text_sha256"]:
        raise ValueError("Rendered episode bytes changed since its action spans were recorded")
    rendered = rendered_bytes.decode("utf-8")
    examples = []
    for index, span in enumerate(record["action_spans"]):
        prefix = tokenizer.apply_chat_template(
            messages[:span["source_message_index"]], tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        completion = rendered[span["start_char"]:span["end_char"]]
        tokenized = tokenizer(prefix + completion, add_special_tokens=False,
                              return_offsets_mapping=True)
        offsets = tokenized["offset_mapping"]
        if any(start < len(prefix) < stop for start, stop in offsets):
            raise ValueError("Token crosses actual generation-prefix boundary")
        mask = [int(stop > len(prefix)) for start, stop in offsets]
        if not sum(mask) or len(mask) > 32768:
            raise ValueError("Empty or over-context official-prefix action example")
        examples.append({
            "schema_version": "gemma-human-action-sft-v2",
            "instance_id": record["instance_id"], "action_index": index,
            "primary_adapter": record["primary_adapter"], "submechanism": record["submechanism"],
            "canonical_repo": record["canonical_repo"], "repair_family_id": record["repair_family_id"],
            "snapshot_group_id": record["snapshot_group_id"],
            "source_dataset": record["source_dataset"], "source_revision": record["source_revision"],
            "physical_split": record["physical_split"], "source_task_sha256": record["source_task_sha256"],
            "tool_call_id": span["tool_call_id"], "tool": span["tool"],
            "purpose": span["purpose"], "evidence_kind": record["evidence_kind"],
            "source_requirement_profile": record["source_requirement_profile"],
            "baseline_evidence_profile": record["baseline_evidence_profile"],
            "visible_prompt_profile": record["visible_prompt_profile"]["kind"],
            "demonstration_authorship": record["demonstration_authorship"],
            "prompt": prefix, "completion": completion,
            "input_ids": tokenized["input_ids"], "assistant_loss_mask": mask,
            "context_tokens": len(mask), "assistant_loss_tokens": sum(mask),
            "generation_template_arguments": {"add_generation_prompt": True, "enable_thinking": False},
            "generation_configuration_for_this_export": {"thinking_budget": 0, "include_thoughts": False},
            "forced_empty_thought_block_is_prompt_only": True,
            "generated_reasoning": None, "old_logprobs": None, "on_policy": False,
            "source_episode": record["messages_source"],
            "source_episode_sha256": record["messages_source_sha256"],
            "system_prompt": record["system_prompt"],
            "official_initial_user_prompt_sha256": record["canonical_initial_user_prompt"]["prompt_sha256"],
            "initial_prompt_augmentation": record["initial_prompt_augmentation"],
        })
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--declarations", type=Path, required=True)
    parser.add_argument("--official-prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.tokenizer / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer / "tokenizer.json"),
                                        chat_template=(args.tokenizer / "chat_template.jinja").read_text(),
                                        **{key: cfg[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")})
    declarations = json.loads(args.declarations.read_text())
    tools = [{"type": "function", "function": {
        "name": d["name"], "description": d.get("description", ""),
        "parameters": schema_for_template(d.get("parameters", {"type": "object", "properties": {}, "required": []})),
    }} for d in declarations]
    official_prompts = {row["instance_id"]: row for row in json.loads(args.official_prompts.read_text())["tasks"]}
    records = [render_episode(p, tokenizer, tools, args.out, official_prompts) for p in args.results]
    target = args.out / "sft-episodes.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    actions = [action for record in records for action in action_examples(record, tokenizer, tools)]
    action_target = args.out / "sft-actions.jsonl"
    action_target.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in actions))
    manifest = {
        "tokenizer_sha256": sha(args.tokenizer / "tokenizer.json"),
        "template_sha256": sha(args.tokenizer / "chat_template.jinja"),
        "CPU_export_runtime_packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "actual_adk_declarations": str(args.declarations), "declarations_sha256": sha(args.declarations),
        "official_initial_prompts": str(args.official_prompts), "official_initial_prompts_sha256": sha(args.official_prompts),
        "episodes": [{k: r[k] for k in ("instance_id", "context_tokens", "assistant_loss_tokens", "replay_result")} for r in records],
        "output_path": str(target), "output_sha256": sha(target),
        "recommended_training_input": str(action_target), "action_output_sha256": sha(action_target),
        "action_examples": len(actions), "unique_task_episodes": len(records),
        "source_requirement_profile_episode_counts": dict(Counter(r["source_requirement_profile"] for r in records)),
        "baseline_evidence_profile_episode_counts": dict(Counter(r["baseline_evidence_profile"] for r in records)),
        "visible_prompt_profile_episode_counts": dict(Counter(r["visible_prompt_profile"]["kind"] for r in records)),
        "episode_counts_by_primary_adapter": dict(Counter(r["primary_adapter"] for r in records)),
        "episode_counts_by_submechanism": dict(Counter(r["submechanism"] for r in records)),
        "actual_tool_call_examples": sum(a["tool"] is not None for a in actions),
        "actual_text_only_completion_examples": sum(a["tool"] is None for a in actions),
        "action_loss_tokens": sum(a["assistant_loss_tokens"] for a in actions),
        "processed_context_tokens_with_repeated_prefixes": sum(a["context_tokens"] for a in actions),
        "maximum_action_context_tokens": max(a["context_tokens"] for a in actions),
        "inference_prefix_alignment": "Actual pinned Gemma template with add_generation_prompt=True; forced empty thought controls masked as prompt",
        "coldstart_export_thinking_budget": 0,
        "final_text_termination_supervision": "Only actually recorded, reviewed human completions are included. Earlier episodes lacking text-only termination remain unaugmented.",
        "gpu_used": False, "model_inference": False, "training_started": False,
        "deployment_system_prompt_and_thinking_policy": "This version is direct-tool coldstart with thinking_budget=0, include_thoughts=False and no added system prompt. Other deployment prompts/budgets require a separately versioned re-export; no generated reasoning is fabricated.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
