"""Unsloth policy and full-prefix scoring for official-task GRPO.

CUDA libraries are imported only when Policy is constructed. Generation uses
HF/Unsloth generate; training losses and optimizer steps belong to the TRL
adapter in grpo_trainer.py. Every model call retains its actual full prefix.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
import threading
import time
import uuid


class ContextLimitError(ValueError):
    """The complete prompt has no space left in the configured context window."""


class GenerationAborted(RuntimeError):
    """The enclosing run expired; this unfinished rollout has no task reward."""


class Policy:
    def __init__(self, config: dict):
        # Import Unsloth before transformers/PEFT so its documented patches apply.
        from unsloth import FastModel
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from transformers.utils.chat_parsing.response_templates import load_response_template

        self.torch = torch
        self.FastModel = FastModel
        self.config = dict(config)
        self.abort_generation = threading.Event()
        self.max_context = int(config.get("max_context", 32768))
        self.chunk_size = int(config.get("logprob_chunk_size", 128))
        self.gradient_checkpointing = config.get("gradient_checkpointing", "unsloth")
        if self.max_context <= 1 or self.max_context > 32768 or self.chunk_size < 1:
            raise ValueError("max_context must be 2..32768 and logprob_chunk_size positive")
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires the separately authorized CUDA server")
        expected_gpus = int(config.get("expected_gpu_count", 2))
        if expected_gpus < 1 or torch.cuda.device_count() != expected_gpus:
            raise RuntimeError(f"Expected {expected_gpus} visible GPUs; found {torch.cuda.device_count()}")
        if float(config.get("lora_dropout", 0.0)) != 0.0:
            raise ValueError("This on-policy baseline requires lora_dropout=0")
        model_path = Path(config["model_path"]).expanduser().resolve(strict=True)
        if not model_path.is_dir():
            raise ValueError("model_path must be a prepared local checkpoint directory")
        raw_config = json.loads((model_path / "config.json").read_text())
        text_config = raw_config.get("text_config", {})
        if raw_config.get("model_type") != "gemma4" or text_config.get("num_hidden_layers") != 60:
            raise ValueError("This implementation is for the corresponding Gemma 4 31B checkpoint")
        if raw_config.get("quantization_config"):
            raise ValueError("Use the official QAT-unquantized BF16 checkpoint, not a packed/NF4 base")
        self.model_path = model_path
        # Validate these small assets before loading the 31B weights. Some official
        # checkpoint versions only contain response_schema, which Transformers
        # 5.17 drops. An explicit official tokenizer-config sidecar can supply its
        # newer response_template without replacing this checkpoint's tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.chat_template:
            raise ValueError("The official chat_template must be present")
        self.response_template = getattr(self.tokenizer, "response_template", None)
        response_template_path = config.get("response_template_path")
        if response_template_path:
            response_metadata = json.loads(
                Path(response_template_path).expanduser().resolve(strict=True).read_text()
            )
            self.response_template = response_metadata.get("response_template")
        if not self.response_template:
            raise ValueError(
                "Transformers 5.17 requires response_template; legacy response_schema is unsupported. "
                "Set response_template_path to an official tokenizer_config.json with the corresponding "
                "Gemma 4 response_template, keeping the checkpoint tokenizer assets unchanged."
            )
        load_response_template(self.response_template)
        self.generation_config = GenerationConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        kwargs = {}
        if config.get("max_memory_gib"):
            kwargs["max_memory"] = {
                int(k): f"{int(v)}GiB" for k, v in config["max_memory_gib"].items()
            }
        self.model, self.processor = FastModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.max_context,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            fast_inference=False,
            device_map=config.get("device_map", "balanced"),
            use_gradient_checkpointing=self.gradient_checkpointing,
            use_exact_model_name=True,
            fix_tokenizer=False,
            trust_remote_code=False,
            **kwargs,
        )
        self.targets = self._lora_targets(self.model, text_config)
        self.model = FastModel.get_peft_model(
            self.model,
            r=int(config.get("lora_rank", 32)),
            lora_alpha=int(config.get("lora_alpha", 64)),
            lora_dropout=0.0,
            bias="none",
            # Unsloth treats a scoped list as leaf names. An exact regex keeps
            # our qualified text-only projections, including the K exclusions.
            target_modules="(?:" + "|".join(map(re.escape, self.targets)) + ")",
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing=self.gradient_checkpointing,
            random_state=int(config.get("seed", 3407)),
            use_rslora=False,
        )
        unexpected = [n for n, p in self.model.named_parameters() if p.requires_grad and ".lora_" not in n]
        if unexpected:
            raise RuntimeError(f"Only LoRA may be trainable; unexpected parameters: {unexpected[:5]}")
        devices = {p.device.type for p in self.model.parameters()}
        if devices - {"cuda"}:
            raise RuntimeError(f"Automatic CPU/disk weight spill is not configured: {sorted(devices)}")
        gpu_indices = {p.device.index for p in self.model.parameters() if p.numel()}
        if len(gpu_indices) != expected_gpus:
            raise RuntimeError(f"Model uses GPUs {sorted(gpu_indices)}, expected {expected_gpus} partitions")
        self.input_device = self.model.get_input_embeddings().weight.device
        self.softcap = float(text_config.get("final_logit_softcapping") or 0.0)
        self._mode = None

    @staticmethod
    def _lora_targets(model, text_config: dict) -> list[str]:
        prefix = "model.language_model.layers."
        targets = []
        layer_types = text_config["layer_types"]
        for name, module in model.named_modules():
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):].split(".")
            if len(rest) != 3:
                continue
            index, group, leaf = rest
            if not index.isdigit():
                continue
            attention = group == "self_attn" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}
            mlp = group == "mlp" and leaf in {"gate_proj", "up_proj", "down_proj"}
            if not (attention or mlp) or not hasattr(module, "weight"):
                continue
            # Full attention shares the K projection with V. Until the scorer's
            # nonzero-adapter mapping is measured, omit these ten K projections.
            if leaf == "k_proj" and layer_types[int(index)] == "full_attention":
                continue
            targets.append(name)
        if len(targets) != 400:
            raise RuntimeError(f"Expected 400 reviewed text projections, found {len(targets)}")
        return targets

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _for_inference(self):
        if self._mode != "inference":
            self.FastModel.for_inference(self.model)
            self.model.eval()
            self._mode = "inference"

    def _for_training(self):
        if self._mode != "training":
            self.FastModel.for_training(
                self.model, use_gradient_checkpointing=self.gradient_checkpointing
            )
            decoder = self.model.get_base_model().model.language_model
            if not self.model.training or not decoder.training:
                raise RuntimeError("Unsloth did not restore training mode")
            if self.gradient_checkpointing and not decoder.gradient_checkpointing:
                raise RuntimeError("Decoder gradient checkpointing was not enabled")
            self._mode = "training"

    @staticmethod
    def _messages_for_template(messages: list[dict]) -> list[dict]:
        result = copy.deepcopy(messages)
        for message in result:
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" for part in content):
                    raise ValueError("This competition policy endpoint accepts text-only messages")
                message["content"] = "".join(part.get("text", "") for part in content)
            if message.get("content") is None:
                message["content"] = ""
            if message.get("reasoning_content") and "thinking" not in message:
                message["thinking"] = message["reasoning_content"]
            for call in message.get("tool_calls", []):
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        return result

    def generate(self, messages: list[dict], tools=None, max_tokens=16384, *, enable_thinking=True) -> dict:
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        torch = self.torch
        if self.abort_generation.is_set():
            raise GenerationAborted("Generation cancelled by the enclosing run budget")
        self._for_inference()
        prompt = self.tokenizer.apply_chat_template(
            self._messages_for_template(messages),
            tools=tools or None,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=bool(enable_thinking),
            truncation=False,
        )
        prompt_ids = list(prompt)
        if len(prompt_ids) >= self.max_context:
            raise ContextLimitError(f"Complete prompt has {len(prompt_ids)} tokens; window is {self.max_context}")
        requested = int(max_tokens)
        if requested < 1:
            raise ValueError("max_tokens must be positive")
        limit = min(requested, self.max_context - len(prompt_ids))
        generation_started = time.perf_counter()

        class BehaviorObserver(LogitsProcessor):
            def __init__(self):
                self.pending = None
                self.pending_length = None
                self.values = []
                self.first_scores_seconds = None

            def collect_previous(self, ids):
                if self.pending is None:
                    return
                if ids.shape[-1] != self.pending_length + 1:
                    raise RuntimeError("Behavior observer requires ordinary one-token-at-a-time generate")
                self.values.append(self.pending.gather(-1, ids[:, -1:].to(self.pending.device)).reshape(-1))

            def __call__(self, ids, scores):
                self.collect_previous(ids)
                if self.first_scores_seconds is None:
                    self.first_scores_seconds = time.perf_counter() - generation_started
                if self.values and len(self.values) % 128 == 0:
                    print(json.dumps({"event": "generation_progress", "prompt_tokens": len(prompt_ids),
                                      "generated_tokens": len(self.values),
                                      "elapsed_seconds": time.perf_counter() - generation_started}), flush=True)
                self.pending = torch.log_softmax(scores.float(), dim=-1).detach()
                self.pending_length = ids.shape[-1]
                return scores

            def finish(self, ids):
                self.collect_previous(ids)
                self.pending = None
                return torch.cat(self.values).cpu().tolist() if self.values else []

        observer = BehaviorObserver()
        abort_event = self.abort_generation

        class StopOnRunAbort(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],), abort_event.is_set(), device=input_ids.device, dtype=torch.bool
                )

        gen = copy.deepcopy(self.generation_config)
        # Explicit research policy: full categorical sampling. There is no extra
        # thinking-token cap. No hidden top-p/top-k or penalty alters its density.
        gen.do_sample = True
        gen.temperature = 1.0
        gen.top_p = 1.0
        gen.top_k = 0
        gen.typical_p = 1.0
        gen.min_p = None
        gen.epsilon_cutoff = 0.0
        gen.eta_cutoff = 0.0
        gen.repetition_penalty = 1.0
        gen.no_repeat_ngram_size = 0
        gen.min_length = 0
        gen.min_new_tokens = None
        gen.forced_bos_token_id = None
        gen.forced_eos_token_id = None
        gen.suppress_tokens = None
        gen.begin_suppress_tokens = None
        gen.bad_words_ids = None
        gen.sequence_bias = None
        gen.num_beams = 1
        gen.num_return_sequences = 1
        gen.max_new_tokens = limit
        gen.return_dict_in_generate = False
        gen.output_scores = False
        gen.output_logits = False
        gen.use_cache = True
        input_ids = torch.tensor([prompt_ids], device=self.input_device, dtype=torch.long)
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                generation_config=gen,
                logits_processor=LogitsProcessorList([observer]),
                stopping_criteria=StoppingCriteriaList([StopOnRunAbort()]),
            )
            behavior = observer.finish(sequences)
        generation_elapsed = time.perf_counter() - generation_started
        if self.abort_generation.is_set():
            raise GenerationAborted("Generation cancelled; discard the unfinished group, not a zero task reward")
        completion_ids = sequences[0, len(prompt_ids):].cpu().tolist()
        if len(behavior) != len(completion_ids) or not all(math.isfinite(x) for x in behavior):
            raise RuntimeError("Generation did not produce one finite behavior logprob per sampled token")
        raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        parsed = self.tokenizer.parse_response(
            completion_ids, schema=self.response_template, prefix=prompt_ids, tools=tools or None
        )
        message = {"role": "assistant", "content": parsed.get("content") or None}
        if parsed.get("thinking"):
            message["reasoning_content"] = parsed["thinking"]
        calls = []
        for call in parsed.get("tool_calls", []) or []:
            function = call["function"]
            arguments = function.get("arguments", {})
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                },
            })
        if calls:
            message["tool_calls"] = calls
        eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
        ended = bool(completion_ids and completion_ids[-1] in eos)
        finish = "tool_calls" if calls else ("stop" if ended else "length")
        return {
            "assistant_message": message,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "behavior_logprobs": behavior,
            "finish_reason": finish,
            "raw_text": raw_text,
            "parsed_message": parsed,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "enable_thinking": bool(enable_thinking)},
            "requested_max_tokens": requested,
            "effective_max_tokens": limit,
            "context_limited": limit < requested,
            "generation_elapsed_seconds": generation_elapsed,
            "first_scores_seconds": observer.first_scores_seconds,
        }

    def per_token_logprobs(self, call: dict, *, compute_entropy: bool = False):
        """Return differentiable selected-token log probabilities for one call.

        Recompute the exact full prefix, including tool observations. The large
        vocabulary projection is checkpointed in chunks; no prefix KV is detached.
        """
        from torch.utils.checkpoint import checkpoint

        torch = self.torch
        prompt_ids = call["prompt_ids"]
        completion_ids = call["completion_ids"]
        if not prompt_ids or not completion_ids:
            raise ValueError("Call must contain its complete prompt and generated IDs")
        if len(prompt_ids) + len(completion_ids) > self.max_context:
            raise ContextLimitError("Stored call exceeds the declared context; no truncation is permitted")
        self._for_training()
        ids = torch.tensor([prompt_ids + completion_ids[:-1]], device=self.input_device, dtype=torch.long)
        base = self.model.get_base_model()
        outputs = base.model.language_model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
            return_dict=True, output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state[:, len(prompt_ids) - 1:, :].squeeze(0)
        head = base.lm_head
        head_device = head.weight.device
        targets = torch.tensor(completion_ids, device=head_device, dtype=torch.long)

        def score_chunk(h, y):
            logits = head(h.to(device=head_device, dtype=head.weight.dtype))
            if self.softcap:
                logits = self.softcap * torch.tanh(logits / self.softcap)
            log_distribution = torch.log_softmax(logits.float(), dim=-1)
            selected = log_distribution.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            if compute_entropy:
                # Entropy is a diagnostic here, not an extra training objective.
                with torch.no_grad():
                    entropy = -(log_distribution.exp() * log_distribution).sum(-1)
                return selected, entropy
            return selected

        pieces, entropy_pieces = [], []
        for start in range(0, len(completion_ids), self.chunk_size):
            stop = min(start + self.chunk_size, len(completion_ids))
            if torch.is_grad_enabled():
                scored = checkpoint(score_chunk, hidden[start:stop], targets[start:stop], use_reentrant=False)
            else:
                scored = score_chunk(hidden[start:stop], targets[start:stop])
            if compute_entropy:
                logp, entropy = scored
                pieces.append(logp)
                entropy_pieces.append(entropy)
            else:
                pieces.append(scored)
        logprobs = torch.cat(pieces)
        if not torch.isfinite(logprobs).all():
            raise FloatingPointError("Learner produced a non-finite selected-token logprob")
        return logprobs, torch.cat(entropy_pieces) if compute_entropy else None

    def score_call(self, call: dict, *, reference: bool = False) -> list[float]:
        """Freeze old-policy/reference probabilities before any cohort update.

        The reference is the frozen official BF16 base with LoRA disabled. This
        run begins with a zero-B adapter, so it is the initial policy and remains
        fixed across all later optimizer updates.
        """
        from contextlib import nullcontext

        self._for_training()
        context = self.model.disable_adapter() if reference else nullcontext()
        with context, self.torch.no_grad():
            logprobs, _ = self.per_token_logprobs(call)
            return logprobs.detach().cpu().tolist()

    def save_adapter(self, path: str | Path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination), safe_serialization=True)
        (destination / "training_base.json").write_text(json.dumps({
            "declared_model_source": self.config["model_name"],
            "declared_model_revision": self.config.get("model_revision"),
            "declared_model_source_uri": self.config.get("model_source_uri"),
            "algorithm": "trl_grpo_with_official_multiturn_rollouts",
            "targets": self.targets,
            "max_context": self.max_context,
            "sampler": "temperature=1, top_p=1, top_k=0; no separate thinking cap",
        }, indent=2) + "\n")
