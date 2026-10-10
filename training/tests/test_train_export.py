"""Focused correctness tests; no downloads and no real Gemma weights.

stdlib tests always run; safetensors tests use tiny actual tensor files.
Numerical Gemma4 + PEFT test is skipped only when the optional ML stack/model
class is unavailable. It creates a random tiny model locally, never from Hub.
Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s training/tests -p test_train_export.py -v
"""
import copy
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout, redirect_stderr

from training.common import (TrainingError, load_config, load_encoded_jsonl,
                             pad_encoded_batch, validate_encoded_record)
from training import train_sft as train
from training import export_peft as export


def encoded(ids=None, labels=None):
    return {"input_ids": ids or [1, 2, 3, 4], "attention_mask": [1] * len(ids or [1, 2, 3, 4]),
            "labels": labels or [-100, -100, 3, 4],
            "meta": {"synthetic": False, "training_eligible": True,
                     "tokenizer_hash": "a" * 64, "chat_template_hash": "b" * 64,
                     "tools_hash": "c" * 64, "protocol_hash": "d" * 64,
                     "protocol_id": "p005-harness-tools-v1"}}


class ValidationTests(unittest.TestCase):
    def test_config_false_and_hash_in_quoted_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text('is_fallback_reference: false\nmodel_name_or_path: "/tmp/a#b" # comment\n')
            result = load_config(path)
            self.assertIs(result["is_fallback_reference"], False)
            self.assertEqual(result["model_name_or_path"], "/tmp/a#b")

    def test_duplicate_config_and_unknown_keys_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("rank: 8\nrank: 16\n")
            with self.assertRaises(TrainingError):
                load_config(path)
        with self.assertRaises(TrainingError):
            train.validate_config({"base_revision": "kaggle-v2"})

    def test_ct_and_fake_hf_revision_rejected(self):
        with self.assertRaises(TrainingError):
            train.validate_config({"model_name_or_path": "google/gemma-4-31b-it-qat-w4a16-ct"})
        with self.assertRaises(TrainingError):
            train.validate_config({"model_revision": "kaggle-v2"})
        with self.assertRaises(TrainingError):
            train.validate_config({"model_name_or_path": "google/gemma-4-31B-it"})
        with self.assertRaises(TrainingError):
            train.source_options("google/a-model", None, True)

    def test_padding_preserves_selected_labels(self):
        short = encoded([1, 2, 3], [-100, -100, 3])
        long = encoded()
        result = pad_encoded_batch([short, long], 0)
        self.assertEqual(result["input_ids"][0], [1, 2, 3, 0])
        self.assertEqual(result["labels"], [[-100, -100, 3, -100], [-100, -100, 3, 4]])
        self.assertEqual(result["attention_mask"][0], [1, 1, 1, 0])

    def test_bad_label_synthetic_and_no_effective_targets_rejected(self):
        for change in ("wrong_label", "synthetic", "empty_targets", "truncated"):
            row = encoded()
            if change == "wrong_label":
                row["labels"][2] = 99
            elif change == "synthetic":
                row["meta"]["synthetic"] = True
            elif change == "empty_targets":
                row["labels"] = [1, -100, -100, -100]
            else:
                row["meta"]["truncated_tool_block"] = True
            with self.subTest(change=change), self.assertRaises(TrainingError):
                validate_encoded_record(row, 32)

    def test_mixed_provenance_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoded.jsonl"
            first, second = encoded(), encoded()
            second["meta"]["tools_hash"] = "f" * 64
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            with self.assertRaisesRegex(TrainingError, "mixed"):
                load_encoded_jsonl(path, 32)

    def test_selection_and_heldout_data_cannot_be_trained(self):
        for split in ("select", "heldout", "heldout-excluded"):
            row = encoded()
            row["meta"]["group"] = {"split": split}
            with self.subTest(split=split), self.assertRaisesRegex(TrainingError, "heldout"):
                validate_encoded_record(row, 32)

    def test_deployment_template_and_tools_hash_must_match(self):
        from training.encoding.gemma_tool_chat import tokenizer_fingerprint, sha256_json
        class Tokenizer:
            special_tokens_map = {}
            def get_vocab(self):
                return {"x": 1}
            def get_chat_template(self, tools=None):
                return "default deployment template"
        tok = Tokenizer()
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.json"
            path.write_text(json.dumps({"protocol_id": "p005-harness-tools-v1", "tools": tools}))
            cfg = train.validate_config({"tool_schema_path": str(path)})
            provenance = dict(encoded()["meta"], tokenizer_hash=tokenizer_fingerprint(tok),
                              tools_hash=sha256_json(tools),
                              chat_template_hash=hashlib.sha256(tok.get_chat_template().encode()).hexdigest())
            self.assertEqual(train.validate_tokenizer_protocol(tok, provenance, cfg)["tools_hash"], sha256_json(tools))
            override = Path(tmp) / "override.jinja"
            override.write_text("different deliberate template")
            provenance["chat_template_hash"] = hashlib.sha256(override.read_bytes()).hexdigest()
            with self.assertRaisesRegex(TrainingError, "template differs"):
                train.validate_tokenizer_protocol(tok, provenance, cfg)
            cfg["chat_template_path"] = str(override)
            self.assertEqual(train.validate_tokenizer_protocol(tok, provenance, cfg)["chat_template_source"], str(override.resolve()))
            provenance["tools_hash"] = "0" * 64
            with self.assertRaisesRegex(TrainingError, "schema hash"):
                train.validate_tokenizer_protocol(tok, provenance, cfg)

    def test_targets_are_exact_text_projection_names(self):
        actual = ["model.language_model.layers.0.self_attn.q_proj",
                  "model.vision_tower.layers.0.self_attn.q_proj",
                  "model.language_model.layers.0.self_attn.q_proj_extra"]
        self.assertEqual(train.resolve_target_modules(actual, ["q_proj"]), actual[:1])

    def test_dry_run_never_imports_training_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{}")
            with mock.patch.object(train, "require_stack", side_effect=AssertionError("must not import")):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(train.main(["--config", str(path), "--dry-run"]), 0)

    def test_missing_stack_fails_not_plan_success(self):
        with mock.patch.object(train.importlib, "import_module", side_effect=ImportError("absent")):
            with self.assertRaisesRegex(TrainingError, "dependencies unavailable"):
                train.require_stack("bf16")

    def test_missing_export_weights_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            manifest = Path(tmp) / "manifest.json"
            result = export.main(["--adapter-dir", tmp, "--manifest-only", "--out", str(manifest)])
            self.assertNotEqual(result, 0)
            self.assertFalse(manifest.exists())


_HAS_SAFE = importlib.util.find_spec("safetensors") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_HAS_SAFE, "safetensors/numpy unavailable")
class RealTensorExportTests(unittest.TestCase):
    def make_adapter(self, path, rank=2):
        import numpy as np
        from safetensors.numpy import save_file
        path.mkdir()
        cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": 4,
               "bias": "none", "base_model_name_or_path": "/local/tiny-model",
               "target_modules": ["q_proj"], "inference_mode": True}
        (path / "adapter_config.json").write_text(json.dumps(cfg))
        prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        save_file({prefix + ".lora_A.weight": np.ones((rank, 4), dtype=np.float32),
                   prefix + ".lora_B.weight": np.zeros((6, rank), dtype=np.float32)},
                  str(path / "adapter_model.safetensors"))
        return cfg

    def test_inspect_real_tensors_and_copy_only_peft_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "src", Path(tmp) / "dst"
            self.make_adapter(source)
            (source / "unrelated.txt").write_text("not part of adapter")
            result = export.export_adapter(source, dest, {"rank": 2, "target_text_modules": ["q_proj"]})
            self.assertEqual(result["adapter"]["tensor_parameters"], 20)
            self.assertEqual(set(p.name for p in dest.iterdir()), {"adapter_config.json", "adapter_model.safetensors"})
            self.assertEqual(export.inspect_adapter(source)["files"], export.inspect_adapter(dest)["files"])
            self.assertFalse(result["runtime_compatibility_verified"])

    def test_wrong_rank_and_corrupt_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            cfg = self.make_adapter(source)
            with self.assertRaises(TrainingError):
                export.inspect_adapter(source, expected_rank=4)
            data = source / "adapter_model.safetensors"
            data.write_bytes(data.read_bytes()[:-4])
            with self.assertRaisesRegex(TrainingError, "invalid safetensors"):
                export.inspect_adapter(source)

    def test_full_weight_or_vision_keys_rejected(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            self.make_adapter(source)
            save_file({"base_model.model.model.embed_tokens.weight": np.ones((4, 4), dtype=np.float32)},
                      str(source / "adapter_model.safetensors"))
            with self.assertRaisesRegex(TrainingError, "unsupported PEFT key"):
                export.inspect_adapter(source)


def numerical_stack():
    try:
        import torch
        import transformers
        import peft
        import safetensors
        import tokenizers
        if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
            return None
        return {"torch": torch, "transformers": transformers, "peft": peft, "safetensors": safetensors}
    except ImportError:
        return None


class NumericalGemmaTests(unittest.TestCase):
    def test_tiny_gemma_lora_training_checkpoint_and_resume(self):
        stack = numerical_stack()
        if stack is None:
            self.skipTest("Gemma4-capable torch/transformers/peft stack unavailable")
        torch, tr = stack["torch"], stack["transformers"]
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        torch.set_num_threads(1)
        torch.manual_seed(7)
        text = tr.Gemma4TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            global_head_dim=8, num_global_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"], sliding_window=16,
            max_position_embeddings=64, hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=32, num_kv_shared_layers=0,
            attention_k_eq_v=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                             "full_attention": {"rope_type": "default", "rope_theta": 10000.0}})
        vision = tr.Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                      position_embedding_size=4)
        config = tr.Gemma4Config(text_config=text, vision_config=vision, audio_config=None,
                                 image_token_id=29, boi_token_id=30, eoi_token_id=31)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            tr.Gemma4ForConditionalGeneration(config).save_pretrained(base)
            backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="3"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = tr.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="0",
                                                  bos_token="1", eos_token="2", unk_token="3")
            tokenizer.save_pretrained(base)
            cfg = train.validate_config({
                "model_name_or_path": str(base), "device": "cpu", "rank": 2, "alpha": 4,
                "epochs": 2, "micro_batch_size": 1, "gradient_accumulation_steps": 2,
                "learning_rate": 0.01, "warmup_ratio": 0.0, "max_sequence_length": 16,
                "gradient_checkpointing": True, "checkpoint_every_n_steps": 1,
            })
            model, tok, _, info = train.load_model_and_tokenizer(cfg, stack)
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            rows = [encoded([1, 4, 5, 6], [-100, -100, 5, 6]),
                    encoded([1, 7, 8], [-100, -100, 8])]
            out = root / "run"
            out.mkdir()
            result = train.run_training_loop(model, rows, cfg, out, torch, tok.pad_token_id, "tiny-unit-test")
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertTrue(math_isfinite(result["loss"]))
            self.assertTrue(any(torch.count_nonzero(p).item() > 0 for n, p in model.named_parameters() if "lora_B" in n))
            for n, p in model.named_parameters():
                if n in frozen:
                    self.assertTrue(torch.equal(frozen[n], p.detach()), n)
            original = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            self.assertGreater(export.inspect_adapter(out / "adapter", 2)["tensor_parameters"], 0)
            cp = out / "checkpoint-000001"
            resumed, tok, _, _ = train.load_model_and_tokenizer(cfg, stack, resume=cp)
            resumed_out = root / "resumed"
            resumed_out.mkdir()
            train.run_training_loop(resumed, rows, cfg, resumed_out, torch, tok.pad_token_id,
                                    "tiny-unit-test", resume=cp)
            for n, p in resumed.named_parameters():
                if n in original:
                    self.assertTrue(torch.equal(original[n], p.detach()), "resume mismatch: " + n)


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    unittest.main()
