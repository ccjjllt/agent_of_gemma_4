"""Actual tiny Gemma4 / PEFT numerical checks; no weights or GPU required.

Run with the pinned ML stack, e.g.:
PYTHONDONTWRITEBYTECODE=1 /tmp/p005-training-check/bin/python -m unittest training.tests.test_memory -v
"""
import inspect
import unittest
from contextlib import ExitStack

import torch
import transformers as tr
from peft import LoraConfig, TaskType, get_peft_model

from unittest import mock

from training.memory import (install_selected_token_loss,
                             prepare_model_for_memory_efficient_kbit_training,
                             selected_token_forward)


def tiny_base(dtype=torch.float32):
    text = tr.Gemma4TextConfig(
        vocab_size=43, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        global_head_dim=8, num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"], sliding_window=16,
        max_position_embeddings=64, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=43, num_kv_shared_layers=0,
        attention_k_eq_v=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        final_logit_softcapping=2.0, attention_dropout=0.0,
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                         "full_attention": {"rope_type": "default", "rope_theta": 10000.0}})
    vision = tr.Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, position_embedding_size=4)
    config = tr.Gemma4Config(text_config=text, vision_config=vision, audio_config=None,
                            image_token_id=40, boi_token_id=41, eoi_token_id=42)
    return tr.Gemma4ForConditionalGeneration(config).to(dtype=dtype)


def tiny_lora(dtype=torch.float32, checkpointing=False):
    torch.manual_seed(123)
    base = tiny_base(dtype)
    prepare_model_for_memory_efficient_kbit_training(base, use_gradient_checkpointing=checkpointing)
    targets = [name for name, _ in base.named_modules()
               if name.startswith("model.language_model.")
               and name.rsplit(".", 1)[-1] in {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2,
                          lora_alpha=4, lora_dropout=0.0, target_modules=targets, bias="none"))
    # Nonzero B makes BOTH A and B gradients meaningful in the comparison.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    return model.train()


def batch():
    return {
        "input_ids": torch.tensor([[1, 4, 5, 6, 7, 8], [1, 9, 10, 11, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        # The first position is intentionally labelled: original HF shifting
        # ignores it, as must our helper. Gaps and last-token supervision test
        # the remaining alignment and sparse-selection cases.
        "labels": torch.tensor([[5, -100, 5, -100, 7, 8], [-100, 9, -100, 11, -100, -100]]),
        "use_cache": False,
    }


class MemoryPreparationTests(unittest.TestCase):
    def test_frozen_large_storage_and_tied_embedding_preserved_norms_cast(self):
        model = tiny_base(torch.bfloat16)
        before = {name: (parameter.data_ptr(), parameter.dtype, parameter.clone())
                  for name, parameter in model.named_parameters()}
        embedding = model.get_input_embeddings().weight
        self.assertIs(embedding, model.lm_head.weight)
        self.assertIs(prepare_model_for_memory_efficient_kbit_training(model), model)
        changed = {entry["name"] for entry in model._p005_memory_prepare_report["norm_casts"]}
        self.assertTrue(changed)
        for name, parameter in model.named_parameters():
            self.assertFalse(parameter.requires_grad)
            pointer, dtype, values = before[name]
            if name in changed:
                self.assertEqual(parameter.dtype, torch.float32)
                torch.testing.assert_close(parameter, values.float(), rtol=0, atol=0)
            else:
                self.assertEqual(parameter.dtype, dtype)
                self.assertEqual(parameter.data_ptr(), pointer)
        self.assertIs(embedding, model.lm_head.weight)
        self.assertFalse(model.config.get_text_config().use_cache)
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.model.language_model.layers[0]._gradient_checkpointing_func.keywords["use_reentrant"])

    def test_params4bit_never_reassigned_even_if_norm_parameter(self):
        # This local subclass exercises the same public class boundary without
        # claiming a CUDA/bitsandbytes check. Real NF4 is tested separately.
        class Params4bit(torch.nn.Parameter):
            pass
        model = tiny_base(torch.bfloat16)
        norm = model.model.language_model.norm
        norm.weight = Params4bit(norm.weight.detach(), requires_grad=False)
        original = norm.weight
        pointer = original.data_ptr()
        prepare_model_for_memory_efficient_kbit_training(model)
        self.assertIs(norm.weight, original)
        self.assertEqual(norm.weight.data_ptr(), pointer)
        self.assertEqual(norm.weight.dtype, torch.bfloat16)
        self.assertEqual(model._p005_memory_prepare_report["unchanged_params4bit_tensors"], 1)

    def test_preparing_after_injection_rejected(self):
        with self.assertRaisesRegex(ValueError, "before LoRA"):
            prepare_model_for_memory_efficient_kbit_training(tiny_lora())


class SelectedLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_loss_and_every_adapter_gradient_match_multiple_chunks(self):
        reference = tiny_lora()
        result = reference(**batch())
        result.loss.backward()
        expected = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()
                    if parameter.requires_grad}
        self.assertTrue(all(value.abs().sum().item() > 0 for value in expected.values()))
        for chunk_size in (1, 2, 20):
            with self.subTest(chunk_size=chunk_size):
                candidate = tiny_lora()
                install_selected_token_loss(candidate, chunk_size=chunk_size)
                actual = candidate(**batch())
                self.assertIsNone(actual.logits)
                torch.testing.assert_close(actual.loss, result.loss, rtol=2e-6, atol=2e-7)
                actual.loss.backward()
                for name, parameter in candidate.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-5, atol=3e-7,
                                                   msg=lambda message: name + ": " + message)

    def test_chunk_graph_does_not_save_vocabulary_logits_and_recomputes(self):
        model = tiny_lora(checkpointing=True)
        install_selected_token_loss(model, chunk_size=2)
        saved_shapes, head_shapes, outer_calls = [], [], []
        base = model.get_base_model()
        head_hook = base.lm_head.register_forward_hook(lambda module, args, output: head_shapes.append(tuple(output.shape)))
        outer_hook = model.register_forward_hook(lambda *args: outer_calls.append(True))
        def save(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor
        try:
            with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                output = model(**batch())
            self.assertEqual(len(outer_calls), 1)
            self.assertEqual(len(head_shapes), 3)  # five selected targets
            self.assertTrue(all(shape[0] <= 2 and shape[-1] == 43 for shape in head_shapes))
            self.assertFalse(any(len(shape) == 2 and shape[-1] == 43 for shape in saved_shapes), saved_shapes)
            output.loss.backward()
            self.assertGreater(len(head_shapes), 3)  # head recomputed in backward
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in grads))
            self.assertTrue(all(grad.abs().sum() > 0 for grad in grads))
        finally:
            head_hook.remove()
            outer_hook.remove()

    def test_explicit_normalization_matches_hf(self):
        model = tiny_lora()
        inputs = dict(batch(), num_items_in_batch=torch.tensor(13))
        expected = model(**inputs).loss
        install_selected_token_loss(model, chunk_size=2)
        torch.testing.assert_close(model(**inputs).loss, expected, rtol=2e-6, atol=2e-7)

    def test_inference_and_signature_unchanged_install_idempotent(self):
        model = tiny_lora().eval()
        base = model.get_base_model()
        signature = inspect.signature(base.forward)
        inputs = batch()
        inputs.pop("labels")
        with torch.no_grad():
            expected = model(**inputs).logits
            install_selected_token_loss(model, chunk_size=2)
            install_selected_token_loss(model, chunk_size=1)
            actual = model(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(inspect.signature(base.forward), signature)
        self.assertEqual(base._p005_loss_chunk_size, 1)

    def test_bf16_nonreentrant_adapter_update_preserves_frozen_weights(self):
        model = tiny_lora(dtype=torch.bfloat16, checkpointing=True)
        reference = tiny_lora(dtype=torch.bfloat16, checkpointing=True)
        reference_loss = reference(**batch()).loss
        reference_loss.backward()
        expected_grads = {name: p.grad.clone() for name, p in reference.named_parameters() if p.requires_grad}
        install_selected_token_loss(model, chunk_size=2)
        frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
        trained = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        result = model(**batch())
        self.assertTrue(torch.isfinite(result.loss))
        torch.testing.assert_close(result.loss, reference_loss, rtol=2e-6, atol=2e-7)
        result.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=3e-5, atol=3e-7)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertFalse(torch.equal(parameter, trained[name]), name)
            else:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)

    def test_unsupported_multimodal_cache_embeds_and_empty_labels_rejected(self):
        model = tiny_lora()
        install_selected_token_loss(model)
        for extra in ({"pixel_values": torch.zeros(1)}, {"mm_token_type_ids": torch.zeros(2, 6)},
                      {"inputs_embeds": torch.zeros(2, 6, 16)}, {"use_cache": True},
                      {"past_key_values": object()}, {"logits_to_keep": 1},
                      {"labels": torch.full((2, 6), -100)},
                      {"input_ids": torch.full((2, 6), 40)}):
            with self.subTest(extra=list(extra)), self.assertRaises(ValueError):
                model(**dict(batch(), **extra))


def _loss_grads_and_checkpoint_calls(model):
    """Observe the actual decoder checkpoint entry points during forward/backward."""
    base = model.get_base_model()
    layers = list(base.model.language_model.layers)
    train_mode = model.training and base.training and all(layer.training for layer in layers)
    with ExitStack() as stack:
        spies = [stack.enter_context(mock.patch.object(
            layer, "_gradient_checkpointing_func",
            wraps=getattr(layer, "_gradient_checkpointing_func", torch.utils.checkpoint.checkpoint),
            create=True,
        )) for layer in layers]
        loss = model(**batch()).loss
        loss.backward()
        calls = [spy.call_count for spy in spies]
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    return loss, gradients, calls, train_mode


def compare_save_on_cpu_grads(checkpointing=False):
    """One-shot tiny-Gemma comparison. Returns the numbers; does not write files."""
    reference = tiny_lora(checkpointing=checkpointing)
    install_selected_token_loss(reference, chunk_size=2, save_on_cpu=False)
    expected_loss, expected, reference_calls, reference_train = _loss_grads_and_checkpoint_calls(reference)
    candidate = tiny_lora(checkpointing=checkpointing)
    install_selected_token_loss(candidate, chunk_size=2, save_on_cpu=True)
    actual_loss, actual, candidate_calls, candidate_train = _loss_grads_and_checkpoint_calls(candidate)
    diffs = {name: float((gradient - expected[name]).abs().max())
             for name, gradient in actual.items()}
    both_gradients = list(expected.values()) + list(actual.values())
    return {
        "loss_baseline": float(expected_loss.detach()),
        "loss_save_on_cpu": float(actual_loss.detach()),
        "loss_abs_diff": float((actual_loss - expected_loss).abs().detach()),
        "n_lora_params": len(diffs),
        "max_lora_grad_abs_diff": max(diffs.values()) if diffs else None,
        "all_lora_grads_nonzero": all(value.abs().sum().item() > 0 for value in both_gradients),
        "all_lora_grads_finite": all(torch.isfinite(value).all().item() for value in both_gradients),
        "checkpointing": checkpointing,
        "baseline_train_mode": reference_train,
        "save_on_cpu_train_mode": candidate_train,
        "baseline_decoder_checkpoint_calls": reference_calls,
        "save_on_cpu_decoder_checkpoint_calls": candidate_calls,
        "save_on_cpu_flag": candidate.get_base_model()._p005_save_on_cpu,
        "default_flag": reference.get_base_model()._p005_save_on_cpu,
        "rtol_loss": 2e-6, "atol_loss": 2e-7, "rtol_grad": 3e-5, "atol_grad": 3e-7,
    }


class SaveOnCpuTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_install_leaves_save_on_cpu_off(self):
        model = tiny_lora()
        install_selected_token_loss(model, chunk_size=2)
        self.assertFalse(model.get_base_model()._p005_save_on_cpu)
        selected_token_forward(model.get_base_model(), chunk_size=2, **batch())

    def test_backbone_save_on_cpu_called_only_when_enabled(self):
        calls = []
        real = torch.autograd.graph.save_on_cpu

        def wrapped(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return real(*args, **kwargs)

        model = tiny_lora()
        install_selected_token_loss(model, chunk_size=2, save_on_cpu=False)
        with mock.patch.object(torch.autograd.graph, "save_on_cpu", wrapped):
            model(**batch())
        self.assertEqual(calls, [])
        install_selected_token_loss(model, chunk_size=2, save_on_cpu=True)
        with mock.patch.object(torch.autograd.graph, "save_on_cpu", wrapped):
            model(**batch())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kwargs"].get("pin_memory"), True)

    def test_save_on_cpu_matches_loss_and_lora_grads(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                reference = tiny_lora(checkpointing=checkpointing)
                install_selected_token_loss(reference, chunk_size=2, save_on_cpu=False)
                expected_loss, expected, reference_calls, reference_train = _loss_grads_and_checkpoint_calls(reference)
                candidate = tiny_lora(checkpointing=checkpointing)
                install_selected_token_loss(candidate, chunk_size=2, save_on_cpu=True)
                self.assertTrue(candidate.get_base_model()._p005_save_on_cpu)
                actual_loss, actual, candidate_calls, candidate_train = _loss_grads_and_checkpoint_calls(candidate)
                self.assertTrue(reference_train)
                self.assertTrue(candidate_train)
                for calls in (reference_calls, candidate_calls):
                    self.assertTrue(calls)
                    self.assertTrue(all(count > 0 if checkpointing else count == 0 for count in calls))
                for gradient in list(expected.values()) + list(actual.values()):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-7)
                for name, gradient in actual.items():
                    torch.testing.assert_close(gradient, expected[name], rtol=3e-5, atol=3e-7,
                                               msg=lambda message: name + ": " + message)


if __name__ == "__main__":
    unittest.main()
