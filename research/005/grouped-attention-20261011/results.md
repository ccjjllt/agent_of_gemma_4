# Global head_dim=512 grouped SDPA — CPU mechanism study

2026-10-11. This study used only inline CPU Python commands. No CUDA, Kaggle job, pretrained weights, or test suite was used. The ongoing GPU repair comparison is independent of this candidate.

The candidate is [`training/attention.py`](../../../training/attention.py). It is **off by default**, is not imported by the regular trainer, and changes no model on import. Raw numerical observations are in [`manual-cpu-results.json`](manual-cpu-results.json).

## Conclusion

Four KV-group calls are mathematically equivalent to the current full-head expansion when dropout is zero and each query head keeps its original KV head and mask. Our local CPU observations support that equivalence, including gradients. The implementation avoids constructing the full repeated K/V tensors before SDPA.

**CUDA peak-memory and throughput improvement remain hypotheses.** Four SDPA backward graphs coexist, output concatenation adds a transient allocation, and a CUDA kernel can internally copy stride-zero views. This CPU result is sufficient to prepare one optional operator candidate, not to claim that 6032/9841/12832-token training now fits.

## Source facts and mechanism

The official 31B config has 32 query heads, four global KV heads and global head dimension 512; global layers occur every sixth decoder layer. In Transformers 5.13.1, `use_gqa_in_sdpa` only admits CUDA native GQA for equal K/V head dimensions at most 256 and no explicit attention mask. The 512-dimensional global layers therefore use `repeat_kv` to expand K/V from four to 32 heads. `repeat_kv` expands a group dimension and reshapes it into the head dimension.

- [Transformers 5.13.1 SDPA integration](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/integrations/sdpa_attention.py): `repeat_kv`, `use_gqa_in_sdpa`, `sdpa_attention_forward`.
- [Gemma4 attention implementation](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/gemma4/modeling_gemma4.py): `Gemma4TextAttention`, global head dimensions, `num_key_value_groups`, and scaling of **1.0**.
- [PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html): per-head attention equation, boolean-mask meaning, causal alignment, backend selection and GQA limits.
- [PyTorch 2.11 CUDA backend selection](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp): FlashAttention and memory-efficient attention have different shape constraints; the efficient backend does not directly admit unequal Q/KV head counts.

For query head `h`, the baseline selects KV head `floor(h / 8)`. The candidate processes groups `g=0..3`: query heads `[8g, 8g+8)` with KV head `g`, expanded as an eight-head **stride-zero view**, then concatenates the four outputs in the original head order. Softmax is over source tokens, never over heads, so the partition changes no attention dependency. Gradient contributions from the eight query heads still sum into their shared KV source through `expand` backward.

Only head groups change. The candidate does not split sequences, truncate history, change causal positions, remove padding, replace the mask, change RoPE, or change scaling. It does not force a CUDA backend or enable native GQA. It presents equal Q/KV head counts within each SDPA call.

At `B=1, L=S=6032, BF16`, the baseline's two materialized 32-head K/V tensors contain **377 MiB**. Their original four-head sources contain **47.125 MiB**. Avoiding materialization removes a potential allocation; it is not a promise of a 377 MiB reduction in peak memory. Existing source lifetimes, backend copies, output assembly and recomputation determine the actual peak.

## Scope and exactness

- Query shape `[B,32,L,512]`, K/V shape `[B,4,S,512]`, matching floating dtype and device.
- Masks: 2D broadcastable `[L,S]`, or 4D `[B|1,32|1,L|1,S|1]`. A 32-head mask is sliced with the corresponding query heads. Shared masks remain shared. Boolean `True` means allowed; additive values are preserved.
- The HF adapter retains HF's dispatch: use causal mode only when query length exceeds one, no explicit mask is present, and the module/caller requests causality. Non-square causal semantics are PyTorch's existing upper-left convention, not a new decode-alignment policy.
- **Nonzero attention dropout is rejected.** Separate calls consume RNG differently, so identical seeds would not guarantee identical dropout masks. The current competition training recipe has zero attention dropout.
- Sliding attention, packed/ragged layouts, multimodal attention masks, returned attention score matrices and KV-sharing model variants are not the candidate's target. Installation rejects incompatible global module settings. The core has no implementation for packed layouts.
- Floating-point kernels may choose different reduction orders, so mathematical equivalence does not imply bitwise equality on CUDA.

## Manual CPU observations

Environment: macOS CPU, Python 3.11, **torch 2.14.1**, Transformers **5.13.1**. The actual Kaggle training stack uses **torch 2.11.0+cu128**. The local default operator was `aten::_scaled_dot_product_flash_attention_for_cpu`; this is not evidence that NVIDIA FlashAttention supports global head_dim=512.

Thirteen small-shape comparisons covered FP64/FP32/BF16, causal and explicit masks, boolean 2D/shared/per-head masks, additive per-head masks, fully masked rows, noncontiguous Q/K/V, non-square causal shapes, forced CPU math SDPA, and shared K/V source tensors. In these observations:

- Maximum forward absolute difference: **0**.
- Maximum Q/K/V gradient absolute difference: **0**.
- The additional shared differentiable-mask case had maximum mask-gradient difference **7.1054e-15**, consistent with reduction-order rounding.

A small randomly initialized two-layer `Gemma4TextModel` (hidden size 16, vocabulary 64, **956,048 parameters**) exercised the actual HF attention dispatcher, native sliding/full mask construction, padding and the official proportional global RoPE settings. Only its global layer was changed. Output and all **27** observed parameter gradients had maximum difference **0**. No pretrained weights were loaded. The initial small-model setup used an incorrect default global RoPE configuration and failed before SDPA; repeating with the official global RoPE configuration resolved that setup error without changing the candidate.

## What backward storage showed

For `[1,32,64,512]` FP32 queries, hooks recorded tensors saved by the standalone attention graph. Storage pointers were deduplicated; expanded views must not be counted as independent allocations.

| CPU backend | Arm | Saved tensor entries | Logical saved MiB | Unique saved storage MiB | Unique storage excluding original inputs MiB |
|---|---|---:|---:|---:|---:|
| Default CPU flash | Repeat to 32 heads | 5 | 16.008 | 16.008 | 12.008 |
| Default CPU flash | Four KV groups | 20 | 16.008 | 9.008 | 4.008 |
| Forced math | Repeat to 32 heads | 5 | 13.000 | 12.500 | 12.500 |
| Forced math | Four KV groups | 20 | 13.000 | 9.000 | 8.500 |

The benefit here comes from storage sharing, **not** four times fewer saved values. All four groups' graphs remain until backward. In math attention the aggregate score-matrix area remains proportional to `32*L*S`, so grouping is not a cure for a math-kernel quadratic-memory fallback.

`torch.cat` must allocate a complete output while group outputs coexist. The HF-required transpose/contiguous output may allocate another buffer, as in the baseline. During real non-reentrant layer checkpointing, original-forward saves and backward-recomputed saves have different lifetimes; these standalone CPU storage numbers do not measure that combined peak. CUDA kernels may also materialize expanded K/V or allocate different backward workspaces. Those effects can reduce or erase the proposed gain.

## Explicit use, still disabled in the trainer

For a deliberately selected future experiment, after loading model and LoRA, before DDP:

```python
from training.attention import enable_grouped_global_sdpa
changed = enable_grouped_global_sdpa(model, enabled=True)
```

Calling with the default `enabled=False` changes nothing. Installation registers one runtime attention interface and copies the config only on supported global attention modules. The language model keeps its ordinary SDPA mask builder and sliding layers keep ordinary SDPA. A PEFT adapter does not serialize this runtime choice; a new experimental process must opt in explicitly. Use a fresh model for the ordinary baseline. No existing training entry point was edited to call it.

## Minimum next CUDA operator study

Do this only when separately scheduled; the current GPU allocation remains for T0/single_state repair comparisons. No 31B model is needed and no default full-model capacity rerun is proposed.

Use the recorded runtime versions, a single L4, and just these inputs:

1. BF16 query `[1,32,6032,512]`, K/V `[1,4,6032,512]`, scale **1.0**, dropout **0**. Reproduce Gemma's `[B,L,H,D] -> [B,H,L,D]` strides; use the same values and upstream gradient for both arms. All Q/K/V require gradients.
2. Preserve the **actual global mask** from the current HF configuration and all-valid 6032-token row. Record whether it is `None`; if not, record shape, dtype, strides and mask semantics. Do not manufacture a different mask to obtain a faster backend.
3. Compare HF 5.13.1 ordinary `sdpa_attention_forward` with `gemma4_grouped_global_attention_forward`. Capture the actual dispatched forward/backward operator names, kernel eligibility warnings when relevant, forward/backward elapsed time, live memory after forward, and phase-specific peak allocated/reserved CUDA bytes. Record output and Q/K/V-gradient absolute/relative differences.

The concrete question is whether avoiding full K/V materialization and reducing per-call head count outweighs output assembly, saved state, copies and four kernel launches on **this** CUDA path. If a different backend is selected, report that as part of the mechanism rather than attributing all savings to grouping. A useful reduction in complete forward-plus-backward peak justifies considering one full-model application; absent such a reduction, keep the candidate off and prioritize independent training problems and real repair gains. Neither result requires another context-window sweep or a new test framework.
