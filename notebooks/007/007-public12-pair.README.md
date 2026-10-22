# P007 Public-12 Paired Evaluation: `public022_v10` vs `behavior_v1`

This carrier implements the P007 12-task single-variable paired evaluation on `dev_future` tasks (`rich_4006`, `rich_3067`, `rich_3180`, `rich_3472`, `fastapi_14482`, `fastapi_14851`, `fastapi_14463`, `fastapi_14459`, `requests_7205`, `requests_6757`, `requests_7427`, `requests_7502`).

| Property | Arm A: `public022_v10` | Arm B: `behavior_v1` |
|---|---|---|
| Source | Exact `flexonafft` Budget-Aware V10 (`scriptVersionId=357955943`, Apache 2.0) | Derived from `public022_v10`; only behavioral verification bullets in `prompts/system.md` replaced |
| Files | 4 files unchanged byte-for-byte | `agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml` unchanged; `prompts/system.md` updated |
| Model & Tools | `gemma-4-31b-it-qat-w4a16-ct` (v2), 9 tools (graph tools discouraged in prompt) | Identical |
| Budgets | `timeout_seconds: 240`, `max_tool_calls: 28`, `max_time_minutes: 8`, `max_turns: 80` | Identical |
| Sampling | `temperature: 0.2`, `top_p: 0.95`, `max_output_tokens: 8192`, `thinking_budget: 4096` | Identical |
| Adapter / Skills | None | None |

## GPU Utilization Optimization

- **Single model startup (`server_start_count = 1`)**: Loads 31B CT v2 once on 4×L4 (`tensor_parallel_size=4`, `gpu_memory_utilization=0.90`, `max_model_len=32768`, `--max-num-seqs=4`).
- **Rolling two-pair concurrency (`MAX_CONCURRENT = 4`)**: Keeps up to 2 pairs (4 isolated `Evaluator` worker processes) active simultaneously. Both arms of each task always launch at the exact same timestamp (balanced `AB` / `BA` across the 12 tasks and `2 AB / 2 BA` within each repository), while freed pair slots are refilled immediately without wave-barrier stalls.

## Files & Usage

- `notebooks/builders/build_007_public12_pair.py`: Stdlib-only generator.
- `notebooks/007/007-public12-pair-single-cell.py`, `notebooks/007/007-public12-pair.ipynb`: Standalone carrier.
- `research/007/public12-pair-20261017/kernel/prepare.py` (`kernel-metadata-cpu.json`, `enable_gpu=false`): Free CPU preparation mode that verifies all 41 v31 wheels, extracts and verifies the 12 frozen `dev_future` tasks and snapshots from `tasks.jsonl`, and compiles both Agent arms (`model_requests=0`, `judge_runs=0`).
- `research/007/public12-pair-20261017/kernel/pair.py` (`kernel-metadata.json`, `enable_gpu=true`): Bounded 24-attempt GPU evaluation.
