#!/usr/bin/env python3
"""Build the P007 12B real-sample training, PEFT export, and official 12B CT v2 probe.

Uses stdlib only at build time. Packages the repository's training modules,
pinned 12B configs, and the frozen single-sample click train / pydicom select
files into a standalone script with both CPU-prepare and GPU execution modes.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
KERNEL = HERE / 'gpu-kernel'
CPU_KERNEL = HERE / 'cpu-kernel'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'

EMBEDDED_PATHS = (
    'training/common.py',
    'training/memory.py',
    'training/export_peft.py',
    'training/train_sft.py',
    'training/encoding/gemma_tool_chat.py',
    'training/configs/sft_12b_qat_nf4.yaml',
    'training/configs/gemma4_12b_sources.json',
    'training/configs/chat_templates/gemma4_12b_ct_v2.jinja',
    'training/configs/harness_tools.json',
    'research/006/data-replay-20261013/data-usage.json',
    'research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl',
    'research/006/data-replay-20261013/outputs/encoded-12b-select-diagnostic.jsonl',
    'research/006/data-replay-20261013/outputs/click-echo/transaction.json',
    'research/006/data-replay-20261013/outputs/pydicom-recovery/transaction.json',
    'research/006/data-replay-20261013/sources/click-buggy-utils.py',
    'research/006/data-replay-20261013/outputs/pydicom-recovery/state-before.py.txt',
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PHASE1_CHILD = r'''
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import peft
from safetensors.torch import load_file, save_file
import torch
import transformers

BUNDLE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CACHE_DIR = Path(sys.argv[3])
CT_DIR = Path(sys.argv[4])

sys.path.insert(0, str(BUNDLE))
from training.common import load_config, load_encoded_jsonl, pad_encoded_batch, sha256_file
from training.export_peft import inspect_adapter
from training.memory import install_selected_token_loss, prepare_model_for_memory_efficient_kbit_training
from training.train_sft import validate_config

assert torch.cuda.is_available(), 'Phase 1 requires CUDA'
device = torch.device('cuda:0')

train_data_path = BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl'
select_data_path = BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-select-diagnostic.jsonl'
cfg_path = BUNDLE / 'training/configs/sft_12b_qat_nf4.yaml'
train_out = OUT / 'train-run'
export_root = OUT / 'exported_adapters'
real_adapter_dir = export_root / 'click_r8_step1'
zero_adapter_dir = export_root / 'zero_r8_ctrl'
export_manifest_path = OUT / 'export_manifest.json'

# 1. Verify data-usage contract: only click is allowed for parameter updates.
usage = json.loads((BUNDLE / 'research/006/data-replay-20261013/data-usage.json').read_text())
assert usage['assignments']['p006-click-echo']['allow_parameter_update'] is True
assert usage['assignments']['p006-pydicom-recovery']['allow_parameter_update'] is False

# 2. Run production training/train_sft.py on the single click sample (3368 input / 519 supervised tokens).
cmd_train = [
    sys.executable, '-m', 'training.train_sft',
    '--config', str(cfg_path),
    '--model', str(CACHE_DIR),
    '--tokenizer', str(CT_DIR),
    '--data', str(train_data_path),
    '--out', str(train_out),
    '--device', 'cuda:0',
]
t0 = time.monotonic()
res_train = subprocess.run(cmd_train, cwd=str(BUNDLE), text=True, capture_output=True, check=False)
train_wall = round(time.monotonic() - t0, 3)
(OUT / 'train_sft_stdout.log').write_text((res_train.stdout or '') + '\n' + (res_train.stderr or ''))
if res_train.returncode != 0:
    raise RuntimeError(f'train_sft failed ({res_train.returncode}): {res_train.stderr or res_train.stdout}')

train_manifest = json.loads((train_out / 'training_manifest.json').read_text())
step_metrics = [json.loads(line) for line in (train_out / 'metrics.jsonl').read_text().splitlines() if line.strip()]
assert len(step_metrics) == 1
step1_loss = float(step_metrics[0]['loss'])
supervised_tokens = int(step_metrics[0]['supervised_tokens'])
step_wall = float(train_manifest['metrics']['wall_seconds_this_run'])
peak_allocated = int(train_manifest['peak_cuda_allocated_bytes'])

# 3. Run production training/export_peft.py to export click_r8_step1.
cmd_export = [
    sys.executable, '-m', 'training.export_peft',
    '--adapter-dir', str(train_out / 'adapter'),
    '--destination', str(real_adapter_dir),
    '--config', str(cfg_path),
    '--training-manifest', str(train_out / 'training_manifest.json'),
    '--data-fingerprint', sha256_file(train_data_path),
    '--out', str(export_manifest_path),
]
res_export = subprocess.run(cmd_export, cwd=str(BUNDLE), text=True, capture_output=True, check=False)
(OUT / 'export_peft_stdout.log').write_text((res_export.stdout or '') + '\n' + (res_export.stderr or ''))
if res_export.returncode != 0:
    raise RuntimeError(f'export_peft failed ({res_export.returncode}): {res_export.stderr or res_export.stdout}')

# 4. Inspect exported tensors and construct matched zero-delta control adapter (zero_r8_ctrl).
zero_adapter_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(real_adapter_dir / 'adapter_config.json', zero_adapter_dir / 'adapter_config.json')
tensors = load_file(str(real_adapter_dir / 'adapter_model.safetensors'), device='cpu')
zero_tensors = {}
a_norms, b_norms, nonzero_b_count = [], [], 0
for key, tensor in tensors.items():
    t_f32 = tensor.detach().float()
    norm = float(torch.linalg.vector_norm(t_f32).item())
    if '.lora_A.' in key:
        a_norms.append(norm)
        zero_tensors[key] = tensor.clone()
    elif '.lora_B.' in key:
        b_norms.append(norm)
        if norm > 0.0:
            nonzero_b_count += 1
        zero_tensors[key] = torch.zeros_like(tensor)
    else:
        raise RuntimeError(f'Unexpected tensor key: {key}')
assert nonzero_b_count > 0, 'Expected non-zero lora_B weights after 1 Adam step'
save_file(zero_tensors, str(zero_adapter_dir / 'adapter_model.safetensors'))
zero_inspection = inspect_adapter(zero_adapter_dir, expected_rank=8)

# 5. Evaluate no_grad NF4 loss before vs after the step on click (train) and pydicom (select diagnostic).
# Load the cached NF4 base once on cuda:0, attach zero_r8_ctrl (base equivalent) and click_r8_step1,
# and compute exact selected-token loss without gradient updates.
train_rows, _ = load_encoded_jsonl(train_data_path, 16384, require_real=True)
select_rows, _ = load_encoded_jsonl(select_data_path, 16384, require_real=False)
tokenizer = transformers.AutoTokenizer.from_pretrained(str(CT_DIR), local_files_only=True)
pad_id = tokenizer.pad_token_id

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
base_model = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
    str(CACHE_DIR),
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={'': 0},
    attn_implementation='sdpa',
)
base_model = prepare_model_for_memory_efficient_kbit_training(base_model, use_gradient_checkpointing=False)
peft_model = peft.PeftModel.from_pretrained(base_model, str(real_adapter_dir), adapter_name='click_r8_step1', is_trainable=False)
peft_model.load_adapter(str(zero_adapter_dir), adapter_name='zero_r8_ctrl', is_trainable=False)
install_selected_token_loss(peft_model, chunk_size=32)
peft_model.eval()


def eval_no_grad(adapter_name, row):
    peft_model.set_adapter(adapter_name)
    batch = {k: torch.tensor(v, dtype=torch.long, device=device)
             for k, v in pad_encoded_batch([row], pad_id).items()}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = peft_model(**batch, use_cache=False)
    return float(out.loss.detach().cpu().item())


nf4_eval = {
    'click_train_3368_519': {
        'zero_r8_ctrl_loss': eval_no_grad('zero_r8_ctrl', train_rows[0]),
        'train_step1_logged_loss': step1_loss,
        'click_r8_step1_post_loss': eval_no_grad('click_r8_step1', train_rows[0]),
    },
    'pydicom_select_2055_226_no_grad': {
        'zero_r8_ctrl_loss': eval_no_grad('zero_r8_ctrl', select_rows[0]),
        'click_r8_step1_post_loss': eval_no_grad('click_r8_step1', select_rows[0]),
    },
}
nf4_eval['click_train_3368_519']['delta_loss'] = (
    nf4_eval['click_train_3368_519']['click_r8_step1_post_loss']
    - nf4_eval['click_train_3368_519']['zero_r8_ctrl_loss']
)
nf4_eval['pydicom_select_2055_226_no_grad']['delta_loss'] = (
    nf4_eval['pydicom_select_2055_226_no_grad']['click_r8_step1_post_loss']
    - nf4_eval['pydicom_select_2055_226_no_grad']['zero_r8_ctrl_loss']
)

# Remove temporary checkpoint optimizer states to keep output directory small and clean.
for pt_file in train_out.rglob('*.pt'):
    pt_file.unlink()
for ckpt_dir in train_out.glob('checkpoint-*'):
    shutil.rmtree(ckpt_dir, ignore_errors=True)

phase1_report = {
    'status': 'completed',
    'train_cli_wall_seconds': train_wall,
    'optimizer_step_wall_seconds': step_wall,
    'input_tokens': 3368,
    'supervised_tokens': supervised_tokens,
    'supervised_tokens_per_second_in_step': round(supervised_tokens / step_wall, 3) if step_wall > 0 else None,
    'input_tokens_per_second_in_step': round(3368 / step_wall, 3) if step_wall > 0 else None,
    'peak_cuda_allocated_bytes': peak_allocated,
    'peak_cuda_allocated_gib': round(peak_allocated / (1024 ** 3), 4),
    'peak_cuda_reserved_bytes_after_eval': int(torch.cuda.max_memory_reserved(device)),
    'adapter_stats': {
        'module_count': len(b_norms),
        'tensor_count': len(tensors),
        'nonzero_lora_b_modules': nonzero_b_count,
        'lora_a_norm_min': min(a_norms),
        'lora_a_norm_max': max(a_norms),
        'lora_b_norm_min': min(b_norms),
        'lora_b_norm_max': max(b_norms),
        'lora_b_norm_mean': sum(b_norms) / len(b_norms),
        'click_r8_step1_files': train_manifest['adapter']['files'],
        'zero_r8_ctrl_files': zero_inspection['files'],
    },
    'nf4_no_grad_loss_comparison': nf4_eval,
}
(OUT / 'phase1-training-report.json').write_text(json.dumps(phase1_report, indent=2) + '\n')
print(json.dumps({'phase1': 'completed', 'peak_gib': phase1_report['peak_cuda_allocated_gib'],
                  'nf4_eval': nf4_eval}), flush=True)
'''


PHASE2_CHILD = r'''
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request

BUNDLE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CT_DIR = Path(sys.argv[3])

os.environ.update({
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
    'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200',
    'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})

import litellm
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters, discover_declared_models
from adk_eval_core.editing import edit_file_content
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True
TARGET_MODEL_NAME = 'gemma-4-12b-it-qat-w4a16-ct'
export_root = OUT / 'exported_adapters'
real_adapter_dir = export_root / 'click_r8_step1'
zero_adapter_dir = export_root / 'zero_r8_ctrl'

# Stage an official agent directory containing both adapters so discover_adapters registers both.
agent_root = Path('/tmp/p007-12b-ct-agent')
if agent_root.exists():
    shutil.rmtree(agent_root)
(agent_root / 'adapters' / 'click_r8_step1').mkdir(parents=True)
(agent_root / 'adapters' / 'zero_r8_ctrl').mkdir(parents=True)
for name, src in (('click_r8_step1', real_adapter_dir), ('zero_r8_ctrl', zero_adapter_dir)):
    for fname in ('adapter_config.json', 'adapter_model.safetensors'):
        shutil.copy2(src / fname, agent_root / 'adapters' / name / fname)

(agent_root / 'agent.yaml').write_text(
    'name: p007_12b_probe\n'
    f'model: {TARGET_MODEL_NAME}\n'
    'adapter: click_r8_step1\n'
    'instruction: Probe agent.\n'
    'tools:\n'
    '  - run_command\n'
    '  - read_file\n'
    '  - edit_file\n'
    '  - write_file\n'
    '  - get_status\n'
    '  - submit_patch\n'
)

declared_models = sorted(discover_declared_models(agent_root))
assert declared_models == [TARGET_MODEL_NAME], f'Unexpected declared models: {declared_models}'
adapters = discover_adapters(str(agent_root), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_names = sorted(adapters.adapters.keys()) if hasattr(adapters, 'adapters') and isinstance(adapters.adapters, dict) else []
assert adapter_names == ['click_r8_step1', 'zero_r8_ctrl'], f'Unexpected discovered adapters: {adapter_names}'


def http_json(url: str, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='GET' if body is None else 'POST')
    req.add_header('Authorization', 'Bearer EMPTY')
    if body is not None:
        req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


gpu_count = torch.cuda.device_count()
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
vllm_cfg = VllmConfig(
    model=str(CT_DIR),
    port=8000,
    host='127.0.0.1',
    tool_call_parser='gemma4',
    reasoning_parser='gemma4',
    max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90,
    enable_auto_tool_choice=True,
    enable_lora=True,
    max_loras=4,
    max_lora_rank=16,
    tensor_parallel_size=tp_size,
    startup_timeout=60 * 15,
    extra_args=['--max-num-seqs', '4'],
)
server = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_cmd = server.build_cmd()
t_start = time.monotonic()
try:
    server.start()
    startup_seconds = round(time.monotonic() - t_start, 3)
    base_url = str(server.base_url).rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = base_url + '/v1'
    models_resp = http_json(base_url + '/models', timeout=30)
    model_ids = [item['id'] for item in models_resp.get('data', [])]
    base_model_id = str(CT_DIR)
    assert base_model_id in model_ids, f'Base model missing from {model_ids}'
    assert 'click_r8_step1' in model_ids, f'click_r8_step1 missing from {model_ids}'
    assert 'zero_r8_ctrl' in model_ids, f'zero_r8_ctrl missing from {model_ids}'

    # 1. Teacher-forced log-probability comparison across all supervised tokens on official 12B CT v2.
    train_row = json.loads((BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl').read_text().splitlines()[0])
    select_row = json.loads((BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-select-diagnostic.jsonl').read_text().splitlines()[0])

    def evaluate_ct_teacher_forced(model_id, row):
        input_ids = row['input_ids']
        labels = row['labels']
        payload = {
            'model': model_id,
            'prompt': input_ids,
            'max_tokens': 1,
            'temperature': 0.0,
            'prompt_logprobs': 1,
        }
        resp = http_json(base_url + '/completions', body=payload, timeout=180)
        choice = resp['choices'][0]
        prompt_lps = choice['prompt_logprobs']
        assert len(prompt_lps) == len(input_ids), f'Expected {len(input_ids)} prompt_logprobs, got {len(prompt_lps)}'
        sup_logprobs = []
        for idx in range(1, len(input_ids)):
            if labels[idx] == -100:
                continue
            token_id_str = str(input_ids[idx])
            entry = prompt_lps[idx]
            assert entry is not None and token_id_str in entry, f'Missing logprob for token {token_id_str} at pos {idx}'
            val = entry[token_id_str]
            lp = float(val['logprob'] if isinstance(val, dict) else val)
            sup_logprobs.append({'pos': idx, 'token_id': input_ids[idx], 'logprob': lp})
        per_step = []
        for span in row['meta'].get('supervised_spans', []):
            step_lps = [x['logprob'] for x in sup_logprobs if span['token_start'] <= x['pos'] < span['token_end'] and x['pos'] >= 1]
            if step_lps:
                per_step.append({
                    'assistant_step': span.get('assistant_step'),
                    'count': len(step_lps),
                    'sum_logprob': sum(step_lps),
                    'mean_nll': -sum(step_lps) / len(step_lps),
                })
        all_vals = [x['logprob'] for x in sup_logprobs]
        return {
            'supervised_tokens': len(all_vals),
            'sum_logprob': sum(all_vals),
            'mean_nll': -sum(all_vals) / len(all_vals),
            'per_step': per_step,
            'raw_logprobs': all_vals,
        }

    ct_teacher_forced = {}
    for sample_name, row in (('click_train_3368_519', train_row), ('pydicom_select_2055_226', select_row)):
        by_model = {}
        for mid_key, mid in (('base', base_model_id), ('zero_r8_ctrl', 'zero_r8_ctrl'), ('click_r8_step1', 'click_r8_step1')):
            by_model[mid_key] = evaluate_ct_teacher_forced(mid, row)
        base_lps = by_model['base'].pop('raw_logprobs')
        zero_lps = by_model['zero_r8_ctrl'].pop('raw_logprobs')
        real_lps = by_model['click_r8_step1'].pop('raw_logprobs')
        diffs_zero = [abs(z - b) for z, b in zip(zero_lps, base_lps)]
        diffs_real = [r - b for r, b in zip(real_lps, base_lps)]
        ct_teacher_forced[sample_name] = {
            'models': by_model,
            'zero_vs_base_max_abs_logprob_diff': max(diffs_zero),
            'zero_vs_base_mean_abs_logprob_diff': sum(diffs_zero) / len(diffs_zero),
            'real_vs_base_max_abs_logprob_diff': max(abs(d) for d in diffs_real),
            'real_vs_base_mean_abs_logprob_diff': sum(abs(d) for d in diffs_real) / len(diffs_real),
            'real_vs_base_sum_logprob_delta': by_model['click_r8_step1']['sum_logprob'] - by_model['base']['sum_logprob'],
            'real_vs_base_mean_nll_delta': by_model['click_r8_step1']['mean_nll'] - by_model['base']['mean_nll'],
            'improved_token_count': sum(1 for d in diffs_real if d > 1e-6),
            'degraded_token_count': sum(1 for d in diffs_real if d < -1e-6),
            'unchanged_token_count': sum(1 for d in diffs_real if abs(d) <= 1e-6),
        }

    # 2. Live tool-call generation + top-5 logprobs + real workspace execution on click and pydicom.
    tools_schema = json.loads((BUNDLE / 'training/configs/harness_tools.json').read_text())
    click_tx = json.loads((BUNDLE / 'research/006/data-replay-20261013/outputs/click-echo/transaction.json').read_text())
    pydicom_tx = json.loads((BUNDLE / 'research/006/data-replay-20261013/outputs/pydicom-recovery/transaction.json').read_text())
    click_buggy_src = (BUNDLE / 'research/006/data-replay-20261013/sources/click-buggy-utils.py').read_text()
    pydicom_state_src = (BUNDLE / 'research/006/data-replay-20261013/outputs/pydicom-recovery/state-before.py.txt').read_text()

    def normalize_messages_for_openai(messages):
        out = []
        for m in messages:
            msg = {'role': m['role'], 'content': m.get('content') or ''}
            if 'tool_calls' in m:
                tcs = []
                for tc in m['tool_calls']:
                    fn = tc['function']
                    args = fn['arguments']
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    tcs.append({'id': tc['id'], 'type': 'function', 'function': {'name': fn['name'], 'arguments': args}})
                msg['tool_calls'] = tcs
            if 'tool_call_id' in m:
                msg['tool_call_id'] = m['tool_call_id']
            if 'name' in m and m['role'] == 'tool':
                msg['name'] = m['name']
            out.append(msg)
        return out

    # Slice messages right before the target edit_file step:
    # In click_tx, messages[0..5] end with the failing run_command observation; messages[6] is the target edit_file.
    # In pydicom_tx, messages[0..5] end with the read_file(70..82) observation; messages[6] is the target edit_file.
    click_prompt_msgs = normalize_messages_for_openai(click_tx['messages'][:6])
    pydicom_prompt_msgs = normalize_messages_for_openai(pydicom_tx['messages'][:6])

    def execute_edit_and_verify(case_name, tool_calls):
        if not tool_calls:
            return {'called_tool': None, 'edit_ok': False, 'reason': 'no_tool_call'}
        tc = tool_calls[0]
        fn = tc.get('function') or {}
        name = fn.get('name')
        try:
            args = json.loads(fn.get('arguments') or '{}')
        except Exception as exc:
            return {'called_tool': name, 'edit_ok': False, 'reason': f'invalid_json_args: {exc}'}
        if name != 'edit_file':
            return {'called_tool': name, 'arguments': args, 'edit_ok': False, 'reason': 'not_edit_file'}
        old_str = args.get('old_string', '')
        new_str = args.get('new_string', '')
        allow_mult = bool(args.get('allow_multiple', False))
        source_text = click_buggy_src if case_name == 'click_pre_edit' else pydicom_state_src
        edit_res = edit_file_content(source_text, old_str, new_str, allow_multiple=allow_mult)
        if not edit_res.success:
            return {
                'called_tool': name,
                'filepath': args.get('filepath'),
                'edit_ok': False,
                'error': edit_res.error,
                'occurrences': edit_res.occurrences,
            }
        new_content = edit_res.new_content
        syntax_ok = True
        try:
            compile(new_content, args.get('filepath') or '<edited>', 'exec')
        except SyntaxError:
            syntax_ok = False
        result_info = {
            'called_tool': name,
            'filepath': args.get('filepath'),
            'edit_ok': True,
            'occurrences': edit_res.occurrences,
            'strategy': edit_res.strategy,
            'syntax_ok': syntax_ok,
        }
        if case_name == 'click_pre_edit' and syntax_ok:
            # Verify whether the edited utils.py makes file=None a no-op when _default_text_stdout/stderr returns None
            result_info['contains_none_return_guard'] = ('if file is None:' in new_str and 'return' in new_str)
        elif case_name == 'pydicom_recovery' and syntax_ok:
            result_info['updated_cid_dict_annotation'] = ('def __getattr__(self, name: str) -> Union[CodeLike, Code]:\n        """Return the ``Code`` for class attribute `name`."""' in new_content)
        return result_info

    def probe_live_generation(case_name, messages):
        by_model = {}
        for mid_key, mid in (('base', base_model_id), ('zero_r8_ctrl', 'zero_r8_ctrl'), ('click_r8_step1', 'click_r8_step1')):
            payload = {
                'model': mid,
                'messages': messages,
                'tools': tools_schema,
                'temperature': 0.0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'logprobs': True,
                'top_logprobs': 5,
            }
            resp = http_json(base_url + '/chat/completions', body=payload, timeout=180)
            choice = resp['choices'][0]
            msg = choice.get('message') or {}
            lp_content = (choice.get('logprobs') or {}).get('content') or []
            token_lps = [float(item['logprob']) for item in lp_content if 'logprob' in item]
            first_diff_tokens = []
            for idx, item in enumerate(lp_content[:16]):
                top_candidates = [
                    {'token': t.get('token'), 'logprob': round(float(t.get('logprob', 0.0)), 6)}
                    for t in (item.get('top_logprobs') or [])[:3]
                ]
                first_diff_tokens.append({
                    'index': idx,
                    'token': item.get('token'),
                    'logprob': round(float(item.get('logprob', 0.0)), 6),
                    'top3': top_candidates,
                })
            exec_outcome = execute_edit_and_verify(case_name, msg.get('tool_calls'))
            by_model[mid_key] = {
                'routed_model_in_response': resp.get('model'),
                'finish_reason': choice.get('finish_reason'),
                'content': msg.get('content'),
                'reasoning_content': msg.get('reasoning_content'),
                'tool_calls': msg.get('tool_calls'),
                'generated_token_count': len(token_lps),
                'sum_generated_logprob': sum(token_lps) if token_lps else None,
                'mean_generated_logprob': (sum(token_lps) / len(token_lps)) if token_lps else None,
                'first_16_token_logprobs': first_diff_tokens,
                'tool_execution': exec_outcome,
            }
        return by_model

    live_generation = {
        'click_pre_edit': probe_live_generation('click_pre_edit', click_prompt_msgs),
        'pydicom_recovery': probe_live_generation('pydicom_recovery', pydicom_prompt_msgs),
    }

    # Archive server log and parse KV cache / concurrency lines.
    log_path = getattr(server, 'log_path', None)
    kv_metrics = {}
    lora_log_lines = []
    if log_path and Path(log_path).is_file():
        log_text = Path(log_path).read_text(errors='replace')
        (OUT / 'vllm_12b_server.log').write_text(log_text)
        for line in log_text.splitlines():
            if 'Available KV cache memory:' in line:
                kv_metrics['available_kv_cache_line'] = line
            elif 'GPU KV cache size:' in line:
                kv_metrics['gpu_kv_cache_tokens_line'] = line
            elif 'Maximum concurrency for' in line:
                kv_metrics['max_concurrency_line'] = line
            elif 'Estimated CUDA graph memory:' in line:
                kv_metrics['estimated_cudagraph_line'] = line
            if any(k in line.lower() for k in ('lora', 'click_r8_step1', 'zero_r8_ctrl')):
                lora_log_lines.append(line)

    phase2_report = {
        'status': 'completed',
        'vllm_startup_seconds': startup_seconds,
        'server_command': server_cmd,
        'registered_models': model_ids,
        'kv_and_cudagraph_metrics': kv_metrics,
        'lora_log_excerpt': lora_log_lines[-40:],
        'teacher_forced_logprob_comparison': ct_teacher_forced,
        'live_generation_and_tool_execution': live_generation,
    }
    (OUT / 'phase2-ct-probe-report.json').write_text(json.dumps(phase2_report, indent=2) + '\n')
    print(json.dumps({
        'phase2': 'completed',
        'vllm_startup_seconds': startup_seconds,
        'kv_metrics': kv_metrics,
        'click_nll_delta': ct_teacher_forced['click_train_3368_519']['real_vs_base_mean_nll_delta'],
        'pydicom_nll_delta': ct_teacher_forced['pydicom_select_2055_226']['real_vs_base_mean_nll_delta'],
    }), flush=True)
finally:
    try:
        server.stop(timeout=10.0)
    except Exception:
        pass
'''


WRAPPER = r'''
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

PREPARE_ONLY = os.environ.get('P007_12B_PREPARE_ONLY') == '1' or '--prepare-only' in sys.argv
OUT = Path('/kaggle/working/p007-12b-probe')
BUNDLE = Path('/tmp/p007-12b-bundle')
INPUT_ROOT = Path('/kaggle/input')
started = time.monotonic()


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_input(marker: str, substr: str = None) -> Path:
    found = []
    for p in INPUT_ROOT.rglob(marker):
        if substr is None or substr in str(p):
            found.append(p.parent)
    unique = sorted(set(found))
    if len(unique) != 1:
        raise RuntimeError(f'Expected 1 input matching marker={marker} substr={substr}, found {unique}')
    return unique[0]


def main():
    if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux':
        raise RuntimeError(f'Requires Starter Linux Python 3.12; got {sys.version}')
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'study': 'P007 12B QAT real-sample NF4 training, PEFT export, and official 12B CT v2 base/zero-ctrl/adapter comparison',
        'mode': 'cpu_prepare' if PREPARE_ONLY else 'gpu_train_and_ct_probe',
        'status': 'started',
        'python': sys.version,
        'platform': platform.platform(),
        'events': [],
    }

    def event(stage, **kw):
        report.update(kw)
        report['events'].append({'stage': stage, 'elapsed_seconds': round(time.monotonic() - started, 3)})
        write_json(OUT / 'summary-report.json', report)
        print(json.dumps({'stage': stage, **kw}), flush=True)

    # 1. Unpack and verify embedded repository bundle.
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    (BUNDLE / 'training').mkdir(parents=True)
    (BUNDLE / 'training/__init__.py').write_text('')
    (BUNDLE / 'training/encoding').mkdir(parents=True)
    (BUNDLE / 'training/encoding/__init__.py').write_text('')
    for rel, content in BUNDLE_FILES.items():
        raw = content.encode('utf-8')
        expected = SOURCE_MANIFEST['files'][rel]
        assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], f'Hash mismatch: {rel}'
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
    write_json(OUT / 'source-manifest.json', SOURCE_MANIFEST)
    event('bundle_unpacked', files=len(BUNDLE_FILES))

    # 2. Resolve mounted inputs: wheelhouse v31, 12B NF4 cache, and 12B CT v2 model.
    wheelhouse_dir = resolve_input('swegemma-0.2.11-py3-none-any.whl')
    cache_dir = resolve_input('cache-manifest.json', 'qat-12b-nf4')
    ct_dir = resolve_input('tokenizer.json', 'gemma-4-12b-it-qat-w4a16-ct/2')
    extra_wheels_manifest_path = cache_dir.parent / 'training-wheels-manifest.json'
    assert extra_wheels_manifest_path.is_file(), 'Missing training-wheels-manifest.json next to qat-12b-nf4'
    event('inputs_resolved', wheelhouse_dir=str(wheelhouse_dir), cache_dir=str(cache_dir), ct_dir=str(ct_dir))

    # 3. Stage and install wheelhouse v31 + peft/accelerate/bitsandbytes wheels without duplicate distributions.
    staging = Path('/tmp/p007-12b-wheels')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged_by_dist = {}
    for w in sorted(wheelhouse_dir.glob('*.whl')):
        if 'cutlass' in w.name.lower():
            continue
        name = w.name.replace('cu128', '+cu128') if 'cu128' in w.name and '+' not in w.name else w.name
        dist = name.split('-')[0].replace('_', '-').lower()
        target = staging / name
        target.symlink_to(w)
        staged_by_dist[dist] = str(target)
    extra_manifest = json.loads(extra_wheels_manifest_path.read_text())
    for item in extra_manifest['wheels']:
        if item['name'].startswith(('peft-', 'accelerate-', 'bitsandbytes-')):
            wpath = cache_dir.parent / 'training-wheels' / item['name']
            assert sha256_file(wpath) == item['sha256'], f"Extra wheel SHA mismatch: {item['name']}"
            dist = item['name'].split('-')[0].replace('_', '-').lower()
            if dist in staged_by_dist:
                old_target = Path(staged_by_dist[dist])
                if old_target.exists() or old_target.is_symlink():
                    old_target.unlink()
            target = staging / item['name']
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(wpath)
            staged_by_dist[dist] = str(target)
    staged_wheels = sorted(staged_by_dist.values())
    pip_res = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', '--no-index', *staged_wheels],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
    )
    if pip_res.returncode != 0:
        raise RuntimeError(f'pip install failed ({pip_res.returncode}):\n{pip_res.stdout}')
    event('wheels_installed', wheel_count=len(staged_wheels))

    # 4. Run dry-run and provenance validation on the mounted 12B NF4 cache & CT v2 assets.
    dry = subprocess.run(
        [
            sys.executable, '-m', 'training.train_sft',
            '--config', str(BUNDLE / 'training/configs/sft_12b_qat_nf4.yaml'),
            '--model', str(cache_dir),
            '--tokenizer', str(ct_dir),
            '--data', str(BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl'),
            '--dry-run',
        ],
        cwd=str(BUNDLE),
        check=True,
        capture_output=True,
        text=True,
    )
    dry_info = json.loads(dry.stdout)

    # Also run validate_pinned_model_source + validate_nf4_cache + validate_tokenizer_protocol in Python.
    sys.path.insert(0, str(BUNDLE))
    import torch
    import transformers
    from training.train_sft import (
        source_options,
        validate_config,
        validate_nf4_cache,
        validate_pinned_model_source,
        validate_tokenizer_protocol,
    )
    from training.common import load_config, load_encoded_jsonl

    raw_cfg = load_config(BUNDLE / 'training/configs/sft_12b_qat_nf4.yaml')
    raw_cfg['model_name_or_path'] = str(cache_dir)
    raw_cfg['tokenizer_name_or_path'] = str(ct_dir)
    raw_cfg['model_source_manifest'] = str(BUNDLE / 'training/configs/gemma4_12b_sources.json')
    raw_cfg['chat_template_path'] = str(BUNDLE / 'training/configs/chat_templates/gemma4_12b_ct_v2.jinja')
    raw_cfg['tool_schema_path'] = str(BUNDLE / 'training/configs/harness_tools.json')
    cfg = validate_config(raw_cfg)
    rows, data_info = load_encoded_jsonl(
        BUNDLE / 'research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl',
        cfg['max_sequence_length'],
    )
    src, opts = source_options(cfg['model_name_or_path'], cfg['model_revision'], cfg['local_files_only'])
    tsrc, topts = source_options(cfg['tokenizer_name_or_path'], cfg['tokenizer_revision'], cfg['local_files_only'])
    pinned = validate_pinned_model_source(src, opts, tsrc, topts, cfg, transformers)
    mcfg = transformers.AutoConfig.from_pretrained(src, **opts)
    cache_check = validate_nf4_cache(src, mcfg, cfg)
    tok = transformers.AutoTokenizer.from_pretrained(tsrc, **topts)
    proto_check = validate_tokenizer_protocol(tok, data_info['provenance'], cfg)
    event('provenance_verified', cache_check=cache_check, protocol_check=proto_check)

    if PREPARE_ONLY:
        assert not torch.cuda.is_available(), 'CPU prepare must not allocate GPU'
        event('prepared', status='prepared', success=True, model_requests=0, optimizer_steps=0)
        return

    assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Requires 4xL4 GPU session'
    assert all('L4' in torch.cuda.get_device_properties(i).name for i in range(4))

    # 5. Execute Phase 1 (NF4 Training on click + PEFT Export + Zero-Delta Control + NF4 Pre/Post Eval) in Child Process 1.
    phase1_script = Path('/tmp/p007_12b_phase1.py')
    phase1_script.write_text(PHASE1_CHILD)
    event('phase1_start')
    p1 = subprocess.run(
        [sys.executable, '-u', str(phase1_script), str(BUNDLE), str(OUT), str(cache_dir), str(ct_dir)],
        cwd=str(BUNDLE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (OUT / 'phase1.log').write_text(p1.stdout or '')
    print(p1.stdout or '', flush=True)
    if p1.returncode != 0:
        raise RuntimeError(f'Phase 1 failed with exit code {p1.returncode}')
    phase1_report = json.loads((OUT / 'phase1-training-report.json').read_text())
    event('phase1_completed', phase1_summary=phase1_report)

    # 6. Execute Phase 2 (Official 12B CT v2 vLLM Server + Base vs Zero-Ctrl vs Real Adapter Probe) in Child Process 2.
    phase2_script = Path('/tmp/p007_12b_phase2.py')
    phase2_script.write_text(PHASE2_CHILD)
    event('phase2_start')
    p2 = subprocess.run(
        [sys.executable, '-u', str(phase2_script), str(BUNDLE), str(OUT), str(ct_dir)],
        cwd=str(BUNDLE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1500,
    )
    (OUT / 'phase2.log').write_text(p2.stdout or '')
    print(p2.stdout or '', flush=True)
    if p2.returncode != 0:
        raise RuntimeError(f'Phase 2 failed with exit code {p2.returncode}')
    phase2_report = json.loads((OUT / 'phase2-ct-probe-report.json').read_text())
    event('completed', status='completed', success=True, phase2_summary=phase2_report)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        (OUT / 'fatal_exception.txt').write_text(tb)
        print(tb, flush=True)
        raise SystemExit(1)
'''


def build():
    bundle_files = {}
    manifest_files = {}
    for rel in EMBEDDED_PATHS:
        path = ROOT / rel
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        if path.suffix == '.py':
            ast.parse(text, filename=rel)
        bundle_files[rel] = text
        manifest_files[rel] = {'bytes': len(raw), 'sha256': sha(raw)}
    ast.parse(PHASE1_CHILD, filename='phase1_child.py')
    ast.parse(PHASE2_CHILD, filename='phase2_child.py')
    ast.parse(WRAPPER, filename='wrapper.py')
    source_manifest = {
        'study': 'P007 12B QAT real-sample training, PEFT export, and official 12B CT v2 verification',
        'image_pin': IMAGE,
        'cache_kernel': 'ccjjllt/p007-12b-cpu-nf4-cache',
        'deployment_model': 'google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2',
        'files': manifest_files,
    }
    full_source = (
        '# Generated by research/007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py\n'
        + 'BUNDLE_FILES = ' + repr(bundle_files) + '\n'
        + 'SOURCE_MANIFEST = ' + repr(source_manifest) + '\n'
        + 'PHASE1_CHILD = ' + repr(PHASE1_CHILD) + '\n'
        + 'PHASE2_CHILD = ' + repr(PHASE2_CHILD) + '\n'
        + WRAPPER
    )
    ast.parse(full_source, filename='probe_12b.py')
    KERNEL.mkdir(parents=True, exist_ok=True)
    CPU_KERNEL.mkdir(parents=True, exist_ok=True)
    (KERNEL / 'probe_12b.py').write_text(full_source)
    (CPU_KERNEL / 'prepare_12b.py').write_text("import os\nos.environ['P007_12B_PREPARE_ONLY'] = '1'\n" + full_source)
    (HERE / 'source-manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')

    base_meta = {
        'id': 'ccjjllt/p007-12b-train-and-ct-probe',
        'title': 'P007 12B Train And CT Probe',
        'code_file': 'probe_12b.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': ['ccjjllt/p007-12b-cpu-nf4-cache'],
        'model_sources': ['google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2'],
        'docker_image': IMAGE,
        'docker_image_pinning_type': 'original',
    }
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(base_meta, indent=2) + '\n')
    cpu_meta = dict(
        base_meta,
        id='ccjjllt/p007-12b-train-and-ct-probe-cpu-prepare',
        title='P007 12B Train And CT Probe CPU Prepare',
        code_file='prepare_12b.py',
        enable_gpu=False,
    )
    (CPU_KERNEL / 'kernel-metadata.json').write_text(json.dumps(cpu_meta, indent=2) + '\n')
    print(json.dumps({
        'gpu_script_sha256': sha(full_source.encode()),
        'embedded_files': len(bundle_files),
        'gpu_kernel': str(KERNEL.relative_to(ROOT)),
        'cpu_kernel': str(CPU_KERNEL.relative_to(ROOT)),
    }))


if __name__ == '__main__':
    build()
