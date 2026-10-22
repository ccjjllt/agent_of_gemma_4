"""Freeze one offline train-only capacity/action carrier; never starts a GPU."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from training.common import load_config, load_encoded_jsonl
from training.train_sft import validate_config

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    source = ROOT/'research/015/data/encoded-lego-31b.jsonl'
    rows, info = load_encoded_jsonl(source, 8192)
    short = sorted((row for row in rows if len(row['input_ids']) <= 4203),
                   key=lambda row: (-len(row['input_ids']), row['meta']['sample_id']))
    long = [row for row in rows if len(row['input_ids']) == 6846]
    recovery = [row for row in short if row['meta']['sample_id'].endswith('sqlfluff-4151')]
    if len(long) != 1 or len(recovery) != 1 or len(short[0]['input_ids']) != 4203:
        raise ValueError('The frozen capacity source no longer has expected complete train transactions')
    chosen = short[:7]
    if recovery[0] not in chosen:
        chosen.append(recovery[0])
    else:
        chosen.append(short[7])
    chosen.append(long[0])
    if len(chosen) != 9 or len({row['meta']['sample_id'] for row in chosen}) != 9:
        raise ValueError('Expected nine independent train issues')
    data = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n'
                   for row in chosen)
    cfg = validate_config(load_config(ROOT/'training/configs/sft_p015_31b_fsdp.yaml'))
    cfg.update(max_sequence_length=6846, gradient_accumulation_steps=1,
               checkpoint_every_n_steps=1, max_steps=3, warmup_ratio=0.0)
    paths = ['training/train_sft.py', 'training/train_ddp.py', 'training/train_fsdp.py',
             'training/fsdp_qlora.py', 'training/fsdp_checkpoint.py', 'training/common.py', 'training/export_peft.py',
             'training/memory.py', 'training/encoding/gemma_tool_chat.py',
             'training/configs/harness_tools.json']
    sources = {path: (ROOT/path).read_text() for path in paths}
    sources.update({'training/__init__.py': '', 'training/encoding/__init__.py': ''})
    manifest = {'purpose': 'FSDP capacity/action-only; not autonomous task SFT or repair benefit',
        'source_data_sha256': info['sha256'], 'selected_data_sha256': digest(data.encode()),
        'grouped_attention': False, 'max_sequence_length_setting': 6846,
        'planned_optimizer_steps': 3, 'micro_batch_per_rank': 1,
        'gradient_accumulation_steps': 1, 'world_size': 4,
        'batch_plan': 'length-grouped two full four-issue batches (including4203), then6846 one-issue tail; empty ranks zero-weight duplicate',
        'selected_issues': [{'sample_id': row['meta']['sample_id'],
                            'length': len(row['input_ids']),
                            'supervised_tokens': sum(x != -100 for x in row['labels'][1:]),
                            'group': row['meta']['group']} for row in chosen],
        'source_sha256': {path: digest(text.encode()) for path, text in sources.items()},
        'worker_timeout_seconds': 2100, 'platform_seconds': 2400,
        'quota_upper_hours_at_2x': 2400*2/3600,
        'initialization': 'each rank loads original uint8 NF4 cache before byte-preserving repack and FSDP shard'}
    bundle = base64.b64encode(zlib.compress(json.dumps({'sources': sources, 'data': data,
        'config': cfg, 'manifest': manifest}, ensure_ascii=False).encode(), level=9)).decode()
    body = (HERE/'fsdp_run.py').read_text().replace('# BUNDLE is embedded at packaging time.',
                                                  'BUNDLE = '+repr(bundle))
    ast.parse(body)
    if len(body.encode()) >= 1000000:
        raise ValueError('Carrier exceeds requested one MB')
    kernel = HERE/'fsdp-kernel'
    kernel.mkdir(exist_ok=True)
    (kernel/'run.py').write_text(body)
    metadata = {'id': 'zhuating/p015-fsdp-capacity', 'title': 'P015 FSDP Capacity',
        'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'dataset_sources': [], 'model_sources': [],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': ['ccjjllt/p005-training-cpu-stage', 'ccjjllt/p005-cpu-nf4-cache',
                           'zhuating/p015-31b-tokenizer-cpu'],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'docker_image_pinning_type': 'original'}
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    manifest.update(carrier_sha256=digest(body.encode()), carrier_bytes=len(body.encode()),
        launch_command=['kaggle','kernels','push','-p',str(kernel.relative_to(ROOT)),
                        '-t','2400','--accelerator','NvidiaL4'])
    (HERE/'fsdp-carrier-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
