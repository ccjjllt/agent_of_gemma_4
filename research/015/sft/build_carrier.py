"""Package a real curated train dataset; never starts a GPU or invents a result."""
import argparse
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

from training.common import load_config, load_encoded_jsonl
from training.train_sft import validate_config
from training.encoding.gemma_tool_chat import normalize_tools, sha256_json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--max-sequence-length', type=int, default=8192)
    parser.add_argument('--max-steps', type=int, default=16)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_sequence_length < 2:
        raise ValueError('SFT packaging requires a positive step cap and sequence length >=2')
    rows, info = load_encoded_jsonl(args.data, args.max_sequence_length)
    data = args.data.read_text()
    dataset = json.loads(args.dataset_manifest.read_text())
    if dataset.get('encoded_data_sha256') != info['sha256']:
        raise ValueError('Curated dataset manifest must name these exact encoded bytes')
    issues = sorted({row['meta']['group']['issue'] for row in rows})
    if any(row['meta']['group']['split'] != 'train' for row in rows):
        raise ValueError('Only explicitly assigned train records may update parameters')
    tools = args.tools.read_text()
    actual_tools, tool_protocol = normalize_tools(json.loads(tools))
    if sha256_json(actual_tools) != info['provenance']['tools_hash']:
        raise ValueError('Actual runtime tools differ from encoded training provenance')
    if tool_protocol and tool_protocol != info['provenance']['protocol_id']:
        raise ValueError('Actual tool manifest protocol differs from encoded training provenance')
    if (dataset.get('actual_tools_file_sha256')
            and sha(args.tools.read_bytes()) != dataset['actual_tools_file_sha256']):
        raise ValueError('Actual tools file differs from the curated dataset manifest')
    cfg = validate_config(load_config(ROOT/'training/configs/sft_p015_31b_fsdp.yaml'))
    cfg.update(protocol_id=info['provenance']['protocol_id'],
               max_sequence_length=args.max_sequence_length, gradient_accumulation_steps=2,
               checkpoint_every_n_steps=4, max_steps=args.max_steps, warmup_ratio=0.0,
               epochs=1)
    cfg = validate_config(cfg)
    paths = ['training/train_sft.py', 'training/train_ddp.py', 'training/train_fsdp.py',
             'training/fsdp_qlora.py', 'training/common.py', 'training/export_peft.py',
             'training/memory.py', 'training/encoding/gemma_tool_chat.py']
    sources = {path: (ROOT/path).read_text() for path in paths}
    # This bundle path deliberately contains the actual captured runtime schema;
    # the repository's older diagnostic schema is not silently reused.
    sources['training/configs/harness_tools.json'] = tools
    sources.update({'training/__init__.py': '', 'training/encoding/__init__.py': ''})
    manifest = {
        'purpose': 'verified on-policy action RFT; not full-task SFT or repair gain',
        'source_data_sha256': info['sha256'], 'selected_data_sha256': sha(data.encode()),
        'dataset_manifest_sha256': sha(args.dataset_manifest.read_bytes()),
        'dataset_manifest': dataset, 'actual_tools_file_sha256': sha(tools.encode()),
        'independent_train_issues': issues, 'encoded_records': len(rows),
        'supervised_tokens': sum(sum(x != -100 for x in row['labels'][1:]) for row in rows),
        'max_actual_sequence_length': max(len(row['input_ids']) for row in rows),
        'max_sequence_length_setting': args.max_sequence_length,
        'planned_optimizer_steps': min(args.max_steps, math.ceil(math.ceil(len(rows)/4)/2)),
        'epochs': 1, 'gradient_accumulation_steps': 2, 'micro_batch_per_rank': 1,
        'world_size': 4, 'grouped_attention': False,
        'source_sha256': {path: sha(text.encode()) for path, text in sources.items()},
        'worker_timeout_seconds': 1500, 'platform_seconds': 1800,
        'quota_upper_hours_at_2x': 1.0,
        'capacity_scope': '6846 already updated; longer real rows must actually complete, not inferred',
    }
    payload = json.dumps({'sources': sources, 'data': data, 'config': cfg,
                          'manifest': manifest}, ensure_ascii=False).encode()
    bundle = base64.b64encode(zlib.compress(payload, level=9)).decode()
    source = (HERE/'run.py').read_text().replace('# BUNDLE is embedded at packaging time.',
                                                 'BUNDLE = '+repr(bundle))
    ast.parse(source)
    if len(source.encode()) >= 1000000:
        raise ValueError('Carrier exceeds one MB; stage exact data separately before launching')
    kernel = HERE/'kernel'; kernel.mkdir(exist_ok=True)
    (kernel/'run.py').write_text(source)
    metadata = {
        'id':'zhuating/p015-31b-task-rft', 'title':'P015 31B Task RFT',
        'code_file':'run.py', 'language':'python', 'kernel_type':'script', 'is_private':True,
        'enable_gpu':True, 'enable_tpu':False, 'enable_internet':False,
        'dataset_sources':[], 'model_sources':[],
        'competition_sources':['gemma-4-developer-agent'],
        'kernel_sources':['ccjjllt/p005-training-cpu-stage', 'ccjjllt/p005-cpu-nf4-cache',
                          'zhuating/p015-31b-tokenizer-cpu'],
        'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'docker_image_pinning_type':'original',
    }
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    manifest.update(carrier_sha256=sha(source.encode()), carrier_bytes=len(source.encode()))
    (HERE/'carrier-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'dataset_manifest'}, indent=2))


if __name__ == '__main__':
    main()
