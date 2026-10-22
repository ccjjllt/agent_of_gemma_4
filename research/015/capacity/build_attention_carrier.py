"""Freeze one bounded GPU attention experiment; building never launches it."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    config_path = ROOT / 'research/005/kaggle-training-probe-20261010/cpu-cache-output/qat-nf4/config.json'
    config = json.loads(config_path.read_text())['text_config']
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    attention = (ROOT / 'training/attention.py').read_text()
    nodes = ast.parse(attention)
    names = ('grouped_global_sdpa', 'gemma4_grouped_global_attention_forward')
    functions = '\n\n'.join(ast.get_source_segment(attention, node)
        for node in nodes.body if isinstance(node, ast.FunctionDef) and node.name in names)
    body = (HERE / 'attention_run.py').read_text()
    marker = '# CONFIG, CONFIG_SHA256 and grouped_attention are embedded in the runnable carrier.'
    embedded = (functions + '\n\ngrouped_attention = gemma4_grouped_global_attention_forward\n'
                + 'CONFIG = ' + repr(config) + '\nCONFIG_SHA256 = ' + repr(config_sha))
    source = body.replace(marker, embedded)
    ast.parse(source)
    kernel = HERE / 'attention-kernel'
    kernel.mkdir(exist_ok=True)
    (kernel / 'run.py').write_text(source)
    metadata = {
        'id': 'zhuating/p015-attention-capacity',
        'title': 'P015 Attention Capacity', 'code_file': 'run.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'dataset_sources': [], 'model_sources': [],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': ['ccjjllt/p005-training-cpu-stage'],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'docker_image_pinning_type': 'original',
    }
    (kernel / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = {'purpose': 'global attention forward/backward capacity; no model weights',
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'source_bytes': len(source.encode()), 'config_sha256': config_sha,
        'attention_source_sha256': hashlib.sha256(attention.encode()).hexdigest(),
        'platform_seconds': 840, 'worker_timeout_seconds': 600,
        'quota_estimate_upper_hours_at_2x': 840 * 2 / 3600,
        'launch_command': ['kaggle', 'kernels', 'push', '-p', str(kernel.relative_to(ROOT)),
                           '-t', '840', '--accelerator', 'NvidiaL4']}
    (HERE / 'attention-carrier-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
