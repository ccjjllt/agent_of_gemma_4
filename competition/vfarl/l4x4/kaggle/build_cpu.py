"""Freeze CPU preparation source and exact selected task environment profiles."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
source = (HERE / 'cpu_stage.py').read_text()
profiles = json.loads((HERE / 'environment-profiles.json').read_text())
source = 'ENVIRONMENT_PROFILES = ' + repr(profiles) + '\n' + source
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--kernel-id', default='qingyu1232/vfarl-l4x4-cpu-stage')
parser.add_argument('--title', default='VFARL L4x4 CPU Stage')
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=False)
(out / 'run.py').write_text(source)
metadata = {
    'id': args.kernel_id, 'title': args.title,
    'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
    'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True,
    'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
    'competition_sources': ['gemma-4-developer-agent'],
    'kernel_sources': ['ccjjllt/p005-training-cpu-stage', 'ccjjllt/p005-cpu-nf4-cache'],
    'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'],
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
    'docker_image_pinning_type': 'original',
}
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps({'kernel': metadata['id'], 'gpu': False,
                  'source_sha256': hashlib.sha256(source.encode()).hexdigest()}))
