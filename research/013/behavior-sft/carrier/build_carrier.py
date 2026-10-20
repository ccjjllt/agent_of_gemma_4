"""Freeze the 34-issue P013 behavior dataset and package one continuous SFT carrier.

Building is CPU-only and never starts a Kaggle session. Code-only bundle
(~tens of KB); the ~2MB encoded file travels as a versioned private dataset
(P012 v4 mode: small bootstrap + data via dataset mount).
"""
import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = HERE.relative_to(ROOT)
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
EXPECTED_NF4_CACHE_MANIFEST = 'e7cde8010895cf3fc4928701c9346ac442c05be97e34c97478b9d6e97eeff4ca'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path,
                        default=ROOT / 'research/013/behavior-sft/dataset-manifest.json')
    parser.add_argument('--platform-wall-seconds', type=int, default=7200)
    parser.add_argument('--training-stage-seconds', type=int, default=5400)
    parser.add_argument('--kernel-id', default='SFT_OWNER/p013-12b-behavior-sft-34')
    parser.add_argument('--dataset-slug', default='SFT_OWNER/p013-behavior-data-34')
    args = parser.parse_args()
    manifest_path = args.dataset_manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit('Freeze the P013 behavior train issues first; dataset manifest missing: ' + str(manifest_path))
    if not manifest_path.is_relative_to(ROOT):
        raise SystemExit('Dataset manifest must be inside this repository')
    if not 600 < args.training_stage_seconds <= args.platform_wall_seconds - 600:
        raise SystemExit('Reserve at least 600 seconds outside training stage for installation/cleanup')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('p013_stage', HERE / 'training_stage.py')
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    data_path, rows, data = stage.load_frozen_data(ROOT, manifest_path)
    if data['data']['provenance']['protocol_id'] != 'p013-behavior-v1':
        raise SystemExit('Carrier admits the p013-behavior-v1 protocol only')
    steps = data['planned_optimizer_steps']
    if steps != 8 * math.ceil(len(rows) / 4) or steps != 72:
        raise SystemExit(f'P013 plan is 34 rows x 8 epochs = 72 steps; got {steps}')
    pins = json.loads((HERE / 'runtime-pins.json').read_text())
    if pins.get('nf4_cache_manifest_sha256') != EXPECTED_NF4_CACHE_MANIFEST:
        raise SystemExit('P007 immutable NF4 cache manifest changed; re-verify ccjjllt/p007-12b-cpu-nf4-cache')
    point_training = steps * (100.37 / 4)
    # This is a planning estimate, not a new throughput measurement. Old work had
    # max length 6089; new length/label distributions and grouping can change cost.
    if point_training + 720 > args.training_stage_seconds:
        raise SystemExit('Requested time envelope is below P009 step-rate estimate + 12min overhead; '
                         'increase --training-stage-seconds and --platform-wall-seconds explicitly')
    lengths = sorted(data['input_lengths'])
    estimate = {
        'reference': 'P009 four L4: 100.37 seconds / 4 optimizer steps; max length 6089',
        'measured_reference_seconds_per_step': 100.37 / 4,
        'optimizer_compute_point_seconds': round(point_training, 2),
        'planning_wall_seconds_1x_to_2x_plus_720': [round(point_training + 720, 2), round(2 * point_training + 720, 2)],
        'assumptions': ['Same model, rank8, global batch4 and selected-chunked loss',
                        'Loading/environment/export allowance 720s, including about 6min model loading',
                        '1x-2x is a planning range; not a measured upper bound for new data',
                        'Sequence lengths, slowest rank and supervised-label density all affect runtime',
                        'Length grouping has not demonstrated GPU throughput gain'],
        'actual_data_input_tokens': data['data']['input_tokens'],
        'actual_data_supervised_tokens': data['data']['supervised_tokens_after_shift'],
        'actual_lengths': {'min': lengths[0], 'mean': sum(lengths) / len(lengths),
                           'p50_nearest_rank': lengths[math.ceil(.5 * len(lengths)) - 1],
                           'p95_nearest_rank': lengths[math.ceil(.95 * len(lengths)) - 1], 'max': lengths[-1]},
        'quota_note': 'If four-L4 account charging remains 2x wall time, multiply hours by 2; account billing is checked at launch.'}
    data_identity = {
        'dataset_slug': args.dataset_slug,
        'dataset_manifest_sha256': stage.sha256(manifest_path),
        'encoded_data_sha256': stage.sha256(data_path),
        'manifest_bytes': manifest_path.stat().st_size,
        'encoded_bytes': data_path.stat().st_size,
    }
    run = {
        'study': 'P013 behavior five-tool SFT', 'kernel': args.kernel_id,
        'dataset_slug': data_identity['dataset_slug'],
        'dataset_manifest_sha256': data_identity['dataset_manifest_sha256'],
        'encoded_data_sha256': data_identity['encoded_data_sha256'],
        'rows': len(rows), 'independent_issues': data['independent_issues'],
        'independent_repos': data['independent_repos'], 'repo_counts': data['repo_counts'],
        'epochs': 8, 'global_batch_size': 4, 'planned_optimizer_steps': steps,
        'ddp_length_grouped_batches': True,
        'platform_wall_seconds': args.platform_wall_seconds,
        'worker_wall_seconds': args.platform_wall_seconds - 120,
        'training_stage_max_seconds': args.training_stage_seconds,
        'diagnostic_runs_planned': 0, 'submission_planned': False,
        'image_pin': IMAGE, 'runtime_pins_sha256': stage.sha256(HERE / 'runtime-pins.json'),
        'nf4_cache_manifest_sha256': EXPECTED_NF4_CACHE_MANIFEST,
        'nf4_cache_kernel': 'ccjjllt/p007-12b-cpu-nf4-cache',
        'new_lora_optimizer': True, 'resume': None,
        'estimate': estimate,
        'data_identity': data_identity,
    }
    stage.write_json(HERE / 'run-manifest.json', run)
    paths = [PREFIX / name for name in ('training_stage.py', 'runner.py', 'supervisor.py',
                                        'runtime-pins.json', 'run-manifest.json')]
    paths += [Path('training') / name for name in (
        'train_sft.py', 'train_ddp.py', 'common.py', 'memory.py', 'attention.py', 'export_peft.py',
        'encoding/gemma_tool_chat.py', 'configs/sft_p013_12b_behavior_ddp.yaml',
        'configs/gemma4_12b_sources.json', 'configs/chat_templates/gemma4_12b_ct_v2.jinja',
        'configs/harness_tools.json')]
    # Five-tool behavior schema (the actual SFT protocol); bundled for provenance.
    # The six-tool harness file above stays as the repo-canonical reference.
    paths += [Path('research/012/agent/repair_tools_v1.json')]
    # Training DATA is NOT embedded (single-file ~1MB push cap; encoded file is ~2MB):
    # it travels as a versioned private dataset (see dataset_sources in kernel
    # metadata) and is hash-verified at runtime against the shas recorded here.
    paths = list(dict.fromkeys(paths))
    carrier = {'image_pin': IMAGE, 'run': run,
               'files': {str(p): {'bytes': (ROOT / p).stat().st_size, 'sha256': stage.sha256(ROOT / p)} for p in paths}}
    manifest_text = json.dumps(carrier, ensure_ascii=False, indent=2) + '\n'
    (HERE / 'carrier-manifest.json').write_text(manifest_text)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            payload = (ROOT / path).read_bytes()
            if path.suffix == '.py':
                ast.parse(payload, filename=str(path))
            archive.writestr(str(path), payload)
        archive.writestr(str(PREFIX / 'carrier-manifest.json'), manifest_text)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    bootstrap = '''# Generated by P013 build_carrier.py. Running this starts one bounded training job.
import base64, hashlib, io, os, runpy, uuid, zipfile
from pathlib import Path
PAYLOAD = {payload!r}
EXPECTED_SHA256 = {digest!r}
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
bundle = Path('/kaggle/working') / ('p013-bundle-' + uuid.uuid4().hex[:8])
bundle.mkdir()
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for member in archive.infolist():
        assert (bundle / member.filename).resolve().is_relative_to(bundle.resolve())
    archive.extractall(bundle)
os.environ['P013_BUNDLE_ROOT'] = str(bundle)
runpy.run_path(str(bundle / {supervisor!r}), run_name='__main__')
'''.format(payload=base64.b64encode(payload).decode(), digest=digest,
           supervisor=str(PREFIX / 'supervisor.py'))
    ast.parse(bootstrap)
    kernel = HERE / 'kernel'
    kernel.mkdir(exist_ok=True)
    (kernel / 'train.py').write_text(bootstrap)
    metadata = {'id': args.kernel_id, 'title': 'P013 12B Behavior SFT 34',
                'code_file': 'train.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse',
                                    args.dataset_slug],
                # L4 eligibility requires competition association. The training
                # runner still reads only its frozen external training data.
                'competition_sources': ['gemma-4-developer-agent'],
                'machine_shape': 'NvidiaL4',
                'kernel_sources': ['ccjjllt/p007-12b-cpu-nf4-cache'],
                'model_sources': ['google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2'],
                'docker_image': IMAGE, 'docker_image_pinning_type': 'original'}
    stage.write_json(kernel / 'kernel-metadata.json', metadata)
    print(json.dumps({'kernel': args.kernel_id, 'issues': len(rows), 'repos': data['independent_repos'],
                      'optimizer_steps': steps, 'compressed_bundle_bytes': len(payload),
                      'platform_seconds': args.platform_wall_seconds,
                      'training_stage_seconds': args.training_stage_seconds,
                      'carrier_sha256': digest,
                      'kernel_files': {name: stage.sha256(kernel / name) for name in
                                       ('train.py', 'kernel-metadata.json')},
                      'estimate': estimate}, ensure_ascii=False))


if __name__ == '__main__':
    main()
