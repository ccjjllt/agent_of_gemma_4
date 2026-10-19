from pathlib import Path
import asyncio
import concurrent.futures
import glob
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile


import signal

def require_supported_runtime():
    # v30 includes cp312-only native wheels. Fail before installing ANY package.
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            'P012 eval requires a fresh Linux CPython 3.12 kernel for wheelhouse v31; '
            f'current={sys.version.split()[0]} executable={sys.executable}. '
            'Select a Python 3.12 Kaggle image/kernel, then rerun. '
            'No packages have been installed. Do not force cp312 wheels into 3.13.')
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('P012 eval wheelhouse v31 requires Linux x86_64; no packages installed.')


require_supported_runtime()

RUN_ID = os.environ['P012_EVAL_RUN_ID']
RUN_ROOT = Path(os.environ['P012_EVAL_RUN_ROOT'])
AGENT_ROOT = Path('/kaggle/working/agent_exec') / RUN_ID
INPUT_ROOT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
TARGET_MODEL_NAME = 'gemma-4-12b-it-qat-w4a16-ct'
SELECTED_VARIANTS = ('a', 'b')
REPEAT_ID = 'r1'
MAX_CONCURRENT = 4
ATTEMPT_WALL_SECONDS = 1020  # 480 agent + 300 judge + up to 240 setup/cleanup.
VERIFICATION_TIMEOUT_SECONDS = 300
PREPARE_ONLY = os.environ.get('P012_EVAL_PREPARE_ONLY') == '1'
EVALUATION_DEADLINE = float(os.environ['P012_EVAL_DEADLINE'])


def set_phase(name, **extra):
    atomic_json(RUN_ROOT / 'phase.json', {'phase': name, 'time': time.time(), **extra})

def input_snapshot(limit=120):
    # Distinguish "no inputs attached" from "wrong path" within the fail-fast window.
    root = Path('/kaggle/input')
    try:
        if not root.is_dir():
            return ['/kaggle/input missing']
        files = sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_file())
        return files[:limit] + ([f'...(+{len(files) - limit} more)'] if len(files) > limit else [])
    except Exception as exc:
        return [f'list-failed: {exc}']


INPUT_ROOT = Path('/kaggle/input')


def resolve_input(candidates, marker, marker_substr=None):
    # Mount layout varies by worker: nested (datasets/metric/…, competitions/…, models/…)
    # or flat (<slug>/). Prefer known candidates, else locate by unique marker file.
    # Defined before first use (module top-level executes in order).
    for cand in candidates:
        if cand.exists():
            print(f'input resolved: {cand}', flush=True)
            return cand
    found = []
    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob(marker):
            if marker_substr is None or marker_substr in str(path):
                found.append(path.parent)
    uniq = sorted(set(found))
    assert len(uniq) == 1, (f'Input {marker} not found: '
                            f'candidates={[str(c) for c in candidates]} scan={uniq[:5]} '
                            f'input_files={input_snapshot()}')
    print(f'input resolved by marker {marker}: {uniq[0]}', flush=True)
    return uniq[0]
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


BUNDLE = %%BUNDLE_PY%%
EXPECTED_CONFIG_HASHES = %%HASHES_PY%%
MANIFEST_TEXT = %%MANIFEST_PY%%
EXPERIMENT = json.loads(MANIFEST_TEXT)
TASK_IDS = %%TASK_IDS_PY%%
MODEL_PIN = {'ref': 'google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2', 'tokenizer_bytes': 32169626, 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'config_sha256': 'ba14713d084391532b285f7c252b954b7c6c0db97f427a9d8fb788edc2949168', 'tokenizer_config_sha256': 'ee66825f1a587ca13bcb90d7e60f59bc99e84f3b8bc0cefdc3ea595cd5abf773'}
WHEELHOUSE_V31_FILES = {'adk_eval_core-0.1.0-py3-none-any.whl': 90060, 'adk_submission-0.2.13-py3-none-any.whl': 66528, 'anthropic-1.4.0-py3-none-any.whl': 1300390, 'apache_tvm_ffi-0.1.13.post3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl': 2955490, 'astor-0.8.1-py2.py3-none-any.whl': 27488, 'bitsandbytes-0.46.1-py3-none-manylinux_2_24_x86_64.whl': 72865511, 'cbor2-6.1.4-cp312-cp312-manylinux_2_28_x86_64.whl': 461193, 'compressed_tensors-0.15.0.1-py3-none-any.whl': 194260, 'depyf-0.20.0-py3-none-any.whl': 39381, 'diskcache-5.6.3-py3-none-any.whl': 45550, 'flashinfer_cubin-0.6.6-py3-none-any.whl': 267661457, 'flashinfer_python-0.6.6-py3-none-any.whl': 7817047, 'gguf-0.19.0-py3-none-any.whl': 118475, 'google_adk-1.36.1-py3-none-any.whl': 2877731, 'google_genai-2.11.0-py3-none-any.whl': 984162, 'ijson-3.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': 149848, 'interegular-0.3.3-py37-none-any.whl': 23635, 'llguidance-1.3.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 3033133, 'lm_format_enforcer-0.11.3-py3-none-any.whl': 45418, 'loguru-0.7.3-py3-none-any.whl': 61595, 'mistral_common-1.11.7-py3-none-any.whl': 6553583, 'model_hosting_container_standards-0.1.16-py3-none-any.whl': 125666, 'msgspec-0.21.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': 225025, 'nvidia_cudnn_frontend-1.18.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': 2159245, 'openai_harmony-0.0.8-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 2953260, 'opentelemetry_exporter_otlp-1.44.0-py3-none-any.whl': 6727, 'opentelemetry_semantic_conventions_ai-0.5.1-py3-none-any.whl': 11250, 'outlines_core-0.2.11-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 2287677, 'partial_json_parser-0.2.1.1.post7-py3-none-any.whl': 10877, 'prometheus_fastapi_instrumentator-8.1.0-py3-none-any.whl': 19649, 'pybase64-1.5.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl': 94681, 'pycountry-26.2.16-py3-none-any.whl': 8044600, 'pydantic_extra_types-2.11.2-py3-none-any.whl': 82851, 'quack_kernels-0.6.4-py3-none-any.whl': 728458, 'safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 506502, 'setproctitle-1.3.7-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl': 32932, 'swegemma-0.2.11-py3-none-any.whl': 119670, 'tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 3274982, 'transformers-5.13.1-py3-none-any.whl': 11503977, 'vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl': 433132704, 'xgrammar-0.2.5.post1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': 48149933}
WORKER_SOURCE = (Path(__file__).parent / 'attempt_worker.py').read_text()


EXPECTED_TASK_META = %%TASK_META_PY%%
VARIANT_CONDITIONS = %%VARIANTS_PY%%
REFERENCE_LICENSE_TEXT = '\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      "License" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      "Licensor" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      "Legal Entity" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      "control" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      "You" (or "Your") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      "Source" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      "Object" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      "Work" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      "Derivative Works" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      "Contribution" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, "submitted"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as "Not a Contribution."\n\n      "Contributor" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a "NOTICE" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an "AS IS" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets "[]"\n      replaced with your own identifying information. (Don\'t include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same "printed page" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the "License");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an "AS IS" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.'

set_phase('setup')
DATA_DIR = Path(os.environ['P012_EVAL_DATA_DIR']) if os.environ.get('P012_EVAL_DATA_DIR') else resolve_input([], 'eval-data-marker.json', 'p012-eval-tasks')
# Datasets uploaded with --dir-mode zip may arrive as <name>.zip instead of <name>/
# directories (the files API lists zip contents virtually, but the kernel mount
# may not extract them). Unpack into scratch when only the zip is present.
for _dirname in ('snapshots', 'deps'):
    _target = DATA_DIR / _dirname
    _zipped = DATA_DIR / (_dirname + '.zip')
    if not _target.is_dir() and _zipped.is_file():
        import zipfile as _zipfile
        _target.mkdir(parents=True)
        with _zipfile.ZipFile(_zipped) as _archive:
            for _member in _archive.infolist():
                _dest = (_target / _member.filename).resolve()
                assert _dest.is_relative_to(_target.resolve())
            _archive.extractall(_target)
        print(f'extracted {_zipped.name} -> {_target}', flush=True)
# Dir-mode zip extraction may nest one level down (snapshots/snapshots/*.tgz);
# collapse a single wrapper subdir that holds the payload.
for _dirname in ('snapshots', 'deps'):
    _target = DATA_DIR / _dirname
    if _target.is_dir():
        _kids = sorted(p for p in _target.iterdir())
        if (len(_kids) == 1 and _kids[0].is_dir()
                and not list(_target.glob('*.whl')) and not list(_target.glob('*.tgz'))):
            _inner = _kids[0]
            for _item in list(_inner.iterdir()):
                _item.rename(_target / _item.name)
            _inner.rmdir()
            print(f'collapsed wrapper dir {_inner.name} in {_target}', flush=True)
WHEELHOUSE_DIR = resolve_input([INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
                               INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
                              'swegemma-0.2.11-py3-none-any.whl')
MODEL_PATH = resolve_input([INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2',
                            INPUT_ROOT / 'gemma-4-12b-it-qat-w4a16-ct/2'],
                           'tokenizer.json', 'gemma-4-12b-it-qat-w4a16-ct/2')
assert MODEL_PATH.name == '2' and 'gemma-4-12b-it-qat-w4a16-ct' in str(MODEL_PATH)
tokenizer = MODEL_PATH / 'tokenizer.json'
assert tokenizer.stat().st_size == MODEL_PIN['tokenizer_bytes']
assert sha256_file(tokenizer) == MODEL_PIN['tokenizer_sha256']
assert sha256_file(MODEL_PATH / 'config.json') == MODEL_PIN['config_sha256']
assert sha256_file(MODEL_PATH / 'tokenizer_config.json') == MODEL_PIN['tokenizer_config_sha256']
assert not AGENT_ROOT.exists()
AGENT_ROOT.mkdir(parents=True)
FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
atomic_text(FROZEN_DIR / 'experiment_manifest.json', MANIFEST_TEXT)
atomic_json(FROZEN_DIR / 'embedded_config_sha256.json', EXPECTED_CONFIG_HASHES)
atomic_json(FROZEN_DIR / 'model_pin.json', MODEL_PIN)
for variant, file_map in BUNDLE.items():
    for relative, content in file_map.items():
        assert hashlib.sha256(content.encode()).hexdigest() == EXPECTED_CONFIG_HASHES[variant][relative]
        atomic_text(FROZEN_DIR / 'variants' / variant / relative, content)
        atomic_text(AGENT_ROOT / variant / relative, content)

schedule = []
for slot, task_id in enumerate(TASK_IDS, 1):
    order = SELECTED_VARIANTS if slot % 2 else tuple(reversed(SELECTED_VARIANTS))
    for launch_order, variant in enumerate(order, 1):
        ordinal = len(schedule) + 1
        attempt_id = f'{ordinal:02d}-{task_id}-{REPEAT_ID}-{variant}'
        schedule.append({'ordinal': ordinal, 'task_slot': slot, 'repeat_id': REPEAT_ID,
                         'within_pair_launch_order': launch_order, 'task_id': task_id,
                         'variant': variant, 'attempt_id': attempt_id,
                         'results_dir': str(RUN_ROOT / variant / REPEAT_ID / 'attempts' / attempt_id),
                         'config_sha256': EXPECTED_CONFIG_HASHES[variant]})
full_schedule = list(schedule)
assert len(full_schedule) == 24
if PREPARE_ONLY:
    schedule = schedule[:2]  # Construct both official agents; no model or judge.
state = {entry['attempt_id']: dict(entry, status='planned') for entry in schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)

atomic_text(FROZEN_DIR / 'P012-LICENSE.txt', REFERENCE_LICENSE_TEXT)
atomic_json(FROZEN_DIR / 'source-arms.json', EXPERIMENT['source_arms'])

def descendant_pids(root_pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            row = next(line for line in path.read_text().splitlines() if line.startswith('PPid:'))
            parents[int(path.parent.name)] = int(row.split()[1])
        except (OSError, ValueError, StopIteration):
            pass
    found, frontier = [], [root_pid]
    while frontier:
        frontier = [pid for pid, parent in parents.items() if parent in frontier and pid not in found]
        found.extend(frontier)
    return found


def stop_process_tree(process, grace=3, own_group=False):
    children = descendant_pids(process.pid)
    pids = list(reversed(children)) + [process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if own_group:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(grace)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
# Install before importing target packages. A warm kernel is rejected, never silently reused.
loaded = [name for name in ('torch', 'swegemma', 'adk_submission', 'adk_eval_core',
                           'google.adk', 'vllm', 'litellm') if name in sys.modules]
assert not loaded, f'Restart the kernel before running: packages already loaded: {loaded}'
os.environ.update({
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
assert WHEELHOUSE_DIR.is_dir(), f'Missing wheelhouse: {WHEELHOUSE_DIR} input_files={input_snapshot()}'
source_wheels = [path for path in sorted(WHEELHOUSE_DIR.glob('*.whl')) if 'cutlass' not in path.name.lower()]
assert source_wheels, 'Wheelhouse is empty'

# Validate CPython/platform tags before installation; wheelhouse v31 is NOT a
# universal Python environment. Packaging normally ships with the Kaggle image.
try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
supported_tags = set(sys_tags())
expected_wheels = WHEELHOUSE_V31_FILES
actual_wheels = {p.name: p.stat().st_size for p in source_wheels}
if actual_wheels != expected_wheels:
    raise RuntimeError('Attached wheelhouse is not the recorded v31 inventory: '
                       f'missing={sorted(set(expected_wheels)-set(actual_wheels))} '
                       f'extra={sorted(set(actual_wheels)-set(expected_wheels))} '
                       f'size_changed={[n for n in actual_wheels.keys() & expected_wheels.keys() if actual_wheels[n] != expected_wheels[n]]}')
incompatible = []
for path in source_wheels:
    fixed_name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    if not parse_wheel_filename(fixed_name)[3] & supported_tags:
        incompatible.append(path.name)
if incompatible:
    raise RuntimeError(f'Incompatible wheels for this Python/platform; no install attempted: {incompatible}')

wheel_manifest = []
seen_distributions = set()
for path in source_wheels:
    distribution = path.name.split('-')[0].replace('_', '-').lower()
    assert distribution not in seen_distributions, f'Duplicate distribution: {distribution}'
    seen_distributions.add(distribution)
    wheel_manifest.append({'name': path.name, 'distribution': distribution,
                           'size': path.stat().st_size, 'sha256': sha256_file(path)})
for distribution, expected in EXPERIMENT['common']['official_packages'].items():
    version, expected_sha = expected.split(' sha256 ')
    matching = [entry for entry in wheel_manifest if entry['distribution'] == distribution]
    assert len(matching) == 1 and matching[0]['sha256'] == expected_sha, f'Official package changed: {distribution}'
atomic_json(FROZEN_DIR / 'wheel_manifest.json', wheel_manifest)
staging = Path('/tmp') / ('wheelhouse-' + RUN_ID)
staging.mkdir(exist_ok=False)
for path in source_wheels:
    name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    target = staging / name
    assert not target.exists() and not target.is_symlink(), f'Staging collision: {name}'
    target.symlink_to(path)
wheels = sorted(str(path) for path in staging.glob('*.whl'))
assert len(wheels) == len(wheel_manifest)
print(f'Installing {len(wheels)} frozen wheels', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

# P012 task dependencies (vendored wheels in the eval-tasks dataset, no internet):
# install only what is missing; never --force-reinstall (must not downgrade the
# harness stack). jinja2-time is deliberately absent (task 1779 needs its absence).
dep_dir = DATA_DIR / 'deps'
# Dir-mode zip extraction may nest wheels one level down (deps/deps/*.whl);
# prefer the level that actually holds wheels.
if dep_dir.is_dir() and not list(dep_dir.glob('*.whl')):
    _nested = sorted(p for p in dep_dir.iterdir() if p.is_dir() and list(p.glob('*.whl')))
    if len(_nested) == 1:
        dep_dir = _nested[0]
if dep_dir.is_dir():
    floors = EXPERIMENT['data']['task_dependencies']['floors']
    found = sorted(p.name for p in dep_dir.glob('*.whl'))
    atomic_json(FROZEN_DIR / 'task-deps-inventory.json',
                {'wheels': found, 'floors': floors})
    if found:
        _dep_log = RUN_ROOT / 'task-dependencies.log'
        _dep_proc = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
             '--find-links', str(dep_dir), *floors],
            capture_output=True, text=True, timeout=300)
        _dep_log.write_text(
            'CMD: ' + ' '.join([sys.executable, '-m', 'pip', 'install', '-q',
                                '--no-index', '--find-links', str(dep_dir), *floors])
            + '\nRETURNCODE: %d\nSTDOUT:\n%s\nSTDERR:\n%s\n'
            % (_dep_proc.returncode, _dep_proc.stdout[-4000:], _dep_proc.stderr[-4000:]))
        if _dep_proc.returncode != 0:
            raise RuntimeError('Task dependency install failed; see task-dependencies.log')
        importlib.invalidate_caches()
else:
    print('WARNING: task deps dir missing; criterion tests may fail on imports',
          flush=True)

# Eval-Job-1 (A/B) ships no adapters: both arms are pure 12B base.
# Fail fast only if OUR adapter datasets are mounted (D=new adapter, C=P011
# adapter ride the dedicated follow-up job). The competition bundle's own
# sample_submission/adapters are official fixtures and are ignored.
stray = sorted({str(p) for p in INPUT_ROOT.rglob('adapter_model.safetensors')
                if 'sample_submission' not in p.parts
                and ('p012' in str(p) or 'p011' in str(p))})
if stray:
    raise RuntimeError('Eval-Job-1 must not see P012/P011 adapter weights: ' + str(stray[:3]))
atomic_json(RUN_ROOT / 'adapter-source.json',
            {'adapter_source': None, 'note': 'A/B are pure base; adapters ride the C/D job'})

# Runtime-only research correction: prefer local src packages in both the agent
# and verification sandboxes. Preserve the exact before/after source and diff.
import difflib
sandbox_path = Path(metadata.distribution('swegemma').locate_file('swegemma/sandbox/subprocess.py'))
sandbox_before = sandbox_path.read_text()
old_import = "            'PYTHONPATH': str(workspace_dir),"
new_import = "            'PYTHONPATH': os.pathsep.join(str(p) for p in (workspace_dir / 'src', workspace_dir) if p.is_dir()),"
if sandbox_before.count(old_import) != 1:
    raise RuntimeError('Official subprocess source changed; src correction anchor absent')
sandbox_after = sandbox_before.replace(old_import, new_import)
atomic_text(FROZEN_DIR / 'subprocess-before.py', sandbox_before)
atomic_text(FROZEN_DIR / 'subprocess-after.py', sandbox_after)
atomic_text(FROZEN_DIR / 'subprocess-src-import.diff', ''.join(difflib.unified_diff(
    sandbox_before.splitlines(True), sandbox_after.splitlines(True), fromfile='official/subprocess.py', tofile='research/subprocess.py')))
sandbox_path.write_text(sandbox_after)
importlib.invalidate_caches()
set_phase('serving_setup')

import torch
import yaml
import litellm
from adk_submission import VllmConfig, VllmServer, discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import (ALLOWED_ADAPTER_EXTENSIONS, ALLOWED_SUBMISSION_EXTENSIONS,
                            MAX_SUBMISSION_SIZE_BYTES, EvalConfig, build_submission_limits)
from swegemma.context import SwegemmaContext
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import save_trace_artifact
from swegemma.models import load_tasks
from adk_submission.discovery import discover_declared_models
from swegemma.results import append_task_result

yaml.SafeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
versions = {name: metadata.version(name) for name in
            ('swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'vllm', 'litellm', 'torch')}
for name, expected in EXPERIMENT['common']['official_packages'].items():
    assert versions[name] == expected.split(' sha256 ')[0], f'Installed version mismatch: {name}'
if PREPARE_ONLY:
    # Prepare runs on the GPU session too (no separate CPU kernel): it builds
    # both agents and compile-checks them without loading the model.
    print('prepare on GPU session; cuda_available=%s' % torch.cuda.is_available(), flush=True)
else:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Requires four L4 GPUs'
    assert all('L4' in torch.cuda.get_device_properties(i).name for i in range(4))
gpus = [{'index': i, 'name': torch.cuda.get_device_properties(i).name,
         'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
assert (DATA_DIR / 'snapshots').is_dir() and MODEL_PATH.is_dir(), \
    f'Competition data or pinned model is missing input_files={input_snapshot()}'

# Read only the three already exposed, frozen diagnostic rows. Never scan the
# competition tasks.jsonl; optional source overrides must contain these same rows.
TASKS_PATH = Path(os.environ.get('P012_EVAL_TASKS_PATH', str(Path(__file__).parent / 'tasks-selected.jsonl')))
subset_bytes = TASKS_PATH.read_bytes()
tasks_sha = hashlib.sha256(subset_bytes).hexdigest()
tasks_size = len(subset_bytes)
assert tasks_sha == EXPERIMENT['data']['source_sha256'], 'Selected diagnostic rows changed'
assert tasks_size == EXPERIMENT['data']['source_bytes']
TASKS_SUBSET_PATH = FROZEN_DIR / 'tasks-selected.jsonl'
TASKS_SUBSET_PATH.write_bytes(subset_bytes)
tasks_subset_sha = tasks_sha
tasks = load_tasks(TASKS_SUBSET_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(tasks) == 12 and set(by_id) == set(TASK_IDS)
from swegemma.deduplication import resolve_task_snapshot_paths
snapshot_verification = {}
for task in tasks:
    assert task.test_patch and task.FAIL_TO_PASS, f'Frozen dev task must carry verifying tests: {task.instance_id}'
    expected_meta = EXPECTED_TASK_META[task.instance_id]
    assert task.repo == expected_meta['repo'] and task.base_commit == expected_meta['base_commit'], (
        f'Task metadata mismatch for {task.instance_id}: {task.repo}@{task.base_commit}')
    assert not task.patch, f'Gold patch must never ship with dev tasks: {task.instance_id}'
    snapshot, base_snapshot, patch_file = resolve_task_snapshot_paths(
        DATA_DIR / 'snapshots', task.instance_id, task.repo)
    assert snapshot.is_file(), f'Missing selected snapshot: {task.instance_id}'
    assert base_snapshot is None or base_snapshot.is_file()
    assert patch_file is None or patch_file.is_file()
    snapshot_verification[task.instance_id] = {
        'repo': task.repo, 'base_commit': task.base_commit,
        'snapshot_name': snapshot.name, 'snapshot_bytes': snapshot.stat().st_size}
atomic_json(FROZEN_DIR / 'task-source.json', {
    **EXPERIMENT['data'], 'selected_sha256': tasks_subset_sha,
    'selected_bytes': len(subset_bytes), 'verified_snapshots': snapshot_verification})


common = EXPERIMENT['common']
for variant in SELECTED_VARIANTS:
    condition = VARIANT_CONDITIONS[variant]
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    assert config.get('adapter') == condition['adapter'] and config['tools'] == condition['tools']
    assert yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml']) == condition['sampling']
    assert yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation'] == condition['budgets']
    assert condition['budgets']['max_time_minutes'] == 8
    assert (config.get('skills') or []) == condition['skills']
    assert discover_declared_models(AGENT_ROOT / variant) == {TARGET_MODEL_NAME}
    assert not any(path.is_symlink() for path in (AGENT_ROOT / variant).rglob('*'))
assert 'verification_timeout_seconds' in inspect.signature(EvalConfig).parameters

eval_signature = inspect.signature(Evaluator.evaluate_task)
assert 'context' in eval_signature.parameters, f'Unsupported evaluate_task API: {eval_signature}'
# Check support before any call; never catch TypeError and rerun an already-started task.
seen_adapters = {}
for variant in SELECTED_VARIANTS:
    found = discover_adapters(str(AGENT_ROOT / variant), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    seen_adapters[variant] = sorted(found.adapters)
    assert not found.adapters, f'Variant {variant} must be adapter-free in Eval-Job-1: {found.adapters}'
adapters = discover_adapters(str(AGENT_ROOT / SELECTED_VARIANTS[0]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
litellm.drop_params = True
gpu_count = torch.cuda.device_count()
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
run_manifest = {
    'run_id': RUN_ID, 'repeats': ['r1'], 'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate', 'selected_variants': list(SELECTED_VARIANTS),
    'full_schedule': full_schedule, 'config_sha256': EXPECTED_CONFIG_HASHES,
    'experiment_manifest_sha256': hashlib.sha256(MANIFEST_TEXT.encode('utf-8')).hexdigest(),
    'tasks_sha256': tasks_sha, 'tasks_bytes': tasks_size, 'selected_tasks_sha256': tasks_subset_sha, 'versions': versions,
    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
    'gpus': gpus, 'tensor_parallel_size': tp_size, 'cuda_build': torch.version.cuda,
    'variant_conditions': VARIANT_CONDITIONS, 'verification_timeout_seconds': VERIFICATION_TIMEOUT_SECONDS, 'research_condition': EXPERIMENT['research_condition'],     'sandbox': 'subprocess',
    'model_path': str(MODEL_PATH), 'model_alias': TARGET_MODEL_NAME,
    'input_dirs': {'data': str(DATA_DIR), 'wheelhouse': str(WHEELHOUSE_DIR), 'model': str(MODEL_PATH)},
    'server_start_count': 0, 'model_requests': 0 if PREPARE_ONLY else 'four isolated concurrent Evaluators (two simultaneous pairs); sampled via /metrics',
    'context_cache': {'min_tokens': 2048, 'ttl_seconds': 1800, 'cache_intervals': 10},
    'events_compaction': {'compaction_interval': 15, 'overlap_size': 2,
                          'token_threshold': 14336, 'event_retention_size': 5},
    'cache_note': 'Shared server may retain prefix cache; schedule position recorded, not claimed cache-equivalent.',
    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
print(f'GPUS={gpus} versions={versions}', flush=True)

# Export each unchanged config independently; these ZIPs are outputs, not competition submissions.
for variant in SELECTED_VARIANTS:
    output_dir = RUN_ROOT / variant / REPEAT_ID
    output_dir.mkdir(parents=True)
    zip_path = output_dir / f'agent-bundle-010-budget-{variant}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((AGENT_ROOT / variant).rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(AGENT_ROOT / variant))
            assert path.suffix.lower() in ALLOWED_SUBMISSION_EXTENSIONS
            archive.write(path, relative)
    with zipfile.ZipFile(zip_path) as archive:
        assert 'agent.yaml' in archive.namelist()
        assert sum(info.file_size for info in archive.infolist()) <= MAX_SUBMISSION_SIZE_BYTES
    atomic_json(output_dir / 'submission_manifest.json', {
        'zip_sha256': sha256_file(zip_path), 'config_sha256': EXPECTED_CONFIG_HASHES[variant]})


def run_sync(function, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(function(**kwargs))).result()
    return asyncio.run(function(**kwargs))


server = None
if not PREPARE_ONLY:
    class ArchivingVllmServer(VllmServer):
        # The official start() can call stop() on failure; archive before that internal cleanup too.
        def stop(self, timeout=10.0):
            log_path = getattr(self, 'log_path', None)
            try:
                if log_path and Path(log_path).is_file():
                    shutil.copyfile(log_path, RUN_ROOT / 'vllm_server.archived.log')
            except Exception:
                atomic_text(RUN_ROOT / 'server_log_archive_error.txt', traceback.format_exc())
            finally:
                super().stop(timeout=timeout)


    server = ArchivingVllmServer(VllmConfig(
        model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
        reasoning_parser='gemma4', max_model_len=32768,
        dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        enable_lora=False, extra_args=['--max-num-seqs', '4', '--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'],
        tensor_parallel_size=tp_size, startup_timeout=60 * 10), adapter_manifest=adapters)

    atomic_json(RUN_ROOT / 'server-log-path.json', {'path': str(server.log_path)})
    run_manifest['server_command'] = server.build_cmd()
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)

results_by_variant = {variant: [] for variant in SELECTED_VARIANTS}
analysis_rows = []
batch_start = time.monotonic()
active = []
failure = None
worker_path = RUN_ROOT / 'attempt_worker.py'
atomic_text(worker_path, WORKER_SOURCE)


def collect(entry, process, log, started, forced=None):
    log.close()
    attempt = Path(entry['results_dir'])
    status_path = attempt / 'attempt_status.json'
    item = json.loads(status_path.read_text()) if status_path.exists() else dict(entry, status='error')
    item.update(exit_code=process.returncode, coordinator_outer_seconds=time.monotonic() - started)
    if forced or process.returncode != 0:
        item.update(status=forced or 'error')
    result_path = attempt / ('prepared.json' if PREPARE_ONLY else 'result.json')
    if not result_path.exists():
        atomic_json(result_path, {'instance_id': entry['task_id'], 'resolved': None,
                                 'error': forced or f'worker exited {process.returncode} before result'})
    result = json.loads(result_path.read_text())
    state[entry['attempt_id']] = item
    atomic_json(status_path, item)
    analysis_rows.append({**entry, **result, 'status': item['status'], 'exit_code': process.returncode})
    atomic_json(RUN_ROOT / 'analysis_rows.json', analysis_rows)
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    files = {str(path.relative_to(attempt)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
             for path in attempt.rglob('*') if path.is_file() and path.name != 'artifact_manifest.json'}
    atomic_json(attempt / 'artifact_manifest.json', files)
    print(json.dumps({'attempt_id': entry['attempt_id'], 'status': item['status'],
                      'resolved': result.get('resolved'), 'time': time.time()}), flush=True)



try:
    if not PREPARE_ONLY:
        set_phase('server_start')
        run_manifest['server_start_count'] = 1
        atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
        startup = time.monotonic()
        server.start()
        run_manifest['server_startup_seconds'] = time.monotonic() - startup
        run_manifest['server_base_url'] = str(server.base_url)
        atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    pair_queue = [schedule[i:i + 2] for i in range(0, len(schedule), 2)]
    while pair_queue or active:
        if pair_queue and time.monotonic() >= EVALUATION_DEADLINE:
            raise TimeoutError('Whole-run evaluation deadline; keep partial evidence, no retry')
        while pair_queue and len(active) + len(pair_queue[0]) <= MAX_CONCURRENT:
            if time.monotonic() >= EVALUATION_DEADLINE:
                break
            pair = pair_queue.pop(0)
            set_phase('cpu_prepare' if PREPARE_ONLY else 'evaluation',
                      task_slot=pair[0]['task_slot'], task_id=pair[0]['task_id'],
                      repeat_id=pair[0]['repeat_id'],
                      active_workers=len(active) + len(pair),
                      remaining_pairs=len(pair_queue))
            for entry in pair:
                attempt = Path(entry['results_dir'])
                attempt.mkdir(parents=True, exist_ok=False)
                scratch = Path('/tmp') / RUN_ID / entry['attempt_id']
                scratch.mkdir(parents=True, exist_ok=False)
                agent = scratch / 'agent'
                shutil.copytree(AGENT_ROOT / entry['variant'], agent)
                specification = {'entry': entry, 'data_dir': str(DATA_DIR), 'agent_dir': str(agent),
                                 'tasks_subset': str(TASKS_SUBSET_PATH), 'scratch': str(scratch),
                                 'base_url': 'http://127.0.0.1:8000/v1' if PREPARE_ONLY else str(server.base_url),
                                 'served_model': str(MODEL_PATH) if PREPARE_ONLY else str(getattr(server, 'effective_model_path', MODEL_PATH)),
                                 'model_alias': TARGET_MODEL_NAME,
                                 'budgets': VARIANT_CONDITIONS[entry['variant']]['budgets'],
                                 'prepare_only': PREPARE_ONLY, 'total_tasks': 24,
                                 'verification_timeout_seconds': VERIFICATION_TIMEOUT_SECONDS}
                spec_path = attempt / 'worker_spec.json'
                atomic_json(spec_path, specification)
                env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                           PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false')
                log = (attempt / 'worker.log').open('w')
                process = subprocess.Popen([sys.executable, '-u', str(worker_path), str(spec_path)],
                                           cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT)
                started = time.monotonic()
                active.append((entry, process, log, started))
                state[entry['attempt_id']].update(status='running', pid=process.pid, started_at=time.time())
                atomic_json(attempt / 'attempt_manifest.json', state[entry['attempt_id']])
                atomic_json(RUN_ROOT / 'attempt_status.json', state)
                print(f'Launched {entry["attempt_id"]} pid={process.pid} active={len(active)}', flush=True)
            assert len(active) <= MAX_CONCURRENT
        for record in active[:]:
            entry, process, log, started = record
            forced = None
            if process.poll() is None:
                if time.monotonic() >= EVALUATION_DEADLINE:
                    forced = 'whole_run_deadline'
                elif time.monotonic() - started >= ATTEMPT_WALL_SECONDS:
                    forced = 'attempt_wall_timeout'
                if forced:
                    stop_process_tree(process)
            if process.poll() is not None:
                collect(entry, process, log, started, forced)
                active.remove(record)
        if active and (not pair_queue or len(active) + len(pair_queue[0]) > MAX_CONCURRENT):
            time.sleep(1)
except BaseException as exc:
    failure = exc
    atomic_text(RUN_ROOT / 'batch_exception.txt', traceback.format_exc())
    if getattr(exc, 'output', None):
        atomic_text(RUN_ROOT / 'server_exception_output.log', str(exc.output))
finally:
    set_phase('cleanup')
    for entry, process, log, started in active:
        stop_process_tree(process)
        collect(entry, process, log, started, 'interrupted')
    if server is not None:
        try:
            server.stop(timeout=10.0)
        except Exception:
            atomic_text(RUN_ROOT / 'server_stop_error.txt', traceback.format_exc())
    for item in state.values():
        if item['status'] == 'planned':
            item['status'] = 'not_run'
    run_manifest['batch_wall_seconds'] = time.monotonic() - batch_start
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    summary = {variant: {
        'planned': sum(entry['variant'] == variant for entry in full_schedule),
        'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate',
        'prepared': sum(s['variant'] == variant and s['status'] == 'prepared' for s in state.values()),
        'completed': sum(s['variant'] == variant and s['status'] == 'completed' for s in state.values()),
        'resolved': None if PREPARE_ONLY else sum(row['variant'] == variant and row.get('resolved') is True for row in analysis_rows),
        'states': {s['attempt_id']: s['status'] for s in state.values() if s['variant'] == variant}}
        for variant in SELECTED_VARIANTS}
    atomic_json(RUN_ROOT / 'joint_summary.json', summary)
    print('FINAL ' + json.dumps(summary), flush=True)
if failure is not None:
    raise failure
expected_state = 'prepared' if PREPARE_ONLY else 'completed'
if any(item['status'] != expected_state for item in state.values()):
    raise RuntimeError('Some attempts are incomplete; retain partial evidence, no automatic retry')
set_phase('prepared' if PREPARE_ONLY else 'completed')
