"""Run this frozen research round on jiyujia; no GPU work occurs on import.

The driver waits for the actual CPU runtime, validates the selected environments,
collects new behavior once, and may consume that exact batch for one update.
It is a live experiment entry point, not a test or a source of synthetic results.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ACCOUNT = 'jiyujia'
DATASET = 'jiyujia/vfarl-newtasks-r1/1'
CONFIG = ROOT / 'training/configs/vfarl_l4x4_16k.yaml'
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ref_parts(value):
    match = re.fullmatch(r'([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)/([0-9]+)', value)
    if not match:
        raise ValueError('An explicit owner/kernel/version is required: ' + value)
    return match.groups()


class Round:
    def __init__(self, args):
        self.args = args
        self.out = HERE / ('execution-' + args.run_name)
        self.out.mkdir(parents=True, exist_ok=True)
        self.selection = load(HERE / 'selection.json')
        self.plan = load(HERE / 'arm-plan.json')
        self.material = load(HERE / 'material-v1/material-build.json')
        if sha(HERE / 'selection.json') != self.material['selection_sha256']:
            raise RuntimeError('The frozen task selection changed after material creation')
        self.notes = []
        self.token = (Path.home() / '.kaggle/access_token_jiyujia').read_text().strip()
        self.env = dict(os.environ, KAGGLE_API_TOKEN=self.token)
        # Apply timeouts in this process only; do not edit the installed SDK.
        os.environ['KAGGLE_API_TOKEN'] = self.token
        import requests
        original_send = requests.Session.send
        def bounded_send(session, request, **kwargs):
            kwargs.setdefault('timeout', (15, 60))
            return original_send(session, request, **kwargs)
        requests.Session.send = bounded_send
        self.requests = requests
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.retry(self.api.authenticate)
        if self.api.config_values.get('username') != ACCOUNT:
            raise RuntimeError('This round only uses the explicitly authorized jiyujia account')

    def note(self, text):
        self.notes.append(text)
        print(json.dumps({'at_utc': utc(), 'research': text}, ensure_ascii=False), flush=True)
        path = self.out / 'results.md'
        if not path.exists():
            path.write_text('# 新任务小轮实际执行\n\n所有题来自S600 train；四道研究验证题不参加focal采集或更新。'
                            '这些小样本结果不构成比赛heldout成绩。\n\n')
        with path.open('a') as stream:
            stream.write(utc() + '：' + text + '\n\n')
        with (HERE / 'execution.md').open('a') as stream:
            stream.write('\n' + utc() + '：' + text + '\n')

    def retry(self, action):
        for attempt in range(3):
            try:
                return action()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))

    def request(self, ref, kind):
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelRequest, ApiGetKernelSessionStatusRequest,
            ApiListKernelSessionOutputRequest)
        classes = {'metadata': ApiGetKernelRequest, 'status': ApiGetKernelSessionStatusRequest,
                   'files': ApiListKernelSessionOutputRequest}
        owner, slug, version = ref_parts(ref)
        req = classes[kind]()
        # The service currently returns404 for numeric version_label on these
        # three endpoints. Read latest only after checking its actual version.
        req.user_name, req.kernel_slug = owner, slug
        return req

    def current_version(self, ref):
        result = self.sdk('get_kernel', self.request(ref, 'metadata'))
        expected = int(ref_parts(ref)[2])
        if result.metadata.current_version_number != expected:
            raise RuntimeError('The notebook version changed; do not read or reuse a different run: ' + ref)
        return result

    def sdk(self, method, request):
        def call():
            with self.api.build_kaggle_client() as client:
                return getattr(client.kernels.kernels_api_client, method)(request)
        return self.retry(call)

    def wait(self, ref, gpu=False):
        started, previous, last_print = time.monotonic(), None, 0
        maximum = 14 * 3600 if gpu else 3600
        while time.monotonic() - started < maximum:
            self.current_version(ref)
            result = self.sdk('get_kernel_session_status', self.request(ref, 'status'))
            state = getattr(result.status, 'name', str(result.status).rsplit('.', 1)[-1])
            if state != previous or time.monotonic() - last_print > 300:
                record = dict(at_utc=utc(), kernel=ref, status=state,
                              failure_message=result.failure_message,
                              observed_seconds=time.monotonic() - started)
                save(self.out / (ref_parts(ref)[1] + '-status.json'), record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                previous, last_print = state, time.monotonic()
            if state in TERMINAL:
                return state
            time.sleep(self.args.poll_seconds)
        raise RuntimeError('The observation interval ended while the real job was still active: ' + ref)

    def quota(self, stage):
        result = self.retry(self.api.quota_view)
        gpu = result.gpu_quota
        used = gpu.time_used.total_seconds() / 3600
        total = gpu.total_time_allowed.total_seconds() / 3600
        record = dict(at_utc=utc(), account=ACCOUNT, used_h=used, total_h=total,
                      remaining_h=max(0.0, total - used),
                      refresh_at=str(result.quota_refresh_time))
        save(self.out / ('quota-' + stage + '.json'), record)
        print(json.dumps({'phase': 'actual_quota', 'stage': stage, **record}), flush=True)
        return record

    def download(self, ref, destination, full=True):
        self.current_version(ref)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / 'download-manifest.json'
        previous = load(manifest_path) if manifest_path.exists() else {}
        cached = {row['path']: row for row in previous.get('files', [])} if previous.get('kernel') == ref else {}
        req = self.request(ref, 'files')
        req.page_size = 200
        records = []
        while True:
            result = self.sdk('list_kernel_session_output', req)
            if result.log:
                raw = result.log if isinstance(result.log, str) else json.dumps(result.log)
                (destination / 'kernel.log').write_text(raw)
            for item in result.files or []:
                relative = PurePosixPath(item.file_name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unsafe output path in the experiment output')
                if any(part in {'sandboxes', 'assets', 'runtime', '__pycache__'} or part.startswith('state-')
                       for part in relative.parts[:-1]):
                    continue
                small_runtime = relative.name in {'runtime-preparation-outcome.json', 'runtime-archive.json',
                    'base-runtime-archive.json', 'vfarl-preparation.json', 'preparation.json', 'exception.txt',
                    'pip-resolver.log', 'unsloth-extension.json'}
                if not (small_runtime if not full else relative.suffix in {'.json', '.jsonl', '.xml', '.txt', '.log', '.patch'}):
                    continue
                target = destination / Path(*relative.parts)
                old = cached.get(item.file_name)
                if not (old and target.is_file() and sha(target) == old['sha256']):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    def transfer():
                        temp = target.with_name(target.name + '.part')
                        with self.requests.get(item.url, stream=True, timeout=(15, 60)) as response:
                            response.raise_for_status()
                            with temp.open('wb') as stream:
                                for block in response.iter_content(1024 * 1024):
                                    if block:
                                        stream.write(block)
                        temp.replace(target)
                    self.retry(transfer)
                records.append(dict(path=item.file_name, bytes=target.stat().st_size, sha256=sha(target)))
            if not result.next_page_token:
                break
            req.page_token = result.next_page_token
        self.current_version(ref)
        save(manifest_path, dict(kernel=ref, at_utc=utc(), files=records,
                                excluded='Model weights, opaque archives and expanded state/sandbox forests'))
        return destination

    def build(self, stage, runtime_sha, prevalidation=None, continuation=None, stop_seconds=0):
        carrier = self.out / (stage + '-carrier')
        if carrier.exists():
            return carrier
        slug = 'vfarl-newtasks-' + stage + '-' + self.args.run_name
        command = [sys.executable, '-m', 'competition.vfarl.l4x4.kaggle.build_runner',
            '--material-build', str(HERE / 'material-v1/material-build.json'),
            '--mode', 'prevalidate' if stage == 'cpu' else 'train', '--out', str(carrier),
            '--kernel-owner', ACCOUNT, '--kernel-id', ACCOUNT + '/' + slug,
            '--title', slug.replace('-', ' ').title(), '--dataset-source', DATASET,
            '--runtime-kernel', self.args.runtime_kernel, '--runtime-sha256', runtime_sha,
            '--context', '16384', '--training-config', str(CONFIG)]
        if stage == 'cpu':
            for task in self.selection['tasks']:
                command += ['--only-task', task['instance_id']]
        else:
            command += ['--policy-backend', 'unsloth', '--learner-backend', 'native',
                '--dynamic-queue', '--focal-states', '6', '--stop-after-mixed-groups', '3',
                '--stop-after-tasks', '12', '--prevalidation-report', str(prevalidation)]
            if stage == 'collect':
                command += ['--collect-only', '--stop-after-seconds', str(stop_seconds)]
            if continuation:
                command += ['--collection-kernel', continuation['kernel'],
                    '--collection-carrier-sha256', continuation['carrier_sha256'],
                    '--collection-focal-sha256', continuation['focal_sha256']]
        result = subprocess.run(command, cwd=ROOT, env=self.env, text=True,
                                capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Carrier build failed: ' + result.stderr[-4000:])
        return carrier

    def launch(self, stage, carrier, gpu=False):
        receipt_path = self.out / (stage + '-push.json')
        if receipt_path.exists():
            receipt = load(receipt_path)
            if not receipt.get('kernel'):
                raise RuntimeError('A prior submission has an unknown outcome; inspect its real receipt before resubmitting')
            ref = receipt['kernel']
        else:
            metadata = load(carrier / 'kernel-metadata.json')
            command = ['kaggle', 'kernels', 'push', '-p', str(carrier)]
            if gpu:
                command += ['--accelerator', 'NvidiaL4']
            receipt = dict(at_utc=utc(), command=command)
            try:
                result = subprocess.run(command, cwd=ROOT, env=self.env, text=True,
                                        capture_output=True, timeout=180)
                receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            except subprocess.TimeoutExpired:
                receipt['error'] = 'Submission timed out; remote outcome unknown. Do not blindly submit another version.'
                save(receipt_path, receipt)
                raise
            match = re.search(r'Kernel version (\d+) successfully pushed', result.stdout)
            if match:
                receipt['kernel'] = metadata['id'] + '/' + match.group(1)
            save(receipt_path, receipt)
            if result.returncode or not match or 'not valid' in result.stdout.lower():
                raise RuntimeError('Submission failed or an expected input was rejected; see ' + str(receipt_path))
            ref = receipt['kernel']
        self.verify_remote(ref, carrier)
        self.note(('GPU' if gpu else 'CPU') + '作业已提交并回读源码/私有设置/输入：' + ref + '。')
        return ref

    def verify_remote(self, ref, carrier):
        result = self.current_version(ref)
        actual, expected = result.metadata, load(carrier / 'kernel-metadata.json')
        readback = dict(kernel=ref, at_utc=utc(), ref=actual.ref,
                        is_private=actual.is_private, enable_gpu=actual.enable_gpu,
                        enable_internet=actual.enable_internet,
                        dataset_sources=list(actual.dataset_data_sources or []),
                        kernel_sources=list(actual.kernel_data_sources or []),
                        competition_sources=list(actual.competition_data_sources or []),
                        model_sources=list(actual.model_data_sources or []),
                        source_sha256=hashlib.sha256(result.blob.source.encode()).hexdigest())
        save(carrier / 'remote-readback.json', readback)
        if (not actual.is_private or bool(actual.enable_gpu) != bool(expected['enable_gpu'])
                or bool(actual.enable_internet) != bool(expected['enable_internet'])
                or readback['source_sha256'] != sha(carrier / 'run.py')):
            raise RuntimeError('Actual notebook source or privacy/hardware settings differ from the frozen carrier')
        for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources'):
            # The service may omit version suffixes; payload SHA checks remain
            # embedded in the carrier before any model is loaded.
            canonical = lambda values: {'/'.join(value.split('/')[:2]).lower() for value in values}
            if canonical(readback[key]) != canonical(expected[key]):
                raise RuntimeError('Actual notebook inputs differ: ' + key)

    def committed_batch(self, focal_sha):
        for path in (ROOT / 'competition/vfarl/l4x4').rglob('optimizer-step.json'):
            record = load(path)
            if record.get('source_data_sha256') == focal_sha and record.get('optimizer_steps', 0) > 0:
                return path
        return None

    def run(self):
        self.note('开始固定12＋4研究轮；平台输出只按实际执行结果报告，不扩充100题。')
        state = self.wait(self.args.runtime_kernel)
        runtime = self.download(self.args.runtime_kernel, self.out / 'runtime-evidence', full=False)
        outcome_path = runtime / 'runtime-preparation-outcome.json'
        if outcome_path.exists():
            outcome = load(outcome_path)
            ready = outcome.get('base_runtime_prepared') and outcome.get('unsloth_extension_completed')
        else:
            preparation = runtime / 'vfarl-unsloth-preparation/preparation.json'
            outcome = load(preparation) if preparation.exists() else {}
            ready = outcome.get('success') is True and any(
                phase.get('phase') == 'archive_completed' for phase in outcome.get('phases', []))
        if state != 'COMPLETE' or not ready:
            raise RuntimeError('The actual runtime is not Unsloth-ready; retained CPU preparation evidence explains the failure')
        archive = load(runtime / 'runtime-archive.json')
        runtime_sha = archive['sha256']
        if runtime_sha != outcome.get('actual_archive_sha256', outcome.get('sha256')):
            raise RuntimeError('Actual runtime output hashes disagree')
        self.note('目标账号CPU运行时实际完成，Unsloth依赖已组装；归档SHA：' + runtime_sha + '。CUDA尚待实测。')
        if self.args.prevalidation_report:
            prevalidation = self.args.prevalidation_report.resolve(strict=True)
            state = 'COMPLETE'
            self.note('复用本轮已实际完成的CPU验收原件：' + str(prevalidation.relative_to(ROOT)) + '；不重复验收。')
        elif self.args.prevalidation_kernel:
            state = self.wait(self.args.prevalidation_kernel)
            cpu = self.download(self.args.prevalidation_kernel, self.out / 'cpu-evidence')
            prevalidation = cpu / 'vfarl-prevalidate/prevalidation.jsonl'
            cpu_carrier = load(cpu / 'vfarl-prevalidate/carrier.json')
            if cpu_carrier['runtime_sha256'] not in {runtime_sha, archive.get('base_runtime_sha256')}:
                raise RuntimeError('CPU task verification does not use this runtime or its unchanged base')
            self.note('复用已提交的独立CPU验收：' + self.args.prevalidation_kernel + '；任务环境沿用同一基础归档。')
        else:
            cpu_ref = self.launch('cpu', self.build('cpu', runtime_sha))
            state = self.wait(cpu_ref)
            cpu = self.download(cpu_ref, self.out / 'cpu-evidence')
            prevalidation = cpu / 'vfarl-prevalidate/prevalidation.jsonl'
        accepted = rows(prevalidation) if prevalidation.exists() else []
        expected = {task['instance_id'] for task in self.selection['tasks']}
        usable = [row for row in accepted if row.get('usable') and
                  row.get('baseline', {}).get('reward') == 0 and row.get('repaired', {}).get('reward') == 1]
        sources = {task['instance_id']: task['task_input_sha256'] for task in self.selection['tasks']}
        if any(row.get('task_input_sha256') != sources.get(row.get('instance_id')) for row in accepted):
            raise RuntimeError('Actual CPU report belongs to different selected task assets')
        self.note('新题Linux CPU实际可用：%d/%d，源作业%s。' % (len(usable), len(expected), state))
        if (state != 'COMPLETE' or len(accepted) != len(expected)
                or {row['instance_id'] for row in accepted} != expected):
            raise RuntimeError('The actual CPU report does not cover the frozen16-task selection')
        usable_ids = {row['instance_id'] for row in usable}
        collection_ids = set(self.selection['research_split']['collection_task_ids'])
        validation_ids = set(self.selection['research_split']['validation_task_ids'])
        if len(usable_ids & collection_ids) < 2:
            self.note('可用新采集任务不足两道，无法形成多个任务的异奖组，本轮保留CPU失败原件。')
            return
        if usable_ids != expected:
            self.note('仅从实际可用环境采集；不可用任务未计作模型失败：' + ', '.join(sorted(expected - usable_ids)) + '。')
        quota = self.quota('before-collect')
        if quota['remaining_h'] <= 10:
            self.note('实际剩余GPU配额%.2fh，未超过10h的更新/验证及在途缓冲预留，本轮未启动新采集。' % quota['remaining_h'])
            return
        deadline = (quota['remaining_h'] - 10) / 2 * 3600
        collect = self.build('collect', runtime_sha, prevalidation, stop_seconds=deadline)
        collect_ref = self.launch('collect', collect, gpu=True)
        state = self.wait(collect_ref, gpu=True)
        evidence = self.download(collect_ref, self.out / 'collect-evidence')
        self.quota('after-collect')
        if state != 'COMPLETE':
            raise RuntimeError('Collection did not complete; partial real evidence was retained and no update was submitted')
        focal = evidence / 'vfarl-train/focal.jsonl'
        carrier = evidence / 'vfarl-train/carrier.json'
        sys.path.insert(0, str(ROOT))
        from training.train_vfarl import prepare_rows
        reward_rows, summary = prepare_rows(focal, 16384)
        save(self.out / 'collection-result.json', dict(kernel=collect_ref, focal_sha256=sha(focal),
             groups=summary, sampled_focal_rows=len(reward_rows)))
        self.note('collection-only实际完成：%d个有效状态组，其中%d个异奖组、%d条参与更新候选动作；原行为batch SHA：%s。' %
                  (summary['valid_group_count'], summary['mixed_group_count'], len(reward_rows), sha(focal)))
        if self.args.through == 'collect' or summary['mixed_group_count'] < 2:
            self.note('按本轮范围保留真实采集结果；未提交RL更新。')
            return
        if not validation_ids <= usable_ids:
            self.note('新异奖数据已保留；独立验证环境尚未全部就绪，本次不提交更新。')
            return
        previous = self.committed_batch(sha(focal))
        if previous:
            self.note('该行为batch已有真实optimizer记录：' + str(previous.relative_to(ROOT)) + '，不重复训练。')
            return
        quota = self.quota('before-update')
        if quota['remaining_h'] < 6:
            self.note('采集后实际剩余%.2fh，未达到本轮更新及独立前后对照约6h预留；数据保留，本次未更新。' % quota['remaining_h'])
            return
        continuation = dict(kernel=collect_ref, carrier_sha256=sha(carrier), focal_sha256=sha(focal))
        update = self.build('update', runtime_sha, prevalidation, continuation=continuation)
        original_sources = load(collect / 'carrier-manifest.json')['source_files']
        update_sources = load(update / 'carrier-manifest.json')['source_files']
        actor_sources = lambda values: {key: value for key, value in values.items()
                                       if key != 'training/train_vfarl.py'}
        if actor_sources(original_sources) != actor_sources(update_sources):
            raise RuntimeError('Actor/tool/preparation source changed since collection; preserve the batch and review before allocating another GPU job')
        update_ref = self.launch('update', update, gpu=True)
        state = self.wait(update_ref, gpu=True)
        updated = self.download(update_ref, self.out / 'update-evidence')
        self.quota('after-update')
        step_path = updated / 'vfarl-train/update/optimizer-step.json'
        step = load(step_path) if step_path.exists() else {}
        pairs_path = updated / 'vfarl-train/research-validation.json'
        pairs = load(pairs_path) if pairs_path.exists() else None
        committed = step.get('optimizer_steps') == 1 and step.get('source_data_sha256') == sha(focal)
        save(self.out / 'update-result.json', dict(kernel=update_ref, platform_status=state,
            optimizer_step_committed=committed, source_data_sha256=sha(focal),
            grad_norm=step.get('grad_norm'), parameter_delta_l2=step.get('global_parameter_delta_l2'),
            independent_validation=pairs))
        if committed:
            self.note('真实optimizer step=1；grad norm=%s，LoRA delta L2=%s。平台最终状态%s。' %
                      (step.get('grad_norm'), step.get('global_parameter_delta_l2'), state))
        else:
            self.note('本轮未取得可核实的一步更新；实际平台状态' + state + '，异常及原件已保存。')
        if pairs:
            items = pairs['pairs']
            before = sum(row.get('before_verified') is True and row.get('before_reward') == 1 for row in items)
            after = sum(row.get('after_verified') is True and row.get('after_reward') == 1 for row in items)
            self.note('独立研究任务实际前后结果：%d/%d → %d/%d；逐题原件见update-evidence/vfarl-train/research-validation.json。' %
                      (before, len(items), after, len(items)))
        elif committed:
            self.note('更新已提交，但尚无完整独立前后对照；不重训该batch，也不宣称泛化提升。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-kernel', required=True,
                        help='Actual completed Unsloth-ready owner/kernel/version; never an earlier failed extension')
    parser.add_argument('--run-name', default='auto-r1')
    parser.add_argument('--through', choices=('collect', 'update'), default='update')
    parser.add_argument('--prevalidation-report', type=Path,
                        help='Actual16-task report from this frozen selection; reuse it after an overlay-only runtime repair')
    parser.add_argument('--prevalidation-kernel',
                        help='Already submitted target-account owner/kernel/version with this16-task CPU verification')
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args()
    ref_parts(args.runtime_kernel)
    if args.prevalidation_kernel:
        ref_parts(args.prevalidation_kernel)
        if not args.prevalidation_kernel.startswith(ACCOUNT + '/'):
            parser.error('Use this round\'s target-account CPU verification')
    if args.prevalidation_kernel and args.prevalidation_report:
        parser.error('Choose the completed CPU report or the submitted CPU kernel, not both')
    if not args.runtime_kernel.startswith(ACCOUNT + '/'):
        parser.error('Use the explicitly authorized target-account runtime')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,35}', args.run_name) or args.poll_seconds < 10:
        parser.error('Choose a short lowercase run name and a polling interval of at least10seconds')
    experiment = Round(args)
    try:
        experiment.run()
    except Exception as exc:
        # File-scoped URLs remain inside requests and never enter provenance.
        detail = re.sub(r'https?://\S+', '<URL omitted>', str(exc))
        experiment.note('实际执行停止：%s: %s。原始输出已保留；未执行的后续步骤不计为研究结果。' %
                        (type(exc).__name__, detail))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
