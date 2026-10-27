"""Submit this authorized frozen experiment once and retain its actual outputs.

Importing this module does not submit a job. The launch receipt is written before
the single mutating request, so an unknown response cannot cause a blind retry.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def save(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def connect(out=None):
    spec = importlib.util.spec_from_file_location('vfarl_round_helper', HERE.parent / 'next-round/execute.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    original = requests.Session.send
    def bounded(session, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (10, 30)
        return original(session, request, **kwargs)
    requests.Session.send = bounded
    run = helper.Round.__new__(helper.Round)
    run.out, run.notes, run.requests = Path(out or HERE), [], requests
    run.out.mkdir(parents=True, exist_ok=True)
    run.args = argparse.Namespace(poll_seconds=60)
    run.api = KaggleApi()
    run.api.config_values[run.api.CONFIG_NAME_USER] = 'jiyujia'
    run.api.config_values[run.api.CONFIG_NAME_TOKEN] = (Path.home() / '.kaggle/access_token_jiyujia').read_text().strip()
    run.api.config_values[run.api.CONFIG_NAME_KEY] = None
    return run


def dataset_ready(run):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    slug = 'jiyujia/vfarl-l4x4-train100-v4'
    state = json.loads(run.retry(lambda: run.api.dataset_status(slug, format='json')))
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = slug.split('/')
    def read():
        with run.api.build_kaggle_client() as client:
            return client.datasets.dataset_api_client.get_dataset(request)
    actual = run.retry(read)
    files = run.retry(lambda: run.api.dataset_list_files(slug + '/1'))
    record = dict(at_utc=now(), dataset=slug, status=state,
                  is_private=actual.is_private, current_version=actual.current_version_number,
                  files=[dict(name=row.name, bytes=row.total_bytes) for row in files.files or []])
    save(run.out / 'dataset-readback.json', record)
    if (state['status'] != 'ready' or actual.current_version_number != 1 or not actual.is_private
            or record['files'] != [dict(name='vfarl-train100.bin', bytes=124593514)]):
        raise RuntimeError('Actual dataset is not the authorized private version and material size')
    return record


def launch(run):
    receipt_path = run.out / 'gpu-push.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get('kernel'):
            raise RuntimeError('Prior submission has an unknown outcome; inspect before any resubmission')
        return receipt['kernel']
    dataset_ready(run)
    quota = run.quota('before-resident-r1')
    if quota['remaining_h'] < run.minimum_quota:
        raise RuntimeError('Actual remaining quota is below the frozen seven-hour new-claim plan')
    carrier = run.carrier
    receipt = dict(at_utc=now(), status='single-submission-started',
                   source_sha256=hashlib.sha256((carrier / 'run.py').read_bytes()).hexdigest(),
                   accelerator='NvidiaL4', session_timeout_seconds=run.session_seconds)
    save(receipt_path, receipt)
    try:
        # Never retry this mutation automatically: a lost response may still
        # correspond to a live four-card session.
        result = run.api.kernels_push(str(carrier), timeout=str(run.session_seconds), acc='NvidiaL4')
        receipt.update(error=result.error, url=result.url, version=result.version_number,
            invalid_dataset_sources=list(result.invalid_dataset_sources or []),
            invalid_kernel_sources=list(result.invalid_kernel_sources or []),
            invalid_competition_sources=list(result.invalid_competition_sources or []))
        match = re.search(r'kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)', result.url or '')
        if match and result.version_number:
            receipt['kernel'] = match.group(1) + '/' + str(result.version_number)
        save(receipt_path, receipt)
        if (result.error or not receipt.get('kernel') or receipt['invalid_dataset_sources']
                or receipt['invalid_kernel_sources'] or receipt['invalid_competition_sources']):
            raise RuntimeError('Submission response was not a complete accepted experiment; inspect receipt')
        run.verify_remote(receipt['kernel'], carrier)
        receipt['status'] = 'submitted-and-read-back'
        save(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        return receipt['kernel']
    except BaseException as error:
        receipt['exception'] = type(error).__name__ + ': ' + str(error)
        save(receipt_path, receipt)
        raise


def observe(run, ref):
    previous, failures = None, 0
    while True:
        try:
            run.current_version(ref)
            result = run.sdk('get_kernel_session_status', run.request(ref, 'status'))
            state = getattr(result.status, 'name', str(result.status).rsplit('.', 1)[-1])
            record = dict(at_utc=now(), kernel=ref, status=state, failure_message=result.failure_message)
            save(run.out / 'status.json', record)
            with (run.out / 'status-history.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if state != previous:
                print(json.dumps(record), flush=True)
                previous = state
            failures = 0
            if state in {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_REQUESTED'}:
                run.download(ref, run.out / 'actual-output', full=True)
                run.quota('after-resident-r1')
                print(json.dumps(dict(phase='actual_outputs_retained', kernel=ref, status=state)), flush=True)
                return
        except Exception as error:
            failures += 1
            print(json.dumps(dict(at_utc=now(), phase='observation_retry', failures=failures,
                error=type(error).__name__ + ': ' + str(error))), flush=True)
            if failures >= 10:
                raise
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run-name', default='')
    parser.add_argument('--carrier', type=Path, default=HERE / 'carrier-r1')
    parser.add_argument('--minimum-quota', type=float, default=16)
    parser.add_argument('--session-seconds', type=int, default=30000)
    args = parser.parse_args()
    if args.run_name and not re.fullmatch(r'[a-z0-9-]+', args.run_name):
        parser.error('Run name must be a simple experiment directory name')
    run = connect(HERE / args.run_name if args.run_name else HERE)
    run.carrier, run.minimum_quota, run.session_seconds = args.carrier, args.minimum_quota, args.session_seconds
    ref = launch(run) if args.launch else json.loads((run.out / 'gpu-push.json').read_text())['kernel']
    observe(run, ref)


if __name__ == '__main__':
    main()
