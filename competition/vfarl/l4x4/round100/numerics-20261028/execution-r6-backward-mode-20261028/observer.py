"""One bounded one-backward mode launch observer; the remote parent owns the hard runtime budget."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import re
import signal
import sys
import time
from urllib.parse import urljoin, urlparse

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'competition/vfarl/l4x4/round100/execute.py').is_file())
sys.path.insert(0, str(ROOT))
from competition.vfarl.l4x4.round100.execute import connect
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest, ApiDownloadKernelOutputRequest)

CONTRACT = json.loads((OUT / 'launch-contract.json').read_text())
EXPECTED = CONTRACT['wrapper_source_sha256']
CHILD = CONTRACT['frozen_child_source_sha256']
if CHILD != 'e46de57695e84807126dccd14b8a81a9992a607ca505681f922c3d46d01e7556':
    raise RuntimeError('Observer contract does not bind the authorized frozen one-backward mode child')
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}
run = connect(OUT)


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(phase, **values):
    row = dict(at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), phase=phase)
    row.update(values)
    with (OUT / 'observer-events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


class LocalDeadline(BaseException):
    pass


def alarm(signum, frame):
    # Existing read helpers retry Exception; this must escape those retries.
    raise LocalDeadline('Observer RPC exceeded its independent local deadline')


signal.signal(signal.SIGALRM, alarm)


def bounded(function, seconds=12):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def metadata(ref):
    # This guards latest-only SDK status/download methods against version drift.
    result = bounded(lambda: run.current_version(ref))
    actual = hashlib.sha256(result.blob.source.encode()).hexdigest()
    if actual != EXPECTED or not result.metadata.is_private or not result.metadata.enable_gpu:
        raise RuntimeError('Actual ref/version/source/private-GPU binding differs')
    return result.metadata


class SessionResolved(Exception):
    pass


def resolve_session(ref):
    owner, slug, version = ref.split('/')
    if owner != 'jiyujia':
        raise ValueError('Only this explicit authorized account is allowed')
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug, request.version_number = owner, slug, int(version)
    found = {}

    def headers_only(response):
        try:
            for part in [*response.history, response]:
                location = part.headers.get('Location')
                if part.status_code != 302 or not location:
                    continue
                parsed = urlparse(urljoin(part.url, location))
                match = re.fullmatch(r'/(?:api/)?v1/kernels/output/download_zip/([1-9][0-9]*)', parsed.path)
                if parsed.scheme == 'https' and parsed.hostname == 'api.kaggle.com' and match:
                    found.update(kernel=ref, source_sha256=EXPECTED,
                        session_id=int(match.group(1)), redirect_host=parsed.hostname,
                        redirect_path=parsed.path, response_body_read=False,
                        request_version_number=int(version), binding='Actual versioned official SDK HTTP302')
        finally:
            response.close()
            for part in response.history:
                part.close()
        raise SessionResolved()

    try:
        with run.api.build_kaggle_client() as client:
            client.http_client()._response_processor = headers_only
            bounded(lambda: client.kernels.kernels_api_client.download_kernel_output(request))
    except SessionResolved:
        pass
    if not found:
        raise RuntimeError('No actual version-bound session redirect available yet')
    save('session-binding.json', found)
    event('actual_session_bound', **found)
    return found


def cancel(binding, ref):
    metadata(ref)
    if binding['kernel'] != ref or binding['source_sha256'] != EXPECTED:
        raise RuntimeError('Ref/source mismatch: never cancel an unrelated session')
    request = ApiCancelKernelSessionRequest()
    request.kernel_session_id = binding['session_id']
    event('watchdog_cancel_request', kernel=ref, session_id=request.kernel_session_id)
    with run.api.build_kaggle_client() as client:
        result = bounded(lambda: client.kernels.kernels_api_client.cancel_kernel_session(request), 10)
    row = dict(kernel=ref, session_id=request.kernel_session_id,
        error_message=result.error_message, source_sha256=EXPECTED,
        at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save('watchdog-cancel.json', row)
    event('watchdog_cancel_response', **row)
    if result.error_message:
        raise RuntimeError(result.error_message)


event('independent_observer_started', pid=os.getpid(), source_sha256=EXPECTED,
    deadline_seconds_from_submit=1200, cancel_margin_seconds=60,
    frozen_child_source_sha256=CHILD,
    hard_runtime_budget='Independent remote Linux bounded parent; API cancellation is auxiliary',
    credentials='Existing execute.connect explicit jiyujia token path; no token embedded')
receipt, ref, binding, previous, cancellation_sent = None, None, None, None, False
deadline_reported = False
while True:
    try:
        path = OUT / 'gpu-push.json'
        if not path.exists():
            time.sleep(1)
            continue
        receipt = json.loads(path.read_text())
        if receipt['source_sha256'] != EXPECTED:
            raise RuntimeError('Receipt source differs; no job may be touched')
        started = dt.datetime.fromisoformat(receipt['at_utc']).timestamp()
        remaining = started + 1200 - time.time()
        ref = receipt.get('kernel')
        if not ref:
            if remaining <= 0:
                event('deadline_without_returned_ref', receipt_status=receipt.get('status'),
                    boundary='Cannot guess a kernel/session id or cancel another job')
                break
            time.sleep(1)
            continue
        if remaining <= 60 and binding is not None and not cancellation_sent:
            cancel(binding, ref)
            cancellation_sent = True
        actual = metadata(ref)
        status = bounded(lambda: run.sdk('get_kernel_session_status', run.request(ref, 'status')))
        state = getattr(status.status, 'name', str(status.status).rsplit('.', 1)[-1])
        row = dict(kernel=ref, status=state, failure_message=status.failure_message,
            source_sha256=EXPECTED, machine_shape=actual.machine_shape,
            remaining_to_local_deadline_seconds=remaining, timeout_readback='Not exposed by public metadata/status API')
        save('observer-status.json', row)
        if state != previous:
            event('actual_session_status', **row)
            previous = state
        if state in TERMINAL:
            event('terminal_session', **row)
            event('artifact_retrieval_deferred', kernel=ref,
                policy='Agent retrieves only exact known research filenames; never generic output traversal')
            try:
                run.quota('after-zero-optimizer-numerics')
            except Exception as error:
                event('quota_after_error', error=type(error).__name__ + ': ' + str(error))
            save('observer-complete.json', row)
            break
        if binding is None:
            try:
                binding = resolve_session(ref)
            except Exception as error:
                event('session_redirect_not_yet_available', kernel=ref,
                    error=type(error).__name__ + ': ' + str(error))
        # Initiate cancellation sixty seconds early to leave bounded network
        # and stop-ack time inside the conservative submit-to-stop1200s window.
        remaining = started + 1200 - time.time()
        if remaining <= 60 and not cancellation_sent:
            if binding is None:
                event('cannot_cancel_without_actual_session_binding', kernel=ref)
            else:
                cancel(binding, ref)
                cancellation_sent = True
        if remaining <= 0 and state not in TERMINAL and not deadline_reported:
            deadline_reported = True
            event('deadline_stop_not_yet_confirmed', kernel=ref, status=state,
                cancellation_sent=cancellation_sent)
        time.sleep(15 if cancellation_sent else (3 if remaining <= 45 else 15))
    except (Exception, LocalDeadline) as error:
        event('observer_error', kernel=ref, error=type(error).__name__ + ': ' + str(error))
        time.sleep(3)
