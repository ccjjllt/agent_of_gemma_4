#!/usr/bin/env python3
"""Stream the fixed private Kaggle NF4 cache directly to an existing SSH host.

Only signed file URLs cross SSH stdin; the account token stays in this process.
Downloads use three workers, restart incomplete .part files, and publish files
only after their fixed size/SHA match. No GPU or Kaggle job is started.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


MANIFEST_SHA = "3a9661d56f0eec51474741b01d7ddb2932fb4836d7f776051d8ece7c333750cf"
SOURCE = "google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
REMOTE = r'''
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

payload = json.load(sys.stdin)
root = Path(payload["destination"])
root.mkdir(parents=True, exist_ok=True)
started = time.monotonic()

def log(name, state, **details):
    print(json.dumps({"file": name, "state": state,
                      "elapsed_seconds": round(time.monotonic() - started, 2),
                      **details}), flush=True)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(item):
    name = item["name"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("unsafe filename")
    url = item["url"]
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("HTTPS required")
    target = root / name
    if target.is_symlink():
        raise ValueError("symlink target")
    if (target.is_file() and target.stat().st_size == item["bytes"]
            and digest(target) == item["sha256"]):
        log(name, "reused", bytes=item["bytes"])
        return
    partial = root / (name + ".part")
    if partial.is_symlink():
        raise ValueError("symlink partial")
    if partial.exists():
        log(name, "restart_partial", previous_bytes=partial.stat().st_size)
    h = hashlib.sha256()
    size = 0
    next_report = 256 * 1024 * 1024
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError("unexpected HTTP status")
            with partial.open("wb") as out:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > item["bytes"]:
                        raise ValueError("size exceeded")
                    out.write(chunk)
                    h.update(chunk)
                    if size >= next_report:
                        log(name, "downloading", bytes=size)
                        next_report += 256 * 1024 * 1024
        if size != item["bytes"] or h.hexdigest() != item["sha256"]:
            raise ValueError("size or SHA256 mismatch")
        partial.replace(target)
        log(name, "complete", bytes=size, sha256=h.hexdigest())
    except Exception as exc:
        # URL-bearing exception messages and tracebacks must never be emitted.
        log(name, "failed", bytes=size, error_type=type(exc).__name__,
            **({"http_status": exc.code} if isinstance(exc, urllib.error.HTTPError) else {}))
        raise RuntimeError("download failed") from None

try:
    # Exact source-manifest bytes are checked before any weight download.
    download(payload["manifest_file"])
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, item): item["name"] for item in payload["files"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                failed = True
    if failed:
        sys.exit(1)
    log("cache", "complete", files=len(payload["files"]),
        bytes=sum(item["bytes"] for item in payload["files"]),
        manifest_sha256=payload["manifest_file"]["sha256"])
except Exception as exc:
    log("cache", "failed", error_type=type(exc).__name__)
    sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH user@host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--remote-log", help="Detach download and write safe progress to this absolute remote log")
    parser.add_argument("--token-file", type=Path,
                        default=Path.home() / ".kaggle/access_token_zhuating")
    parser.add_argument("--destination", default="/workspace/models/p005-cpu-nf4-cache/qat-nf4")
    args = parser.parse_args()
    if args.host.startswith("-") or not 1 <= args.port <= 65535:
        parser.error("invalid SSH host/port")
    if not args.destination.startswith("/"):
        parser.error("destination must be an absolute remote path")
    if args.remote_log and not args.remote_log.startswith("/"):
        parser.error("remote log must be an absolute path")

    repo = Path(__file__).resolve().parents[3]
    manifest_path = repo / "research/005/kaggle-training-probe-20261010/cpu-cache-output/qat-nf4/cache-manifest.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("local fixed manifest changed")
    manifest = json.loads(raw)
    if manifest["source"] != SOURCE:
        raise ValueError("unexpected source")
    wanted = {"qat-nf4/" + item["name"] for item in manifest["files"]}
    wanted.add("qat-nf4/cache-manifest.json")

    # Direct SDK avoids kaggle package auto-authentication and never downloads locally.
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    urls = {}
    with KaggleClient(api_token=args.token_file.read_text().strip(), verbose=False) as client:
        page_token = ""
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = "ccjjllt"
            request.kernel_slug = "p005-cpu-nf4-cache"
            request.page_size = 200
            request.page_token = page_token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name in wanted:
                    if item.file_name in urls:
                        raise ValueError("duplicate output filename")
                    urls[item.file_name] = item.url
            page_token = response.next_page_token
            if not page_token:
                break
    if set(urls) != wanted:
        raise ValueError("required cache output files missing")

    payload = {
        "destination": args.destination,
        "manifest_file": {"name": "cache-manifest.json", "bytes": len(raw),
                          "sha256": MANIFEST_SHA, "url": urls["qat-nf4/cache-manifest.json"]},
        "files": [{**item, "url": urls["qat-nf4/" + item["name"]]}
                  for item in manifest["files"]],
    }
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-p", str(args.port)]
    if args.ssh_key:
        ssh.extend(["-i", str(args.ssh_key)])
    remote_code = REMOTE
    if args.remote_log:
        # URLs stay in stdin/memory; the child has no SSH-owned descriptors.
        remote_code = ("import json,subprocess,sys\nfrom pathlib import Path\n"
                       "payload=json.load(sys.stdin)\n"
                       "log=Path(" + repr(args.remote_log) + ")\n"
                       "log.parent.mkdir(parents=True,exist_ok=True)\n"
                       "with log.open('xb') as out:\n"
                       " p=subprocess.Popen([sys.executable,'-c'," + repr(REMOTE) + "],stdin=subprocess.PIPE,stdout=out,stderr=out,start_new_session=True)\n"
                       " p.stdin.write(json.dumps(payload).encode());p.stdin.close()\n"
                       "print(json.dumps({'state':'download_started','pid':p.pid,'log':str(log)}),flush=True)\n")
    ssh.extend([args.host, "python3 -c " + shlex.quote(remote_code)])
    print(json.dumps({"state": "sending_download_plan", "files": len(payload["files"]),
                      "bytes": manifest["total_bytes"], "manifest_sha256": MANIFEST_SHA}), flush=True)
    return subprocess.run(ssh, input=json.dumps(payload), text=True).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # SDK/network exceptions may contain a signed URL; emit only safe metadata.
        print(json.dumps({"state": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
