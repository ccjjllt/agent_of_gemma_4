"""Acquire F5-fresh Gym/V2 buggy workspaces (base commit = buggy state)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-fresh"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire

JOBS = [
 ("f5-fresh-dask-trim-chunks", "dask/dask", "1d4064d0ef5794f7dc61bf3e29b5ae234d04ed4d"),
 ("f5-fresh-bokeh-stream-rollover", "bokeh/bokeh", "06a6bf5d7544500431d0fa106c9fec50f15436bf"),
 ("f5-fresh-k8s-iter-lines", "kubernetes-client/python", "76ec6f5ce5572f66d82717e296480dc42584f0b7"),
 ("f5-fresh-ably-release", "ably/ably-python", "9eae66dd720b0efeae7097ad00c12c42348f1f0b"),
 ("f5-fresh-k8sasyncio-str-log", "tomplus/kubernetes_asyncio", "7222c989d38c82c1eabfc7b4342c611cafd20a7a"),
 ("f5-fresh-flit-desc-missing", "takluyver/flit", "8498aa7b97f4e58d99d5b06d20ff8220f83de8e5"),
 ("f5-fresh-wtforms-option-validators", "wtforms/wtforms", "9cc7c7f4d32c8e69c67460e8b8b2aa5ed411da2e"),
 ("f5-fresh-piptools-close-unlink", "jazzband/pip-tools", "49efbec42ba941b7e3cc56770d916b05ec425076"),
 ("f5-fresh-nibabel-writable-buffer", "nipy/nibabel", "e7e3668e52eda7869bafd43f0cda462800ee37f0"),
 ("f5-fresh-copier-hidden-filter", "copier-org/copier", "321662864d0f9070578985aff3206cd75d504e5d"),
 ("f5-fresh-sdk-visited-stack", "meltano/sdk", "ae326fe394ec15d528ec2f1e396b8bb3dd5e99a2"),
 ("f5-fresh-hub-datetime-normalize", "huggingface/huggingface_hub", "d0a948fc2a32ed6e557042a95ef3e4af97ec4a7c"),
]


def main(keys=None):
    for key, repo, base in JOBS:
        if keys and key not in keys:
            continue
        dest = ART / (key + "-source")
        print(f"acquire {repo} {base[:8]} -> {key}", flush=True)
        got = acquire(repo, base, dest)
        assert got["resolved_revision"] == base, (key, got)
        print(f"ready {key}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
