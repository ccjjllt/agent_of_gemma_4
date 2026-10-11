#!/usr/bin/env python3
"""T2 span state: per-task nonce + handle records in the execution sandbox temp dir.

Stdlib only. No network. Never reads reference patches, heldout content, host
model/data directories, or other attempts.

Why this file exists (§6细则2):
  ADK SkillToolset extracts each skill into a TemporaryDirectory, chdirs there,
  executes, then restores cwd and recycles the directory on normal exit. State
  kept in the skill directory or inferred from bare cwd therefore does not
  survive across skill calls and cannot identify the task repository. There is
  also no declared per-task nonce injected into skill argv (cf. ADK
  skill_toolset.py:625-709 TemporaryDirectory semantics + swegemma0.2.7
  agent_runner.py:327,392 showing no automatic nonce plumbing).

What this module does instead:
  - Takes the explicit --root passed to source_edit.py (default /workspace),
    validates it exactly like the frozen navigation skill (resolve, reject
    symlink root, reject non-/workspace roots unless --allow-non-workspace-root,
    reject missing dirs, reject escapes/symlinks per file).
  - Resolves the true task root via `git rev-parse --show-toplevel` when inside
    a git checkout (fallback: resolved --root), and reads HEAD via
    `git rev-parse HEAD` (fallback "no-git"). Both come from live sandbox
    execution, not from any injected variable.
  - Constructs the state directory in code under the sandbox system temp base
    ($TMPDIR or /tmp via tempfile.gettempdir()), e.g.
    <tmp>/p002_t2_spans/root_<sha16>/store.json. The model cannot specify this
    root; no --state-root flag exists. The directory is outside the repo, so it
    never enters `git diff`, and it persists across skill calls within one task
    sandbox (unlike the per-call skill TemporaryDirectory).
  - First access creates a random nonce (secrets.token_hex) persisted in
    store.json alongside root identity + HEAD; later calls reload and reuse it.
    If the live root identity or HEAD differs (next task reusing the same /tmp
    on one host), the old spans are discarded and a fresh nonce is created:
    handles never cross tasks. Env TASK_NONCE-style injection is never trusted.

Record key (细则1): span binds nonce + repo toplevel + HEAD + relative path +
file byte sha256 + 1-based line range + byte interval + original slice bytes
(kept as decoded text; file was strict-UTF-8 validated, so re-encoding is
lossless for byte accounting).

Safety: atomic JSON writes (tmp + os.replace). Cap 64 spans, prune oldest.
"""
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path
from workspace_paths import resolve_task_root, resolve_issue_input
from typing import Dict, Optional, Tuple

STORE_DIRNAME = "p002_t2_spans"
STORE_FILENAME = "store.json"
MAX_SPANS = 64


def system_temp_base() -> Path:
    for key in ("TMPDIR", "TEMP", "TMP"):
        v = os.environ.get(key)
        if v:
            try:
                p = Path(v)
                if p.is_dir():
                    return p
            except OSError:
                continue
    try:
        return Path(tempfile.gettempdir())
    except Exception:
        return Path("/tmp")


def validate_task_root(root_arg, allow_non_ws):
    return resolve_task_root(root_arg, allow_non_ws)


def _git_out(args, cwd: Path, timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def repo_identity(root: Path) -> Tuple[str, str]:
    """Return (toplevel_str, head_sha) from live sandbox execution."""
    top = _git_out(["rev-parse", "--show-toplevel"], cwd=root)
    if top:
        try:
            toplevel = str(Path(top).resolve())
        except OSError:
            toplevel = top
    else:
        toplevel = str(root)
    head = _git_out(["rev-parse", "HEAD"], cwd=root) or "no-git"
    return toplevel, head


def state_dir_for(toplevel: str) -> Path:
    base = system_temp_base() / STORE_DIRNAME
    h = hashlib.sha256(toplevel.encode("utf-8")).hexdigest()[:16]
    d = base / f"root_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_path_for(toplevel: str) -> Path:
    return state_dir_for(toplevel) / STORE_FILENAME


def _blank_store(nonce: str, toplevel: str, head: str) -> dict:
    return {
        "meta": {"nonce": nonce, "toplevel": toplevel, "head": head},
        "spans": {},
    }


def load_or_init(root: Path) -> Tuple[dict, Path]:
    """Load store for this task; create/reset nonce on first access or on
    root/HEAD mismatch (next task must not reuse handles)."""
    toplevel, head = repo_identity(root)
    spath = store_path_for(toplevel)
    if spath.is_file():
        try:
            raw = spath.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            meta = data.get("meta", {})
            if (
                meta.get("toplevel") == toplevel
                and meta.get("head") == head
                and isinstance(meta.get("nonce"), str)
                and meta["nonce"]
                and isinstance(data.get("spans"), dict)
            ):
                return data, spath
        except Exception:
            pass
        # Mismatch or corrupt: fall through to fresh store (old handles die).
    data = _blank_store(secrets.token_hex(8), toplevel, head)
    _atomic_write_json(spath, data)
    return data, spath


def _atomic_write_json(spath: Path, data: dict) -> None:
    spath.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False)
    tmp = spath.parent / (STORE_FILENAME + f".{os.getpid()}.tmp")
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, spath)


def save_store(spath: Path, data: dict) -> None:
    if len(data.get("spans", {})) > MAX_SPANS:
        # Prune oldest by created_seq.
        items = sorted(
            data["spans"].items(), key=lambda kv: kv[1].get("created_seq", 0)
        )
        data["spans"] = dict(items[-MAX_SPANS:])
    _atomic_write_json(spath, data)


def new_span_id(existing: dict) -> str:
    for _ in range(32):
        sid = "sp_" + secrets.token_hex(8)
        if sid not in existing:
            return sid
    raise RuntimeError("span id space exhausted")
