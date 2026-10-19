"""Stage P012 dev-task source subsets (task prep only; never training input).

For each frozen dev12 item: atomically fetch the trajectory row
(instance_id+traj_id+row_index), extract production str_replace paths,
resolve the fixed buggy commit via `git ls-remote` on the swesmith mirror,
download those files + a license file, and write provenance.json.

Writes only under research/012/evaluation/task-env/. No behavioral
verification here; PRE/POST discriminative checks are separate one-off runs
recorded in task-prep-pilot.md (pilots) or follow-up notes.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FREEZE = HERE / "dev12-freeze.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(iid: str) -> str:
    prefix = re.split(r"\.[0-9a-f]{8}\.", iid)[0]
    return prefix.replace("__", "/", 1).lower()


def production(path: str) -> bool:
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and p.suffix == ".py"
        and not any(
            x.lower() in ("test", "tests", "test-data", "docs", "doc", "examples", ".github")
            for x in p.parts
        )
        and not p.name.lower().startswith(
            ("test", "reproduce", "debug", "verify", "check", "temp", "setup")
        )
        and not p.name.endswith("_test.py")
    )


def fetch_messages(source_file: str, row_index: int, iid: str, traj: str) -> str:
    import duckdb

    c = duckdb.connect()
    got = c.execute(
        "SELECT messages FROM read_parquet(?) WHERE file_row_number=? AND instance_id=? AND traj_id=?",
        [str(ROOT / source_file), row_index, iid, traj],
    ).fetchone()
    if not got:
        raise ValueError("atomic source identity mismatch")
    return got[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    items = json.loads(FREEZE.read_text())["items"]
    done = []
    for it in items[: args.limit or len(items)]:
        iid = it["instance_id"]
        out = HERE / "task-env" / iid
        if args.skip_existing and (out / "provenance.json").is_file():
            done.append(dict(iid=iid, status="skipped_existing"))
            continue
        messages = fetch_messages(
            it["source_file"], it["row_index"], iid, it["selected_traj_id"]
        )
        msgs = json.loads(messages)
        paths: set[str] = set()
        for m in msgs:
            calls = list(m.get("tool_calls", []) or [])
            content = m.get("content", "")
            if isinstance(content, str):
                for name, body in re.findall(
                    r"<function=([^>]+)>(.*?)</function>", content, re.S
                ):
                    params = {
                        k: v
                        for k, v in re.findall(
                            r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S
                        )
                    }
                    calls.append({"function": {"name": name, "arguments": params}})
            for call in calls:
                f = call.get("function", {})
                raw = f.get("arguments", {})
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError):
                    continue
                if a.get("command") != "str_replace":
                    continue
                p = a.get("path", "").removeprefix("/testbed/")
                if production(p):
                    paths.add(p)
        if not paths:
            done.append(dict(iid=iid, status="no_production_paths"))
            continue
        mirror = "swesmith/" + iid.split(".pr_")[0]
        ref = subprocess.run(
            ["git", "ls-remote", f"https://github.com/{mirror}.git", f"refs/heads/{iid}"],
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        ).stdout
        commit = ref.split()[0]
        if not re.fullmatch("[0-9a-f]{40}", commit):
            raise ValueError(f"bad commit for {iid}: {ref!r}")
        srcdir = out / "source"
        files = []
        for p in sorted(paths):
            url = f"https://raw.githubusercontent.com/{mirror}/{commit}/{p}"
            with urllib.request.urlopen(url, timeout=45) as r:
                data = r.read(2_000_001)
            dest = srcdir / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            files.append({"path": p, "bytes": len(data), "sha256": sha(data)})
        lic = None
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            try:
                with urllib.request.urlopen(
                    f"https://raw.githubusercontent.com/{mirror}/{commit}/{name}",
                    timeout=45,
                ) as r:
                    data = r.read(2_000_001)
            except Exception:
                continue
            dest = srcdir / name
            dest.write_bytes(data)
            lic = {"path": name, "bytes": len(data), "sha256": sha(data)}
            files.append(lic)
            break
        prov = {
            "instance_id": iid,
            "repo": it["repo"],
            "mirror": mirror,
            "buggy_commit": commit,
            "git_ls_remote_stdout": ref.strip(),
            "atomic_source": {
                "source_file": it["source_file"],
                "row_index": it["row_index"],
                "traj_id": it["selected_traj_id"],
                "teacher_source": it["teacher_source"],
                "match": "instance_id+traj_id+row verified",
            },
            "files": files,
            "license_note": "stored" if lic else "no top-level license file at buggy commit",
            "scope": "fixed-commit production source subset for task prep; discriminative PRE/POST recorded separately. No training transactions written.",
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "provenance.json").write_text(
            json.dumps(prov, ensure_ascii=False, indent=2) + "\n"
        )
        done.append(
            dict(iid=iid, status="staged", commit=commit, files=len(files))
        )
        print(json.dumps(done[-1]), flush=True)
    print(json.dumps({"staged": done}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
