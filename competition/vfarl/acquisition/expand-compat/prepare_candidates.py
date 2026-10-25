"""Acquire pinned R4 train candidates and exact original verification assets.

This is training-source preparation, not an authored test or verification suite.
"""
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "competition/vfarl/replay"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire

HERE = Path(__file__).resolve().parent
ART = REPLAY / "artifacts/expand-compat"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
CANDIDATES = {
    "cloudpickle": ("cloudpipe__cloudpickle.6220b0ce.lm_rewrite__6cyqq4th", "8cf03dbd0b013111866b195f673a663773875efe", "cloudpipe/cloudpickle", "6220b0ce"),
    "apispec-ref": ("marshmallow-code__apispec.8b421526.pr_958", "44bba1131ed3f88c1b9e3dfcfd8df894a6c09fda", "marshmallow-code/apispec", "8b421526"),
    "apispec-null": ("marshmallow-code__apispec.8b421526.pr_904", "bae3ce82c3600e88b459959df4bab6ef44ad7f39", "marshmallow-code/apispec", "8b421526"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def prepare(key):
    iid, commit, repo, base = CANDIDATES[key]
    task_path = HERE / (iid + ".task.json")
    if not task_path.exists():
        for page in (HERE / "pages").glob("*.json"):
            for record in json.loads(page.read_text())["rows"]:
                if record["row"]["instance_id"] == iid:
                    save(task_path, {"source_dataset": "SWE-bench/SWE-smith-py", "revision": REV,
                                     "split": "train", "row_idx": record["row_idx"], "row": record["row"]})
    row = json.loads(task_path.read_text())["row"]
    source = ART / (key + "-source")
    buggy = acquire(row["repo"], commit, source)
    upstream_root = ART / (repo.replace("/", "__") + "-upstream")
    upstream = acquire(repo, base, upstream_root)
    files, missing = [], []
    for path in sorted(upstream_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(upstream_root)
        if not any(p in {"tests", "test", "testing"} for p in rel.parts) and str(rel) != "conftest.py":
            continue
        raw = path.read_bytes()
        actual = source / rel
        if actual.exists() and actual.read_bytes() != raw:
            raise ValueError("Original verification source differs: " + str(rel))
        files.append({"path": str(rel), "sha256": sha(path), "original_revision": upstream["resolved_revision"],
                      "present_in_buggy_source": actual.exists()})
        if not actual.exists():
            text = raw.decode("utf-8")
            missing.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + str(rel)))
    targets = sorted({node.split("::", 1)[0] for node in row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]})
    assets = {f["path"] for f in files}
    if not all((source / target).exists() or target in assets for target in targets):
        raise ValueError("Original required file unavailable")
    task = {"instance_id": iid, "repo": row["repo"], "base_commit": commit,
            "problem_statement": row["problem_statement"], "patch": "", "test_patch": "".join(missing),
            "FAIL_TO_PASS": row["FAIL_TO_PASS"], "PASS_TO_PASS": row["PASS_TO_PASS"],
            "verification_target_files": targets}
    adapted = ART / (key + "-task.json")
    save(adapted, task)
    license_path = next((upstream_root / name for name in ("LICENSE", "LICENSE.md", "LICENSE.rst", "LICENSE.txt")
                         if (upstream_root / name).exists()), None)
    if license_path is None:
        raise ValueError("Original source license missing")
    copied_license = HERE / (repo.replace("/", "__") + ".LICENSE")
    copied_license.write_bytes(license_path.read_bytes())
    result = {"instance_id": iid, "canonical_repo": repo, "source_dataset": "SWE-bench/SWE-smith-py",
              "source_revision": REV, "physical_split": "train", "original_task": str(task_path.relative_to(ROOT)),
              "original_task_sha256": sha(task_path), "execution_commit": commit, "buggy_source": buggy,
              "upstream_source": upstream, "workspace": str(source.relative_to(ROOT)),
              "adapted_task": str(adapted.relative_to(ROOT)), "adapted_task_sha256": sha(adapted),
              "verification_target_files": targets, "verification_assets": files,
              "source_FAIL_TO_PASS_count": len(row["FAIL_TO_PASS"]), "source_PASS_TO_PASS_count": len(row["PASS_TO_PASS"]),
              "primary_adapter": "R", "submechanism": "R4", "original_patch_semantics": "bug_introducing",
              "original_verifier_requirements_relaxed": False, "required_node_identifiers_verbatim_preserved": True,
              "source_license": {"path": str(copied_license.relative_to(ROOT)), "sha256": sha(copied_license),
                                 "revision": upstream["resolved_revision"]},
              "dataset_license": {"path": "competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt",
                                  "sha256": sha(ROOT / "competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt"),
                                  "declared_card_license": "MIT", "revision": REV}}
    save(REPLAY / ("expand-r4-" + key + "-source-preparation.json"), result)
    return result


if __name__ == "__main__":
    for name in CANDIDATES:
        result = prepare(name)
        print(json.dumps({key: result[key] for key in ("instance_id", "execution_commit", "workspace", "source_FAIL_TO_PASS_count", "source_PASS_TO_PASS_count")} ), flush=True)
