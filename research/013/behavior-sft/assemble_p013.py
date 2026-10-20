"""Assemble P013 behavior SFT input (p013-behavior-v1, 38 admitted transactions).

Reads the 38 admitted canonical-replay transactions under
research/013/behavior-data/outputs/ (batch-01 unprefixed dirs + b02-*/b03-*
dirs; refusal-only dirs contain no transaction.json and are ignored),
gates each on the live behavior evidence (requirement_done, verification
pass, int non-zero pre exit, zero post exit, same command SHA, zero-based
canonical-arc targets, train split, train-source guard), encodes the full
tool sequences with the 12B CT tokenizer + frozen chat template + five-tool
repair schema (--thinking on --thought-loss mask, whole-transaction 8K
admission, oversize listed separately and NEVER truncated), and freezes
encoded-train-12b-behavior.jsonl + dataset-manifest.json.

The old P012 256-row file is referenced as an auxiliary layer by path+SHA
with a sampling-weight plan only; it is NEVER copied or concatenated here
(its rows carry protocol p012-repair-v1 while these carry p013-behavior-v1,
and training/common.py::load_encoded_jsonl rejects mixed-protocol files).

Refuses to overwrite a frozen output. CPU-only, no heldout reads, no GPU.
"""
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BEHAVIOR_OUT = ROOT / "research/013/behavior-data/outputs"
P012_MANIFEST = ROOT / "research/012/training-data/dataset-manifest.json"
PROTOCOL = "p013-behavior-v1"
MAX_SEQ = 8192

TOKENIZER = ROOT / "training/artifacts/tokenizers/gemma4-12b-ct-v2"
TOOLS = ROOT / "research/012/agent/repair_tools_v1.json"
CHAT_TEMPLATE = ROOT / "training/configs/chat_templates/gemma4_12b_ct_v2.jinja"
ENCODER = ROOT / "training/encoding/gemma_tool_chat.py"

RESERVED_REPOS = {
    "conan-io/conan", "cookiecutter/cookiecutter", "encode/httpx", "fastapi/fastapi",
    "keleshev/schema", "kurtmckee/feedparser", "modin-project/modin", "pallets/click",
    "pallets/markupsafe", "pandas-dev/pandas", "psf/requests", "pydicom/pydicom",
    "pytest-dev/iniconfig", "textualize/rich", "tkrajina/gpxpy",
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))


def load_denylist():
    splits = json.loads((ROOT / "research/task_splits.json").read_text())
    denied = {r["instance_id"] for r in splits["records"] if r["split"] != "train"}
    denied |= set(splits.get("dev_panel_ids", []))
    return denied


def gate_transaction(tx, path, denylist):
    """Admit only live-verified behavior recoveries. Raises on any gate failure."""
    if tx.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol mismatch: %s in %s" % (tx.get("protocol_id"), path))
    group = tx.get("group") or {}
    if group.get("split") != "train":
        raise ValueError("non-train split in %s" % path)
    repo = group.get("repo") or ""
    if repo.lower() in RESERVED_REPOS:
        raise ValueError("reserved repo refused: %s" % repo)
    issue = group.get("issue") or ""
    if issue in denylist:
        raise ValueError("non-train split instance refused: %s" % issue)
    handover = tx.get("handover") or {}
    if handover.get("requirement_done") is not True:
        raise ValueError("requirement_done is not true in %s" % path)
    if handover.get("verification_status") != "pass":
        raise ValueError("verification_status != pass in %s" % path)
    verification = handover.get("verification") or {}
    pre = verification.get("pre_exit_code")
    post = verification.get("post_exit_code")
    if type(pre) is not int or pre == 0:
        raise ValueError("pre_exit_code must be int non-zero in %s" % path)
    if post != 0:
        raise ValueError("post_exit_code must be 0 in %s" % path)
    if verification.get("pre_command_sha256") != verification.get("post_command_sha256"):
        raise ValueError("pre/post command differ in %s" % path)
    observed = tx.get("observed_outcome") or {}
    if observed.get("behavior_verified") is not True:
        raise ValueError("behavior_verified is not true in %s" % path)
    if observed.get("task_resolved") is not None:
        raise ValueError("task_resolved must stay null in %s" % path)
    if tx.get("synthetic") is not False:
        raise ValueError("synthetic must be false in %s" % path)
    targets = tx.get("target_assistant_steps")
    messages = tx.get("messages") or []
    n_asst = sum(1 for m in messages if m.get("role") == "assistant")
    if (not isinstance(targets, list) or not targets
            or any(type(t) is not int or t < 0 or t >= n_asst for t in targets)):
        raise ValueError("bad target_assistant_steps in %s" % path)
    kinds = [m.get("tool_calls", [{}])[0].get("function", {}).get("name")
             if m.get("tool_calls") else None for m in messages if m.get("role") == "assistant"]
    flat = [k for k in kinds if k]
    if "read_file" not in flat or flat.count("run_command") < 2 or "edit_file" not in flat:
        raise ValueError("tool arc incomplete (need read/pre-run/edit/post-run) in %s" % path)
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        raise ValueError("final message must be the handover assistant turn in %s" % path)
    tail = json.loads(last.get("content") or "{}")
    if tail.get("requirement_done") is not True:
        raise ValueError("handover tail not requirement_done in %s" % path)
    return issue


def main():
    final = HERE / "encoded-train-12b-behavior.jsonl"
    if final.exists():
        raise RuntimeError("Frozen output exists; create a new version instead of overwriting")
    denylist = load_denylist()
    paths = sorted(set(glob.glob(str(BEHAVIOR_OUT / "*" / "transaction.json")))
                   | set(glob.glob(str(BEHAVIOR_OUT / "b*" / "transaction.json"))))
    if not paths:
        raise RuntimeError("no transaction.json found under %s" % BEHAVIOR_OUT)
    transactions, issues = [], []
    for path in paths:
        tx = json.loads(Path(path).read_text(encoding="utf-8"))
        issues.append(gate_transaction(tx, path, denylist))
        transactions.append(tx)
    order = sorted(range(len(transactions)), key=lambda i: transactions[i].get("sample_id", ""))
    transactions = [transactions[i] for i in order]
    issues = [issues[i] for i in order]
    if len(transactions) != 38:
        raise RuntimeError("expected 38 admitted behavior transactions, found %d" % len(transactions))
    if len(set(issues)) != 38:
        raise ValueError("duplicate issue across admitted transactions")
    inputs = HERE / "transactions-p013-behavior.jsonl"
    write_rows(inputs, transactions)
    encoded = HERE / "encoded-all-p013-behavior.jsonl"
    subprocess.run(
        [sys.executable, str(ENCODER), "encode",
         "--input", str(inputs), "--output", str(encoded),
         "--tokenizer", str(TOKENIZER),
         "--tools", str(TOOLS),
         "--chat-template", str(CHAT_TEMPLATE),
         "--thinking", "on", "--thought-loss", "mask",
         "--max-seq-len", str(MAX_SEQ), "--on-oversize", "skip"],
        check=True)
    report = json.loads((Path(str(encoded) + ".report.json")).read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in encoded.read_text(encoding="utf-8").splitlines() if x.strip()]
    skipped = report.get("skipped_oversize", [])
    # Oversize trajectories are NEVER truncated: they are frozen out of this
    # file and listed verbatim for later long-context training (P012 plan s4).
    if len(rows) + len(skipped) != 38:
        raise RuntimeError("encode coverage gap: %d written + %d skipped != 38 admitted"
                           % (len(rows), len(skipped)))
    provenance = {key: rows[0]["meta"][key]
                  for key in ("tokenizer_hash", "chat_template_hash", "tools_hash",
                              "protocol_id", "protocol_hash")}
    if provenance["protocol_id"] != PROTOCOL:
        raise ValueError("encoded protocol is not %s" % PROTOCOL)
    for row in rows:
        if any(row["meta"][key] != value for key, value in provenance.items()):
            raise ValueError("mixed provenance across encoded rows")
        if row["meta"].get("synthetic") is not False \
                or row["meta"].get("training_eligible") is not True:
            raise ValueError("non-training-eligible row: %s" % row["meta"].get("transaction_id"))
        if row["meta"].get("truncated_tool_block"):
            raise ValueError("truncated tool block must not enter training")
    write_rows(final, rows)
    taken_ids = [r["meta"]["group"]["issue"] for r in rows]
    oversize_ids = sorted(set(issues) - set(taken_ids))
    if len(taken_ids) != len(set(taken_ids)):
        raise ValueError("duplicate issue in encoded rows")
    p012 = json.loads(P012_MANIFEST.read_text(encoding="utf-8"))
    p012_file = {"path": p012["encoded"]["path"], "sha256": p012["encoded"]["sha256"],
                 "rows": p012["rows"], "protocol_id": p012["provenance"]["protocol_id"]}
    lengths = sorted(len(r["input_ids"]) for r in rows)
    manifest = {
        "status": "frozen", "split": "train",
        "encoded": {"path": str(final.relative_to(ROOT)), "sha256": sha(final)},
        "rows": len(rows), "independent_issues": len(set(taken_ids)),
        "independent_repos": len({r["meta"]["group"]["repo"] for r in rows}),
        "max_sequence_length": max(lengths), "training_sequence_limit": MAX_SEQ,
        "provenance": provenance,
        "input_tokens": sum(lengths),
        "supervised_tokens": sum(r["meta"]["supervised_tokens"] for r in rows),
        "length_distribution": {
            "min": lengths[0], "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))], "max": lengths[-1],
            "le6144": sum(n <= 6144 for n in lengths), "le8192": len(lengths)},
        "selection": {
            "order": "sample_id-sorted admitted behavior transactions; whole-transaction "
                     "encodable (<=8192) taken, oversize listed verbatim, never truncated",
            "admitted_transactions": len(transactions),
            "encodable_taken": len(rows),
            "skipped_oversize_before_taken": len(skipped),
            "skipped_oversize": skipped,
            "taken_issues": sorted(taken_ids),
            "oversize_issues": oversize_ids},
        "layers": {
            "p013_behavior_primary": {
                "rows": len(rows), "behavior_verified": True, "task_resolved": None,
                "note": "real tool sequences (read/pre-run/edit/post-run) + verified "
                        "handover; zero-based canonical-arc targets; --thinking on "
                        "--thought-loss mask"},
            "p013_behavior_oversize_deferred": {
                "rows": len(oversize_ids), "issues": oversize_ids,
                "behavior_verified": True, "task_resolved": None,
                "note": "whole-transaction token length exceeds 8192; tool blocks "
                        "never truncated. Listed for later long-context training, "
                        "not in this file."},
            "p012_repair_auxiliary": {
                "rows": p012_file["rows"], "encoded": p012_file,
                "behavior_verified": False, "task_resolved": None,
                "sampling": "reference only, NOT concatenated: protocol "
                            "p012-repair-v1 differs from p013-behavior-v1 and the "
                            "trainer rejects mixed-protocol files. Proposed follow-up "
                            "schedule: stage A trains this 34-row file alone; stage B "
                            "trains the P012 file separately (new optimizer state). "
                            "Suggested sampling weight if a future single-protocol "
                            "mix is built: primary:auxiliary = 1:1 by step "
                            "(alternate global batches), never by row duplication."}},
        "truncated": False, "gpu_or_teacher_calls_for_data": 0,
        "training_executed_by_this_producer": False,
        "source_manifest": {
            "path": str((HERE / "admitted-source-manifest.json").relative_to(ROOT)),
            "sha256": ""},
        "frozen_policy": "Do not overwrite this encoded file after training is launched. "
                         "New work requires another version."}
    entries = []
    by_issue = {}
    repo_of = {}
    for p in map(Path, paths):
        tx = json.loads(p.read_text(encoding="utf-8"))
        gid = (tx.get("group") or {}).get("issue")
        by_issue[gid] = p.parent
        repo_of[gid] = (tx.get("group") or {}).get("repo")
    for issue in sorted(issues):
        folder = by_issue[issue]
        assets = [p for p in sorted(folder.rglob("*")) if p.is_file()]
        entries.append({"issue": issue, "repo": repo_of[issue],
                        "tier": "behavior_recovery", "behavior_verified": True,
                        "task_resolved": None,
                        "encodable_le8192": issue in set(taken_ids),
                        "files": [{"path": str(p.relative_to(ROOT)),
                                   "bytes": p.stat().st_size, "sha256": sha(p)} for p in assets]})
    dump(HERE / "admitted-source-manifest.json", entries)
    manifest["source_manifest"]["sha256"] = sha(HERE / "admitted-source-manifest.json")
    dump(HERE / "dataset-manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("rows", "independent_issues", "independent_repos",
                       "max_sequence_length", "input_tokens", "supervised_tokens",
                       "length_distribution")}, ensure_ascii=False))
    print("manifest:", HERE / "dataset-manifest.json")


if __name__ == "__main__":
    main()
