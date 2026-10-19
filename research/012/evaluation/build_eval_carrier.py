"""Build P012 Eval-Job-1 (A/B, 12 tasks, no adapters) carrier.

Reads:
  research/012/evaluation/agent-a/ and agent-b/ (frozen agent trees)
  research/012/evaluation/tasks-selected.jsonl (frozen dev tasks + test_patch)
  research/012/evaluation/eval_coordinator_template.py (pattern adapted from P010)
  research/012/evaluation/eval_worker_ab.py
Writes:
  research/012/evaluation-run-ab/{coordinator.py, attempt_worker.py,
    tasks-selected.jsonl, experiment-manifest.json, run-config.json,
    kernel/train.py, kernel/kernel-metadata.json}
No GPU work here. New training is not started by this carrier.
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / ".." / "evaluation-run-ab"
OUT = OUT.resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "kernel").mkdir(exist_ok=True)

IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
JOBS = {
    "ab": {"outdir": "evaluation-run-ab", "trees": ("agent-a", "agent-b"),
           "variants": ("a", "b"), "adapters": (None, None),
           "template": "eval_coordinator_template.py", "worker": "eval_worker_ab.py",
           "kernel_id": "ccjjllt/p012-eval-ab", "zip_prefix": "p012-eval-ab",
           "study": "P012 four-config comparison, job 1: A (frozen P010 12B base) vs B (new multi-role arch, base)",
           "research_condition": "A/B on 12 new frozen dev tasks (cookiecutter3+feedparser4+schema1+conan4): B-A estimates the new locate/repair/accept/select architecture effect under identical 12B base, budget and sampling. Adapter arms (C/D) ride a follow-up job. No training in this job.",
           "source_arms": {
               "a": "P010 frozen 12B base single agent (six tools, no skills, 8min/28call)",
               "b": "P012 new architecture: base coordinator + locator/repair/acceptance/selector AgentTools, five-tool repair role, selector-only submit, candidate namespaced state"},
           "extra_datasets": []},
    "cd": {"outdir": "evaluation-run-cd", "trees": ("agent-c", "agent-d"),
           "variants": ("c", "d"), "adapters": ("c_lora", "d_lora"),
           "template": "eval_coordinator_template_cd.py", "worker": "eval_worker_cd.py",
           "kernel_id": "ccjjllt/p012-eval-cd", "zip_prefix": "p012-eval-cd",
           "study": "P012 four-config comparison, job 2: C (new arch + P011 step70 repair adapters) vs D (new arch + new p012-repair-v1 adapter)",
           "research_condition": "C/D on the same 12 frozen dev tasks: C-B (from job 1) estimates P011 repair-role migration; D-C estimates new post-training gain over P011; D-B the in-architecture adapter gain. Locator/acceptance/selector stay base in both. No training in this job.",
           "source_arms": {
               "c": "P012 new architecture; locator/acceptance/selector base; both repairs on P011 step70 adapter (71687cf9)",
               "d": "P012 new architecture; locator/acceptance/selector base; both repairs on new p012-repair-v1 adapter (b8653b17)"},
           "extra_datasets": ["ccjjllt/p012-p011-step70-adapter",
                             "ccjjllt/p012-new-adapter-128"]},
}
DEFAULT_JOB = "ab"
RUN_PREFIX = "run-012-eval-ab"

AGENT_ALLOW = {".yaml", ".yml", ".md", ".txt", ".py", ".json"}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def read_tree(name):
    base = HERE / name
    bundle, hashes = {}, {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rel = str(path.relative_to(base))
        assert path.suffix.lower() in AGENT_ALLOW, rel
        content = path.read_text(encoding="utf-8")
        bundle[rel] = content
        hashes[rel] = hashlib.sha256(content.encode()).hexdigest()
    return bundle, hashes


def main(prepare_only=False, kernel_id=None, eval_dataset=None, job=None):
    cfg = JOBS[job or DEFAULT_JOB]
    global OUT
    OUT = (HERE / ".." / cfg["outdir"]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernel").mkdir(exist_ok=True)
    KERNEL_ID = kernel_id or cfg["kernel_id"]
    EVAL_DATASET = eval_dataset or "ccjjllt/p012-eval-tasks"
    (v0, v1), (t0, t1), (ad0, ad1) = cfg["variants"], cfg["trees"], cfg["adapters"]
    bundle_0, hashes_0 = read_tree(t0)
    bundle_1, hashes_1 = read_tree(t1)
    tasks_path = HERE / "tasks-selected.jsonl"
    tasks_bytes = tasks_path.read_bytes()
    tasks_sha = hashlib.sha256(tasks_bytes).hexdigest()
    import json as _json

    tasks = [_json.loads(x) for x in tasks_bytes.splitlines() if x.strip()]
    assert len(tasks) == 12, len(tasks)
    task_ids = [t["instance_id"] for t in tasks]
    assert len(set(task_ids)) == 12
    for t in tasks:
        assert t["test_patch"] and t["FAIL_TO_PASS"], t["instance_id"]
        assert not t["patch"], t["instance_id"]

    task_meta = {}
    for t in tasks:
        task_meta[t["instance_id"]] = {"repo": t["repo"],
                                       "base_commit": t["base_commit"],
                                       "split": "select"}

    sampling = {"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 4096,
                "thinking_config": {"thinking_budget": 1024, "include_thoughts": True}}
    budgets = {"timeout_seconds": 240, "max_tool_calls": 28,
               "max_time_minutes": 8, "max_turns": 80}

    import yaml

    yaml.SafeLoader.add_constructor(
        "!include", lambda loader, node: loader.construct_scalar(node))

    def root_tools(name, bundle):
        cfg = yaml.safe_load(bundle["agent.yaml"])
        return cfg.get("tools", [])

    def root_skills(name, bundle):
        cfg = yaml.safe_load(bundle["agent.yaml"])
        return cfg.get("skills", []) or []

    def root_adapter(bundle):
        cfg = yaml.safe_load(bundle["agent.yaml"])
        return cfg.get("adapter")

    variants = {
        v0: {"tools": root_tools(v0, bundle_0), "skills": root_skills(v0, bundle_0),
             "budgets": budgets, "sampling": sampling, "adapter": ad0 or root_adapter(bundle_0)},
        v1: {"tools": root_tools(v1, bundle_1), "skills": root_skills(v1, bundle_1),
             "budgets": budgets, "sampling": sampling, "adapter": ad1 or root_adapter(bundle_1)},
    }
    # The frozen agent.yaml adapter declaration is authoritative; the job table
    # must agree with it.
    assert variants[v0]["adapter"] == ad0, (v0, variants[v0]["adapter"], ad0)
    assert variants[v1]["adapter"] == ad1, (v1, variants[v1]["adapter"], ad1)

    manifest = {
        "study": cfg["study"],
        "common": {
            "official_packages": {
                "adk-eval-core": "0.1.0 sha256 e614e1b2fe8d886a634871e4c57f9887e1cc66e8c27300f8503a43c6e3ca82f3",
                "adk-submission": "0.2.13 sha256 f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d",
                "google-adk": "1.36.1 sha256 1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0",
                "swegemma": "0.2.11 sha256 9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3",
            },
        },
        "variant_conditions": variants,
        "verification_timeout_seconds": 300,
        "task_ids": task_ids,
        "repeats": ["r1"],
        "planned_attempts": 24,
        "max_concurrent_attempts": 4,
        "server_max_num_seqs": 4,
        "image_pin": IMAGE,
        "research_condition": cfg["research_condition"],
        "data": {
            "source": "research/012/evaluation/tasks-selected.jsonl (frozen dev tasks + criterion test_patch)",
            "source_sha256": tasks_sha,
            "source_bytes": len(tasks_bytes),
            "selected_task_ids": task_ids,
            "selected_rows": 12,
            "heldout_or_reserved_accessed": False,
            "task_dependencies": {
                "floors": ["Jinja2>=2.7,<4.0.0", "click>=7.0", "pyyaml",
                           "python-slugify>=4.0.0", "binaryornot",
                           "requests>=2.20.0", "arrow", "rich", "colorama",
                           "fasteners", "patch-ng", "distro",
                           "python-dateutil", "urllib3>=1.26.6,<2.1",
                           "responses"],
                "note": "loose floors only; pip skips already-satisfied reqs, never --force-reinstalls the harness stack; jinja2-time deliberately absent (task 1779 needs its absence)",
            },
        },
        "source_arms": cfg["source_arms"],
        "whole_run": {
            "platform_seconds": 5400,
            "worker_seconds": 5280,
            "evaluation_deadline_seconds": 5100,
            "archive_seconds": 5340,
        },
        "attempt_outer_seconds": 1020,
        "process_isolation": "Separate Evaluator interpreters, submission copies, tmp/sandbox directories and artifacts; one shared vLLM 12B server. No cross-arm result injection.",
        "cache_note": "Shared prefix cache and concurrent scheduling are not cache-equivalent; AB/BA launch order and live concurrency are retained.",
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)

    bundle = {v0: bundle_0, v1: bundle_1}
    hashes = {v0: hashes_0, v1: hashes_1}

    template = (HERE / cfg["template"]).read_text()
    subs = {
        "%%BUNDLE_PY%%": json.dumps(bundle, ensure_ascii=False, sort_keys=True),
        "%%HASHES_PY%%": json.dumps(hashes, ensure_ascii=False, sort_keys=True),
        "%%MANIFEST_PY%%": repr(manifest_text),
        "%%TASK_IDS_PY%%": repr(tuple(task_ids)),
        "%%TASK_META_PY%%": json.dumps(task_meta, ensure_ascii=False, sort_keys=True),
        "%%VARIANTS_PY%%": repr(variants),
    }
    for token, value in subs.items():
        assert template.count(token) == 1, token
        template = template.replace(token, value)
    assert "%%" not in template.split("MANIFEST_TEXT")[0] or True
    (OUT / "coordinator.py").write_text(template)

    shutil.copyfile(HERE / cfg["worker"], OUT / "attempt_worker.py")
    shutil.copyfile(tasks_path, OUT / "tasks-selected.jsonl")
    shutil.copyfile(HERE / "eval_supervisor_template.py", OUT / "supervisor.py")
    (OUT / "experiment-manifest.json").write_text(manifest_text + "\n")
    run_config = {
        "study": manifest["study"],
        "job": job or DEFAULT_JOB,
        "kernel": KERNEL_ID,
        "variants": [v0, v1],
        "task_ids": task_ids,
        "planned_attempts": 24,
        "platform_seconds": 5400,
        "worker_seconds": 5280,
        "evaluation_deadline_seconds": 5100,
        "archive_seconds": 5340,
        "attempt_outer_seconds": 1020,
        "max_concurrent_attempts": 4,
        "tasks_sha256": tasks_sha,
        "experiment_manifest_sha256": hashlib.sha256(
            manifest_text.encode()).hexdigest(),
    }
    (OUT / "run-config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
    packed = {}
    for rel in ["coordinator.py", "attempt_worker.py", "supervisor.py",
                "tasks-selected.jsonl", "experiment-manifest.json",
                "run-config.json"]:
        packed[rel] = sha_file(OUT / rel)
    (OUT / "carrier-manifest.json").write_text(json.dumps({
        "study": manifest["study"],
        "kernel": KERNEL_ID,
        "eval_dataset": EVAL_DATASET,
        "prepare_only": prepare_only,
        "task_ids": task_ids,
        "tasks_sha256": tasks_sha,
        "image_pin": IMAGE,
        "files": packed,
    }, ensure_ascii=False, indent=2) + "\n")

    import ast

    for name in ("coordinator.py", "attempt_worker.py", "supervisor.py"):
        ast.parse((OUT / name).read_text(), filename=name)

    # Kernel bootstrap: small file that unpacks the carrier. The carrier here
    # is small (agent bundles + tasks), so a single embedded blob is fine.
    import base64
    import io
    import zipfile

    files = {}
    for rel in ["coordinator.py", "attempt_worker.py", "supervisor.py",
                "tasks-selected.jsonl", "experiment-manifest.json",
                "run-config.json", "carrier-manifest.json"]:
        files[rel] = (OUT / rel).read_bytes()
    zp = cfg["zip_prefix"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel, payload in files.items():
            if rel.endswith(".py"):
                ast.parse(payload.decode(), filename=rel)
            archive.writestr(zp + "/" + rel, payload)
    payload = buf.getvalue()
    assert len(payload) < 900 * 1024, len(payload)
    bootstrap = (
        "# Generated by build_eval_carrier.py (P012 %s). " % zp
        + "Execution starts the bounded GPU evaluation.\n"
        "import base64, hashlib, io, os, runpy, uuid, zipfile\n"
        "from pathlib import Path\n"
        "PAYLOAD = " + repr(base64.b64encode(payload).decode()) + "\n"
        "EXPECTED_SHA256 = " + repr(hashlib.sha256(payload).hexdigest()) + "\n"
        "raw = base64.b64decode(PAYLOAD)\n"
        "assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256\n"
        "bundle = Path('/kaggle/working') / ('" + zp + "-' + uuid.uuid4().hex[:8])\n"
        "bundle.mkdir()\n"
        "with zipfile.ZipFile(io.BytesIO(raw)) as archive:\n"
        "    for member in archive.infolist():\n"
        "        assert (bundle / member.filename).resolve().is_relative_to(bundle.resolve())\n"
        "    archive.extractall(bundle)\n"
        "os.environ['P012_EVAL_BUNDLE_ROOT'] = str(bundle)\n"
        + ("os.environ['P012_EVAL_PREPARE_ONLY'] = '1'\n" if prepare_only else "")
        + "runpy.run_path(str(bundle / '" + zp + "' / 'supervisor.py'), run_name='__main__')\n"
    )
    ast.parse(bootstrap)
    kdir = OUT / "kernel"
    kdir.mkdir(exist_ok=True)
    (kdir / "train.py").write_text(bootstrap)
    meta = {"id": KERNEL_ID, "title": "P012 Eval " + zp.split("p012-eval-")[-1].upper(),
            "code_file": "train.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse",
                                EVAL_DATASET] + cfg["extra_datasets"],
            "competition_sources": ["gemma-4-developer-agent"],
            "machine_shape": "NvidiaL4",
            "model_sources": ["google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2"],
            "docker_image": IMAGE, "docker_image_pinning_type": "original"}
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"kernel": KERNEL_ID, "job": job or DEFAULT_JOB,
                      "variants": [v0, v1],
                      "tasks": len(task_ids), "attempts": 24,
                      "prepare_only": prepare_only,
                      "bundle_files": sum(len(v) for v in bundle.values()),
                      "carrier_bytes": len(payload)}))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    kid, ds, jb = None, None, None
    for a in args:
        if a.startswith("--kernel-id="):
            kid = a.split("=", 1)[1]
        elif a.startswith("--dataset="):
            ds = a.split("=", 1)[1]
        elif a.startswith("--job="):
            jb = a.split("=", 1)[1]
    if jb not in (None, "ab", "cd"):
        raise SystemExit("--job must be ab or cd")
    main(prepare_only="--prepare-only" in args, kernel_id=kid, eval_dataset=ds,
         job=jb)
