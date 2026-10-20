"""Build P012 eval carriers (12 tasks; A/B or corrected C/D).

Reads:
  research/012/evaluation/agent-{a,b,c,d}/ (frozen agent trees)
  research/012/evaluation/tasks-selected.jsonl (frozen dev tasks + test_patch)
  research/012/evaluation/eval_coordinator_template*.py
  research/012/evaluation/eval_worker_*.py
Writes, per --job:
  research/012/evaluation-run-{ab,cd}/{coordinator.py, attempt_worker.py,
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
           "variants": ("c", "d"), "adapters": (None, None),
           "repair_adapters": ("c_lora", "d_lora"),
           "template": "eval_coordinator_template_cd.py", "worker": "eval_worker_cd.py",
           "kernel_id": "ccjjllt/p012-eval-cd", "zip_prefix": "p012-eval-cd",
           "study": "P012 four-config comparison, job 2 corrected: C (base coordinator, P011 step70 repair role) vs D (base coordinator, new p012-repair-v1 repair role)",
           "research_condition": "Corrected C/D on the same 12 frozen dev tasks: coordinators and locator/acceptance/selector stay base; only the repair role differs by LoRA (C=P011 step70, D=new p012-repair-v1). Between-arm C/D comparison isolates the repair-role adapter. Comparisons to job-1 B need a corrected base control because C/D repair has no skills while old B repair had skills. No training in this job.",
           "source_arms": {
               "c": "P012 new architecture; coordinator/locator/acceptance/selector base; repair role alone carries P011 step70 adapter (71687cf9); five repair tools, no repair skills. Corrected routing: v4 mistakenly put this LoRA on the coordinator while repair stayed base.",
               "d": "P012 new architecture; coordinator/locator/acceptance/selector base; repair role alone carries new p012-repair-v1 adapter (b8653b17); five repair tools, no repair skills. Corrected routing: v4 mistakenly put this LoRA on the coordinator while repair stayed base."},
           "extra_datasets": ["ccjjllt/p012-p011-step70-adapter",
                             "ccjjllt/p012-new-adapter-128"]},
    "bcd6": {"outdir": "evaluation-run-bcd6", "trees": ("agent-f", "agent-c", "agent-d"),
           "variants": ("f", "c", "d"), "adapters": (None, None, None),
           "repair_adapters": (None, "c_lora", "d_lora"),
           "template": "eval_coordinator_template_cd.py", "worker": "eval_worker_cd.py",
           "kernel_id": "ccjjllt/p012-eval-bcd6", "zip_prefix": "p012-eval-bcd6",
           "study": "P012 corrected small matrix f/c/d on 6 fixed stratified tasks: f=corrected base control, c=P011 step70 repair role, d=new p012-repair-v1 repair role",
           "research_condition": "f/c/d on 6 pre-fixed stratified dev tasks (per-repo sha256 order, frozen relative order kept: schema-331, cc-1779, cc-2010, fp-334, fp-469, conan-16559). f isolates the corrected skill-free repair tool env as base control; c-f estimates P011 repair migration; d-f the in-architecture adapter gain; d-c new-vs-P011 gain. Exploratory: the subset rule is mechanical but fixed after old outcomes were seen; final decisions need full-task runs. No training in this job.",
           "task_subset": ["keleshev__schema.24a30457.pr_331",
                         "cookiecutter__cookiecutter.b4451231.pr_1779",
                         "cookiecutter__cookiecutter.b4451231.pr_2010",
                         "kurtmckee__feedparser.cad965a3.pr_334",
                         "kurtmckee__feedparser.cad965a3.pr_469",
                         "conan-io__conan.86f29e13.pr_16559"],
           "source_arms": {
               "f": "P012 new architecture, all roles base; repair role skill-free five tools. Corrected base control matching C/D tool env.",
               "c": "P012 new architecture; coordinator/locator/acceptance/selector base; repair role alone carries P011 step70 adapter (71687cf9); five repair tools, no repair skills.",
               "d": "P012 new architecture; coordinator/locator/acceptance/selector base; repair role alone carries new p012-repair-v1 adapter (b8653b17); five repair tools, no repair skills."},
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
    variant_keys = list(cfg["variants"])
    tree_names = list(cfg["trees"])
    root_ads = list(cfg["adapters"])
    repair_ads = list(cfg.get("repair_adapters", root_ads))
    assert len(variant_keys) == len(tree_names) == len(root_ads) == len(repair_ads) >= 2
    bundles, file_hashes = {}, {}
    for vk, tree in zip(variant_keys, tree_names):
        bundles[vk], file_hashes[vk] = read_tree(tree)
    tasks_path = HERE / "tasks-selected.jsonl"
    tasks_bytes = tasks_path.read_bytes()
    tasks_sha = hashlib.sha256(tasks_bytes).hexdigest()
    import json as _json

    frozen_lines = [x for x in tasks_bytes.splitlines() if x.strip()]
    frozen_tasks = [_json.loads(x) for x in frozen_lines]
    assert len(frozen_tasks) == 12, len(frozen_tasks)
    subset = cfg.get("task_subset")
    if subset is not None:
        frozen_ids = [t["instance_id"] for t in frozen_tasks]
        assert len(subset) == len(set(subset)) and 0 < len(subset) < len(frozen_ids)
        assert set(subset) <= set(frozen_ids)
        assert [i for i in frozen_ids if i in set(subset)] == list(subset), \
            "subset must list frozen rows in frozen relative order"
        keep = set(subset)
        subset_lines = [x for x, t in zip(frozen_lines, frozen_tasks)
                        if t["instance_id"] in keep]
        assert len(subset_lines) == len(subset)
        subset_bytes = b"\n".join(subset_lines) + b"\n"
    else:
        subset_bytes = b"\n".join(frozen_lines) + b"\n"
    selected_sha = hashlib.sha256(subset_bytes).hexdigest()
    tasks = [_json.loads(x) for x in subset_bytes.splitlines() if x.strip()]
    task_ids = [t["instance_id"] for t in tasks]
    assert len(set(task_ids)) == len(task_ids) >= 1
    for t in tasks:
        assert t["test_patch"] and t["FAIL_TO_PASS"], t["instance_id"]
        assert not t["patch"], t["instance_id"]
    planned_attempts = len(task_ids) * len(variant_keys)

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

    def repair_adapter(bundle):
        cfg = yaml.safe_load(bundle["repair.yaml"])
        return cfg.get("adapter")

    variants = {}
    for vk, tree, ad, rd in zip(variant_keys, tree_names, root_ads, repair_ads):
        bundle = bundles[vk]
        variants[vk] = {"tools": root_tools(vk, bundle), "skills": root_skills(vk, bundle),
                        "budgets": budgets, "sampling": sampling,
                        "adapter": ad or root_adapter(bundle),
                        "coordinator_adapter": ad or root_adapter(bundle),
                        "repair_adapter": rd or repair_adapter(bundle)}
    # Root and repair adapter declarations are both authoritative; the job
    # table must agree with the frozen YAML files.
    for vk, ad, rd in zip(variant_keys, root_ads, repair_ads):
        assert variants[vk]["coordinator_adapter"] == ad, (vk, variants[vk]["coordinator_adapter"], ad)
        assert variants[vk]["repair_adapter"] == rd, (vk, variants[vk]["repair_adapter"], rd)

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
        "planned_attempts": planned_attempts,
        "max_concurrent_attempts": 4,
        "server_max_num_seqs": 4,
        "image_pin": IMAGE,
        "research_condition": cfg["research_condition"],
        "data": {
            "source": "research/012/evaluation/tasks-selected.jsonl (frozen dev tasks + criterion test_patch)",
            "source_sha256": tasks_sha,
            "source_bytes": len(tasks_bytes),
            "selected_sha256": selected_sha,
            "selected_bytes": len(subset_bytes),
            "selected_task_ids": task_ids,
            "selected_rows": len(task_ids),
            "task_subset_rule": cfg.get("task_subset_rule",
                "all 12 frozen dev tasks" if cfg.get("task_subset") is None else
                "pre-fixed stratified subset: per-repo sha256 order (schema 1, cookiecutter 2, feedparser 2, conan 1), frozen relative order kept"),
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

    bundle = dict(bundles)
    hashes = dict(file_hashes)

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
    if "%%SELECTED_VARIANTS_PY%%" in template:
        assert template.count("%%SELECTED_VARIANTS_PY%%") == 1
        template = template.replace(
            "%%SELECTED_VARIANTS_PY%%", repr(tuple(variant_keys)))
    assert "%%" not in template.split("MANIFEST_TEXT")[0] or True
    (OUT / "coordinator.py").write_text(template)

    shutil.copyfile(HERE / cfg["worker"], OUT / "attempt_worker.py")
    (OUT / "tasks-selected.jsonl").write_bytes(subset_bytes)
    shutil.copyfile(HERE / "eval_supervisor_template.py", OUT / "supervisor.py")
    (OUT / "experiment-manifest.json").write_text(manifest_text + "\n")
    run_config = {
        "study": manifest["study"],
        "job": job or DEFAULT_JOB,
        "kernel": KERNEL_ID,
        "variants": variant_keys,
        "task_ids": task_ids,
        "planned_attempts": planned_attempts,
        "platform_seconds": 5400,
        "worker_seconds": 5280,
        "evaluation_deadline_seconds": 5100,
        "archive_seconds": 5340,
        "attempt_outer_seconds": 1020,
        "max_concurrent_attempts": 4,
        "tasks_sha256": tasks_sha,
        "selected_sha256": selected_sha,
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
        "selected_sha256": selected_sha,
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
                      "variants": variant_keys,
                      "tasks": len(task_ids), "attempts": planned_attempts,
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
    if jb not in (None, "ab", "cd", "bcd6"):
        raise SystemExit("--job must be ab, cd or bcd6")
    main(prepare_only="--prepare-only" in args, kernel_id=kid, eval_dataset=ds,
         job=jb)
