"""F5-fresh finish: unified publish + catalog + normalize + canonical manifest.

Covers Smith (collect_fresh) and Gym/V2 (collect_fresh_gymv2) contracts in one
pass. Reads *-preparation.json + run-001 summaries + JUnit XMLs, writes:
  reports/f5-fresh-<key>-attempt001-replay-result.json
  reviewed-candidates.json (merged, all pools)
  catalog.json (all entries, final SHAs)
  <key>-normalize.json per report (like f2-gymv2c)
  replay/artifacts/f5-fresh/canonical-results/ + canonical-results-manifest.json
Old dirs, f1-f4, S600 untouched.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4")
HERE = ROOT / "competition/vfarl/acquisition/f5-new/f5-fresh"
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-fresh"
REPORTS = HERE / "reports"
CANON = ART / "canonical-results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(path):
    return str(Path(path).relative_to(ROOT))


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def junit_report_of(out, original, adapted):
    sys.path.insert(0, str(REPLAY))
    from normalize_results import junit_report
    phases = {}
    for phase in ("baseline", "repaired"):
        matches = list(out.glob(phase + "-_swegemma_junit_*.xml"))
        phases[phase] = junit_report(matches[0], original, adapted) if len(matches) == 1 else None
    return phases


def main():
    contracts = json.loads((HERE / "selected-contracts.json").read_text())["tasks"]
    reviews, reports = [], []
    for config in contracts:
        key = config["key"]
        prep_path = HERE / (key + "-preparation.json")
        if not prep_path.exists():
            print(f"skip {key}: no preparation", flush=True)
            continue
        out = ART / (key + "-run-001")
        if not (out / "run-summary.json").exists():
            print(f"skip {key}: no run-001", flush=True)
            continue
        prep = json.loads(prep_path.read_text())
        summary_path = out / "run-summary.json"
        result = json.loads(summary_path.read_text())
        original = json.loads((ROOT / prep["original_task"]).read_text())["row"]
        adapted = json.loads((ROOT / prep["adapted_task"]).read_text())
        phases = junit_report_of(out, original, adapted)
        events_path = out / "tool-events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        observations = [{"index": e["index"], "actual_response": e["response"]}
                        for e in events if e["tool"] == "run_command"]
        rev = prep.get("source_revision") or prep.get("source_train_revision")
        ds = prep.get("source_dataset")
        result.update({
            "original_task": prep["original_task"],
            "original_task_sha256": prep["original_task_sha256"],
            "adapted_task": prep["adapted_task"], "recipe": prep["recipe"],
            "replay_output": label(out), "reports": phases,
            "canonical_repo": prep["canonical_repo"], "source_dataset": ds,
            "source_revision": rev, "physical_split": "train",
            "primary_adapter": prep["primary_adapter"], "submechanism": prep["submechanism"],
            "course_membership": prep["course_membership"],
            "execution_commit": prep["execution_commit"],
            "repair_family_id": prep["repair_family_id"],
            "snapshot_group_id": prep["snapshot_group_id"],
            "mechanism": {"adapter": prep["primary_adapter"],
                          "submechanism": prep["submechanism"],
                          "evidence": prep["mechanism_evidence"]},
            "original_verifier_requirements_relaxed": False,
            "required_node_identifiers_verbatim_preserved": True,
            "actual_mechanism_observations": observations,
            "source_preparation": label(prep_path)})
        reviews.append({**prep, "source_review_complete": True,
            "actual_current_run": label(out),
            "actual_run_summary_sha256": sha(summary_path),
            "actual_recipe_sha256": sha(ROOT / prep["recipe"]),
            "actual_events_sha256": sha(events_path),
            "actual_reports": phases, "actual_observations": observations,
            "resolved_by_local_official_verify_task": result["resolved_by_local_official_verify_task"]})
        reports.append((REPORTS / ("f5-fresh-" + key + "-attempt001-replay-result.json"), result))
    # merged source review (frozen anew each finish; single writer)
    review_path = HERE / "reviewed-candidates.json"
    save(review_path, {"date": "2026-10-25",
        "source_revision": "smith-77cab905/gym-bb94ed9e/v2-10483de0",
        "physical_split": "train",
        "scope": "F600 F5 event/stream/lifecycle timing contracts; Smith-py/Gym/V2 train only; four competition repositories excluded.",
        "independence_basis": "One reviewed production-function contract per selected mutation, with full source delta confirmation; same upstream snapshot grouped.",
        "tasks": reviews})
    review_sha = sha(review_path)
    for target, result in reports:
        result.update({"source_review_path": label(review_path),
                       "source_review_sha256": review_sha})
        save(target, result)
        print(json.dumps({"instance_id": result["instance_id"],
            "resolved": result["resolved_by_local_official_verify_task"],
            "baseline_F2P_failed": (result["reports"]["baseline"] or {}).get("all_original_F2P_observed_failed")}), flush=True)
    # catalog (final SHAs)
    cat_tasks = []
    for entry in reviews:
        task_path = ROOT / entry["original_task"]
        row = json.loads(task_path.read_text())["row"]
        ps = row["problem_statement"]
        cat_tasks.append({
            "canonical_repo": entry["canonical_repo"],
            "course_membership": "F600_F5",
            "environment": {
                "buggy_source": entry["buggy_source"],
                "upstream_source": entry.get("upstream_source", entry["buggy_source"])},
            "execution_commit": entry["execution_commit"],
            "instance_id": row["instance_id"] if "instance_id" in row else entry["instance_id"],
            "license_evidence": {"source": entry["source_license"]},
            "mechanism_evidence": {
                "bug_patch_direction": entry["mechanism_evidence"].get(
                    "bug_patch_direction", "introduces_defect_not_repair_target"),
                "causal_contract": entry["mechanism_evidence"]["causal_contract"],
                "primary_mechanism": "F5",
                "production_symbol": entry["mechanism_evidence"]["production_symbol"],
                "source_diff_reviewed": True},
            "original_FAIL_TO_PASS_count": entry["original_FAIL_TO_PASS_count"],
            "original_PASS_TO_PASS_count": entry["original_PASS_TO_PASS_count"],
            "original_patch_semantics": entry.get("original_patch_semantics", "bug_introducing"),
            "physical_split": "train", "primary_adapter": "F",
            "repair_family_id": entry["repair_family_id"],
            "snapshot_group_id": entry["snapshot_group_id"],
            "source_dataset": entry["source_dataset"],
            "source_resolution": entry.get("source_revision"),
            "source_review_complete": True,
            "source_review_path": label(review_path),
            "source_review_sha256": review_sha,
            "source_revision": entry.get("source_revision"),
            "source_task_path": entry["original_task"],
            "source_task_sha256": sha(task_path),
            "submechanism": "F5", "use_partition": "train_only",
            "visible_prompt_profile": {
                "kind": "verbatim_publisher_description",
                "original_problem_statement_sha256": hashlib.sha256(ps.encode()).hexdigest()}})
    cat_tasks.sort(key=lambda t: t["instance_id"])
    save(HERE / "catalog.json", {"date": "2026-10-25", "physical_split": "train",
        "schema_version": "reviewed-supplemental-training-catalog-v1",
        "selection_policy": "F5-fresh new snapshots small full-suite F2P+P2P<50 single-file single-hunk, 75 banned +5 used excluded, skip-precheck CLEAN only, temporal chain sequence, whitelist prompt, JUnit dual, six tools, replay/.venv dual; train_only.",
        "tasks": cat_tasks})
    print(f"catalog tasks={len(cat_tasks)} review_sha={review_sha[:12]}", flush=True)
    # normalize each report (supplemental = f5-fresh catalog)
    spec = importlib.util.spec_from_file_location("f5_norm", REPLAY / "normalize_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    catalog = {t["instance_id"]: t for t in cat_tasks}
    approvals = {e["instance_id"]: e for e in
                 json.loads((REPLAY / "approved-source-requirement-mappings.json").read_text())["approvals"]}
    CANON.mkdir(parents=True, exist_ok=True)
    eligible, reasons_count, canon_paths = [], {}, []
    for target, _ in reports:
        res = mod.normalize(target, catalog, approvals)
        save(HERE / (target.stem.replace("f5-fresh-", "").replace("-attempt001-replay-result", "") + "-normalize.json"),
             {"instance_id": res.get("instance_id"),
              "eligible_for_human_off_policy_SFT": res["eligible_for_human_off_policy_SFT"],
              "ineligibility_reasons": res["ineligibility_reasons"],
              "complete_official_replay": res.get("complete_official_replay", False),
              "baseline_verified_original_F2P_failure": res.get("baseline_verified_original_F2P_failure", False),
              "strict_verbatim_required_ids_passed": res.get("strict_verbatim_required_ids_passed", False)})
        for r in res["ineligibility_reasons"]:
            reasons_count[r] = reasons_count.get(r, 0) + 1
        if res["eligible_for_human_off_policy_SFT"]:
            eligible.append(res["instance_id"])
            dest = CANON / target.name
            dest.write_text(target.read_text())
    manifest = {"schema_version": "canonical-official-replay-manifest-v1",
        "date": "2026-10-25",
        "supplemental_catalog": "competition/vfarl/acquisition/f5-new/f5-fresh/catalog.json",
        "supplemental_catalog_sha256": sha(HERE / "catalog.json"),
        "supplemental_catalog_task_count": len(cat_tasks),
        "source_report_count": len(reports),
        "eligible_count": len(eligible),
        "strict_verbatim_eligible_count": len(eligible),
        "reasons": reasons_count,
        "eligible_instance_ids": sorted(eligible),
        "canonical_results": sorted([label(p) for p in CANON.glob("f5-fresh-*-replay-result.json")])}
    save(ART / "canonical-results-manifest.json", manifest)
    print(json.dumps({"source_report_count": len(reports), "eligible_count": len(eligible),
                      "reasons": reasons_count, "eligible": sorted(eligible)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
