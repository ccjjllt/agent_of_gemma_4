"""Refresh admitted data exports after an explicitly reviewed source batch.

This runs data serializers only. It executes no task, model, training or GPU job.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
VFARL = Path(__file__).resolve().parent
REPLAY = VFARL / "replay"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    python = REPLAY / ".venv/bin/python"
    run = lambda args, **kwargs: subprocess.run([str(x) for x in args], cwd=ROOT, check=True, **kwargs)
    run([python, REPLAY / "normalize_results.py"])
    canonical = read(REPLAY / "canonical-results-manifest.json")
    results = canonical["eligible_results"]
    if not results:
        raise ValueError("No admitted episodes to export")
    prompts = REPLAY / "artifacts/gemma-sft/official-task-prompts.json"
    declarations = REPLAY / "adk-six-tool-declarations.json"
    run([python, REPLAY / "export_prompts.py", "--results", *results,
         "--declarations", declarations.relative_to(ROOT), "--out", prompts.relative_to(ROOT)])
    with (REPLAY / "artifacts/gemma-sft/export-current.log").open("w") as log:
        run([REPLAY / ".venv-format/bin/python", VFARL / "format_replays.py", "--results", *results,
            "--tokenizer", "research/015/tokenizer-assets/official-31b-ct-v2-tokenizer",
            "--declarations", declarations.relative_to(ROOT), "--official-prompts", prompts.relative_to(ROOT),
            "--out", "competition/vfarl/replay/artifacts/gemma-sft"],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            stdout=log, stderr=subprocess.STDOUT)
    shutil.copyfile(REPLAY / "artifacts/gemma-sft/manifest.json", REPLAY / "gemma-sft-manifest.json")
    run([python, VFARL / "export_curricula.py", "--manifest", "competition/vfarl/replay/gemma-sft-manifest.json",
         "--out", "competition/vfarl/replay/artifacts/adapter-curricula"])
    shutil.copyfile(REPLAY / "artifacts/adapter-curricula/manifest.json", REPLAY / "adapter-curricula-manifest.json")
    run([python, VFARL / "assemble_pilot.py"], stdout=subprocess.DEVNULL)
    run([python, VFARL / "prepare_s_course.py"])
    sft = read(REPLAY / "gemma-sft-manifest.json")
    course = read(VFARL / "s600/course-manifest.json")
    primary = read(VFARL / "pilot96/catalog.json")
    supplemental = read(VFARL / "acquisition/supplemental/catalog.json")
    pids = {e["instance_id"] for e in primary["tasks"]}
    sids = {e["instance_id"] for e in supplemental["tasks"]}
    catalog_by_id = {e["instance_id"]: e for e in primary["tasks"] + supplemental["tasks"]}
    rows = canonical["tasks"]
    ps = [row for row in rows if row["instance_id"] in pids]
    ss = [row for row in rows if row["instance_id"] in sids]
    summary_path = REPLAY / "replay-summary.json"
    summary = read(summary_path)
    summary.update({"prepared_on": "2026-10-25", "active_course": "S600_questions_and_data_only",
        "reviewed_training_candidates": len(pids | sids), "supplemental_reviewed_train_tasks": len(sids),
        "independent_tasks_replayed_including_reserve_audits": canonical["unique_reported_tasks"],
        "pilot96_tasks_with_actual_tool_events": sum(row["tool_events"] > 0 for row in ps),
        "pilot96_tasks_with_both_official_verification_results_and_JUnit": sum(row["complete_official_replay"] for row in ps),
        "pilot96_successful_full_task_demonstrations": sum(row["eligible"] for row in ps),
        "pilot96_not_admitted": sum(not row["eligible"] for row in ps),
        "pilot96_complete_but_not_admitted": sum(not row["eligible"] and row["complete_official_replay"] for row in ps),
        "pilot96_verification_incomplete": sum(not row["complete_official_replay"] for row in ps),
        "successful_supplemental_demonstrations": sum(row["eligible"] for row in ss),
        "successful_full_task_demonstrations": canonical["eligible_count"],
        "successful_full_task_by_adapter": sft["episode_counts_by_primary_adapter"],
        "successful_full_task_by_submechanism": sft["episode_counts_by_submechanism"],
        "source_requirement_profiles": sft["source_requirement_profile_episode_counts"],
        "baseline_evidence_profiles": sft["baseline_evidence_profile_episode_counts"],
        "visible_prompt_profiles": sft["visible_prompt_profile_episode_counts"],
        "source_repo_count": len({catalog_by_id[row["instance_id"]]["canonical_repo"].lower()
            for row in rows if row["eligible"]}),
        "reviewed_source_repo_count": len({e["canonical_repo"].lower() for e in catalog_by_id.values()}),
        "actual_tool_events_in_current_selected_training_runs": sum(row["tool_events"] for row in rows if row["instance_id"] in pids | sids),
        "actual_tool_events_in_current_runs_including_reserve_audits": sum(row["tool_events"] for row in rows),
        "canonical_admission_manifest_sha256": sha(REPLAY / "canonical-results-manifest.json"),
        "supplemental_catalog_sha256": sha(VFARL / "acquisition/supplemental/catalog.json"),
        "gemma_sft_manifest_sha256": sha(REPLAY / "gemma-sft-manifest.json"),
        "adapter_curricula_manifest_sha256": sha(REPLAY / "adapter-curricula-manifest.json"),
        "S600_course_manifest": "competition/vfarl/s600/course-manifest.json",
        "S600_course_manifest_sha256": sha(VFARL / "s600/course-manifest.json"),
        "S600_common_tasks": course["courses"]["common"]["admitted_unique_tasks"],
        "S600_specialist_tasks": course["courses"]["specialist"]["admitted_unique_tasks"],
        "SFT_task_episodes": sft["unique_task_episodes"], "SFT_decision_examples": sft["action_examples"],
        "SFT_tool_call_examples": sft["actual_tool_call_examples"],
        "SFT_text_only_completion_examples": sft["actual_text_only_completion_examples"],
        "SFT_action_loss_tokens": sft["action_loss_tokens"], "SFT_maximum_action_context_tokens": sft["maximum_action_context_tokens"],
        "model_rollouts": canonical["model_rollouts"], "verified_focal_pairs": canonical["verified_focal_pairs"],
        "GPU_training_started": canonical["GPU_training_started"],
        "tasks": rows})
    six = {"get_status", "read_file", "write_file", "edit_file", "run_command", "submit_patch"}
    attempts = {}
    event_paths = {ROOT / row["actual_run_path"] / "tool-events.jsonl"
        for row in canonical["source_reports"]
        if row["instance_id"] in pids | sids and row.get("actual_run_path")}
    for path in sorted(event_paths):
        if not path.is_file():
            continue
        digest = sha(path)
        if digest in attempts:
            continue
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if events and all(isinstance(e, dict) and e.get("tool") in six and "tool_call_id" in e and "budget_before" in e for e in events):
            attempts[digest] = len(events)
    summary.pop("all_unique_retained_attempt_event_files", None)
    summary.pop("actual_tool_events_in_all_unique_retained_attempts", None)
    summary["selected_train_retained_attempt_event_files"] = len(attempts)
    summary["actual_tool_events_in_selected_train_retained_attempts"] = sum(attempts.values())
    summary["retained_attempt_accounting_scope"] = "Only report-linked attempts belonging to catalog-selected train task identities; unrelated and reserve event files are not opened."
    summary["evidence_notes"] = [note for note in summary.get("evidence_notes", []) if not note.startswith("Current addition:")]
    for index, note in enumerate(summary["evidence_notes"]):
        if note.startswith("Four originally empty task descriptions") or note.startswith("Versioned behavioral clarifications:"):
            curated = sft["visible_prompt_profile_episode_counts"].get("curator_authored_behavioral_clarification", 0)
            summary["evidence_notes"][index] = (
                f"Versioned behavioral clarifications: {curated} admitted tasks use an explicitly reviewed curator prompt "
                "and actual execution under those task bytes. Full disclosure equals the reviewed catalog profile and enters "
                "its stable source fingerprint. Original source rows and any earlier actual runs remain preserved; an "
                "original empty or incorrect prompt is not claimed to have been executed when it was not.")
        elif note.startswith("Original official verifier and all repaired required-node specifications"):
            summary["evidence_notes"][index] = (
                "Original official verifier and all repaired required-node specifications remain intact. Per-source "
                "task Python/dependency profiles are separate from the modern ADK parent and do not establish competition Docker/Linux equivalence.")
    save(summary_path, summary)
    priority_path = VFARL / "pilot96/priority16-review.json"
    priority = read(priority_path)
    priority["current_replay_followup"]["canonical_manifest_sha256"] = sha(REPLAY / "canonical-results-manifest.json")
    save(priority_path, priority)
    common = course["courses"]["common"]
    specialist = course["courses"]["specialist"]
    names = {"S1": "编码、解析与表示", "S2": "坐标、边界与跨度", "S3": "多值、集合与别名",
        "S4": "算法与数值不变量", "S5": "结构、文法与组合表达"}
    specialist_rows = "\n".join(
        "| " + sub + " " + names[sub] + " | " + str(specialist["submechanism_task_counts"].get(sub, 0))
        + " | 40 |" for sub in ("S1", "S2", "S3", "S4", "S5"))
    (VFARL / "s600/README.md").write_text(
        "# S600 题目与数据准备\n\n"
        "按用户最新指示，只准备题目和CPU真实工具数据，暂不处理训练。优先S，其他adapter的训练计划暂不推进。\n\n"
        f"当前完成准入和对白导出 **{course['admitted_unique_tasks']} / 600** 个独立任务，"
        f"仍缺{course['remaining_unique_tasks']}题。共有{sft['action_examples']}条决策、"
        f"{sft['action_loss_tokens']:,}个助手监督tokens。来源缓存、未审阅候选和局部修复均不计入完成数。\n\n"
        "| 课程 | 已准备 | 目标 | 剩余 |\n| --- | ---: | ---: | ---: |\n"
        f"| 共同 | {common['admitted_unique_tasks']} | 400 | {common['remaining_unique_tasks']} |\n"
        f"| S专科 | {specialist['admitted_unique_tasks']} | 200 | {specialist['remaining_unique_tasks']} |\n\n"
        "两个名单互不重复。共同课程保留每题真实G/S/F/R机制，不把提供给S使用的流转或运行时题伪标为S专科。"
        "初始81题分为65共同和16专科；相同题的工具动作、重放次数与恢复分支不增加题数。"
        "同family/snapshot分组保留，当前未创建评测随机划分。\n\n"
        "| S子机制 | 当前专科题数 | 搜索参考目标 |\n| --- | ---: | ---: |\n"
        + specialist_rows + "\n\n"
        "40/机制用于指导搜索，不能为了配额放宽来源或验收要求。共同题中的S机制不再重复算进此表。\n\n"
        f"已准入题来自{summary['source_repo_count']}个仓库，数据集原split全部为train，四个比赛仓库未进入本课程。"
        "原要求/基线差异的少量逐题审阅profile单独保留，详见证据manifest；没有新增全局判分例外。\n\n"
        f"可直接审阅[共同题面](../replay/artifacts/s600/S_common_questions.jsonl)与"
        f"[S专科题面](../replay/artifacts/s600/S_specialist_questions.jsonl)。每行只有一道题，绑定实际官方初始对白、"
        "来源及审阅哈希、实际重放结果；外层分类元数据不作为策略提示，未把参考补丁、验证节点或事后标签追加到策略题面。\n\n"
        "[课程实物与哈希](course-manifest.json)、[原始来源与审阅](../acquisition/supplemental/catalog.json)、"
        "[逐题官方工具证据](../replay/canonical-results-manifest.json)、"
        "[官方数据及对白要求](../s600-official-data-alignment.md)。\n\n"
        "当前是Codex编排、审阅并真实执行的off-policy冷启动示范；仍不是Gemma当前策略采样或同状态关键动作对。"
        "共同课现有不少短修复，尚不能据此声称定位、反馈恢复或复杂组合能力已覆盖充分；六工具课没有委派工具，不能伪造委派回包。"
        "官方SubprocessManager本地CPU证据不证明比赛Docker/Linux环境等价。未启动GPU或训练。\n")
    print(json.dumps({"S600_common": summary["S600_common_tasks"], "S600_specialist": summary["S600_specialist_tasks"],
        "unique_tasks": sft["unique_task_episodes"], "decisions": sft["action_examples"], "loss_tokens": sft["action_loss_tokens"]}))


if __name__ == "__main__":
    main()
