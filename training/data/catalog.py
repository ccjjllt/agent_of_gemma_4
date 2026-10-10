#!/usr/bin/env python3
"""P005 训练数据目录 (catalog) —— 交接 §4 D1。

只记录来源元数据, 不下载语料本身。每个条目必须可追溯:
source_uri / license / revision / teacher / issue / repo / commit / content_hash。

来源类型 (source_type):
  external    公开外部轨迹 (如 SWE-Gym / SWE-smith, 作者数据卡标 MIT)
  gemma-self  本项目 Gemma 自采样轨迹
  corrected   人工 / teacher 纠正的轨迹
  oracle      oracle 演示 (仅有限格式热身, 不作语义大头)

许可边界 (主办方讨论 746046, 见训练报告 §5.2):
  - 开源 teacher 与现成公开非 Gemma 轨迹: 明确允许, 首版可用。
  - 向外部 API 发送比赛 released tasks: 746251 尚无明确答复, 首版不依赖该路径。
  数据卡许可不自动覆盖内嵌仓库内容及教师供应商条款, 来源字段与原许可须一起保留。

索引非语料: research/002/candidates/training_prep/fragments_index.json
只是回查起点, 必须回查原始轨迹补全实际观察, 不能从最终补丁倒推伪造。

stdlib only. 用法:
  python3 catalog.py add --source-type external --source-uri ... --license MIT ...
  python3 catalog.py list --manifest source_manifest.jsonl
  python3 catalog.py verify --manifest source_manifest.jsonl
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

CATALOG_VERSION = 1

SOURCE_TYPES = ("external", "gemma-self", "corrected", "oracle")

REQUIRED_FIELDS = (
    "source_uri",
    "license",
    "revision",
    "teacher",
    "issue",
    "repo",
    "commit",
    "content_hash",
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, limit_bytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            if limit_bytes is not None and h.digest_size:
                pass
    return h.hexdigest()


def make_record(source_type, source_uri, license, revision, teacher, issue,
                repo, commit, content_hash, note=""):
    if source_type not in SOURCE_TYPES:
        raise ValueError("unknown source_type %r, want one of %s" % (source_type, SOURCE_TYPES))
    record = {
        "catalog_version": CATALOG_VERSION,
        "source_type": source_type,
        "source_uri": source_uri,
        "license": license,
        "revision": revision,
        "teacher": teacher,
        "issue": issue,
        "repo": repo,
        "commit": commit,
        "content_hash": content_hash,
        "note": note,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    missing = [k for k in REQUIRED_FIELDS if not record.get(k)]
    if missing:
        raise ValueError("missing required fields: %s" % ",".join(missing))
    return record


def cmd_add(args):
    record = make_record(
        source_type=args.source_type,
        source_uri=args.source_uri,
        license=args.license,
        revision=args.revision,
        teacher=args.teacher,
        issue=args.issue,
        repo=args.repo,
        commit=args.commit,
        content_hash=args.content_hash,
        note=args.note or "",
    )
    with open(args.manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "manifest": args.manifest,
                      "content_hash": record["content_hash"]}, ensure_ascii=False))
    return 0


def cmd_list(args):
    n = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if args.source_type and record.get("source_type") != args.source_type:
                continue
            n += 1
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"ok": True, "count": n}, ensure_ascii=False))
    return 0


def cmd_verify(args):
    n = 0
    bad = 0
    by_type = {}
    for line in open(args.manifest, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        n += 1
        missing = [k for k in REQUIRED_FIELDS if not record.get(k)]
        st = record.get("source_type")
        by_type[st] = by_type.get(st, 0) + 1
        if missing or st not in SOURCE_TYPES:
            bad += 1
            print(json.dumps({"bad_record": record, "missing": missing}, ensure_ascii=False))
    print(json.dumps({"ok": bad == 0, "count": n, "bad": bad,
                      "by_source_type": by_type}, ensure_ascii=False))
    return 0 if bad == 0 else 1


def build_parser():
    p = argparse.ArgumentParser(description="P005 training data catalog (D1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="append one source record to a manifest JSONL")
    a.add_argument("--manifest", required=True)
    a.add_argument("--source-type", required=True, choices=SOURCE_TYPES)
    a.add_argument("--source-uri", required=True)
    a.add_argument("--license", required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--teacher", required=True)
    a.add_argument("--issue", required=True)
    a.add_argument("--repo", required=True)
    a.add_argument("--commit", required=True)
    a.add_argument("--content-hash", required=True)
    a.add_argument("--note", default="")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="print manifest records as JSONL")
    l.add_argument("--manifest", required=True)
    l.add_argument("--source-type", default=None, choices=SOURCE_TYPES)
    l.set_defaults(func=cmd_list)

    v = sub.add_parser("verify", help="check required fields and source types")
    v.add_argument("--manifest", required=True)
    v.set_defaults(func=cmd_verify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
