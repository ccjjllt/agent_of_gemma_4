#!/usr/bin/env python3
"""P005 working state: issue contract + sourced facts + at most two hypotheses.

Stdlib only. No network. Never reads reference patches, heldout tasks, host
model/data directories, or other attempts' outputs.

What this script is (handoff doc I2):
  A tiny record/show store for the model's *current* working state. It records
  what the model was told and what tools returned; it never infers a root
  cause. Update it when preparing an edit or when an assumption was overturned
  by a new observation -- not on every step.

Schema (schema_version 1, exactly as specified in I2):
  {
    "schema_version": 1,
    "issue_contract": {"trigger": ..., "expected": ..., "preserve": ...},
    "facts": [{"id": "f1", "source_ref": "span or raw tool reply",
               "claim": "fact supported by that evidence"}],
    "hypotheses": [{"id": "h1", "layer": "producer|transformer|consumer|boundary",
                    "claim": "...", "support": ["f1"], "contradiction": []},
                   (at most 2)],
    "chosen": "h1",
    "unknown": "which unknown fact would change the choice",
    "next_action": {"kind": "inspect|probe|edit", "target": "...",
                    "purpose": "..."},
    "current_patch_id": "..."
  }

Subcommands (all print one complete JSON object to stdout; JSON is never
sliced mid-string -- fields shrink or calls fail, see emit()):
  record          update contract/facts/hypotheses/choice/next action
  show            display state plus changed_fields and current next_action
  record-issue    (handoff use) store visible problem text passed by the model
  write-handoff   (handoff use) compose handoff.json from issue + re-read code
                  + live worktree diff + explicitly passed hypotheses
  read-handoff    (executor first action) read handoff.json back by handle

State lives in the sandbox system temp dir bound to the task root
(<tmp>/p005_working_state/root_<hash>/), outside the repo so it never enters
`git diff`; the model cannot set the state root. Same binding idea as the T2
span_store (explicit --root, no symlink root, per-task reset), reused here --
see source_context/README.md. Default rejects non-/workspace --root; local
diagnosis must pass --allow-non-workspace-root.

Structured arguments use "|" separators to stay a single run_command string
without nested JSON quoting (I1 escaping rule):
  --fact "f1|span:sp_abc123|None becomes 'None' at return"
  --hypothesis "h1|transformer|converter drops None|f1|o2"
Facts/hypotheses upsert by id. To overturn an assumption, re-record it with a
contradiction entry or drop it via --drop-hypothesis, then set next_action to
the discriminating step. Claims must not contain "|"; for exact text with "|"
use --facts-json/--hyps-json (JSON arrays) instead.

P012 candidate namespace (backward compatible):
  record/show/record-issue/write-handoff accept --candidate ID --stage STAGE.
  Both default to "" (absent) and preserve the legacy single-file behavior:
  state.json / state.prev.json / issue.json / handoff.json. When given, files
  are namespaced by filename suffix so repair-A, acceptance-A and repair-B
  records never overwrite each other, e.g. --candidate A --stage repair uses
  state_A_repair.json / state_A_repair.prev.json / issue_A_repair.json /
  handoff_A_repair.json; --candidate A alone uses state_A.json, etc.
  Tokens reuse the state id charset (1-32 chars of [A-Za-z0-9_-]).
  read-handoff stays handle-based and needs no namespace flags.

patch_store candidate binding (no storage change here):
  patch_store.py already keys snapshots by patch_id with an explicit
  parent_patch_id chain (snapshot --parent p_xxx|none). Candidate binding is
  the caller's job: pass each candidate's snapshot id as the parent of its
  follow-up snapshots (A-chain vs B-chain), and record the resulting
  patch_id in that candidate's namespaced state via --current-patch-id.
  Snapshot existence never means correctness; restore is explicit per id.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from workspace_paths import resolve_task_root, resolve_issue_input

STATE_DIRNAME = "p005_working_state"
STATE_FILENAME = "state.json"
PREV_FILENAME = "state.prev.json"
ISSUE_FILENAME = "issue.json"
HANDOFF_FILENAME = "handoff.json"


def check_ns_token(v):
    """Namespace token: None/"" means legacy default; else state id charset."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and 1 <= len(v) <= 32 and \
            all(c in ID_RE for c in v):
        return v
    return False


def resolve_ns(args):
    cand = getattr(args, "candidate", None) or ""
    stage = getattr(args, "stage", None) or ""
    c = check_ns_token(cand)
    s = check_ns_token(stage)
    if c is False:
        return None, "bad --candidate (want 1-32 chars of [A-Za-z0-9_-])"
    if s is False:
        return None, "bad --stage (want 1-32 chars of [A-Za-z0-9_-])"
    return ((c or "", s or ""), None)


def ns_suffix(candidate, stage):
    if candidate and stage:
        return "%s_%s" % (candidate, stage)
    return candidate or stage or ""


def state_filename(candidate="", stage=""):
    sfx = ns_suffix(candidate, stage)
    return "state_%s.json" % sfx if sfx else STATE_FILENAME


def prev_filename(candidate="", stage=""):
    sfx = ns_suffix(candidate, stage)
    return "state_%s.prev.json" % sfx if sfx else PREV_FILENAME


def issue_filename(candidate="", stage=""):
    sfx = ns_suffix(candidate, stage)
    return "issue_%s.json" % sfx if sfx else ISSUE_FILENAME


def handoff_filename(candidate="", stage=""):
    sfx = ns_suffix(candidate, stage)
    return "handoff_%s.json" % sfx if sfx else HANDOFF_FILENAME


def state_fp(d, candidate="", stage=""):
    return d / state_filename(candidate, stage)


def prev_fp(d, candidate="", stage=""):
    return d / prev_filename(candidate, stage)


def issue_fp(d, candidate="", stage=""):
    return d / issue_filename(candidate, stage)


def handoff_fp(d, candidate="", stage=""):
    return d / handoff_filename(candidate, stage)

LAYERS = ("producer", "transformer", "consumer", "boundary")
NEXT_KINDS = ("inspect", "probe", "edit")
MAX_HYPOTHESES = 2
ID_RE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"

OUTPUT_CAP_ACK = 4000
OUTPUT_CAP_SHOW = 12000
OUTPUT_CAP_HANDOFF_READ = 20000
HANDOFF_FILE_CAP = 16000  # ~2-4K tokens advisory; code kept, prose trimmed
CODE_RANGE_CAP = 8000
DIFF_CAP = 6000


def emit(obj, cap=OUTPUT_CAP_ACK):
    blob = json.dumps(obj, ensure_ascii=False)
    if len(blob) > cap:
        minimal = {"status": "error", "error": "output-too-large",
                   "hint": "narrow the record (fewer/shorter claims) and retry"}
        blob = json.dumps(minimal, ensure_ascii=False)
    sys.stdout.write(blob)


def fail(error, **extra):
    emit({"status": "error", "error": error, **extra})


def system_temp_base():
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


def _git_out(args, cwd, timeout=10):
    try:
        r = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                           text=True, timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def repo_identity(root):
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


def state_dir_for(toplevel):
    base = system_temp_base() / STATE_DIRNAME
    h = hashlib.sha256(toplevel.encode("utf-8")).hexdigest()[:16]
    d = base / ("root_" + h)
    d.mkdir(parents=True, exist_ok=True)
    return d


def blank_state():
    return {
        "schema_version": 1,
        "issue_contract": {"trigger": "", "expected": "", "preserve": ""},
        "facts": [],
        "hypotheses": [],
        "chosen": "",
        "unknown": "",
        "next_action": {"kind": "", "target": "", "purpose": ""},
        "current_patch_id": "none",
    }


def load_state_fp(fp):
    if fp.is_file():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("schema_version") == 1:
                return data
        except Exception:
            pass
    return None


def save_state_fp(cur_fp, prev_fp_, data):
    if cur_fp.is_file():
        try:
            prev_fp_.write_bytes(cur_fp.read_bytes())
        except OSError:
            pass
    tmp = cur_fp.parent / (cur_fp.name + ".%d.tmp" % os.getpid())
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cur_fp)


def load_prev_fp(fp):
    if fp.is_file():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def load_state(d):
    return load_state_fp(d / STATE_FILENAME)


def save_state(d, data):
    save_state_fp(d / STATE_FILENAME, d / PREV_FILENAME, data)


def load_prev(d):
    return load_prev_fp(d / PREV_FILENAME)


def changed_fields(old, new):
    if old is None:
        return ["(first-record: all fields new)"]
    fields = []

    def blob(v):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)

    for k in ("trigger", "expected", "preserve"):
        if blob((old.get("issue_contract") or {}).get(k, "")) != \
           blob((new.get("issue_contract") or {}).get(k, "")):
            fields.append("issue_contract." + k)
    for k in ("facts", "hypotheses", "chosen", "unknown", "next_action",
              "current_patch_id"):
        if blob(old.get(k)) != blob(new.get(k)):
            fields.append(k)
    return fields


def check_id(v):
    return isinstance(v, str) and 1 <= len(v) <= 32 and \
        all(c in ID_RE for c in v)


def parse_fact(s):
    parts = s.split("|", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        return None, "bad --fact (want 'id|source_ref|claim', claim without '|')"
    fid, src, claim = (p.strip() for p in parts)
    if not check_id(fid):
        return None, "bad fact id '%s'" % fid
    return {"id": fid, "source_ref": src, "claim": claim}, None


def parse_hyp(s):
    parts = s.split("|", 4)
    if len(parts) != 5 or not parts[0].strip() or not parts[2].strip():
        return None, ("bad --hypothesis (want "
                      "'id|layer|claim|support|contradiction', claim without '|')")
    hid, layer, claim, sup, con = (p.strip() for p in parts)
    if not check_id(hid):
        return None, "bad hypothesis id '%s'" % hid
    if layer not in LAYERS:
        return None, "bad layer '%s' (want one of %s)" % (layer, "|".join(LAYERS))
    support = [x.strip() for x in sup.split(",") if x.strip()] if sup else []
    contra = [x.strip() for x in con.split(",") if x.strip()] if con else []
    return {"id": hid, "layer": layer, "claim": claim, "support": support,
            "contradiction": contra}, None


def cmd_record(args):
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    ns, err = resolve_ns(args)
    if err:
        fail(err)
        return
    candidate, stage = ns
    toplevel, _head = repo_identity(root)
    d = state_dir_for(toplevel)
    cur_fp, prev_fp_ = state_fp(d, candidate, stage), prev_fp(d, candidate, stage)
    st = load_state_fp(cur_fp) or blank_state()
    pre_ids = [h.get("id") for h in st.get("hypotheses", [])]

    if args.trigger is not None:
        st["issue_contract"]["trigger"] = args.trigger
    if args.expected is not None:
        st["issue_contract"]["expected"] = args.expected
    if args.preserve is not None:
        st["issue_contract"]["preserve"] = args.preserve

    # Drops first: overturn-then-replace in one call stays within budget.
    for drop in (args.drop_fact or []):
        st["facts"] = [f for f in st["facts"] if f.get("id") != drop]
    for drop in (args.drop_hypothesis or []):
        st["hypotheses"] = [h for h in st["hypotheses"]
                            if h.get("id") != drop]

    for fs in (args.fact or []):
        rec, err = parse_fact(fs)
        if err:
            fail(err)
            return
        st["facts"] = [f for f in st["facts"] if f.get("id") != rec["id"]]
        st["facts"].append(rec)
    if args.facts_json:
        try:
            arr = json.loads(args.facts_json)
        except Exception:
            fail("bad --facts-json (must be a JSON array)")
            return
        if not isinstance(arr, list):
            fail("bad --facts-json (must be a JSON array)")
            return
        for rec in arr:
            if not isinstance(rec, dict) or not check_id(rec.get("id", "")) \
               or not rec.get("source_ref") or "claim" not in rec:
                fail("bad --facts-json entry (need id/source_ref/claim)")
                return
            st["facts"] = [f for f in st["facts"] if f.get("id") != rec["id"]]
            st["facts"].append({"id": rec["id"],
                               "source_ref": rec["source_ref"],
                               "claim": rec["claim"]})

    for hs in (args.hypothesis or []):
        rec, err = parse_hyp(hs)
        if err:
            fail(err)
            return
        rest = [h for h in st["hypotheses"] if h.get("id") != rec["id"]]
        if len(rest) >= MAX_HYPOTHESES:
            fail("too many hypotheses (%d stored, max %d); drop or overturn "
                 "one first (at most two competing explanations)" %
                 (len(st["hypotheses"]), MAX_HYPOTHESES),
                 kept=[h.get("id") for h in st["hypotheses"]])
            return
        st["hypotheses"] = rest
        st["hypotheses"].append(rec)
    if args.hyps_json:
        try:
            arr = json.loads(args.hyps_json)
        except Exception:
            fail("bad --hyps-json (must be a JSON array)")
            return
        if not isinstance(arr, list):
            fail("bad --hyps-json (must be a JSON array)")
            return
        for rec in arr:
            if not isinstance(rec, dict) or not check_id(rec.get("id", "")) \
               or rec.get("layer") not in LAYERS or "claim" not in rec:
                fail("bad --hyps-json entry (need id/layer/claim)")
                return
            st["hypotheses"] = [h for h in st["hypotheses"]
                                if h.get("id") != rec["id"]]
            st["hypotheses"].append(
                {"id": rec["id"], "layer": rec["layer"],
                 "claim": rec["claim"],
                 "support": rec.get("support", []) or [],
                 "contradiction": rec.get("contradiction", []) or []})
    if len(st["hypotheses"]) > MAX_HYPOTHESES:
        fail("too many hypotheses (%d stored, max %d); drop or overturn one "
             "first (at most two competing explanations)" %
             (len(pre_ids), MAX_HYPOTHESES),
             kept=pre_ids)
        return

    if args.chosen is not None:
        if args.chosen and args.chosen not in \
                [h.get("id") for h in st["hypotheses"]]:
            fail("chosen '%s' matches no recorded hypothesis id" % args.chosen,
                 known=[h.get("id") for h in st["hypotheses"]])
            return
        st["chosen"] = args.chosen
    if args.unknown is not None:
        st["unknown"] = args.unknown
    if args.next_kind is not None:
        if args.next_kind not in NEXT_KINDS:
            fail("bad --next-kind (want one of %s)" % "|".join(NEXT_KINDS))
            return
        st["next_action"]["kind"] = args.next_kind
    if args.next_target is not None:
        st["next_action"]["target"] = args.next_target
    if args.next_purpose is not None:
        st["next_action"]["purpose"] = args.next_purpose
    if args.current_patch_id is not None:
        st["current_patch_id"] = args.current_patch_id

    old = load_state_fp(cur_fp)
    save_state_fp(cur_fp, prev_fp_, st)
    emit({"status": "ok", "command": "record",
          "candidate": candidate, "stage": stage,
          "changed_fields": changed_fields(old, st),
          "next_action": st["next_action"],
          "hypotheses": [h.get("id") for h in st["hypotheses"]],
          "note": "stored only; no root cause was inferred by this tool"})


def cmd_show(args):
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    ns, err = resolve_ns(args)
    if err:
        fail(err)
        return
    candidate, stage = ns
    toplevel, _head = repo_identity(root)
    d = state_dir_for(toplevel)
    cur_fp, prev_fp_ = state_fp(d, candidate, stage), prev_fp(d, candidate, stage)
    st = load_state_fp(cur_fp)
    if st is None:
        emit({"status": "ok", "command": "show", "empty": True,
              "candidate": candidate, "stage": stage,
              "state": blank_state(), "changed_fields": [],
              "next_action": blank_state()["next_action"],
              "hint": "no record yet for this task root"})
        return
    prev = load_prev_fp(prev_fp_)
    out = {"status": "ok", "command": "show", "empty": False,
           "candidate": candidate, "stage": stage, "state": st,
           "changed_fields": changed_fields(prev, st),
           "next_action": st.get("next_action", {})}
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > OUTPUT_CAP_SHOW:
        fail("state-too-large",
             hint="drop a hypothesis/fact or shorten claims, then re-record")
        return
    sys.stdout.write(blob)


def resolve_issue_file(root, spec, allow_non_ws):
    try:
        return resolve_issue_input(root, spec), None
    except ValueError as error:
        return None, str(error)


def cmd_record_issue(args):
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    rp, err = resolve_issue_file(root, args.issue_file,
                                 args.allow_non_workspace_root)
    if err:
        fail(err)
        return
    try:
        raw = rp.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        fail("cannot read --issue-file as UTF-8 text")
        return
    if not text.strip():
        fail("issue text is empty; pass the visible problem text, "
             "do not summarize a long issue into one silent sentence")
        return
    toplevel, _head = repo_identity(root)
    d = state_dir_for(toplevel)
    ns, err = resolve_ns(args)
    if err:
        fail(err)
        return
    candidate, stage = ns
    rec = {"schema_version": 1,
           "text": text,
           "chars": len(text),
           "source": args.source or "explorer-visible-problem-text",
           "budget_note": args.budget_note or ""}
    try:
        issue_fp(d, candidate, stage).write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    except OSError:
        fail("cannot write issue record")
        return
    emit({"status": "ok", "command": "record-issue", "chars": len(text),
          "candidate": candidate, "stage": stage,
          "source": rec["source"],
          "note": "issue stored verbatim as passed by the model; the skill "
                  "never reads host task files and no host auto-copy is claimed"})


def resolve_code_range(root, spec):
    """'relpath:start-end' -> (relpath, start, end)."""
    if ":" not in spec:
        return None, "bad --code (want 'relpath:start-end')"
    rel, _, rng = spec.rpartition(":")
    if "-" not in rng:
        return None, "bad --code (want 'relpath:start-end')"
    s, _, e = rng.partition("-")
    try:
        start, end = int(s), int(e)
    except ValueError:
        return None, "bad --code range (need integers)"
    if start < 1 or end < start:
        return None, "bad --code range (need 1 <= start <= end)"
    parts = rel.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None, "code path must be repo-relative without .."
    fp = root / rel
    try:
        rp = fp.resolve()
    except OSError:
        return None, "cannot resolve code path '%s'" % rel
    try:
        rp.relative_to(root)
    except ValueError:
        return None, "code path escapes root"
    if rp.is_symlink() or not rp.is_file():
        return None, "missing or symlink code path '%s'" % rel
    return (rel, start, end, rp), None


def cmd_write_handoff(args):
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    toplevel, head = repo_identity(root)
    d = state_dir_for(toplevel)
    ns, nserr = resolve_ns(args)
    if nserr:
        fail(nserr)
        return
    candidate, stage = ns
    issue_file = issue_fp(d, candidate, stage)
    if not issue_file.is_file():
        fail("no-issue",
             hint="explorer must record-issue first (visible problem text); "
                  "the skill never invents the issue")
        return
    try:
        issue = json.loads(issue_file.read_text(encoding="utf-8"))
    except Exception:
        fail("stored issue unreadable; re-run record-issue")
        return
    st = load_state_fp(state_fp(d, candidate, stage))
    if st is None:
        fail("no-state",
             hint="explorer must record facts/hypotheses/next action first")
        return
    if not st.get("hypotheses"):
        fail("no-hypotheses",
             hint="pass at least one explicit hypothesis with support; "
                  "empty claims are not a handoff")
        return
    if not (st.get("next_action") or {}).get("kind"):
        fail("no-next-action",
             hint="a handoff without one next action is a status card, "
                  "not an executable handoff")
        return

    code = []
    for spec in (args.code or []):
        parsed, err = resolve_code_range(root, spec)
        if err:
            fail(err)
            return
        rel, start, end, rp = parsed
        try:
            raw = rp.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            fail("cannot read code '%s' as UTF-8" % rel)
            return
        lines = text.splitlines(keepends=True)
        total = len(lines)
        if end > total:
            fail("code range %s exceeds file (%d lines)" % (spec, total))
            return
        body = "".join(lines[start - 1:end])
        if len(body) > CODE_RANGE_CAP:
            fail("code-range-too-large",
                 hint="keep key functions verbatim: narrow '%s' "
                      "(over %d chars)" % (spec, CODE_RANGE_CAP))
            return
        code.append({"path": rel, "range": [start, end],
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "text": body, "complete": True})

    diff_text = _git_out(["diff", "HEAD", "--"], cwd=root)
    if diff_text is None:
        diff_text = ""
    if len(diff_text) > DIFF_CAP:
        diff_text = diff_text[:DIFF_CAP]
        diff_capped = True
    else:
        diff_capped = False
    stat = _git_out(["diff", "HEAD", "--stat", "--"], cwd=root) or ""

    handoff = {"schema_version": 1,
               "issue": issue,
               "code": code,
               "diff": {"head": head, "stat": stat, "text": diff_text,
                        "diff_capped": diff_capped},
               "state": st,
               "handoff_trimmed": False}
    # Over budget: keep code and full issue, trim prose first.
    prose = [("state", "unknown", 200),
             ("state.next_action", "purpose", 200)]
    blob = json.dumps(handoff, ensure_ascii=False)
    while len(blob) > HANDOFF_FILE_CAP:
        shrunk = False
        for section, key, lim in prose:
            if section == "state":
                v = handoff["state"].get(key, "")
            else:
                v = handoff["state"].get("next_action", {}).get(key, "")
            if isinstance(v, str) and len(v) > lim:
                if section == "state":
                    handoff["state"][key] = v[:lim]
                else:
                    handoff["state"]["next_action"][key] = v[:lim]
                handoff["handoff_trimmed"] = True
                shrunk = True
                break
        if not shrunk:
            for h in handoff["state"].get("hypotheses", []):
                if isinstance(h.get("claim"), str) and len(h["claim"]) > 200:
                    h["claim"] = h["claim"][:200]
                    handoff["handoff_trimmed"] = True
                    shrunk = True
                    break
        if not shrunk:
            break
        blob = json.dumps(handoff, ensure_ascii=False)
    if len(blob) > HANDOFF_FILE_CAP:
        fail("handoff-too-large",
             hint="keep required code, cut excluded-search narrative; "
                  "current chars=%d" % len(blob))
        return
    hp = handoff_fp(d, candidate, stage)
    try:
        hp.write_text(blob, encoding="utf-8")
    except OSError:
        fail("cannot write handoff file")
        return
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    emit({"status": "ok", "command": "write-handoff",
          "candidate": candidate, "stage": stage,
          "handle": str(hp), "sha256": digest, "chars": len(blob),
          "token_estimate": max(1, len(blob) // 4),
          "handoff_trimmed": handoff["handoff_trimmed"],
          "note": "explorer final text must be this handle only; output_key "
                  "saves that text, not an auto-merged JSON"})


def cmd_read_handoff(args):
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    cand = Path(args.handle)
    try:
        rp = cand.resolve()
    except OSError:
        fail("cannot resolve handle")
        return
    allowed = False
    tmp = system_temp_base()
    try:
        rp.relative_to(tmp.resolve())
        allowed = True
    except ValueError:
        pass
    try:
        rp.relative_to(root)
        allowed = True
    except ValueError:
        pass
    if not allowed:
        fail("handle rejected (must live under the task sandbox temp dir "
             "or the task root)")
        return
    if rp.is_symlink() or not rp.is_file():
        fail("missing or symlink handle")
        return
    try:
        blob = rp.read_text(encoding="utf-8")
        data = json.loads(blob)
    except Exception:
        fail("handle is not readable JSON")
        return
    if not isinstance(data, dict) or data.get("schema_version") != 1 \
       or "issue" not in data or "code" not in data or "state" not in data:
        fail("handle is not a P005 handoff file")
        return
    out = {"status": "ok", "command": "read-handoff", "handoff": data}
    out_blob = json.dumps(out, ensure_ascii=False)
    if len(out_blob) > OUTPUT_CAP_HANDOFF_READ:
        fail("handoff-too-large-to-display",
             hint="read code ranges individually via source_probe inspect")
        return
    sys.stdout.write(out_blob)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="working_state.py",
        description="P005 working state record/show + handoff compose/read. "
                    "Stdlib only, complete JSON output.")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--allow-non-workspace-root", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_r = sub.add_parser("record", help="update working state (no inference)")
    p_r.add_argument("--trigger", default=None)
    p_r.add_argument("--expected", default=None)
    p_r.add_argument("--preserve", default=None)
    p_r.add_argument("--fact", action="append", default=None,
                     help="'id|source_ref|claim' (upsert by id)")
    p_r.add_argument("--facts-json", default=None)
    p_r.add_argument("--drop-fact", action="append", default=None)
    p_r.add_argument("--hypothesis", action="append", default=None,
                     help="'id|layer|claim|support|contradiction' (max 2)")
    p_r.add_argument("--hyps-json", default=None)
    p_r.add_argument("--drop-hypothesis", action="append", default=None)
    p_r.add_argument("--chosen", default=None)
    p_r.add_argument("--unknown", default=None)
    p_r.add_argument("--next-kind", default=None)
    p_r.add_argument("--next-target", default=None)
    p_r.add_argument("--next-purpose", default=None)
    p_r.add_argument("--current-patch-id", default=None)
    p_r.add_argument("--candidate", default="",
                     help="P012 namespace: candidate id A/B (default: legacy state.json)")
    p_r.add_argument("--stage", default="",
                     help="P012 namespace: stage e.g. repair/acceptance (default: legacy)")
    p_r.set_defaults(func=cmd_record)

    p_s = sub.add_parser("show", help="show state + changed_fields + next")
    p_s.add_argument("--candidate", default="")
    p_s.add_argument("--stage", default="")
    p_s.set_defaults(func=cmd_show)

    p_i = sub.add_parser("record-issue", help="store model-passed issue text")
    p_i.add_argument("--issue-file", required=True)
    p_i.add_argument("--budget-note", default=None)
    p_i.add_argument("--source", default=None)
    p_i.add_argument("--candidate", default="")
    p_i.add_argument("--stage", default="")
    p_i.set_defaults(func=cmd_record_issue)

    p_w = sub.add_parser("write-handoff", help="compose handoff.json")
    p_w.add_argument("--code", action="append", default=None,
                     help="'relpath:start-end' (re-read verbatim, repeatable)")
    p_w.add_argument("--candidate", default="")
    p_w.add_argument("--stage", default="")
    p_w.set_defaults(func=cmd_write_handoff)

    p_h = sub.add_parser("read-handoff", help="read handoff.json by handle")
    p_h.add_argument("--handle", required=True)
    p_h.set_defaults(func=cmd_read_handoff)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
