#!/usr/bin/env python3
"""P005 decision-oriented source reading + on-demand path probe. Stdlib only.

Three stable entries (handoff doc I3), each prints one complete JSON object;
output budgeting shrinks text fields first and sets truncated flags -- the
serialized JSON string itself is never sliced (this is the fix for the
T2-copied navigation read_range `print(blob[:4000])` truncation; frozen T1/T2
files are untouched, see source_context/README.md):

  locate  queries[1..4] + question (+ optional scope): a few candidates with
          hit basis. Relations from source references are labelled
          textual_reference only; only probe output may claim observed_call.
  inspect targets[1..3] + question, mode exact|outline: complete small
          functions (exact) or signatures (outline). Truncated output NEVER
          carries an editable handle -- this script issues no span_id at all;
          only source_edit inspect does, and only for complete displays.
  probe   command + question + timeout: run one command for trigger/path
          evidence. Uses sys.setprofile/threading.setprofile filtered to the
          task repo when --trace is set and the command is a plain
          `python3 <script> [args]` run; reports ordered functions + counts +
          a short stack + named assertions. exit 0 NEVER implies reproduced or
          fixed. Probe scripts live in the sandbox temp dir; no host files are
          collected; no full call graph is built.

Reuses the T4 layered_lookup ideas: text/AST search, AST-boundary complete
reads, per-call candidate caps, shared output budget. No vectors, no network,
no reference patches, no heldout reads.
"""

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from workspace_paths import resolve_task_root, resolve_issue_input

SKIP_DIRS = {"__pycache__", "venv", ".venv", "node_modules", ".git", ".hg"}
MAX_FILE_BYTES = 512 * 1024
MAX_SNIPPET_CHARS = 400
TRIMMED_SNIPPET_CHARS = 120
OUTPUT_CAP = 4000
MAX_HITS_HARD = 8
MAX_QUERIES = 4
MAX_TARGETS = 3
MAX_FUNCTIONS_HARD = 2
MAX_CALLS = 96
STDOUT_TAIL = 1500
STDERR_TAIL = 800
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300


def err(msg):
    return {"status": "error", "error": msg}


def emit(obj, cap=OUTPUT_CAP):
    sys.stdout.write(fit_json(obj, cap))


def check_question(q):
    return isinstance(q, str) and 1 <= len(q.strip()) <= 400


def safe_root(root_arg, allow_non_ws):
    return resolve_task_root(root_arg, allow_non_ws)


def resolve_under(root, rel):
    if rel.startswith("/") or rel.startswith("\\") or ".." in Path(rel).parts:
        raise ValueError("path must be repo-relative without ..")
    fp = (root / rel)
    try:
        rp = fp.resolve()
    except OSError:
        raise ValueError("resolve failed")
    try:
        rp.relative_to(root)
    except ValueError:
        raise ValueError("path escapes root")
    return rp


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in SKIP_DIRS]
        dirnames[:] = [d for d in dirnames
                       if not Path(dirpath, d).is_symlink()]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = Path(dirpath, fn)
            if fp.is_symlink():
                continue
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield fp


class DefVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.defs = []

    def _record(self, node, kind):
        name = node.name
        qual = ".".join(self.stack + [name]) if self.stack else name
        end = getattr(node, "end_lineno", node.lineno)
        self.defs.append({"name": name, "qualname": qual, "kind": kind,
                          "line": node.lineno, "end_line": end})

    def visit_ClassDef(self, node):
        self._record(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._record(node, "def")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self._record(node, "async def")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def parse_defs(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    v = DefVisitor()
    v.visit(tree)
    return v.defs


def signature_of(lines, lineno, width=160):
    if 1 <= lineno <= len(lines):
        sig = lines[lineno - 1].strip()
        if sig:
            return sig[:width]
    return ""


def snippet_around(lines, lineno, width=3):
    lo = max(1, lineno - width)
    hi = lineno + width
    return "".join(lines[lo - 1:hi])[:MAX_SNIPPET_CHARS]


def fit_json(out, cap=OUTPUT_CAP):
    """Complete-JSON budget: trim text fields first, set flags, never slice."""
    def trim(o, aggressive=False):
        changed = False
        lim = 80 if aggressive else TRIMMED_SNIPPET_CHARS
        for key in ("candidates",):
            for c in o.get(key, []) or []:
                if len(c.get("snippet", "")) > lim:
                    c["snippet"] = c["snippet"][:lim]
                    changed = True
        for q in o.get("queries", []) or []:
            for c in q.get("hits", []) or []:
                if len(c.get("snippet", "")) > lim:
                    c["snippet"] = c["snippet"][:lim]
                    changed = True
        for f in o.get("sources", []) or []:
            if not f.get("complete") and len(f.get("text", "")) > 800:
                f["text"] = f["text"][:800]
                f["chars"] = len(f["text"])
                changed = True
        for s in o.get("definitions", []) or []:
            if len(s.get("signature", "")) > lim:
                s["signature"] = s["signature"][:lim]
                changed = True
        if isinstance(o.get("stdout_tail"), str) and \
                len(o["stdout_tail"]) > STDOUT_TAIL:
            o["stdout_tail"] = o["stdout_tail"][-STDOUT_TAIL:]
            o["tail_trimmed"] = True
            changed = True
        if aggressive:
            for key in ("candidates", "calls"):
                if isinstance(o.get(key), list) and len(o[key]) > 1:
                    o[key] = o[key][:max(1, len(o[key]) // 2)]
                    changed = True
            # queries[].hits duplicates candidate dicts; halve them too or
            # the budget loop cannot converge.
            for q in o.get("queries", []) or []:
                if isinstance(q.get("hits"), list) and len(q["hits"]) > 1:
                    q["hits"] = q["hits"][:max(1, len(q["hits"]) // 2)]
                    changed = True
        return changed

    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) <= cap:
        return blob
    out["truncated"] = True
    if "is_truncated" in out:
        out["is_truncated"] = True
    trim(out)
    blob = json.dumps(out, ensure_ascii=False)
    while len(blob) > cap:
        if not trim(out, aggressive=True):
            break
        blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > cap:
        keep = {"status": out.get("status", "ok"), "truncated": True,
                "error": "output budget exceeded; narrow scope and retry"}
        for k in ("command", "question", "path", "symbol"):
            if k in out:
                keep[k] = out[k]
        blob = json.dumps(keep, ensure_ascii=False)
        if len(blob) > cap:
            blob = json.dumps({"status": "error",
                               "error": "output budget exceeded"},
                              ensure_ascii=False)
    return blob


# ---------------- locate ----------------

def find_hits_for_query(scope_root, query):
    per_file = []
    for fp in iter_py_files(scope_root):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines(keepends=True)
        try:
            rel = fp.relative_to(scope_root).as_posix()
        except ValueError:
            continue
        cands = []
        for d in parse_defs(text):
            if d["name"] == query:
                cands.append({"path": rel, "kind": d["kind"],
                              "symbol": query,
                              "line": d["line"], "end_line": d["end_line"],
                              "snippet": snippet_around(lines, d["line"]),
                              "source": "ast"})
        file_text_hits = 0
        for i, ln in enumerate(lines, 1):
            if query in ln and file_text_hits < 4:
                if any(abs(c["line"] - i) <= 1 and c["path"] == rel
                       for c in cands):
                    continue
                cands.append({"path": rel, "kind": "text", "symbol": query,
                              "line": i, "end_line": i,
                              "snippet": ln.strip()[:MAX_SNIPPET_CHARS],
                              "source": "text"})
                file_text_hits += 1
        per_file.extend(cands)
        if len(per_file) >= 64:
            break
    return per_file


def cmd_locate(args, root):
    if not check_question(args.question):
        print(json.dumps(err("a non-empty --question (<=400 chars) is "
                             "required: what decision should this answer?"),
                         ensure_ascii=False))
        return
    queries = args.query or []
    if not queries or len(queries) > MAX_QUERIES:
        print(json.dumps(err("provide 1-4 --query values"), ensure_ascii=False))
        return
    for q in queries:
        q = (q or "").strip()
        if not q or len(q) > 256 or any(c in q for c in "\n\r\x00"):
            print(json.dumps(err("bad query"), ensure_ascii=False))
            return
    sources = args.source or []
    if sources and len(sources) != len(queries):
        print(json.dumps(err("--source count must match --query count"),
                         ensure_ascii=False))
        return
    for s in sources:
        if not (s or "").strip() or len(s) > 200:
            print(json.dumps(err("bad source provenance"), ensure_ascii=False))
            return
    scope = args.scope or "."
    try:
        scope_root = resolve_under(root, scope)
    except ValueError as e:
        print(json.dumps(err(str(e)), ensure_ascii=False))
        return
    if scope_root.is_symlink() or not scope_root.is_dir():
        print(json.dumps(err("scope must be a directory under root"),
                         ensure_ascii=False))
        return
    default_prov = "question:" + args.question.strip()[:60]
    max_hits = min(max(1, args.max_hits), MAX_HITS_HARD)
    all_cands = []
    for qi, q in enumerate(queries):
        prov = sources[qi].strip() if sources else default_prov
        for c in find_hits_for_query(scope_root, q.strip()):
            c = dict(c)
            c["query_index"] = qi
            c["query"] = q.strip()
            c["provenance"] = prov
            all_cands.append(c)

    def rank(c):
        return (c["query_index"], 0 if c["source"] == "ast" else 1,
                len(c["path"]), c["line"])

    all_cands.sort(key=rank)
    total_raw = len(all_cands)
    picked = all_cands[:max_hits]
    per_query = []
    for qi, q in enumerate(queries):
        prov = sources[qi].strip() if sources else default_prov
        per_query.append({"query": q.strip(), "source": prov,
                          "hits": [c for c in picked
                                   if c["query_index"] == qi]})
    links = [{"from_query": c["query"],
              "to": "%s:%s" % (c["path"], c["symbol"]),
              "kind": "textual_reference"} for c in picked]
    files = {c["path"] for c in picked}
    out = {"status": "ok", "command": "locate", "question": args.question,
           "queries": per_query, "candidates": picked,
           "links": links, "total_raw": total_raw,
           "truncated": total_raw > len(picked),
           "static_approx": True, "ambiguous": len(files) > 1,
           "note": "same-name hits are static approximations, NOT proven call "
                   "edges; kind is always textual_reference here -- only probe "
                   "output may claim observed_call; snippets are NOT complete "
                   "functions"}
    sys.stdout.write(fit_json(out))


# ---------------- inspect ----------------

def parse_target(spec):
    """'path' (outline) or 'path:symbol[:line]' (exact)."""
    parts = spec.split(":")
    if len(parts) == 1:
        return (spec, None, None)
    if len(parts) in (2, 3):
        path, symbol = parts[0], parts[1]
        line = None
        if len(parts) == 3:
            try:
                line = int(parts[2])
            except ValueError:
                return None
        if path and symbol:
            return (path, symbol, line)
    return None


def cmd_inspect(args, root):
    if not check_question(args.question):
        print(json.dumps(err("a non-empty --question (<=400 chars) is "
                             "required: what decision should this answer?"),
                         ensure_ascii=False))
        return
    targets = args.target or []
    if not targets or len(targets) > MAX_TARGETS:
        print(json.dumps(err("provide 1-3 --target values"), ensure_ascii=False))
        return
    if args.mode not in ("exact", "outline"):
        print(json.dumps(err("mode must be exact|outline"), ensure_ascii=False))
        return
    max_chars = min(max(args.max_chars, 256), OUTPUT_CAP)
    sources = []
    omitted = []
    for spec in targets:
        parsed = parse_target(spec)
        if parsed is None:
            print(json.dumps(err("bad --target '%s' (want 'path' for outline "
                                 "or 'path:symbol[:line]' for exact)" % spec),
                             ensure_ascii=False))
            return
        rel, symbol, line = parsed
        if args.mode == "outline" and symbol is not None:
            print(json.dumps(err("outline mode takes file/dir paths, not "
                                 "path:symbol"), ensure_ascii=False))
            return
        if args.mode == "exact" and symbol is None:
            print(json.dumps(err("exact mode needs 'path:symbol[:line]'"),
                             ensure_ascii=False))
            return
        try:
            rp = resolve_under(root, rel)
        except ValueError as e:
            print(json.dumps(err(str(e)), ensure_ascii=False))
            return
        if rp.is_symlink() or not rp.exists():
            print(json.dumps(err("missing or symlink target '%s'" % rel),
                             ensure_ascii=False))
            return
        if args.mode == "outline":
            if rp.is_dir():
                defs, _files = [], []
                for fp in iter_py_files(rp):
                    try:
                        text = fp.read_text(encoding="utf-8",
                                            errors="replace")
                    except OSError:
                        continue
                    lines = text.splitlines()
                    try:
                        fprel = fp.relative_to(root).as_posix()
                    except ValueError:
                        continue
                    for d in parse_defs(text)[:4]:
                        defs.append({"path": fprel, "kind": d["kind"],
                                     "name": d["name"],
                                     "qualname": d["qualname"],
                                     "line": d["line"],
                                     "end_line": d["end_line"],
                                     "signature": signature_of(
                                         lines, d["line"]) or d["qualname"]})
                    if len(defs) >= 16:
                        omitted.append({"path": rel, "why": "budget"})
                        break
                sources.append({"path": rel, "mode": "outline",
                                "definitions": defs,
                                "note": "signatures only, not bodies"})
            elif rp.suffix == ".py" and rp.is_file():
                try:
                    text = rp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    print(json.dumps(err("read failed"), ensure_ascii=False))
                    return
                lines = text.splitlines()
                defs = [{"path": rel, "kind": d["kind"], "name": d["name"],
                         "qualname": d["qualname"], "line": d["line"],
                         "end_line": d["end_line"],
                         "signature": signature_of(lines, d["line"])
                         or d["qualname"]} for d in parse_defs(text)[:24]]
                sources.append({"path": rel, "mode": "outline",
                                "definitions": defs,
                                "note": "signatures only, not bodies"})
            else:
                print(json.dumps(err("outline target must be a dir or .py"),
                                 ensure_ascii=False))
                return
            continue
        # exact mode
        if rp.suffix != ".py" or not rp.is_file():
            print(json.dumps(err("exact target must be a .py file"),
                             ensure_ascii=False))
            return
        try:
            raw = rp.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            print(json.dumps(err("read failed (need UTF-8 .py)"),
                             ensure_ascii=False))
            return
        digest = hashlib.sha256(raw).hexdigest()
        lines = text.splitlines(keepends=True)
        defs = [d for d in parse_defs(text) if d["name"] == symbol]
        if line is not None:
            defs = [d for d in defs if d["line"] <= line <= d["end_line"]]
            if not defs:
                print(json.dumps(
                    {"status": "error",
                     "error": "no definition spans --line; retry with outline "
                              "line numbers", "path": rel, "symbol": symbol},
                    ensure_ascii=False))
                return
        if not defs:
            print(json.dumps(
                {"status": "error", "error": "symbol not found in file",
                 "path": rel,
                 "hint": "use outline mode first, then retry with exact name"},
                ensure_ascii=False))
            return
        if len(defs) > MAX_FUNCTIONS_HARD:
            print(json.dumps(
                {"status": "ambiguous",
                 "error": "multiple definitions; retry with path:symbol:line",
                 "path": rel, "symbol": symbol,
                 "candidates": [{"kind": d["kind"], "name": d["name"],
                                 "qualname": d["qualname"], "line": d["line"],
                                 "end_line": d["end_line"]} for d in defs],
                 "note": "bodies withheld to protect the output budget"},
                ensure_ascii=False))
            return
        for d in defs[:MAX_FUNCTIONS_HARD]:
            s, e = d["line"], d["end_line"]
            full = "".join(lines[s - 1:e])
            if len(full) <= max_chars:
                sources.append({"path": rel, "mode": "exact",
                                "symbol": symbol, "range": [s, e],
                                "sha256": digest, "text": full,
                                "complete": True, "is_truncated": False})
            else:
                buf, used, last = [], 0, s - 1
                for i in range(s, e + 1):
                    ln = lines[i - 1] if 1 <= i <= len(lines) else ""
                    if used + len(ln) > max_chars and buf:
                        break
                    buf.append(ln)
                    used += len(ln)
                    last = i
                sources.append({"path": rel, "mode": "exact",
                                "symbol": symbol, "range": [s, last],
                                "full_range": [s, e], "sha256": digest,
                                "text": "".join(buf),
                                "complete": False, "is_truncated": True,
                                "next_start": last + 1 if last < e else None,
                                "warning": "TRUNCATED head only: carries NO "
                                           "editable handle; re-inspect a "
                                           "narrower target"})
                omitted.append({"path": rel, "target": symbol, "why": "budget"})
    out = {"status": "ok", "command": "inspect", "question": args.question,
           "mode": args.mode, "sources": sources, "omitted": omitted,
           "truncated": any(s.get("is_truncated") for s in sources
                            if "is_truncated" in s),
           "note": "this tool issues no span_id; only source_edit inspect "
                   "issues editable handles, and only for complete displays"}
    sys.stdout.write(fit_json(out))


# ---------------- probe ----------------

def _git_head(root):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return "no-git"
    if r.returncode != 0:
        return "no-git"
    return (r.stdout or "").strip() or "no-git"


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


TRACER_TEMPLATE = """\
import os, runpy, sys, json, threading
ROOT = %%ROOT%%
TARGET = %%TARGET%%
ARGS = %%ARGS%%
OUT = %%OUT%%
MAX_FUNCS = 96
_order = []
_counts = {}
def _hook(frame, event, arg):
    try:
        if event != "call":
            return _hook
        fn = frame.f_code.co_filename
        if not os.path.isabs(fn) or fn.startswith("<"):
            return _hook
        try:
            rf = os.path.realpath(fn)
        except Exception:
            return _hook
        if not (rf == ROOT or rf.startswith(ROOT + os.sep)):
            return _hook
        if os.path.basename(rf).startswith("p005_trace_"):
            return _hook
        key = (rf, frame.f_code.co_name, frame.f_code.co_firstlineno)
        if key not in _counts:
            _counts[key] = 0
            if len(_order) < MAX_FUNCS:
                _order.append(key)
        _counts[key] += 1
    except Exception:
        pass
    return _hook
def _main():
    sys.argv = [TARGET] + list(ARGS)
    here = os.path.dirname(os.path.abspath(TARGET))
    if here not in sys.path:
        sys.path.insert(0, here)
    sys.setprofile(_hook)
    try:
        threading.setprofile(_hook)
    except Exception:
        pass
    exc = None
    try:
        runpy.run_path(TARGET, run_name="__main__")
    except BaseException as e:  # noqa: BLE001 - record, do not hide
        import traceback
        exc = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    finally:
        try:
            sys.setprofile(None)
            threading.setprofile(None)
        except Exception:
            pass
    try:
        payload = {"order": [[r, n, l] for (r, n, l) in _order],
                   "counts": [int(_counts[k]) for k in _order],
                   "truncated_funcs": len(_counts) > len(_order),
                   "exception": exc}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
    if exc is not None:
        sys.exit(1)
_main()
"""


def match_trace_command(root, command):
    """Match `python3 <script> [args...]` with script under root."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if len(parts) < 2 or parts[0] not in ("python3", "python"):
        return None
    if parts[1].startswith("-"):
        return None
    try:
        rp = resolve_under(root, parts[1])
    except ValueError:
        return None
    if rp.is_symlink() or not rp.is_file() or rp.suffix != ".py":
        return None
    return str(rp), parts[2:]


def shorten_stack(text, root, max_frames=12):
    if not text:
        return ""
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("File \""):
            try:
                path = s.split("\"")[1]
            except IndexError:
                kept.append(ln)
                continue
            try:
                rp = str(Path(path).resolve())
                rr = str(root.resolve())
                if rp == rr or rp.startswith(rr + os.sep):
                    kept.append(ln + "  [repo]")
                else:
                    kept.append("File <outside-repo> ...")
            except OSError:
                kept.append(ln)
        else:
            kept.append(ln)
    if len(kept) > max_frames + 4:
        kept = kept[:2] + ["... (%d frames omitted; no full call graph kept)"
                           % (len(kept) - max_frames - 2)] + kept[-max_frames:]
    return "\n".join(kept)


def cmd_probe(args, root):
    if not check_question(args.question):
        print(json.dumps(err("a non-empty --question (<=400 chars) is "
                             "required: what should this run decide?"),
                         ensure_ascii=False))
        return
    timeout = args.timeout if args.timeout else DEFAULT_TIMEOUT
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        print(json.dumps(err("bad timeout"), ensure_ascii=False))
        return
    timeout = min(max(1, timeout), MAX_TIMEOUT)
    import secrets
    obs_id = "obs_" + secrets.token_hex(6)
    head = _git_head(root)

    traced = match_trace_command(root, args.command) if args.trace else None
    calls = []
    trace_note = "trace off"
    stdout_text, stderr_text, ret = "", "", None
    timed_out = False
    trace_exc = None
    if traced is not None:
        target, targv = traced
        pdir = system_temp_base() / "p005_probe"
        try:
            pdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        tag = "p005_trace_%d" % os.getpid()
        tpath = pdir / (tag + ".py")
        opath = pdir / (tag + ".json")
        src = TRACER_TEMPLATE.replace("%%ROOT%%",
                                      repr(str(root.resolve())))
        src = src.replace("%%TARGET%%", repr(target))
        src = src.replace("%%ARGS%%", repr(list(targv)))
        src = src.replace("%%OUT%%", repr(str(opath)))
        try:
            tpath.write_text(src, encoding="utf-8")
            r = subprocess.run([sys.executable, str(tpath)],
                               cwd=str(root), capture_output=True, text=True,
                               timeout=timeout)
            stdout_text, stderr_text, ret = r.stdout or "", r.stderr or "", \
                r.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout_text = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr_text = (e.stderr or "") if isinstance(e.stderr, str) else ""
        except Exception as e:
            stdout_text, stderr_text, ret = "", "probe setup failed: %s" % e, 99
        try:
            if opath.is_file():
                payload = json.loads(opath.read_text(encoding="utf-8"))
                order = payload.get("order", [])[:MAX_CALLS]
                counts = payload.get("counts", [])
                rr = str(root.resolve())
                for i, triple in enumerate(order):
                    try:
                        rp, fn, ln = triple
                        rel = Path(rp).resolve().relative_to(rr).as_posix()
                    except (ValueError, OSError):
                        continue
                    calls.append({"path": rel, "function": fn,
                                  "first_line": ln,
                                  "count": counts[i] if i < len(counts) else 1,
                                  "kind": "observed_call"})
                if payload.get("truncated_funcs"):
                    trace_note = "traced (function list capped, no full graph)"
                else:
                    trace_note = "traced"
                trace_exc = payload.get("exception")
        except Exception:
            pass
        try:
            tpath.unlink(missing_ok=True)
            opath.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        if args.trace:
            trace_note = ("trace skipped: command is not plain "
                          "'python3 <script.py> [args]' under root; ran "
                          "untraced")
        try:
            r = subprocess.run(args.command, cwd=str(root), shell=True,
                               capture_output=True, text=True, timeout=timeout)
            stdout_text, stderr_text, ret = r.stdout or "", r.stderr or "", \
                r.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out = e.stdout if isinstance(e.stdout, str) else ""
            errt = e.stderr if isinstance(e.stderr, str) else ""
            stdout_text, stderr_text = out or "", errt or ""
        except Exception as e:
            stdout_text, stderr_text, ret = "", "probe setup failed: %s" % e, 99

    combined = (stdout_text or "") + "\n" + (stderr_text or "")
    # Traced runs keep the traceback in the tracer payload, not stderr;
    # it is equally observed output for classification and assertions.
    searchable = combined + "\n" + (trace_exc or "")
    assertions = []
    for spec in (args.assertion or []):
        if ":" not in spec:
            print(json.dumps(err("bad --assert (want 'name:substring')"),
                             ensure_ascii=False))
            return
        name, _, sub = spec.partition(":")
        name, sub = name.strip(), sub
        if not name or not sub:
            print(json.dumps(err("bad --assert (want 'name:substring')"),
                             ensure_ascii=False))
            return
        hit = None
        for ln in searchable.splitlines():
            if sub in ln:
                hit = ln.strip()[:200]
                break
        assertions.append({"name": name, "substring": sub,
                           "observed": hit is not None,
                           "matched_line": hit or ""})
    if timed_out:
        exit_kind = "timeout"
    elif ret == 0 and "AssertionError" in searchable:
        exit_kind = "assertion"
    elif ret == 0:
        exit_kind = "ok"
    elif "AssertionError" in searchable:
        exit_kind = "assertion"
    elif ("ModuleNotFoundError" in searchable or "ImportError" in searchable) \
            and "Traceback" in searchable and len(searchable.strip()) < 2000:
        exit_kind = "setup_error"
    else:
        exit_kind = "exception"
    stack_src = trace_exc or combined
    short_stack = ""
    if exit_kind in ("exception", "assertion") and "Traceback" in stack_src:
        tail = stack_src[stack_src.rfind("Traceback"):]
        short_stack = shorten_stack(tail, root)
    out = {"status": "ok", "command": "probe",
           "observation_id": obs_id, "question": args.question,
           "workspace_patch_id": head, "exit_kind": exit_kind,
           "returncode": ret if ret is not None else "timeout",
           "calls": calls, "assertions": assertions,
           "short_stack": short_stack,
           "stdout_tail": (stdout_text or "")[-STDOUT_TAIL:],
           "stderr_tail": (stderr_text or "")[-STDERR_TAIL:],
           "omitted": len(combined) > STDOUT_TAIL + STDERR_TAIL,
           "trace": trace_note,
           "note": "exit 0 means the command exited 0; it NEVER by itself "
                   "means the issue reproduced or the fix works -- only the "
                   "named assertions above count as observed checks; call "
                   "entries here are observed_call, unlike locate/inspect "
                   "textual_reference"}
    sys.stdout.write(fit_json(out, cap=OUTPUT_CAP))


def build_parser():
    ap = argparse.ArgumentParser(
        prog="source_probe.py",
        description="P005 locate/inspect/probe. Stdlib only, complete JSON "
                    "output (never sliced).")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--allow-non-workspace-root", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_l = sub.add_parser("locate", help="1-4 queries + question, few hits")
    p_l.add_argument("--query", action="append", required=True)
    p_l.add_argument("--source", action="append", default=None,
                     help="provenance per query; defaults to the question")
    p_l.add_argument("--question", required=True)
    p_l.add_argument("--scope", default=".")
    p_l.add_argument("--max-hits", type=int, default=8)
    p_l.set_defaults(func=cmd_locate)

    p_i = sub.add_parser("inspect", help="1-3 targets + question, exact|outline")
    p_i.add_argument("--target", action="append", required=True,
                     help="'path' (outline) or 'path:symbol[:line]' (exact)")
    p_i.add_argument("--question", required=True)
    p_i.add_argument("--mode", default="exact")
    p_i.add_argument("--max-chars", type=int, default=4000)
    p_i.set_defaults(func=cmd_inspect)

    p_p = sub.add_parser("probe", help="run one command + named assertions")
    p_p.add_argument("--command", required=True,
                     help="single command string (no nested quoting layers)")
    p_p.add_argument("--question", required=True)
    p_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p_p.add_argument("--assert", dest="assertion", action="append",
                     default=None, help="'name:substring' (repeatable)")
    p_p.add_argument("--trace", action="store_true",
                     help="setprofile-trace plain python runs, repo-filtered")
    p_p.set_defaults(func=cmd_probe)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    try:
        root = safe_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        print(json.dumps(err(str(e)), ensure_ascii=False))
        return
    args.func(args, root)


if __name__ == "__main__":
    main()
