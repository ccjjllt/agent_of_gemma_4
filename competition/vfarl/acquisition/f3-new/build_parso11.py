"""Build F3-new parso 11 with expanded obs."""
import hashlib, json, subprocess, tarfile, urllib.request, difflib, re
from pathlib import Path
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ=ROOT/'competition/vfarl/acquisition/f3-new'
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/f3-new'
REV='77cab9055d42ab4a5c25c89a8f937096db13558e'
UPSTREAM_FULL='338a57602740ad0645b2881e8c105ffdc959e90d'
CANONICAL='davidhalter/parso'
SNAP=f"{CANONICAL}::{UPSTREAM_FULL}"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()

EXPANDED = """from parso.cache import parser_cache, _set_cache_item, _NodeCacheItem, _get_cache_directory_path, _get_hashed_path, clear_cache, _touch, _remove_cache_and_update_lock, _get_cache_clear_lock_path
from pathlib import Path
import tempfile, os, time
parser_cache.clear()
item_a = _NodeCacheItem(node="node-a", lines=["l1"], change_time=None)
assert item_a.node == "node-a" and item_a.lines == ["l1"]
assert item_a.change_time is not None and item_a.last_used is not None
print("item_fields", True)
item_b = _NodeCacheItem(node="node-b", lines=["l2"], change_time=None)
_set_cache_item("grammar1", "/tmp/a.py", item_a)
_set_cache_item("grammar1", "/tmp/b.py", item_b)
assert "grammar1" in parser_cache
assert "/tmp/a.py" in parser_cache["grammar1"]
assert "/tmp/b.py" in parser_cache["grammar1"]
assert "/tmp/c.py" not in parser_cache.get("grammar1", {})
print("set_get_isolation", True)
_set_cache_item("grammar1", "/tmp/a.py", item_a)
assert parser_cache["grammar1"]["/tmp/a.py"] is item_a
print("equivalence", True)
with tempfile.TemporaryDirectory() as d:
    cp = Path(d)
    h1 = _get_hashed_path("g1", "/tmp/a.py", cache_path=cp)
    h2 = _get_hashed_path("g1", "/tmp/a.py", cache_path=cp)
    h3 = _get_hashed_path("g1", "/tmp/b.py", cache_path=cp)
    assert h1 == h2
    assert h1 != h3
    assert str(h1).endswith(".pkl")
    print("hashed_isolation", True)
    p1 = _get_cache_directory_path(cache_path=cp)
    assert p1.exists()
    p2 = _get_cache_directory_path(cache_path=cp)
    assert p1 == p2
    print("dir_scope", True)
    from parso.cache import _save_to_file_system, _load_from_file_system
    item = _NodeCacheItem(node="n", lines=["l"], change_time=time.time())
    _save_to_file_system("g1", "/tmp/a.py", item, cache_path=cp)
    loaded = _load_from_file_system("g1", "/tmp/a.py", 0, cache_path=cp)
    assert loaded == "n", f"hit {loaded!r}"
    stale = _load_from_file_system("g1", "/tmp/a.py", time.time()+10000, cache_path=cp)
    assert stale is None
    miss = _load_from_file_system("g1", "/tmp/b.py", 0, cache_path=cp)
    assert miss is None
    print("file_staleness_isolation", True)
    tf = cp / "touchfile"
    assert _touch(tf) is True
    assert tf.exists()
    assert _touch(tf) is True
    print("touch_scope", True)
    r1 = _remove_cache_and_update_lock(cache_path=cp)
    lock = _get_cache_clear_lock_path(cache_path=cp)
    assert lock.exists()
    m1 = os.path.getmtime(lock)
    time.sleep(0.05)
    _remove_cache_and_update_lock(cache_path=cp)
    m2 = os.path.getmtime(lock)
    assert m2 == m1
    print("expire_threshold", True)
    clear_cache(cache_path=cp)
    print("cleared", True)
    parser_cache.clear()
    _set_cache_item("grammar1", "/tmp/a.py", item_a)
    assert parser_cache["grammar1"]["/tmp/a.py"] is item_a
    print("recompute_after_invalidate", True)
"""

FAMILIES=[
 ('davidhalter__parso.338a5760.func_basic__3iwakld6','_get_cache_directory_path','directory_creation_scope'),
 ('davidhalter__parso.338a5760.func_basic__81yk55jm','_set_cache_item','eviction_threshold_and_key_order'),
 ('davidhalter__parso.338a5760.func_basic__e1d2xiy9','_remove_cache_and_update_lock','clear_threshold_and_lock_time'),
 ('davidhalter__parso.338a5760.func_basic__euo60bxd','_load_from_file_system','hashed_path_order_and_staleness'),
 ('davidhalter__parso.338a5760.func_basic__lxdtozlg','_touch','file_creation_and_return_scope'),
 ('davidhalter__parso.338a5760.func_basic__mkplb5q1','_get_hashed_path','hash_input_and_extension_scope'),
 ('davidhalter__parso.338a5760.func_basic__zcabcuzz','_NodeCacheItem.__init__','node_lines_and_time_scope'),
 ('davidhalter__parso.338a5760.func_pm_ctrl_shuffle__cg3az98k','_NodeCacheItem.__init__','assignment_ordering_and_last_used'),
 ('davidhalter__parso.338a5760.func_pm_ctrl_shuffle__jrz9j8ww','_touch','early_return_vs_file_creation'),
]

def inverse_edits(patch):
    edits=[];old=[];new=[]
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@ '):
            if old or new: edits.append((''.join(old),''.join(new)))
            old=[];new=[]
        elif line.startswith(('diff --git ','index ','--- ','+++ ','\\')): continue
        elif line.startswith(' '): old.append(line[1:]);new.append(line[1:])
        elif line.startswith('+'): old.append(line[1:])
        elif line.startswith('-'): new.append(line[1:])
    if old or new: edits.append((''.join(old),''.join(new)))
    return [e for e in edits if e[0]!=e[1]]

def acquire(repo, rev, dest):
    dest.mkdir(parents=True, exist_ok=True)
    arch=dest.with_suffix('.tgz')
    url=f"https://codeload.github.com/{repo}/tar.gz/{rev}"
    if not arch.exists():
        print(f" downloading {repo}@{rev}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            arch.write_bytes(r.read())
    with tarfile.open(arch) as bundle:
        members=bundle.getmembers()
        top=members[0].name.split('/')[0]
        for m in members:
            parts=Path(m.name).parts
            if not parts or parts[0]!=top or '..' in parts: continue
            rel=Path(*parts[1:])
            if not rel.parts: continue
            tgt=dest/rel
            if m.isdir(): tgt.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                payload=bundle.extractfile(m).read()
                if tgt.exists() and tgt.read_bytes()!=payload:
                    raise ValueError(f"altered {tgt}")
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(payload)
    return {"repo":repo,"requested_revision":rev,"resolved_revision":rev,"archive_path":str(arch.relative_to(ROOT)),"archive_sha256":digest(arch.read_bytes()),"download_url":url,"immutable_url":url,"commit_resolution":None}

def hunk_start(patch):
    m=re.search(r'@@ -\d+(?:,\d+)? \+(\d+)', patch)
    return int(m.group(1)) if m else 100

upstream_dest=ART/'upstream-davidhalter__parso'
print("upstream ready", flush=True)
idx={json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl')}
prepared=[]
for num,(iid,symbol,contract) in enumerate(FAMILIES, start=26):
    print(f"\n=== {num}/40 {iid} ===", flush=True)
    found=None
    for pg in (ACQ/'artifacts/pages').glob('davidhalter*.json'):
        if pg.name.endswith('.receipt.json'): continue
        d=json.load(open(pg))
        for item in d['rows']:
            if item['row']['instance_id']==iid:
                found=(item,pg); break
        if found: break
    assert found, iid
    item,pg=found
    row=item['row']
    orig=ACQ/(iid+'.task.json')
    if not orig.exists():
        save(orig, {'row':row,'source':idx[iid],'saved_page':str(pg.relative_to(ROOT)),'saved_page_sha256':sha(pg)})
    remote=subprocess.check_output(['git','ls-remote','https://github.com/'+row['repo']+'.git','refs/heads/'+iid],text=True).strip()
    commit,ref=remote.split()
    key=f"f3-new-parso-{num:02d}"
    ws=ART/(key+'-source')
    buggy=acquire(row['repo'], commit, ws)
    edits=inverse_edits(row['patch'])
    assert edits
    cur=(ws/'parso/cache.py').read_text()
    for o,n in edits:
        assert cur.count(o)==1, f"edit not unique {iid}"
        cur=cur.replace(o,n)
    assert cur==(upstream_dest/'parso/cache.py').read_text(), f"mismatch {iid}"
    print(f" edits {len(edits)} verified", flush=True)
    required=list(row['FAIL_TO_PASS'])+list(row['PASS_TO_PASS'])
    targets=sorted({n.split('::',1)[0] for n in required})
    missing=[]
    for t in targets:
        if not (ws/t).exists() and (upstream_dest/t).exists():
            try: text=(upstream_dest/t).read_text()
            except UnicodeDecodeError: raise ValueError(f"binary {t}")
            missing.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile='/dev/null', tofile='b/'+t))
    test_patch=''.join(missing)
    task={'instance_id':iid,'repo':row['repo'],'base_commit':commit,'problem_statement':row['problem_statement'],'patch':'','test_patch':test_patch,'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
    adapted=ART/(key+'-task.json')
    save(adapted, task)
    start=max(1, hunk_start(row['patch'])-12)
    end=start+35
    binding=("from pathlib import Path\nimport hashlib, importlib\n"
             f"expected=(Path.cwd() / 'parso/cache.py').resolve()\n"
             "try:\n module = importlib.import_module('parso.cache')\n"
             "except Exception as import_error:\n traceback=import_error.__traceback__\n"
             " while traceback is not None:\n  file=traceback.tb_frame.f_globals.get('__file__')\n"
             "  if file and Path(file).resolve()==expected:\n   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')\n"
             "  traceback=traceback.tb_next\n raise\n"
             "actual=Path(module.__file__).resolve()\n"
             'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
             "assert actual == expected, (actual, expected)\n")
    cmd="python3 - <<'PYREPLAY'\n"+binding+EXPANDED+"\nPYREPLAY"
    def act(tool,args,exp="ok"):
        v={"tool":tool,"arguments":args,"expected_status":exp}
        if tool=="run_command": v["reviewed_command_sha256"]=digest(args["command"].encode())
        return v
    actions=[act("get_status",{}), act("read_file",{"filepath":"parso/cache.py","start_line":start,"end_line":end}), act("run_command",{"command":cmd},"error")]
    cur2=(ws/'parso/cache.py').read_text()
    for o,n in edits:
        assert cur2.count(o)==1
        cur2=cur2.replace(o,n)
        actions.append(act("edit_file",{"filepath":"parso/cache.py","old_string":o,"new_string":n}))
    actions.extend([act("run_command",{"command":cmd}), act("get_status",{}), act("submit_patch",{})])
    recipe_path=REPLAY/(key+'-reviewed-recipe.json')
    review=f"Parso cache {symbol} must preserve {contract}. Set stores grammar+path composite keys with equivalence and isolation, file save/load respects staleness, directory/hashed paths isolate, touch/lock/threshold scope expire correctly, clear invalidates and recompute restores. Restore exact logic without hardcoding or changing tests."
    save(recipe_path, {"evidence_kind":"human_reviewed_off_policy_demonstration","reviewed_by":"Codex root source contract and actual command review2026-10-25","review_notes":review+" All original F2P/P2P and complete target files retained. Smith patch introduces the defect. Off-policy demonstration, not Gemma sample.","actions":actions,"final_response":f"Restored {symbol} {contract}. Inline cache observation completed after edit. Patch submitted."})
    fam=f"parso.{symbol}::{contract}"
    prep={"instance_id":iid,"key":key,"original_task":str(orig.relative_to(ROOT)),"original_task_sha256":sha(orig),"adapted_task":str(adapted.relative_to(ROOT)),"workspace":str(ws.relative_to(ROOT)),"source_dataset":"SWE-bench/SWE-smith-py","source_train_revision":REV,"physical_split":"train","course_membership":"F600_F3_new","primary_adapter":"F","submechanism":"F3","canonical_repo":CANONICAL,"repair_family_id":fam,"snapshot_group_id":SNAP,"execution_commit":commit,"git_ls_remote":remote,"buggy_source":buggy,"upstream_source":{"repo":CANONICAL,"requested_revision":"338a5760","resolved_revision":UPSTREAM_FULL,"archive_path":str((ART/'upstream-davidhalter__parso.tgz').relative_to(ROOT)),"archive_sha256":digest((ART/'upstream-davidhalter__parso.tgz').read_bytes())},"recipe":str(recipe_path.relative_to(ROOT)),"recipe_sha256":sha(recipe_path),"F2P_count":len(row['FAIL_TO_PASS']),"P2P_count":len(row['PASS_TO_PASS']),"mechanism_evidence":f"Parso {symbol} {contract}: set->get hit, key equivalence and isolation, expire/invalidate via clear/threshold then recompute. Second observation differs after invalidation.","verification_target_files":targets}
    save(REPLAY/(key+'-source-preparation.json'), prep)
    prepared.append(prep)
    print(f" prepared {key} {fam}", flush=True)
save(ACQ/'f3-new-parso11-preparation.json', {"tasks":prepared})
print(f"PREPARED {len(prepared)}", flush=True)
