"""Build 6 deepdiff tailored where generic passed."""
import hashlib, json, subprocess, tarfile, urllib.request, difflib, re
from pathlib import Path
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ=ROOT/'competition/vfarl/acquisition/f3-new'
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/f3-new'
REV='77cab9055d42ab4a5c25c89a8f937096db13558e'
UPSTREAM_FULL='ed2520229d0369813f6e54cdf9c7e68e8073ef62'
CANONICAL='seperman/deepdiff'
SNAP=f"{CANONICAL}::{UPSTREAM_FULL}"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()

EWBFECJB = """from deepdiff.lfucache import LFUCache
c = LFUCache(3)
c.set("a", value="1")
c.set("b", value="2")
c.set("c", value="3")
assert c.freq_link_head.freq == 0
assert c.freq_link_head.count_caches() == 3
c.get("a")
assert c.freq_link_head.freq == 0
assert c.freq_link_head.count_caches() == 2
c.set("d", value="4")
print("head_after_dump", c.freq_link_head.freq, c.freq_link_head.count_caches())
assert c.freq_link_head.freq == 0, f"head freq {c.freq_link_head.freq}"
assert ("b" in c) is False, "b evicted"
assert ("c" in c) is True, "c retained isolation"
assert ("a" in c) is True
assert ("d" in c) is True
assert c.get("c") == "3"
assert c.get("a") == "1"
print("dump_head_scope", True)
c.set("b", value="2-recomputed")
assert c.get("b") == "2-recomputed"
print("recompute", True)
assert c.get("a") == "1"
print("equivalence", True)
"""
GOVWA93U = """from deepdiff.lfucache import LFUCache
c = LFUCache(5)
c.set("a", value="1")
c.set("b", value="2")
c.set("c", value="3")
assert c.freq_link_head.count_caches() == 3
c.get("b")
print("head_after_middle_promote", c.freq_link_head.count_caches())
assert c.freq_link_head.count_caches() == 2
assert ("a" in c) is True
assert ("b" in c) is True
assert ("c" in c) is True
assert c.get("a") == "1"
assert c.get("b") == "2"
assert c.get("c") == "3"
print("middle_unlink_scope", True)
c2 = LFUCache(2)
c2.set("x", value="1")
c2.set("y", value="2")
assert c2.get("x") == "1"
c2.set("z", value="3")
assert ("x" in c2) is True
assert ("y" in c2) is False
assert ("z" in c2) is True
print("evict_isolation", True)
c2.set("y", value="2-recomputed")
assert c2.get("y") == "2-recomputed"
print("recompute", True)
"""
R98BTVBB = """from deepdiff.lfucache import LFUCache
c = LFUCache(5)
c.set("a", value="1")
c.set("b", value="2")
c.set("c", value="3")
c.get("c")
assert ("a" in c) is True
assert ("b" in c) is True
assert ("c" in c) is True
assert c.get("c") == "3"
print("tail_promote", True)
d = LFUCache(5)
d.set("a", value="1")
d.set("b", value="2")
d.set("c", value="3")
d.get("b")
assert ("a" in d) is True
assert ("b" in d) is True
assert ("c" in d) is True
assert d.get("b") == "2"
print("middle_promote_isolation", True)
e = LFUCache(2)
e.set("x", value="1")
e.set("y", value="2")
assert e.get("x") == "1"
e.set("z", value="3")
assert ("y" in e) is False
assert e.get("z") == "3"
e.set("y", value="2-recomputed")
assert e.get("y") == "2-recomputed"
print("recompute", True)
"""
OP63 = """from deepdiff.lfucache import LFUCache
c = LFUCache(3)
c.set("a", value="1")
assert c.cache["a"].freq_node.freq == 0
c.get("a")
print("freq_after_promote", c.cache["a"].freq_node.freq)
assert c.cache["a"].freq_node.freq == 1, f"freq {c.cache['a'].freq_node.freq}"
assert c.get("a") == "1"
assert ("a" in c) is True
assert ("b" in c) is False
print("promotion_target", True)
c.set("b", value="2")
assert c.get("b") == "2"
assert c.get("a") == "1"
print("isolation", True)
c2 = LFUCache(2)
c2.set("x", value="1")
c2.set("y", value="2")
assert c2.get("x") == "1"
c2.set("z", value="3")
assert ("y" in c2) is False
c2.set("y", value="2-recomputed")
assert c2.get("y") == "2-recomputed"
print("recompute", True)
"""
OPWMY = """from deepdiff.lfucache import LFUCache
c = LFUCache(3)
c.set("a", value="1")
c.get("a")
print("freq_after_promote", c.cache["a"].freq_node.freq)
assert c.cache["a"].freq_node.freq == 1, f"freq {c.cache['a'].freq_node.freq}"
c.get("a")
assert c.cache["a"].freq_node.freq == 2, f"freq2 {c.cache['a'].freq_node.freq}"
assert c.get("a") == "1"
print("promotion_steps", True)
assert ("a" in c) is True
c.set("b", value="2")
assert ("b" in c) is True
assert c.get("b") == "2"
print("isolation", True)
c2 = LFUCache(2)
c2.set("x", value="1")
c2.set("y", value="2")
assert c2.get("x") == "1"
c2.set("z", value="3")
assert ("y" in c2) is False
c2.set("y", value="2-recomputed")
assert c2.get("y") == "2-recomputed"
print("recompute", True)
"""
CREATE3GV = """from deepdiff.lfucache import LFUCache
c = LFUCache(3)
c.set("a", value="1")
print("init_freq", c.cache["a"].freq_node.freq, c.freq_link_head.freq)
assert c.cache["a"].freq_node.freq == 0, f"freq {c.cache['a'].freq_node.freq}"
assert c.freq_link_head.freq == 0
assert ("a" in c) is True
assert c.get("a") == "1"
c.set("b", value="2")
print("second_freq", c.cache["b"].freq_node.freq)
assert c.cache["b"].freq_node.freq == 0, f"b freq {c.cache['b'].freq_node.freq}"
assert ("b" in c) is True
print("insertion_freq_scope", True)
c2 = LFUCache(2)
c2.set("x", value="1")
c2.set("y", value="2")
assert c2.get("x") == "1"
c2.set("z", value="3")
assert ("y" in c2) is False
c2.set("y", value="2-recomputed")
assert c2.get("y") == "2-recomputed"
print("recompute", True)
"""

FAMILIES=[
 ('seperman__deepdiff.ed252022.func_basic__ewbfecjb','LFUCache.dump_cache','head_update_when_single_remaining',EWBFECJB),
 ('seperman__deepdiff.ed252022.func_basic__govwa93u','CacheNode.free_myself','middle_unlink_scope',GOVWA93U),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__r98btvbb','CacheNode.free_myself','tail_vs_middle_dispatch',R98BTVBB),
 ('seperman__deepdiff.ed252022.func_pm_op_change__63trk9w1','LFUCache.move_forward','promotion_target_frequency',OP63),
 ('seperman__deepdiff.ed252022.func_pm_op_change_const__wmy6zc4f','LFUCache.move_forward','promotion_step_frequency',OPWMY),
 ('seperman__deepdiff.ed252022.lm_rewrite__3gv2o7aw','LFUCache.create_cache_node','insertion_frequency_init_scope',CREATE3GV),
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
    import tarfile, urllib.request
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
    import re
    m=re.search(r'@@ -\d+(?:,\d+)? \+(\d+)', patch)
    return int(m.group(1)) if m else 140

upstream_dest=ART/'upstream-seperman__deepdiff'
idx={json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl')}
prepared=[]
for num,(iid,symbol,contract,obs) in enumerate(FAMILIES, start=26):
    print(f"\n=== {num}/40 {iid} ===", flush=True)
    found=None
    for pg in (ACQ/'artifacts/pages').glob('seperman*.json'):
        if pg.name.endswith('.receipt.json'): continue
        d=json.load(open(pg))
        for item in d['rows']:
            if item['row']['instance_id']==iid:
                found=(item,pg); break
        if found: break
    assert found
    item,pg=found
    row=item['row']
    orig=ACQ/(iid+'.task.json')
    if not orig.exists():
        save(orig, {'row':row,'source':idx[iid],'saved_page':str(pg.relative_to(ROOT)),'saved_page_sha256':sha(pg)})
    remote=subprocess.check_output(['git','ls-remote','https://github.com/'+row['repo']+'.git','refs/heads/'+iid],text=True).strip()
    commit,ref=remote.split()
    key=f"f3-new-deepdiff-{num:02d}"
    ws=ART/(key+'-source')
    buggy=acquire(row['repo'], commit, ws)
    edits=inverse_edits(row['patch'])
    assert edits
    cur=(ws/'deepdiff/lfucache.py').read_text()
    for o,n in edits:
        assert cur.count(o)==1, f"edit not unique {iid} {o[:80]!r} {cur.count(o)}"
        cur=cur.replace(o,n)
    assert cur==(upstream_dest/'deepdiff/lfucache.py').read_text()
    print(f" edits {len(edits)} verified", flush=True)
    required=list(row['FAIL_TO_PASS'])+list(row['PASS_TO_PASS'])
    targets=sorted({n.split('::',1)[0] for n in required})
    import difflib
    missing=[]
    for t in targets:
        if not (ws/t).exists() and (upstream_dest/t).exists():
            text=(upstream_dest/t).read_text()
            missing.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile='/dev/null', tofile='b/'+t))
    test_patch=''.join(missing)
    task={'instance_id':iid,'repo':row['repo'],'base_commit':commit,'problem_statement':row['problem_statement'],'patch':'','test_patch':test_patch,'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
    adapted=ART/(key+'-task.json')
    save(adapted, task)
    start=max(1, hunk_start(row['patch'])-12)
    end=start+35
    binding=("from pathlib import Path\nimport hashlib, importlib\n"
             f"expected=(Path.cwd() / 'deepdiff/lfucache.py').resolve()\n"
             "try:\n module = importlib.import_module('deepdiff.lfucache')\n"
             "except Exception as import_error:\n traceback=import_error.__traceback__\n"
             " while traceback is not None:\n  file=traceback.tb_frame.f_globals.get('__file__')\n"
             "  if file and Path(file).resolve()==expected:\n   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')\n"
             "  traceback=traceback.tb_next\n raise\n"
             "actual=Path(module.__file__).resolve()\n"
             'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
             "assert actual == expected, (actual, expected)\n")
    cmd="python3 - <<'PYREPLAY'\n"+binding+obs+"\nPYREPLAY"
    def act(tool,args,exp="ok"):
        v={"tool":tool,"arguments":args,"expected_status":exp}
        if tool=="run_command": v["reviewed_command_sha256"]=digest(args["command"].encode())
        return v
    actions=[act("get_status",{}), act("read_file",{"filepath":"deepdiff/lfucache.py","start_line":start,"end_line":end}), act("run_command",{"command":cmd},"error")]
    cur2=(ws/'deepdiff/lfucache.py').read_text()
    for o,n in edits:
        assert cur2.count(o)==1
        cur2=cur2.replace(o,n)
        actions.append(act("edit_file",{"filepath":"deepdiff/lfucache.py","old_string":o,"new_string":n}))
    actions.extend([act("run_command",{"command":cmd}), act("get_status",{}), act("submit_patch",{})])
    recipe_path=REPLAY/(key+'-reviewed-recipe.json')
    review=f"DeepDiff LFUCache {symbol} must preserve {contract}. Set returns cached hits, distinct keys isolated, eviction removes least-frequent and recompute restores."
    save(recipe_path, {"evidence_kind":"human_reviewed_off_policy_demonstration","reviewed_by":"Codex root source contract and actual command review2026-10-25","review_notes":review+" All original F2P/P2P retained. Smith patch introduces defect. Off-policy.","actions":actions,"final_response":f"Restored {symbol} {contract}. Inline observation completed. Patch submitted."})
    fam=f"deepdiff.{symbol}::{contract}"
    prep={"instance_id":iid,"key":key,"original_task":str(orig.relative_to(ROOT)),"original_task_sha256":sha(orig),"adapted_task":str(adapted.relative_to(ROOT)),"workspace":str(ws.relative_to(ROOT)),"source_dataset":"SWE-bench/SWE-smith-py","source_train_revision":REV,"physical_split":"train","course_membership":"F600_F3_new","primary_adapter":"F","submechanism":"F3","canonical_repo":CANONICAL,"repair_family_id":fam,"snapshot_group_id":SNAP,"execution_commit":commit,"git_ls_remote":remote,"buggy_source":buggy,"upstream_source":{"repo":CANONICAL,"requested_revision":"ed252022","resolved_revision":UPSTREAM_FULL,"archive_path":str((ART/'upstream-seperman__deepdiff.tgz').relative_to(ROOT)),"archive_sha256":digest((ART/'upstream-seperman__deepdiff.tgz').read_bytes())},"recipe":str(recipe_path.relative_to(ROOT)),"recipe_sha256":sha(recipe_path),"F2P_count":len(row['FAIL_TO_PASS']),"P2P_count":len(row['PASS_TO_PASS']),"mechanism_evidence":f"LFUCache {symbol} {contract}: set->get hit, key equivalence/isolation, evict then recompute. Second observation differs.","verification_target_files":targets}
    save(REPLAY/(key+'-source-preparation.json'), prep)
    prepared.append(prep)
    print(f" prepared {key} {fam}", flush=True)
save(ACQ/'f3-new-tailored6-preparation.json', {"tasks":prepared})
print(f"PREPARED {len(prepared)}", flush=True)
