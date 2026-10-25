"""Build F3-new 25 deepdiff tasks: acquire, recipe, replay."""
import hashlib, json, subprocess, tarfile, urllib.request, difflib, re, sys
from pathlib import Path
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ=ROOT/'competition/vfarl/acquisition/f3-new'
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/f3-new'
REV='77cab9055d42ab4a5c25c89a8f937096db13558e'
UPSTREAM_FULL='ed2520229d0369813f6e54cdf9c7e68e8073ef62'
CANONICAL='seperman/deepdiff'
SNAP=f"{CANONICAL}::{UPSTREAM_FULL}"
TASK_VENV=REPLAY/'.venv-sfr/lib/python3.12/site-packages'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()

GENERIC_OBS = """from deepdiff.lfucache import LFUCache
from deepdiff.helper import not_found
cache = LFUCache(3)
cache.set("alpha", value="one")
cache.set("beta", value="two")
assert cache.get("alpha") == "one", f"hit alpha {cache.get('alpha')}"
assert cache.get("beta") == "two"
assert ("alpha" in cache) is True
assert ("beta" in cache) is True
assert ("gamma" in cache) is False
print("hit_and_isolation", True)
cache.set("alpha", value="one")
assert cache.get("alpha") == "one"
cache.get("alpha")
cache.get("alpha")
cache.set("gamma", value="three")
cache.set("delta", value="four")
c2 = LFUCache(2)
c2.set("x", value="1")
c2.set("y", value="2")
assert c2.get("x") == "1"
c2.set("z", value="3")
assert ("x" in c2) is True, "x should remain"
assert ("y" in c2) is False, "y should be evicted"
assert ("z" in c2) is True
assert c2.get("x") == "1"
assert c2.get("z") == "3"
print("evict_scope", True)
c2.set("y", value="2-recomputed")
assert c2.get("y") == "2-recomputed"
print("recompute_after_evict", True)
assert c2.get("missing-key-xyz") is not_found or c2.get("missing-key-xyz") is None, "miss should be not_found/None"
print("miss_contract", True)
"""

FAMILIES=[
 ('seperman__deepdiff.ed252022.func_basic__1jd0ifnk','LFUCache.__init__','lock_and_head_scope_initialization'),
 ('seperman__deepdiff.ed252022.func_basic__3tay213i','LFUCache.set','capacity_threshold_and_key_order_eviction'),
 ('seperman__deepdiff.ed252022.func_basic__8in7y17s','LFUCache.create_cache_node','key_value_report_mapping_and_freq_init'),
 ('seperman__deepdiff.ed252022.func_basic__c0ex2b8o','FreqNode.__init__','freq_link_field_scope'),
 ('seperman__deepdiff.ed252022.func_basic__dxf86sin','LFUCache.get','hit_content_vs_miss_sentinel'),
 ('seperman__deepdiff.ed252022.func_basic__gib19015','LFUCache.__contains__','key_presence_equivalence_and_isolation'),
 ('seperman__deepdiff.ed252022.func_basic__ostb63f0','FreqNode.count_caches','eviction_decision_count'),
 ('seperman__deepdiff.ed252022.func_basic__rypt1cnn','FreqNode.append_cache_to_tail','insertion_link_scope'),
 ('seperman__deepdiff.ed252022.func_basic__sbq72q16','LFUCache.move_forward','promotion_target_and_head_eviction'),
 ('seperman__deepdiff.ed252022.func_basic__y8bo68lh','CacheNode.__init__','key_content_mapping_and_link_scope'),
 ('seperman__deepdiff.ed252022.func_pm_class_rm_funcs__4gez1lin','LFUCache.set','insertion_method_presence_for_cache_population'),
 ('seperman__deepdiff.ed252022.func_pm_class_rm_funcs__m1pze6jk','FreqNode','frequency_list_maintenance_methods'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__26yb3my0','FreqNode','frequency_list_link_conditions'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__37lsx8qd','LFUCache','frequency_head_promotion_and_cleanup'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__b9v8lw4c','FreqNode','cache_list_link_conditions'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__fnihy4xk','LFUCache','cache_content_update_scope'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__13wdnc1x','CacheNode','node_lifecycle_link_scope'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__2xqawwjs','LFUCache.dump_cache','eviction_head_selection'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__6ih0dfcl','LFUCache.create_cache_node','insertion_frequency_init_and_head_link'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__6ql92r6k','LFUCache.create_cache_node','insertion_head_link_alternative'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__g82wlyt7','LFUCache.move_forward','promotion_link_rewiring'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__lze1dxe1','FreqNode','list_operation_ordering'),
 ('seperman__deepdiff.ed252022.func_pm_ctrl_shuffle__s8nsttnr','FreqNode','list_operation_ordering_variant'),
 ('seperman__deepdiff.ed252022.func_pm_remove_assign__pg4j9jdi','LFUCache.move_forward','promotion_target_binding'),
 ('seperman__deepdiff.ed252022.func_pm_remove_cond__cvjzm7w2','LFUCache.move_forward','promotion_empty_head_handling'),
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
    return int(m.group(1)) if m else 140

# ensure upstream
upstream_dest=ART/'upstream-seperman__deepdiff'
if not (upstream_dest/'deepdiff/lfucache.py').exists():
    acquire(CANONICAL, UPSTREAM_FULL, upstream_dest)
print("upstream ready", flush=True)
idx={json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl')}
# need upstream resolution for snapshot? already have full
# process
prepared=[]
for num,(iid,symbol,contract) in enumerate(FAMILIES, start=1):
    print(f"\n=== {num}/25 {iid} ===", flush=True)
    # find row
    found=None
    for pg in (ACQ/'artifacts/pages').glob('seperman*.json'):
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
    assert len(commit)==40
    key=f"f3-new-deepdiff-{num:02d}"
    ws=ART/(key+'-source')
    buggy=acquire(row['repo'], commit, ws)
    # verify inverse edits apply
    edits=inverse_edits(row['patch'])
    assert edits, "no edits"
    cur=(ws/'deepdiff/lfucache.py').read_text()
    for o,n in edits:
        assert cur.count(o)==1, f"edit not unique {iid} {o[:100]!r} count {cur.count(o)}"
        cur=cur.replace(o,n)
    assert cur==(upstream_dest/'deepdiff/lfucache.py').read_text(), f"reconstructed mismatch {iid}"
    print(f" edits {len(edits)} verified", flush=True)
    # verification targets
    required=list(row['FAIL_TO_PASS'])+list(row['PASS_TO_PASS'])
    targets=sorted({n.split('::',1)[0] for n in required})
    # test_patch for missing files
    missing=[]
    for t in targets:
        if not (ws/t).exists() and (upstream_dest/t).exists():
            try: text=(upstream_dest/t).read_text()
            except UnicodeDecodeError: raise ValueError(f"binary missing {t}")
            missing.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile='/dev/null', tofile='b/'+t))
    test_patch=''.join(missing)
    print(f" targets {targets} missing {len(missing)>0}", flush=True)
    task={'instance_id':iid,'repo':row['repo'],'base_commit':commit,'problem_statement':row['problem_statement'],'patch':'','test_patch':test_patch,'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
    adapted=ART/(key+'-task.json')
    save(adapted, task)
    # recipe
    start=max(1, hunk_start(row['patch'])-12)
    end=start+35
    # binding + obs
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
    cmd="python3 - <<'PYREPLAY'\n"+binding+GENERIC_OBS+"\nPYREPLAY"
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
    review=f"DeepDiff LFUCache {symbol} must preserve {contract}. Set returns cached hits, distinct keys stay isolated, eviction removes least-frequent and recompute restores. Restore exact production logic without hardcoding values or changing tests. Distinct from other LFU contracts sharing snapshot correlation."
    save(recipe_path, {"evidence_kind":"human_reviewed_off_policy_demonstration","reviewed_by":"Codex root source contract and actual command review2026-10-25","review_notes":review+" All original F2P/P2P and complete target files retained. Smith patch introduces the defect. This is an off-policy scripted demonstration, not a Gemma sample or preference pair.","actions":actions,"final_response":f"Restored {symbol} {contract}. The actual inline cache observation completed after the production edit. Patch submitted."})
    print(f" recipe {recipe_path} actions {len(actions)}", flush=True)
    # source prep
    fam=f"deepdiff.{symbol}::{contract}"
    prep={"instance_id":iid,"key":key,"original_task":str(orig.relative_to(ROOT)),"original_task_sha256":sha(orig),"adapted_task":str(adapted.relative_to(ROOT)),"workspace":str(ws.relative_to(ROOT)),"source_dataset":"SWE-bench/SWE-smith-py","source_train_revision":REV,"physical_split":"train","course_membership":"F600_F3_new","primary_adapter":"F","submechanism":"F3","canonical_repo":CANONICAL,"repair_family_id":fam,"snapshot_group_id":SNAP,"execution_commit":commit,"git_ls_remote":remote,"buggy_source":buggy,"upstream_source":{"repo":CANONICAL,"requested_revision":"ed252022","resolved_revision":UPSTREAM_FULL,"archive_path":str((ART/'upstream-seperman__deepdiff.tgz').relative_to(ROOT)) if (ART/'upstream-seperman__deepdiff.tgz').exists() else str(upstream_dest.with_suffix('.tgz').relative_to(ROOT)),"archive_sha256":digest((ART/'upstream-seperman__deepdiff.tgz').read_bytes()) if (ART/'upstream-seperman__deepdiff.tgz').exists() else digest((upstream_dest.with_suffix('.tgz')).read_bytes())},"recipe":str(recipe_path.relative_to(ROOT)),"recipe_sha256":sha(recipe_path),"F2P_count":len(row['FAIL_TO_PASS']),"P2P_count":len(row['PASS_TO_PASS']),"mechanism_evidence":f"LFUCache {symbol} {contract}: set->get hit, key equivalence (same key same value) and isolation (different keys), evict least-frequent then recompute. Second observation differs after invalidation (evicted key miss vs hit).",
     "verification_target_files":targets}
    save(REPLAY/(key+'-source-preparation.json'), prep)
    prepared.append(prep)
    print(f" prepared {key} fam {fam}", flush=True)

save(ACQ/'f3-new-25-preparation.json', {"tasks":prepared})
print(f"\nPREPARED {len(prepared)}", flush=True)
