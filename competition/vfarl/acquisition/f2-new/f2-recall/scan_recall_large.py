"""F2 f2-recall: recall nontrivial large rows (gym 770 / v2 1254) reranked by strict-copy + is."""
import hashlib, json, re, sys
from pathlib import Path
ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT/'competition/vfarl'
ACQ = VFARL/'acquisition/f2-new'
OUT = ACQ/'f2-recall'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV='bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV='10483de0f50fe5da545942705a76c6150171af7f'
COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
CLONE_SIG = re.compile(r'\bclone\b|\.clone\(|clone\(|copyInto|copy_into|copyFrom|copy_from|__copy\b')
OWN_SIG = re.compile(r'alias|shar(?:ed|ing)|mutat|isolat|ownership|frozen|immutable|singleton|memo\b|shallow|deep.?copy|copy_on_write|copy-on-write', re.I)
F2P_NAME_SIG = re.compile(r'copy|clone|deep|shallow|alias|ident|isolat|shared|mutat|ownership|immutable|frozen|singleton|same|original|pollut|leak|retain|preserv', re.I)
TEST_IS_SIG = re.compile(r'\bis\s+(?:not\s+)?(?:None|True|False|[A-Za-z_][\w.]*)\b|assertIs|assertIsNot|is not|isinstance')
# strict is: is/is not/assertIs bound to an identifier (not None/True/False), exclude isinstance
STRICT_IS = re.compile(r'assertIs(?:Not)?\b|\bis\s+(?:not\s+)?(?!None\b|True\b|False\b)[A-Za-z_][\w.]*\b|\bis\s+not\b(?!None\b)')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def load_forbidden():
    forb_i,forb_s,forb_f=set(),set(),set()
    notes=[]
    base=json.loads((ACQ/'dedup-baseline.json').read_text())
    forb_i|=set(base['forbidden_instance_ids']); forb_s|=set(base.get('forbidden_snapshot_group_ids',[])); forb_f|=set(base['forbidden_repair_family_ids'])
    notes.append(f"common-400: {len(base['forbidden_instance_ids'])}i")
    f1=json.loads((VFARL/'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks',[]):
        forb_i.add(t['instance_id'])
        if t.get('snapshot_group_id'): forb_s.add(t['snapshot_group_id'])
        if t.get('repair_family_id'): forb_f.add(t['repair_family_id'])
    notes.append(f"F1-40: {len(f1.get('tasks',[]))}")
    f5=json.loads((VFARL/'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks',[]):
        forb_i.add(t.get('instance_id',t.get('canonical_task_id','')))
        for k in ('snapshot_group_id','snapshot_id'):
            if t.get(k): forb_s.add(t[k])
        for k in ('repair_family_id','family_id'):
            if t.get(k): forb_f.add(t[k])
    notes.append(f"F5-old40: {len(f5.get('tasks',[]))}")
    for rel in ['f2-replace/catalog.json','f2-replace/reviewed-candidates.json','candidate-list-40.json','reviewed-candidates.json','catalog.json']:
        p=ACQ/rel
        try: d=json.loads(p.read_text())
        except: continue
        for t in d.get('tasks',d.get('candidates',[])):
            if t.get('instance_id'): forb_i.add(t['instance_id'])
            for k in ('snapshot_group_id','snapshot_group_id_provisional'):
                if t.get(k): forb_s.add(t[k])
            for k in ('repair_family_id','repair_family_id_provisional'):
                if t.get(k): forb_f.add(t[k])
        notes.append(f"f2prior:{rel}: {len(d.get('tasks',d.get('candidates',[])))}")
    forb_s|={'seperman/deepdiff::ed252022','pallets/click::fde47b4b','pydantic/pydantic::acb0f10f','stanfordnlp/dspy::651a4c71','sunpy/sunpy::f8edfd5c','mido/mido::a0158ff9','cantools/cantools::0c6a7871'}
    notes.append("explicit round1 snapshot bans: 7")
    for kdir in ['f2-gymv2','f2-gymv2b','f2-gymv2c']:
        g2keep=json.loads((ACQ/kdir/'keep-list.json').read_text())
        g2fam=json.loads((ACQ/kdir/'keep-families.json').read_text())
        for k in g2keep:
            forb_i.add(k['instance_id']); forb_s.add(f"{k['repo']}::{k['base_commit']}")
        forb_f|=set(g2fam.values())
        notes.append(f"{kdir}: {len(g2keep)} keeps")
    return forb_i,forb_s,forb_f,notes
def snap_hit(repo,commit,forb_s):
    if not commit: return None
    for s in forb_s:
        if '::' not in s: continue
        r,rev=s.split('::',1)
        if r.lower()!=repo.lower(): continue
        c=commit.lower(); rr=rev.lower()
        if c.startswith(rr) or rr.startswith(c): return s
    return None
def base_ok_nontrivial(r,forb_i):
    iid=r['instance_id']
    if iid in forb_i: return False
    repo=(r.get('repo')or'').strip()
    if repo.lower() in EXCLUDED_REPOS: return False
    f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
    if not f2p or set(f2p)&set(p2p): return False
    if not (r.get('problem_statement')or'').strip(): return False
    patch=r.get('patch')or''
    files=re.findall(r'diff --git a/(\S+)',patch)
    if any('..' in f for f in files): return False
    return True
def changed_hits(patch,sig):
    chg=[l for l in patch.splitlines() if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++','---'))]
    return [l.strip()[:220] for l in chg if sig.search(l)]
def is_lines(text,pat):
    return [l.strip()[:220] for l in (text or '').splitlines() if pat.search(l)][:8]
if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'scan'
    OUT.mkdir(parents=True,exist_ok=True)
    if which=='dedup':
        fi,fs,ff,notes=load_forbidden()
        save(OUT/'dedup-evidence.json',{'date':'2026-10-25','scope':'F600-F2 f2-recall vs common-400 + F1-40 + F5-old40 + f2-round1/2 + f2-gymv2-16 + f2-gymv2b-5 + f2-gymv2c-2 disjointness','forbidden_instance_count':len(fi),'forbidden_snapshot_count':len(fs),'forbidden_family_count':len(ff),'sources':notes,'method':'instance exact-match; snapshot repo-equality + commit prefix either way; family exact-match; four competition repos excluded','physical_split':'train only'})
        print(f'forbidden: {len(fi)}i/{len(fs)}s/{len(ff)}f')
        for n in notes: print('  ',n)
    elif which=='scan':
        import pyarrow.parquet as pq
        fi,fs,_,_=load_forbidden()
        # gym
        rows=pq.read_table(str(GYM_PARQUET)).to_pylist()
        gym_nontrivial=[]; gym_le200=[]
        for idx,r in enumerate(rows):
            if not base_ok_nontrivial(r,fi): continue
            f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
            total=len(f2p)+len(p2p)
            if total<=150: continue
            patch=r.get('patch')or''; tp=r.get('test_patch')or''
            sc=changed_hits(patch,COPY_SIG)
            strict=bool(sc)
            ti=is_lines(tp,TEST_IS_SIG)
            si=is_lines(tp,STRICT_IS)
            f2pn=[n for n in f2p if F2P_NAME_SIG.search(n)]
            entry={'instance_id':r['instance_id'],'pool':'gym','parquet_idx':idx,'repo':(r.get('repo')or'').strip(),'base_commit':r.get('base_commit'),'F2P':len(f2p),'P2P':len(p2p),'total':total,'strict_copy':strict,'strict_hits':sc[:4],'test_is_any':bool(ti),'test_is_strict':bool(si),'test_is_lines':si[:4] if si else ti[:4],'f2p_name_hits':len(f2pn),'snapshot_collision':snap_hit((r.get('repo')or'').strip(),r.get('base_commit')or'',fs)}
            gym_nontrivial.append(entry)
            if total<=200: gym_le200.append(entry)
        # v2
        f=pq.ParquetFile(str(V2_PARQUET))
        v2_nontrivial=[]; v2_le200=[]; n_py=0
        for rg in range(f.num_row_groups):
            tbl=f.read_row_group(rg)
            for j,r in enumerate(tbl.to_pylist()):
                if (r.get('language')or'').lower()!='python': continue
                n_py+=1
                if not base_ok_nontrivial(r,fi): continue
                f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
                total=len(f2p)+len(p2p)
                if total<=150: continue
                patch=r.get('patch')or''; tp=r.get('test_patch')or''
                sc=changed_hits(patch,COPY_SIG)
                strict=bool(sc)
                ti=is_lines(tp,TEST_IS_SIG); si=is_lines(tp,STRICT_IS)
                f2pn=[n for n in f2p if F2P_NAME_SIG.search(n)]
                entry={'instance_id':r['instance_id'],'pool':'v2','row_group':rg,'repo':(r.get('repo')or'').strip(),'base_commit':r.get('base_commit'),'F2P':len(f2p),'P2P':len(p2p),'total':total,'strict_copy':strict,'strict_hits':sc[:4],'test_is_any':bool(ti),'test_is_strict':bool(si),'test_is_lines':si[:4] if si else ti[:4],'f2p_name_hits':len(f2pn),'snapshot_collision':snap_hit((r.get('repo')or'').strip(),r.get('base_commit')or'',fs)}
                v2_nontrivial.append(entry)
                if total<=200: v2_le200.append(entry)
        def rank_key(d):
            # strict+strict_is first, then strict+any_is, then strict, then strict_is, then any_is, then rest; smaller total first
            tier=5
            if d['strict_copy'] and d['test_is_strict']: tier=0
            elif d['strict_copy'] and d['test_is_any']: tier=1
            elif d['strict_copy']: tier=2
            elif d['test_is_strict']: tier=3
            elif d['test_is_any']: tier=4
            return (tier,d['total'],d['F2P'])
        gym_nontrivial.sort(key=rank_key); v2_nontrivial.sort(key=rank_key)
        gym_le200.sort(key=rank_key); v2_le200.sort(key=rank_key)
        save(OUT/'gym-recall-large-scan.json',{'date':'2026-10-25','source':'SWE-Gym/SWE-Gym','revision':GYM_REV,'split':'train','rows_scanned':len(rows),'parquet_sha256':hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),'mode':'recall-nontrivial-total_gt_150','count':len(gym_nontrivial),'le200_count':len(gym_le200),'hits':gym_nontrivial,'le200':gym_le200})
        save(OUT/'v2-recall-large-scan.json',{'date':'2026-10-25','source':'nebius/SWE-rebench-V2','revision':V2_REV,'split':'train','language':'python','python_rows_scanned':n_py,'parquet_sha256':hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),'mode':'recall-nontrivial-total_gt_150','count':len(v2_nontrivial),'le200_count':len(v2_le200),'hits':v2_nontrivial,'le200':v2_le200})
        print(f"gym nontrivial={len(gym_nontrivial)} le200={len(gym_le200)}")
        for d in gym_le200[:60]:
            print(f"  t{d['total']:3d} F{d['F2P']:2d} strict={int(d['strict_copy'])} is_strict={int(d['test_is_strict'])} is_any={int(d['test_is_any'])} f2pn={d['f2p_name_hits']} {d['instance_id']}")
        print(f"v2 nontrivial={len(v2_nontrivial)} le200={len(v2_le200)}")
        for d in v2_le200[:80]:
            print(f"  t{d['total']:3d} F{d['F2P']:2d} strict={int(d['strict_copy'])} is_strict={int(d['test_is_strict'])} is_any={int(d['test_is_any'])} f2pn={d['f2p_name_hits']} {d['instance_id']} {d['repo']}")
        # tier breakdown
        from collections import Counter
        print('gym tiers',Counter([(int(d['strict_copy']),int(d['test_is_strict']),int(d['test_is_any'])) for d in gym_nontrivial]))
        print('v2 tiers',Counter([(int(d['strict_copy']),int(d['test_is_strict']),int(d['test_is_any'])) for d in v2_nontrivial]))
        print('gym le200 tiers',Counter([(int(d['strict_copy']),int(d['test_is_strict'])) for d in gym_le200]))
        print('v2 le200 tiers',Counter([(int(d['strict_copy']),int(d['test_is_strict'])) for d in v2_le200]))
