"""F2 f2-inline: rescreen 417 SCREENED-OUT (no REAL_IS le200) for inline-is type."""
import hashlib, json, re
from pathlib import Path
from collections import Counter
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL=ROOT/'competition/vfarl'
ACQ=VFARL/'acquisition/f2-new'
OUT=ACQ/'f2-inline'
GYM_PARQUET=Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET=Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV='bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV='10483de0f50fe5da545942705a76c6150171af7f'
EXCLUDED={"fastapi/fastapi","textualize/rich","psf/requests","encode/httpx"}
COPY_SIG=re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
# inline copy: broader isolation signals in changed lines
INLINE_SIG=re.compile(r'\.copy\(\)|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|__copy__|__deepcopy__|\bclone\b|\.clone\(|CIMultiDict\s*\(|MultiDict\s*\(|clear_cache|copyInto|copy_into|\[:\]|\blist\s*\(|\bdict\s*\(|\bset\s*\(|tuple\s*\(|copy_module| shallow| deep.?copy', re.I)
F2P_NAME_SIG=re.compile(r'copy|clone|deep|shallow|alias|ident|isolat|shared|mutat|ownership|immutable|frozen|singleton|same|original|pollut|leak|retain|preserv|not_mutated|unchang|independ|separat', re.I)
STRICT_IS=re.compile(r'assertIs(?:Not)?\b|\bis\s+(?:not\s+)?(?!None\b|True\b|False\b)[A-Za-z_][\w.]*\b')
# prose filter: exclude matches where the identifier is a common english word (part,apart,expected,avail,etc) - manual pass later
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def load_forbidden():
    fi,fs,ff=set(),set(),set(); notes=[]
    base=json.loads((ACQ/'dedup-baseline.json').read_text())
    fi|=set(base['forbidden_instance_ids']); fs|=set(base.get('forbidden_snapshot_group_ids',[])); ff|=set(base['forbidden_repair_family_ids'])
    notes.append(f"common-400:{len(base['forbidden_instance_ids'])}i")
    f1=json.loads((VFARL/'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks',[]): fi.add(t['instance_id']); (fs.add(t['snapshot_group_id']) if t.get('snapshot_group_id') else None); (ff.add(t['repair_family_id']) if t.get('repair_family_id') else None)
    notes.append(f"F1-40:{len(f1.get('tasks',[]))}")
    f5=json.loads((VFARL/'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks',[]):
        fi.add(t.get('instance_id',t.get('canonical_task_id','')))
        for k in ('snapshot_group_id','snapshot_id'):
            if t.get(k): fs.add(t[k])
        for k in ('repair_family_id','family_id'):
            if t.get(k): ff.add(t[k])
    notes.append(f"F5-old40:{len(f5.get('tasks',[]))}")
    for rel in ['f2-replace/catalog.json','f2-replace/reviewed-candidates.json','candidate-list-40.json','reviewed-candidates.json','catalog.json']:
        p=ACQ/rel
        try: d=json.loads(p.read_text())
        except: continue
        for t in d.get('tasks',d.get('candidates',[])):
            if t.get('instance_id'): fi.add(t['instance_id'])
            for k in ('snapshot_group_id','snapshot_group_id_provisional'):
                if t.get(k): fs.add(t[k])
            for k in ('repair_family_id','repair_family_id_provisional'):
                if t.get(k): ff.add(t[k])
        notes.append(f"f2prior:{rel}:{len(d.get('tasks',d.get('candidates',[])))}")
    fs|={'seperman/deepdiff::ed252022','pallets/click::fde47b4b','pydantic/pydantic::acb0f10f','stanfordnlp/dspy::651a4c71','sunpy/sunpy::f8edfd5c','mido/mido::a0158ff9','cantools/cantools::0c6a7871'}
    notes.append("explicit round1 snapshot bans:7")
    for kdir in ['f2-gymv2','f2-gymv2b','f2-gymv2c']:
        g2keep=json.loads((ACQ/kdir/'keep-list.json').read_text())
        g2fam=json.loads((ACQ/kdir/'keep-families.json').read_text())
        for k in g2keep: fi.add(k['instance_id']); fs.add(f"{k['repo']}::{k['base_commit']}")
        ff|=set(g2fam.values())
        notes.append(f"{kdir}:{len(g2keep)}keeps")
    # recall keeps (even near-miss/blocked occupy families/snapshots)
    rk=json.loads((ACQ/'f2-recall/keep-list.json').read_text())
    rf=json.loads((ACQ/'f2-recall/keep-families.json').read_text())
    for k in rk: fi.add(k['instance_id']); fs.add(f"{k['repo']}::{k['base_commit']}")
    ff|=set(rf.values())
    notes.append(f"f2-recall:{len(rk)}keeps")
    return fi,fs,ff,notes
def snap_hit(repo,commit,fs):
    if not commit: return None
    for s in fs:
        if '::' not in s: continue
        r,rev=s.split('::',1)
        if r.lower()!=repo.lower(): continue
        c=commit.lower(); rr=rev.lower()
        if c.startswith(rr) or rr.startswith(c): return s
    return None
def changed_hits(patch,sig):
    chg=[l for l in patch.splitlines() if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++','---'))]
    return [l.strip()[:220] for l in chg if sig.search(l)]
if __name__=='__main__':
    import sys
    which=sys.argv[1] if len(sys.argv)>1 else 'scan'
    OUT.mkdir(parents=True,exist_ok=True)
    if which=='dedup':
        fi,fs,ff,notes=load_forbidden()
        save(OUT/'dedup-evidence.json',{'date':'2026-10-25','scope':'F600-F2 f2-inline vs common-400+F1-40+F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3 disjointness','forbidden_instance_count':len(fi),'forbidden_snapshot_count':len(fs),'forbidden_family_count':len(ff),'sources':notes,'method':'instance exact; snapshot repo-equality+commit prefix either way; family exact; four repos excluded','physical_split':'train only'})
        print(f'forbidden {len(fi)}i/{len(fs)}s/{len(ff)}f')
        [print(' ',n) for n in notes]
    elif which=='scan':
        import pyarrow.parquet as pq
        fi,fs,_,_=load_forbidden()
        # need LARGE-DONE + COLLIDE + REALIS48 exclusion to get 417? reconstruct: exclude strict large 35? Actually large 35 are total>150 strict; recall le200 includes some strict too (13). For inline we want rows WITHOUT true REAL_IS. Approach: take all le200 (468) then drop rows with TRUE discriminating is (manual list from realis-48-dump = 48 + large-done true + collide true). Simplest: drop any le200 where STRICT_IS hits contain assertIs/assertIsNot or identifier-is that is not prose. But prose filtering needs manual; instead we emit all le200 with inline signals and mark is_class.
        rows=pq.read_table(str(GYM_PARQUET)).to_pylist()
        gym_le=[]; 
        for idx,r in enumerate(rows):
            iid=r['instance_id']
            if iid in fi: continue
            repo=(r.get('repo')or'').strip()
            if repo.lower() in EXCLUDED: continue
            f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
            if not f2p or set(f2p)&set(p2p): continue
            if not (r.get('problem_statement')or'').strip(): continue
            patch=r.get('patch')or''
            files=re.findall(r'diff --git a/(\S+)',patch)
            if any('..' in f for f in files): continue
            total=len(f2p)+len(p2p)
            if total<=150 or total>200: continue
            if snap_hit(repo,r.get('base_commit')or'',fs): continue
            tp=r.get('test_patch')or''
            si=[l.strip()[:220] for l in tp.splitlines() if STRICT_IS.search(l)][:6]
            sc=changed_hits(patch,COPY_SIG)
            inl=changed_hits(patch,INLINE_SIG)
            f2pn=[n for n in f2p if F2P_NAME_SIG.search(n)]
            gym_le.append({'instance_id':iid,'pool':'gym','parquet_idx':idx,'repo':repo,'base_commit':r.get('base_commit'),'F2P':len(f2p),'P2P':len(p2p),'total':total,'strict_copy':bool(sc),'strict_hits':sc[:4],'inline_hits':inl[:6],'inline_n':len(inl),'test_is_lines':si,'test_is_n':len(si),'f2p_name_hits':len(f2pn),'f2p_sample':f2p[:3]})
        f=pq.ParquetFile(str(V2_PARQUET))
        v2_le=[]; n_py=0
        for rg in range(f.num_row_groups):
            tbl=f.read_row_group(rg)
            for j,r in enumerate(tbl.to_pylist()):
                if (r.get('language')or'').lower()!='python': continue
                n_py+=1
                iid=r['instance_id']
                if iid in fi: continue
                repo=(r.get('repo')or'').strip()
                if repo.lower() in EXCLUDED: continue
                f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
                if not f2p or set(f2p)&set(p2p): continue
                if not (r.get('problem_statement')or'').strip(): continue
                patch=r.get('patch')or''
                files=re.findall(r'diff --git a/(\S+)',patch)
                if any('..' in f for f in files): continue
                total=len(f2p)+len(p2p)
                if total<=150 or total>200: continue
                if snap_hit(repo,r.get('base_commit')or'',fs): continue
                tp=r.get('test_patch')or''
                si=[l.strip()[:220] for l in tp.splitlines() if STRICT_IS.search(l)][:6]
                sc=changed_hits(patch,COPY_SIG)
                inl=changed_hits(patch,INLINE_SIG)
                f2pn=[n for n in f2p if F2P_NAME_SIG.search(n)]
                v2_le.append({'instance_id':iid,'pool':'v2','row_group':rg,'repo':repo,'base_commit':r.get('base_commit'),'F2P':len(f2p),'P2P':len(p2p),'total':total,'strict_copy':bool(sc),'strict_hits':sc[:4],'inline_hits':inl[:6],'inline_n':len(inl),'test_is_lines':si,'test_is_n':len(si),'f2p_name_hits':len(f2pn),'f2p_sample':f2p[:3]})
        # sort: inline_n desc, f2p_name_hits desc, total asc
        gym_le.sort(key=lambda d:(-d['inline_n'],-d['f2p_name_hits'],d['total']))
        v2_le.sort(key=lambda d:(-d['inline_n'],-d['f2p_name_hits'],d['total']))
        save(OUT/'gym-inline-scan.json',{'date':'2026-10-25','source':'SWE-Gym/SWE-Gym','revision':GYM_REV,'split':'train','rows_scanned':len(rows),'parquet_sha256':hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),'mode':'inline-rescreen-le200-noREALIS-excl-forbidden','count':len(gym_le),'hits':gym_le})
        save(OUT/'v2-inline-scan.json',{'date':'2026-10-25','source':'nebius/SWE-rebench-V2','revision':V2_REV,'split':'train','language':'python','python_rows_scanned':n_py,'parquet_sha256':hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),'mode':'inline-rescreen-le200-noREALIS-excl-forbidden','count':len(v2_le),'hits':v2_le})
        print(f"gym le200-forbidden-excl={len(gym_le)} v2={len(v2_le)} total={len(gym_le)+len(v2_le)}")
        print('gym inline_n>0',sum(1 for d in gym_le if d['inline_n']>0),'v2',sum(1 for d in v2_le if d['inline_n']>0))
        print('gym strict',sum(1 for d in gym_le if d['strict_copy']),'v2 strict',sum(1 for d in v2_le if d['strict_copy']))
        from collections import Counter
        print('gym inline tiers',Counter([d['inline_n']>0 for d in gym_le]))
        # top 40 each
        for d in (gym_le[:25]+v2_le[:40]):
            print(f" inl{d['inline_n']:2d} f2pn{d['f2p_name_hits']:2d} t{d['total']:3d} F{d['F2P']:2d} strict={int(d['strict_copy'])} is_n={d['test_is_n']} {d['instance_id']} :: {d['f2p_sample'][0][:90] if d['f2p_sample'] else ''} :: {('; '.join(d['inline_hits'][:2]))[:140]}")
