import sys, re
import pyarrow.parquet as pq
from pathlib import Path
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
STRICT_IS = re.compile(r'assertIs(?:Not)?\b|\bis\s+(?:not\s+)?(?!None\b|True\b|False\b)[A-Za-z_][\w.]*\b')
def show(pool,iid):
    if pool=='gym':
        rows=pq.read_table(str(GYM_PARQUET)).to_pylist()
        hit=[r for r in rows if r['instance_id']==iid]
    else:
        f=pq.ParquetFile(str(V2_PARQUET))
        hit=[]
        for rg in range(f.num_row_groups):
            for r in f.read_row_group(rg).to_pylist():
                if r['instance_id']==iid:
                    hit.append(r); break
            if hit: break
    assert len(hit)==1,(pool,iid)
    r=hit[0]
    print('='*100)
    print(f"{r['instance_id']} repo={r.get('repo')} base={r.get('base_commit')}")
    f2p=r.get('FAIL_TO_PASS')or[]; p2p=r.get('PASS_TO_PASS')or[]
    print(f"F2P({len(f2p)}):")
    for n in f2p[:25]: print('   F',n)
    print(f"P2P({len(p2p)}):")
    for n in p2p[:6]: print('   P',n)
    print('--- problem_statement (first 2200) ---')
    print((r.get('problem_statement')or'')[:2200])
    print('--- patch (first 5500) ---')
    print((r.get('patch')or'')[:5500])
    print('--- test_patch is-lines (STRICT_IS, up to 30) ---')
    cnt=0
    for line in (r.get('test_patch')or'').splitlines():
        if STRICT_IS.search(line):
            print('   T:',line.strip()[:220]); cnt+=1
            if cnt>=30: break
    if cnt==0: print('   (no STRICT_IS lines; ANY_IS fallback)')
    print('--- test_patch (first 3500) ---')
    print((r.get('test_patch')or'')[:3500])
if __name__=='__main__':
    show(sys.argv[1],sys.argv[2])
