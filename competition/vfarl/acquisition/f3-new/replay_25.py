"""Replay F3-new 25 deepdiff."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/f3-new'
prep=json.load(open(ROOT/'competition/vfarl/acquisition/f3-new/f3-new-25-preparation.json'))
tasks=prep['tasks']
print(f"replaying {len(tasks)}", flush=True)
for t in tasks:
    key=t['key']
    adapted=ROOT/t['adapted_task']
    ws=ROOT/t['workspace']
    recipe=ROOT/t['recipe']
    out=ART/(key+'-run-001')
    log=ART/(key+'-run-001.log')
    if (out/'run-summary.json').exists():
        print(f"SKIP {key} already done", flush=True)
        continue
    cmd=[sys.executable, str(REPLAY/'run_official.py'),'--task',str(adapted),'--workspace',str(ws),'--recipe',str(recipe),'--output',str(out),'--verify','--verify-baseline','--source-layout','.', '--task-site-packages',str(REPLAY/'.venv-sfr/lib/python3.12/site-packages'),'--task-site-packages-mode','copy']
    print(f"RUN {key} -> {out}", flush=True)
    with open(log,'w') as s:
        r=subprocess.run(cmd, stdout=s, stderr=subprocess.STDOUT)
    print(f"DONE {key} rc={r.returncode} log={log}", flush=True)
    if r.returncode!=0:
        print(f"FAIL {key} see {log}", flush=True)
print("all done", flush=True)
