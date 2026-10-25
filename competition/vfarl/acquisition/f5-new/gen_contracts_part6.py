"""Draft F5 contracts part6: 11 backfills (tenacity x5, ptyprocess x2, dataset x4)."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
C = []

def add(key, iid, canon, snap, rel, layout, module, symbol, family, contract, body, profile):
    assert "assert" in body and ("==" in body), key
    C.append({"canonical_repo": canon, "snapshot": snap, "relative_path": rel,
              "source_layout": layout, "task_profile": profile,
              "source_license": "UPSTREAM",
              "key": key, "instance_id": iid, "symbol": symbol,
              "primary_adapter": "F", "submechanism": "F5",
              "family_contract": family, "contract": contract, "body": body})

SFR = ".venv-sfr/lib/python3.12/site-packages"
PTY = ".venv-expand-f4-task/lib/python3.12/site-packages"
DB = ".venv-expand-f4-db-task/lib/python3.12/site-packages"

add("tenacity-stop-call", "jd__tenacity.0d40e76f.func_basic__uyzg7360",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_after_attempt.__call__",
    "stop_call_fires_on_reaching_max_attempt",
    "stop_after_attempt must fire once the attempt number reaches the max so the final attempt ends the cycle; requiring strictly more runs every schedule one attempt long across the attempt-stop-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_after_attempt
stop=stop_after_attempt(3)
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3,4):
    state.attempt_number=attempt
    decisions.append(stop(state))
print('stop_call_sequence',decisions)
assert decisions==[False,False,True,True]""", SFR)

add("tenacity-retry-never", "jd__tenacity.0d40e76f.func_basic__p6nagieg",
    "jd/tenacity", "0d40e76f", "tenacity/retry.py", ".", "tenacity.retry", "_retry_never.__call__",
    "retry_never_rejects_every_outcome",
    "The never-retry gate must reject every outcome so nothing retries; gating on a positive attempt number retries every cycle across the outcome-decision chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.retry import retry_never
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    decisions.append(retry_never(state))
print('retry_never_sequence',decisions)
assert decisions==[False,False,False]""", SFR)

add("tenacity-jitter-call", "jd__tenacity.0d40e76f.func_basic__rmesyjqx",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_exponential_jitter.__call__",
    "jitter_call_adds_and_clamps_within_max",
    "Jittered waits must add the jitter to the exponential and clamp into range; subtracting and clamping outside collapses every sleep to zero across the attempt-sleep-end chain.",
"""import random
from tenacity import RetryCallState, Retrying
from tenacity.wait import wait_exponential_jitter
random.seed(7)
strat=wait_exponential_jitter()
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    values.append(strat(state))
print('jitter_call_sequence',values)
assert values==[1.3238327648331625, 2.150849173924502, 4.650934473039854]""", SFR)

add("tenacity-future-construct", "jd__tenacity.0d40e76f.func_basic__6xjkrkm1",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "Future.construct",
    "future_construct_preserves_result_or_exception_branches",
    "Outcome construction must wrap results as results and exceptions as exceptions so downstream predicates inspect the right branch; swapping the branches and returning None breaks the produce-inspect-decide chain.",
"""from tenacity import Future
ok=Future.construct(1, 'v', False)
err=Future.construct(1, ValueError('x'), True)
events=[ok.result(), ok.failed, err.failed, type(err.exception()).__name__]
print('future_construct_sequence',events)
assert events==['v', False, True, 'ValueError']""", SFR)

add("tenacity-stop-all-call", "jd__tenacity.0d40e76f.func_basic__tu1b0qsw",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_all.__call__",
    "stop_all_requires_every_stop_across_attempts",
    "stop_all must require every sub-stop so the latest deadline governs; using any ends at the earliest deadline across the attempt-stop-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_all, stop_after_attempt
combined=stop_all(stop_after_attempt(2), stop_after_attempt(5))
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3,5,6):
    state.attempt_number=attempt
    decisions.append(combined(state))
print('stop_all_call_sequence',decisions)
assert decisions==[False,False,False,True,True]""", SFR)

add("pty-waitnoecho-invert", "pexpect__ptyprocess.1067dbda.func_basic__vykdimae",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.waitnoecho",
    "waitnoecho_returns_true_when_cleared_false_on_timeout",
    "waitnoecho must report true once echo clears and false on timeout expiry; inverting both outcomes breaks the spawn-poll-detect chain for password prompts.",
"""from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['sleep','5'])
try:
    proc.setecho(False)
    first=proc.waitnoecho(timeout=5)
    proc.setecho(True)
    second=proc.waitnoecho(timeout=1)
    proc.setecho(False)
    third=proc.waitnoecho(timeout=5)
    print('waitnoecho_invert_sequence',[first,second,third])
    assert [first,second,third]==[True,False,True]
finally:
    try: proc.terminate(force=True)
    except Exception: pass
    try: proc.wait()
    except Exception: pass
    try: proc.close()
    except Exception: pass""", PTY)

add("pty-getecho-roundtrip", "pexpect__ptyprocess.1067dbda.func_basic__7xo9xc1a",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.getecho",
    "getecho_reports_actual_echo_flag",
    "Reading the echo flag must report the actual terminal state after each set call; returning the inverted flag from the wrong bit breaks the set-poll-detect chain.",
"""from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['sleep','5'])
try:
    proc.setecho(True)
    on_state=proc.getecho()
    proc.setecho(False)
    off_state=proc.getecho()
    waited=proc.waitnoecho(timeout=5)
    print('getecho_roundtrip_sequence',[on_state,off_state,waited])
    assert [on_state,off_state,waited]==[True,False,True]
finally:
    try: proc.terminate(force=True)
    except Exception: pass
    try: proc.wait()
    except Exception: pass
    try: proc.close()
    except Exception: pass""", PTY)

add("table-bulk-update", "pudo__dataset.5c2dc8d3.func_pm_remove_assign__wisytcfw",
    "pudo/dataset", "5c2dc8d3", "dataset/table.py", ".", "dataset.table", "Table.update_many",
    "bulk_update_executes_statement_per_chunk",
    "Bulk update must build and execute the update statement for every full chunk and the tail; deleting the statement leaves rows unpersisted across the produce-execute-read chain.",
"""from dataset import Database
db=Database('sqlite:///:memory:')
t=db['t']
t.insert({'id':1,'v':'a'})
t.insert({'id':2,'v':'b'})
t.insert({'id':3,'v':'c'})
t.update_many([{'id':1,'v':'a1'},{'id':2,'v':'b1'},{'id':3,'v':'c1'}], ['id'])
rows=sorted([(r['id'],r['v']) for r in t.all()])
print('bulk_update_sequence',rows)
assert rows==[(1,'a1'),(2,'b1'),(3,'c1')]""", DB)

add("table-bulk-update-loop", "pudo__dataset.5c2dc8d3.func_pm_remove_loop__iagokng5",
    "pudo/dataset", "5c2dc8d3", "dataset/table.py", ".", "dataset.table", "Table.update_many",
    "bulk_update_iterates_every_row",
    "Bulk update must iterate every supplied row into chunks for execution; deleting the loop persists nothing across the produce-execute-read chain.",
"""from dataset import Database
db=Database('sqlite:///:memory:')
t=db['t']
t.insert({'id':1,'v':'a'})
t.insert({'id':2,'v':'b'})
t.update_many([{'id':1,'v':'a1'},{'id':2,'v':'b1'}], ['id'])
rows=sorted([(r['id'],r['v']) for r in t.all()])
print('bulk_update_loop_sequence',rows)
assert rows==[(1,'a1'),(2,'b1')]""", DB)

add("table-insert-many", "pudo__dataset.5c2dc8d3.lm_rewrite__jazezf2x",
    "pudo/dataset", "5c2dc8d3", "dataset/table.py", ".", "dataset.table", "Table.insert_many",
    "insert_many_syncs_columns_then_persists_in_order",
    "Bulk insert must sync columns from all rows then persist every row in order; syncing only the first row and reordering drops columns and rows across the produce-persist-read chain.",
"""from dataset import Database
db=Database('sqlite:///:memory:')
t=db['t']
t.insert_many([{'id':1,'v':'a'},{'id':2,'v':'b','w':'c'},{'id':3,'v':'d'}])
rows=sorted([(r['id'],r['v']) for r in t.all()])
print('insert_many_sequence',rows)
assert rows==[(1,'a'),(2,'b'),(3,'d')]""", DB)

add("table-insert-ignore", "pudo__dataset.5c2dc8d3.func_pm_ctrl_shuffle__ap17dl1l",
    "pudo/dataset", "5c2dc8d3", "dataset/table.py", ".", "dataset.table", "Table.insert_ignore",
    "insert_ignore_inserts_only_missing_rows",
    "Conditional insert must persist a missing row and skip an existing one by key; always returning false without inserting loses the check-persist-read chain.",
"""from dataset import Database
db=Database('sqlite:///:memory:')
t=db['t']
t.insert({'id':1,'v':'a'})
dup=t.insert_ignore({'id':1,'v':'b'}, ['id'])
count_dup=t.count()
val=[r['v'] for r in t.all()][0]
fresh=t.insert_ignore({'id':2,'v':'c'}, ['id'])
count_fresh=t.count()
print('insert_ignore_sequence',[dup,count_dup,bool(fresh),count_fresh,val])
assert [dup,count_dup,bool(fresh),count_fresh,val]==[False,1,True,2,'a']""", DB)

(HERE / "contracts-part6.json").write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote part6 {len(C)}")
