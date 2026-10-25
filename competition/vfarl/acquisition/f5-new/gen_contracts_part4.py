"""Draft F5 contracts part4: 7 replacements (tenacity x5, ptyprocess waitnoecho, dataset flush-order)."""
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

add("tenacity-stop-any-call", "jd__tenacity.0d40e76f.func_basic__9dsev46m",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_any.__call__",
    "stop_any_fires_when_any_stop_fires",
    "stop_any must fire when any sub-stop fires so the earliest deadline ends retries; requiring all stops delays the attempt-stop-end chain to the latest deadline.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_any, stop_after_attempt
combined=stop_any(stop_after_attempt(2), stop_after_attempt(5))
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3,5,6):
    state.attempt_number=attempt
    decisions.append(combined(state))
print('stop_any_call_sequence',decisions)
assert decisions==[False,True,True,True,True]""", SFR)

add("tenacity-jitter-init", "jd__tenacity.0d40e76f.func_basic__nmd66t7l",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_exponential_jitter.__init__",
    "jitter_init_keeps_initial_max_base_roles",
    "Jittered exponential wait must keep initial/max/exp_base/jitter roles so sleeps grow exponentially within bounds; rotating the four arguments collapses every sleep to the cap across the attempt-sleep-end chain.",
"""import random
from tenacity import RetryCallState, Retrying
from tenacity.wait import wait_exponential_jitter
random.seed(1)
strat=wait_exponential_jitter()
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    values.append(strat(state))
print('jitter_init_sequence',values)
assert values==[1.134364244112401, 2.8474337369372327, 4.763774618976614]""", SFR)

add("pty-waitnoecho", "pexpect__ptyprocess.1067dbda.func_pm_remove_loop__exlxfztz",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.waitnoecho",
    "waitnoecho_polls_echo_until_timeout",
    "Waiting for no-echo must poll the echo flag until it clears or the timeout expires; deleting the poll loop returns None immediately, breaking the spawn-poll-detect chain.",
"""from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['sleep','5'])
try:
    proc.setecho(False)
    first=proc.waitnoecho(timeout=5)
    proc.setecho(True)
    second=proc.waitnoecho(timeout=1)
    proc.setecho(False)
    third=proc.waitnoecho(timeout=5)
    print('waitnoecho_sequence',[first,second,third])
    assert [first,second,third]==[True,False,True]
finally:
    try: proc.terminate(force=True)
    except Exception: pass
    try: proc.wait()
    except Exception: pass
    try: proc.close()
    except Exception: pass""", PTY)

add("chunked-flush-order", "pudo__dataset.5c2dc8d3.func_pm_ctrl_shuffle__gfu7q6ma",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "ChunkedUpdate.flush",
    "update_flush_calls_back_sorts_groups_then_clears",
    "Update flush must invoke the callback on queued rows, sort and group by keys into update_many calls, then clear the queue; clearing first persists nothing and reports empty across the queue-flush-persist chain.",
"""from dataset.chunked import ChunkedUpdate
seen=[]; calls=[]
class T:
    def update_many(self, rows, keys): calls.append((list(rows),keys))
t=T()
up=ChunkedUpdate(t, keys=['id'], chunksize=100, callback=lambda q: seen.append(len(q)))
with up as u:
    enter_ok=(u is up)
    u.update({'id':1,'v':'a'})
    u.update({'id':2,'v':'b'})
print('flush_order_sequence',[enter_ok,seen,len(calls),len(up.queue)])
assert [enter_ok,seen,len(calls),len(up.queue)]==[True,[2],1,0]
assert calls[0]==([{'id':1,'v':'a'},{'id':2,'v':'b'}],['id'])""", DB)

add("tenacity-add-action", "jd__tenacity.0d40e76f.func_basic__q2o1oigs",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "BaseRetrying._add_action_func",
    "add_action_appends_every_action_once",
    "Registering iteration actions must append each action exactly once in order; duplicating the first and dropping the rest corrupts the begin-retry-post execution chain.",
"""from tenacity import Retrying, RetryCallState, stop_after_attempt
ctl=Retrying(stop=stop_after_attempt(3))
calls=[]
def f1(rs): calls.append('f1')
def f2(rs): calls.append('f2')
ctl._add_action_func(f1)
n1=len(ctl.iter_state.actions)
ctl._add_action_func(f2)
n2=len(ctl.iter_state.actions)
state=RetryCallState(ctl,None,(),{})
for a in list(ctl.iter_state.actions):
    a(state)
print('add_action_sequence',[n1,n2,calls])
assert [n1,n2,calls]==[1,2,['f1','f2']]""", SFR)

add("tenacity-stop-delay", "jd__tenacity.0d40e76f.func_basic__khos7c1u",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_after_delay.__init__",
    "stop_delay_keeps_exact_max_seconds",
    "stop_after_delay must keep the exact max seconds so the deadline fires on time; taking abs plus a tenth delays every stop decision across the attempt-time-stop chain.",
"""import time
from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_after_delay
stop=stop_after_delay(10)
assert stop.max_delay==10
ctl=Retrying()
decisions=[]
for dt in (9.9,10.0,10.1):
    state=RetryCallState(ctl,None,(),{})
    state.start_time=time.monotonic()-dt
    state.set_result('v')
    decisions.append(stop(state))
print('stop_delay_sequence',decisions)
assert decisions==[False,True,True]""", SFR)

add("tenacity-async-add-order", "jd__tenacity.0d40e76f.func_basic__101pcgqa",
    "jd/tenacity", "0d40e76f", "tenacity/asyncio/__init__.py", ".", "tenacity.asyncio", "AsyncRetrying._add_action_func",
    "async_add_action_appends_in_registration_order",
    "Async action registration must append in order so begin runs before retry checks; prepending reverses the begin-retry-post execution chain.",
"""import asyncio
from tenacity import RetryCallState, Retrying, stop_after_attempt
from tenacity.asyncio import AsyncRetrying
ctl=AsyncRetrying(stop=stop_after_attempt(3))
order=[]
def f1(rs): order.append('first')
def f2(rs): order.append('second')
ctl._add_action_func(f1)
ctl._add_action_func(f2)
state=RetryCallState(ctl,None,(),{})
async def drive():
    for a in list(ctl.iter_state.actions):
        await a(state)
asyncio.run(drive())
print('async_add_order_sequence',order)
assert order==['first','second']""", SFR)

(HERE / "contracts-part4.json").write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote part4 {len(C)}")
