"""Draft F5 contracts part2: ptyprocess, dataset, trio (10). Async-timeout removed (mutations live in <3.11-only dead code on the 3.12 replay runtime)."""
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

PTY = ".venv-expand-f4-task/lib/python3.12/site-packages"
DB = ".venv-expand-f4-db-task/lib/python3.12/site-packages"
TRIO = ".venv-sfr/lib/python3.12/site-packages"

# ---------- ptyprocess 4 ----------
add("pty-close-order", "pexpect__ptyprocess.1067dbda.func_basic__6olomd3a",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.close",
    "close_flushes_then_reaps_then_marks_closed",
    "Closing must flush, close the fd, wait for kernel status, terminate only if alive, then mark fd -1 and closed. Moving flush outside and gating fd assignment on aliveness raises on clean exits instead of closing.",
"""from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['true'])
proc.wait()
alive_after_wait=proc.isalive()
proc.close()
events=[alive_after_wait, proc.closed, proc.fd, proc.exitstatus]
print('close_sequence',events)
assert events==[False,True,-1,0]""", PTY)

add("pty-wait-alive", "pexpect__ptyprocess.1067dbda.func_pm_ctrl_invert_if__ujgscofd",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.wait",
    "wait_blocks_when_alive_returns_cached_when_dead",
    "Waiting on a live child must block in waitpid and return its status; waiting on a reaped child must return the cached status. Inverting the branch returns None immediately and blocks forever on dead children.",
"""from ptyprocess import PtyProcess
import time
proc=PtyProcess.spawn(['sleep','0.2'])
start=time.time()
first=proc.wait()
elapsed=time.time()-start
second=proc.wait()
print('wait_sequence',[first,second,round(elapsed,1)])
assert first==0 and second==0 and elapsed>=0.15""", PTY)

add("pty-wait-status", "pexpect__ptyprocess.1067dbda.func_pm_ctrl_shuffle__80usvwd2",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.wait",
    "wait_reaps_before_reading_status",
    "Wait must check aliveness and reap via waitpid before interpreting the status word. Reading an unassigned status breaks the spawn-wait-reap chain with an unbound variable.",
"""from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['true'])
result=proc.wait()
print('wait_reap_sequence',[result,proc.exitstatus,proc.terminated])
assert [result,proc.exitstatus,proc.terminated]==[0,0,True]""", PTY)

add("pty-isalive-signal", "pexpect__ptyprocess.1067dbda.func_pm_remove_assign__ivbnyzsb",
    "pexpect/ptyprocess", "1067dbda", "ptyprocess/ptyprocess.py", ".", "ptyprocess.ptyprocess", "PtyProcess.isalive",
    "isalive_records_signal_termination",
    "A child killed by a signal must report not-alive with exitstatus None, the signal number, and terminated true. Dropping those assignments leaves stale exit state across the spawn-signal-reap chain.",
"""import time, signal
from ptyprocess import PtyProcess
proc=PtyProcess.spawn(['sh','-c','kill -TERM $$'])
time.sleep(0.5)
alive=proc.isalive()
events=[alive,proc.exitstatus,proc.signalstatus,proc.terminated]
print('signal_reap_sequence',events)
assert events==[False,None,signal.SIGTERM,True]""", PTY)

# ---------- dataset 4 ----------
add("chunker-enter", "pudo__dataset.5c2dc8d3.func_basic__gfn1ybos",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "_Chunker.__enter__",
    "chunker_enter_returns_owner_for_use",
    "Entering a chunked writer must return the writer so queued rows can be added before the exit flush. Returning None breaks the enter-produce-flush chain.",
"""from dataset.chunked import ChunkedInsert
class T:
    def __init__(self): self.rows=[]
    def insert_many(self, rows): self.rows.extend([dict(r) for r in rows])
t=T()
ch=ChunkedInsert(t, chunksize=100)
with ch as ins:
    enter_ok=(ins is ch)
    ins.insert({'a':1})
    ins.insert({'a':2})
print('chunker_enter_sequence',[enter_ok,len(t.rows),len(ch.queue)])
assert [enter_ok,len(t.rows),len(ch.queue)]==[True,2,0]""", DB)

add("chunker-exit", "pudo__dataset.5c2dc8d3.func_basic__u6ujevkp",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "_Chunker.__exit__",
    "chunker_exit_flushes_on_clean_exit",
    "Exiting a chunked writer must flush queued rows on both clean and exceptional exits. Skipping the flush on clean exits loses the produce-flush-persist chain.",
"""from dataset.chunked import ChunkedInsert
class T:
    def __init__(self): self.rows=[]
    def insert_many(self, rows): self.rows.extend([dict(r) for r in rows])
t=T()
ch=ChunkedInsert(t, chunksize=100)
with ch as ins:
    enter_ok=(ins is ch)
    ins.insert({'a':1})
    ins.insert({'a':2})
print('chunker_exit_sequence',[enter_ok,len(t.rows),len(ch.queue)])
assert [enter_ok,len(t.rows),len(ch.queue)]==[True,2,0]""", DB)

add("chunked-update", "pudo__dataset.5c2dc8d3.func_basic__9n4wdmp6",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "ChunkedUpdate.update",
    "chunked_update_queues_actual_items",
    "Update must queue the supplied row so the exit flush persists it. Queuing None breaks the produce-queue-flush chain and crashes grouping.",
"""from dataset.chunked import ChunkedUpdate
class T:
    def __init__(self): self.batches=[]
    def update_many(self, rows, keys): self.batches.append((list(rows),keys))
t=T()
up=ChunkedUpdate(t, keys=['id'], chunksize=100)
with up as u:
    enter_ok=(u is up)
    u.update({'id':1,'v':'a'})
    u.update({'id':2,'v':'b'})
print('chunked_update_sequence',[enter_ok,t.batches,len(up.queue)])
assert enter_ok is True
assert t.batches==[([{'id':1,'v':'a'},{'id':2,'v':'b'}],['id'])]
assert up.queue==[]""", DB)

add("chunked-insert", "pudo__dataset.5c2dc8d3.func_basic__2ygbxps8",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "ChunkedInsert.flush",
    "chunked_insert_flush_persists_queued_rows",
    "Flushing must backfill missing fields and insert the queued rows, then clear the queue. Inserting an empty list loses the queue-flush-persist chain.",
"""from dataset.chunked import ChunkedInsert
class T:
    def __init__(self): self.rows=[]
    def insert_many(self, rows): self.rows.extend([dict(r) for r in rows])
t=T()
ch=ChunkedInsert(t, chunksize=100)
with ch as ins:
    enter_ok=(ins is ch)
    ins.insert({'a':1})
    ins.insert({'a':2,'b':3})
    ins.insert({'b':4})
print('chunked_insert_sequence',[enter_ok,t.rows,len(ch.queue)])
assert enter_ok is True
assert t.rows==[{'a':1,'b':None},{'a':2,'b':3},{'a':None,'b':4}]
assert ch.queue==[]""", DB)

# ---------- trio 2 ----------
add("trio-put-fairness", "python-trio__trio.cfbbe2c1.lm_rewrite__75hte8fe",
    "python-trio/trio", "cfbbe2c1", "src/trio/_core/_unbounded_queue.py", "src", "trio._core._unbounded_queue", "UnboundedQueue.put_nowait",
    "put_nowait_wakes_exactly_one_waiter_and_reserves_batches",
    "Putting to an empty queue with a parked getter must wake exactly one waiter without marking readable, reserving the batch for the waiter; setting readable plus waking lets a second consumer steal the batch across the put-wakeup-consume chain.",
"""import trio
from trio._core._unbounded_queue import UnboundedQueue
from trio.testing import wait_all_tasks_blocked
async def main():
    q=UnboundedQueue()
    q.put_nowait(1)
    q.put_nowait(2)
    first=q.get_batch_nowait()
    result=[]
    async def get_batch():
        result.append(await q.get_batch())
    async with trio.open_nursery() as nursery:
        nursery.start_soon(get_batch)
        await wait_all_tasks_blocked()
        q.put_nowait(3)
        q.put_nowait(4)
        try:
            spare=q.get_batch_nowait()
            spare_ok=False
        except trio.WouldBlock:
            spare_ok=True
    print('fair_sequences',[first,spare_ok,result])
    assert [first,spare_ok,result]==[[1,2],True,[[3,4]]]
trio.run(main)""", TRIO)

add("trio-cancel-enter", "python-trio__trio.cfbbe2c1.lm_rewrite__o9uv20ij",
    "python-trio/trio", "cfbbe2c1", "src/trio/_core/_run.py", "src", "trio._core._run", "CancelScope.__enter__",
    "cancel_scope_enter_clears_relative_deadline_and_arms",
    "Entering a cancel scope must clear a preset relative deadline into an armed absolute deadline so expiry cancels at the next checkpoint; keeping the relative value breaks the enter-checkpoint-cancel chain, and single-use scopes must reject reuse.",
"""import math
import trio
async def main():
    events=[]
    with trio.CancelScope() as s:
        events.append('enter_ok')
        await trio.sleep(0)
        events.append('checkpoint_ok')
    sc=trio.CancelScope()
    sc._relative_deadline=5.0
    with sc:
        events.append('rel_cleared' if sc._relative_deadline==math.inf else 'rel_kept')
        await trio.sleep(0)
        events.append('checkpoint2_ok')
    with trio.CancelScope(deadline=trio.current_time()-1):
        try:
            await trio.sleep(0)
            events.append('not_cancelled')
        except trio.Cancelled:
            events.append('cancelled')
    sc2=trio.CancelScope()
    with sc2:
        pass
    try:
        with sc2:
            pass
        events.append('reuse_ok')
    except RuntimeError:
        events.append('reuse_rejected')
    print('cancel_enter_sequence',events)
    assert events==['enter_ok','checkpoint_ok','rel_cleared','checkpoint2_ok','cancelled','reuse_rejected']
trio.run(main)""", TRIO)

(HERE / "contracts-part2.json").write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote part2 {len(C)}")
