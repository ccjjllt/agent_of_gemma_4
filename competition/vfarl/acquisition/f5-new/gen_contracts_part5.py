"""Draft F5 contracts part5: 6 final replacements (tenacity x4, click File.convert, dataset flush-grouping)."""
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
CLICK = ".venv-click-task/lib/python3.12/site-packages"
DB = ".venv-expand-f4-db-task/lib/python3.12/site-packages"

add("tenacity-stop-before-delay", "jd__tenacity.0d40e76f.func_basic__hwyzze0l",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_before_delay.__init__",
    "stop_before_delay_keeps_exact_max_seconds",
    "stop_before_delay must keep the exact max seconds so the pre-sleep deadline check fires on time; shaving one second stops every cycle early across the attempt-time-stop chain.",
"""import time
from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_before_delay
stop=stop_before_delay(10)
assert stop.max_delay==10
ctl=Retrying()
decisions=[]
for dt in (9.9,10.0,10.1):
    state=RetryCallState(ctl,None,(),{})
    state.start_time=time.monotonic()-dt
    state.set_result('v')
    state.upcoming_sleep=0.0
    decisions.append(stop(state))
print('stop_before_delay_sequence',decisions)
assert decisions==[False,True,True]""", SFR)

add("tenacity-retry-not-result", "jd__tenacity.0d40e76f.func_basic__ct1cp1fi",
    "jd/tenacity", "0d40e76f", "tenacity/retry.py", ".", "tenacity.retry", "retry_if_not_result.__init__",
    "retry_if_not_result_keeps_predicate_polarity",
    "retry_if_not_result must keep the predicate polarity so refuting results retry and matching results stop; negating at registration inverts every decision across the outcome-decision chain.",
"""from tenacity import RetryCallState, Retrying, Future
from tenacity.retry import retry_if_not_result
combined=retry_if_not_result(lambda x: x==1)
def outcome_of(value=None, exc=None):
    st=RetryCallState(Retrying(),None,(),{})
    if exc is not None:
        st.outcome=Future.construct(1, exc, True)
    else:
        st.outcome=Future.construct(1, value, False)
    return st
decisions=[combined(outcome_of(value=1)), combined(outcome_of(value=2)), combined(outcome_of(exc=ValueError('x')))]
print('retry_not_result_sequence',decisions)
assert decisions==[False,True,False]""", SFR)

add("click-file-convert", "pallets__click.fde47b4b.func_basic__41e71b4c",
    "pallets/click", "fde47b4b", "src/click/types.py", "src", "click.types", "File.convert",
    "file_convert_keeps_lazy_and_context_branches",
    "File conversion must take the lazy path for lazy flags and the eager path otherwise, auto-closing only with a context; inverting both branches opens eagerly, wraps lazily, and leaks or double-closes across the open-convert-close chain.",
"""import tempfile, os
from click import Context, Command
from click.types import File
from click.utils import LazyFile
fd, path=tempfile.mkstemp(); os.close(fd)
try:
    with open(path,'w') as f: f.write('hello')
    ctx1=Context(Command('x'))
    r1=File('w').convert(path, None, ctx1)
    lazy_kind=isinstance(r1, LazyFile)
    ctx1.close()
    ctx2=Context(Command('y'))
    f2=File('r').convert(path, None, ctx2)
    eager_kind=(not isinstance(f2, LazyFile))
    content=f2.read()
    ctx2.close()
    eager_closed=f2.closed
    f3=File('r').convert(path, None, None)
    direct=f3.read()
    f3.close()
    print('file_convert_sequence',[lazy_kind,eager_closed,content,direct])
    assert [lazy_kind,eager_closed,content,direct]==[True,True,'hello','hello']
finally:
    try: os.unlink(path)
    except OSError: pass""", CLICK)

add("tenacity-wait-exp", "jd__tenacity.0d40e76f.func_basic__rreoa6fp",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_exponential.__init__",
    "exponential_wait_keeps_multiplier_and_base",
    "Exponential wait must keep a positive multiplier and base so sleeps double within bounds; negating the multiplier and bumping the base collapses every sleep to zero across the attempt-sleep-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.wait import wait_exponential
strat=wait_exponential(multiplier=1, min=0, max=1000, exp_base=2)
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    values.append(strat(state))
print('exponential_wait_sequence',values)
assert values==[1,2,4]""", SFR)

add("chunked-flush-grouping", "pudo__dataset.5c2dc8d3.func_basic__unvpw8ok",
    "pudo/dataset", "5c2dc8d3", "dataset/chunked.py", ".", "dataset.chunked", "ChunkedUpdate.flush",
    "update_flush_groups_by_key_sets_not_values",
    "Update flush must group queued rows by key sets so same-schema rows persist in one call; grouping by values splits every row and persists key lists instead of rows across the queue-flush-persist chain.",
"""from dataset.chunked import ChunkedUpdate
calls=[]
class T:
    def update_many(self, rows, keys): calls.append((list(rows),keys))
t=T()
up=ChunkedUpdate(t, keys=['id'], chunksize=100)
with up as u:
    enter_ok=(u is up)
    u.update({'id':1,'v':'a'})
    u.update({'id':2,'v':'b'})
print('flush_grouping_sequence',[enter_ok,len(calls),len(up.queue)])
assert [enter_ok,len(calls),len(up.queue)]==[True,1,0]
assert calls[0]==([{'id':1,'v':'a'},{'id':2,'v':'b'}],['id'])""", DB)

add("tenacity-stop-attempt-init", "jd__tenacity.0d40e76f.func_basic__nm22ottl",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_after_attempt.__init__",
    "stop_attempt_keeps_exact_max_count",
    "stop_after_attempt must keep the exact max count so the Nth attempt stops; decrementing at registration stops every cycle one attempt early across the attempt-stop-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_after_attempt
stop=stop_after_attempt(3)
assert stop.max_attempt_number==3
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    decisions.append(stop(state))
print('stop_attempt_init_sequence',decisions)
assert decisions==[False,False,True]""", SFR)

(HERE / "contracts-part5.json").write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote part5 {len(C)}")
