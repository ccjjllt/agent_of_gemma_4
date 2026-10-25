"""Draft F5 40 contracts with event-sequence observations (manual review)."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "selected-contracts.json"

# Each: key suffix, instance_id, canonical, snapshot, rel, layout, module, symbol, family, contract, body
# Bodies must contain production->consumption->end sequence assertion (list == expected).
C = []

def add(key, iid, canon, snap, rel, layout, module, symbol, family, contract, body):
    assert "assert" in body and ("==" in body), key
    # must have sequence: list literal with at least 3 elements or order check
    C.append({"canonical_repo": canon, "snapshot": snap, "relative_path": rel,
              "source_layout": layout, "task_profile": ".venv-sfr/lib/python3.12/site-packages",
              "source_license": "UPSTREAM",  # resolved per repo during prepare
              "key": key, "instance_id": iid, "symbol": symbol,
              "primary_adapter": "F", "submechanism": "F5",
              "family_contract": family, "contract": contract, "body": body})

# ---------- tenacity 10 ----------
add("tenacity-run-stop", "jd__tenacity.0d40e76f.func_basic__9b67b0zh",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "BaseRetrying._run_stop",
    "run_stop_forwards_live_retry_state",
    "Begin attempts one through four with stop_after_attempt(3). The stop predicate must receive the live retry state so early attempts continue and later attempts stop.",
"""from tenacity import Retrying, RetryCallState, stop_after_attempt
ctl=Retrying(stop=stop_after_attempt(3))
decisions=[]
for attempt in (1,2,3,4):
    state=RetryCallState(ctl,None,(),{})
    state.attempt_number=attempt
    ctl._run_stop(state)
    decisions.append(bool(ctl.iter_state.stop_run_result))
print('attempt_stop_sequence',decisions)
assert decisions==[False,False,True,True]""")

add("tenacity-post-stop", "jd__tenacity.0d40e76f.lm_rewrite__nrufeeyo",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "BaseRetrying._post_stop_check_actions",
    "post_stop_schedules_sleep_and_next_attempt_accounting",
    "After a non-stopping decision the controller must schedule sleep, advance idle_for and attempt_number, then emit DoSleep; after a stopping decision it must surface the retry error. The rewrite drops next-attempt accounting.",
"""from tenacity import Retrying, RetryCallState, stop_after_attempt, wait_fixed
attempts=[]
def fn():
    attempts.append(len(attempts)+1)
    if len(attempts)<3:
        raise ValueError('fail')
    return 'ok'
ctl=Retrying(stop=stop_after_attempt(5), wait=wait_fixed(0))
result=ctl(fn)
print('retry_attempt_sequence',attempts)
assert attempts==[1,2,3]
assert result=='ok'
assert ctl.statistics['attempt_number']==3""")

add("tenacity-begin-iter", "jd__tenacity.0d40e76f.func_pm_ctrl_shuffle__ied06l4m",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "BaseRetrying._begin_iter",
    "begin_iter_resets_before_branching_on_outcome",
    "A fresh call with no outcome must reset iteration state before selecting DoAttempt; a failed TryAgain outcome must take the explicit-retry branch. Resetting after reading the outcome corrupts the branch.",
"""from tenacity import Retrying, RetryCallState, stop_after_attempt
from tenacity import TryAgain, Future
ctl=Retrying(stop=stop_after_attempt(3))
flags=[]
for outcome_kind in ('none','tryagain','normal_fail'):
    st=RetryCallState(ctl,None,(),{})
    if outcome_kind=='none':
        st.outcome=None
    elif outcome_kind=='tryagain':
        st.outcome=Future.construct(1, TryAgain(), True)
    else:
        st.outcome=Future.construct(1, ValueError('x'), True)
    ctl._begin_iter(st)
    flags.append(bool(ctl.iter_state.is_explicit_retry))
print('begin_explicit_sequence',flags)
assert flags==[False,True,False]""")

add("tenacity-stop-any", "jd__tenacity.0d40e76f.func_basic__7470eiay",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "stop_any.__init__",
    "stop_any_preserves_all_stops_across_attempts",
    "stop_any must keep every stop so the first firing stop ends retries. Dropping the first stop delays the attempt-stop-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_any, stop_after_attempt
combined=stop_any(stop_after_attempt(2), stop_after_attempt(5))
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3,5,6):
    state.attempt_number=attempt
    decisions.append(combined(state))
print('stop_any_sequence',decisions)
assert decisions==[False,True,True,True,True]""")

add("tenacity-wait-random-exp", "jd__tenacity.0d40e76f.func_basic__aafgyqmn",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_random_exponential.__call__",
    "random_exponential_draws_within_growing_bounds",
    "Random-exponential wait must draw uniformly between min and the exponential high for each attempt. Using an undefined bound breaks the attempt-sleep-end chain.",
"""import random
from tenacity import RetryCallState, Retrying
from tenacity.wait import wait_random_exponential
random.seed(0)
strat=wait_random_exponential(multiplier=1, min=0, max=30)
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (1,2,3):
    state.attempt_number=attempt
    values.append(strat(state))
print('random_exp_sequence',values)
assert values==[0.8444218515250481, 1.515908805880605, 1.68228632332338]""")

add("tenacity-wait-inc", "jd__tenacity.0d40e76f.func_basic__m9tzhblx",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_incrementing.__init__",
    "incrementing_wait_preserves_start_increment_max_roles",
    "Incrementing wait must keep start/increment/max in order so attempt sleeps grow linearly and cap at max. Rotating the three arguments corrupts the attempt-sleep-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity import wait_incrementing
strat=wait_incrementing(start=10, increment=5, max=100)
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (1,2,3,4,20):
    state.attempt_number=attempt
    values.append(strat(state))
print('incrementing_wait_sequence',values)
assert values==[10,15,20,25,100]""")

add("tenacity-iter-begin", "jd__tenacity.0d40e76f.func_pm_ctrl_shuffle__ni48riat",
    "jd/tenacity", "0d40e76f", "tenacity/__init__.py", ".", "tenacity", "BaseRetrying.__iter__",
    "blocking_iter_begins_new_call_before_first_attempt",
    "Blocking iteration must begin a new call and build attempt state before the first DoAttempt. Beginning after the loop leaves attempt_number/idle uninitialized for the first attempt-sleep-end cycle.",
"""from tenacity import Retrying, stop_after_attempt, wait_fixed
calls=[]; tries=[]
def fn():
    tries.append(len(tries)+1)
    if len(tries)<2:
        raise ValueError('retry once')
    return 'done'
ctl=Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0), before=lambda rs: calls.append(rs.attempt_number))
for attempt in ctl:
    with attempt:
        result=fn()
print('iter_attempt_sequence',calls)
print('fn_attempt_sequence',tries)
assert [calls,tries,result]==[[1,2],[1,2],'done']""")

add("tenacity-stop-never", "jd__tenacity.0d40e76f.func_basic__7azvfd3s",
    "jd/tenacity", "0d40e76f", "tenacity/stop.py", ".", "tenacity.stop", "_stop_never.__call__",
    "never_stop_stays_false_across_attempt_sequence",
    "The never-stop predicate must stay false for every attempt so retries continue until other stops fire. Returning true stops the first attempt-sleep-end cycle immediately.",
"""from tenacity import RetryCallState, Retrying
from tenacity.stop import stop_never
ctl=Retrying()
state=RetryCallState(ctl,None,(),{})
decisions=[]
for attempt in (1,2,3,4):
    state.attempt_number=attempt
    decisions.append(stop_never(state))
print('never_stop_sequence',decisions)
assert decisions==[False,False,False,False]""")

add("tenacity-retry-all", "jd__tenacity.0d40e76f.func_basic__1l3yf5xv",
    "jd/tenacity", "0d40e76f", "tenacity/retry.py", ".", "tenacity.retry", "retry_all.__call__",
    "retry_all_requires_every_predicate_across_outcomes",
    "retry_all must require every sub-predicate; a single success must not retry while mixed outcomes follow all/any truth tables. Using any retries when only one predicate fires.",
"""from tenacity import RetryCallState, Retrying, Future
from tenacity.retry import retry_all, retry_if_result, retry_if_exception_type
combined=retry_all(retry_if_result(lambda x: x==1), retry_if_exception_type(ValueError))
def outcome_of(value=None, exc=None):
    st=RetryCallState(Retrying(),None,(),{})
    if exc is not None:
        st.outcome=Future.construct(1, exc, True)
    else:
        st.outcome=Future.construct(1, value, False)
    return st
decisions=[combined(outcome_of(value=1)), combined(outcome_of(value=2)), combined(outcome_of(exc=ValueError('x')))]
print('retry_all_sequence',decisions)
assert decisions==[False,False,False]""")

add("tenacity-wait-chain", "jd__tenacity.0d40e76f.func_basic__aavzza2x",
    "jd/tenacity", "0d40e76f", "tenacity/wait.py", ".", "tenacity.wait", "wait_chain.__init__",
    "wait_chain_preserves_stage_order_across_attempts",
    "Chained waits must keep stage order so early attempts use the first stage and late attempts reuse the last. Reversing stages swaps early/late sleeps across the attempt-sleep-end chain.",
"""from tenacity import RetryCallState, Retrying
from tenacity import wait_chain, wait_fixed
strat=wait_chain(wait_fixed(2), wait_fixed(5))
state=RetryCallState(Retrying(),None,(),{})
values=[]
for attempt in (0,1,2,3,99):
    state.attempt_number=attempt
    values.append(strat(state))
print('chained_wait_stages',values)
assert values==[2,2,5,5,5]""")

OUT.write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(C)}")
