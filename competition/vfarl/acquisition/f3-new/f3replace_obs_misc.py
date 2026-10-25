"""F3-replace Cloudpickle/Typeguard/Dask observations.

Each OBS_<key> runs after the source-binding preamble. It must FAIL (nonzero)
on the buggy tree and PASS on the repaired tree.
"""

OBS_cp_01 = r"""
import sys, types
import cloudpickle
from cloudpickle.cloudpickle import (register_pickle_by_value, unregister_pickle_by_value,
                                     list_registry_pickle_by_value)
mod = types.ModuleType('f3r_mod_a')
mod.VALUE = 41
sys.modules['f3r_mod_a'] = mod
register_pickle_by_value(mod)
assert 'f3r_mod_a' in list_registry_pickle_by_value()
print("get_hit", True)
register_pickle_by_value(mod)
assert sum(1 for m in list_registry_pickle_by_value() if m == 'f3r_mod_a') == 1
print("equivalence", True)
mod2 = types.ModuleType('f3r_mod_b')
sys.modules['f3r_mod_b'] = mod2
register_pickle_by_value(mod2)
assert 'f3r_mod_b' in list_registry_pickle_by_value()
assert 'f3r_mod_a' in list_registry_pickle_by_value()
print("isolation", True)
unregister_pickle_by_value(mod)
assert 'f3r_mod_a' not in list_registry_pickle_by_value()
assert 'f3r_mod_b' in list_registry_pickle_by_value()
print("invalidate", True)
register_pickle_by_value(mod)
assert 'f3r_mod_a' in list_registry_pickle_by_value()
unregister_pickle_by_value(mod)
unregister_pickle_by_value(mod2)
print("recompute", True)
"""

OBS_cp_02 = r"""
import abc, pickle
import cloudpickle
MyClass = abc.ABCMeta('MyClass', (), {})
class MyUnrelatedClass:
    pass
class MyRelatedClass:
    pass
MyClass.register(MyRelatedClass)
assert not issubclass(MyUnrelatedClass, MyClass)
assert issubclass(MyRelatedClass, MyClass)
s = cloudpickle.dumps(MyClass)
assert b'MyUnrelatedClass' not in s, "unrelated-class cache must not leak into the payload"
assert b'MyRelatedClass' in s
print("get_hit", True)
assert pickle.loads(cloudpickle.dumps(MyClass)).__name__ == 'MyClass'
print("equivalence", True)
Other = abc.ABCMeta('Other', (), {})
assert pickle.loads(cloudpickle.dumps(Other)).__name__ == 'Other'
print("isolation", True)
from cloudpickle.cloudpickle import _class_getstate
state, slotstate = _class_getstate(MyClass)
assert '_abc_cache' not in state, "abc caches must be excluded from pickled state"
assert '_abc_negative_cache' not in state
assert '_abc_negative_cache_version' not in state
print("invalidate", True)
depickled = pickle.loads(s)
assert not issubclass(MyUnrelatedClass, depickled)
assert issubclass(MyRelatedClass, depickled)
print("recompute", True)
"""

OBS_cp_03 = OBS_cp_02

OBS_cp_04 = r"""
import pickle
import cloudpickle
def make(n):
    def fn(x, label=n):
        return (x, label)
    return fn
f = make('k')
data1 = cloudpickle.dumps(f)
data2 = cloudpickle.dumps(f)
assert data1 == data2, "chained pickling must be deterministic"
print("equivalence", True)
g = pickle.loads(data1)
assert g(2) == (2, 'k')
print("get_hit", True)
h = make('other')
assert pickle.loads(cloudpickle.dumps(h))(3) == (3, 'other')
assert pickle.loads(cloudpickle.dumps(f))(3) == (3, 'k')
print("isolation", True)
"""

OBS_cp_05 = r"""
import abc, pickle
import cloudpickle
MyClass = abc.ABCMeta('MyClass', (), {})
class MyUnrelatedClass:
    pass
assert not issubclass(MyUnrelatedClass, MyClass)
s = cloudpickle.dumps(MyClass)
assert b'MyUnrelatedClass' not in s, "negative cache must not leak into the payload"
print("get_hit", True)
assert pickle.loads(s).__name__ == 'MyClass'
print("equivalence", True)
Other = abc.ABCMeta('Other', (), {})
assert pickle.loads(cloudpickle.dumps(Other)).__name__ == 'Other'
print("isolation", True)
from cloudpickle.cloudpickle import _class_getstate
state, slotstate = _class_getstate(MyClass)
assert '_abc_cache' not in state
assert '_abc_negative_cache' not in state
print("invalidate", True)
assert not issubclass(MyUnrelatedClass, pickle.loads(s))
print("recompute", True)
"""

OBS_cp_06 = r"""
import functools, pickle
import cloudpickle
def deco(f):
    @functools.wraps(f)
    def w(*a, **k):
        return f(*a, **k)
    return w
@deco
def base(a, b, *args, c=None, **kw):
    'doc'
    return (a, b, args, c, kw)
import math as _math
def uses_submodule(x):
    return _math.floor(x) + 1
for fn, check in ((base, lambda g: g(1, 2, 3, c=4, extra=5) == (1, 2, (3,), 4, {'extra': 5}) and g.__name__ == 'base'),
                  (uses_submodule, lambda g: g(2.5) == 3)):
    data = cloudpickle.dumps(fn)
    g = pickle.loads(data)
    assert check(g), fn
print("get_hit", True)
assert pickle.loads(cloudpickle.dumps(base))(1, 2) == (1, 2, (), None, {})
print("equivalence", True)
assert pickle.loads(cloudpickle.dumps(uses_submodule))(2.5) == 3
print("isolation", True)
assert base.__doc__ == pickle.loads(cloudpickle.dumps(base)).__doc__ == 'doc'
print("recompute", True)
"""

OBS_cp_07 = r"""
import pickle
import cloudpickle
class Base:
    val = 1
    def meth(self):
        return 'base'
class Sub(Base):
    val = 2
    def meth(self):
        return 'sub'
data = cloudpickle.dumps(Sub)
back = pickle.loads(data)
assert back.val == 2
assert back().meth() == 'sub'
print("get_hit", True)
assert pickle.loads(cloudpickle.dumps(Sub)).val == 2
print("equivalence", True)
class Sub2(Base):
    val = 3
back2 = pickle.loads(cloudpickle.dumps(Sub2))
assert back2.val == 3 and back.val == 2
print("isolation", True)
from cloudpickle.cloudpickle import _extract_class_dict
d = _extract_class_dict(Sub)
assert 'val' in d and d['val'] == 2
print("recompute", True)
"""

OBS_cp_08 = OBS_cp_02

OBS_tg_01 = r"""
import ast
from typeguard._transformer import TypeguardTransformer
t = TypeguardTransformer()
tree = ast.parse('import os as _os\n')
t.visit(tree)
assert t._memo.local_names == {'_os'}, sorted(t._memo.local_names)
assert t._memo.imported_names == {'_os': 'os'}, dict(t._memo.imported_names)
print("get_hit", True)
t1b = TypeguardTransformer()
t1b.visit(ast.parse('import os as _os\n'))
assert t1b._memo.local_names == t._memo.local_names
assert t1b._memo.imported_names == t._memo.imported_names
print("equivalence", True)
t2 = TypeguardTransformer()
t2.visit(ast.parse('import json as _json\n'))
assert t2._memo.local_names == {'_json'}, sorted(t2._memo.local_names)
assert t2._memo.imported_names == {'_json': 'json'}
assert 'os' not in t2._memo.local_names and 'json' not in t._memo.local_names
print("isolation", True)
fresh = TypeguardTransformer()
assert fresh._memo.local_names == set() and fresh._memo.imported_names == {}
print("invalidate", True)
fresh.visit(ast.parse('import os as _os\n'))
assert fresh._memo.local_names == {'_os'}
assert fresh._memo.imported_names == {'_os': 'os'}
print("recompute", True)
"""

OBS_dask_01 = r"""
from concurrent.futures import ThreadPoolExecutor
from dask.local import get_async, start_state_from_dask
calls = []
def inc(x):
    calls.append(x)
    return x + 1
dsk = {'x': 1, 'y': (inc, 'x'), 'z': (inc, 'y')}
with ThreadPoolExecutor(1) as ex:
    assert get_async(ex.submit, 1, dict(dsk), 'z', cache={}) == 3
assert calls == [1, 2], calls
print("get_hit", True)
state = start_state_from_dask(dict(dsk), {'x': 1})
assert 'y' in state['ready'] and 'x' not in state['ready'], state['ready']
state_b = start_state_from_dask(dict(dsk), {'x': 1})
assert state_b['ready'] == state['ready']
print("equivalence", True)
dsk2 = {'a': (inc, 5), 'b': (inc, 'a')}
s_full = start_state_from_dask(dict(dsk2), {'a': 6})
assert 'b' in s_full['ready'], s_full['ready']
print("isolation", True)
s_empty = start_state_from_dask(dict(dsk2), {})
assert s_empty['ready'] == ['a'] and set(s_empty['ready']) != set(s_full['ready']), (s_empty['ready'], s_full['ready'])
print("invalidate", True)
s_re = start_state_from_dask(dict(dsk2), {'a': 6})
assert set(s_re['ready']) == set(s_full['ready'])
print("recompute", True)
"""

OBS_dask_02 = r"""
from dask.diagnostics import CacheProfiler
from dask.threaded import get as tget
def inc(x):
    return x + 1
dsk = {'x': 1, 'y': (inc, 'x')}
prof = CacheProfiler()
assert prof.start_time is None and prof.end_time is None
print("get_hit", True)
with prof:
    tget(dict(dsk), 'y')
assert prof.results, "run must record profiler results"
assert prof.start_time is not None and prof.end_time is not None
assert prof.end_time >= prof.start_time
print("equivalence", True)
prof2 = CacheProfiler()
assert prof2.results == []
assert prof2.start_time is None
print("isolation", True)
prof.clear()
assert prof.results == []
assert prof.start_time is None and prof.end_time is None, "clear must invalidate timing state"
print("invalidate", True)
with prof:
    tget(dict(dsk), 'y')
assert prof.results, "re-run must recompute results"
assert prof.start_time is not None and prof.end_time is not None
print("recompute", True)
"""

OBS_dask_03 = r"""
from dask.optimization import inline_functions
def inc(x):
    return x + 1
def add(a, b):
    return a + b
class NonHashableCallable:
    def __call__(self, a):
        return a + 1
    def __hash__(self):
        raise TypeError("Not hashable")
dsk = {'a': 1, 'b': (inc, 'a'), 'c': (NonHashableCallable(), 'b'), 'd': (inc, 'c')}
result = inline_functions(dict(dsk), [], fast_functions={inc})
assert result['c'][0].__class__.__name__ == 'NonHashableCallable', result
assert result['c'][1] == (inc, 'a'), result
assert 'b' not in result
print("get_hit", True)
again = inline_functions(dict(dsk), [], fast_functions={inc})
assert again['c'] == result['c'] and ('b' not in again)
print("equivalence", True)
dsk2 = {'x': 1, 'y': 1, 'i': (inc, 'x'), 'd': (inc, 'y'), 'out': (add, 'i', 'd')}
out2 = inline_functions(dict(dsk2), ['i', 'out'], fast_functions={inc})
assert out2['i'] == (inc, 'x'), "output keys must be protected from inlining"
assert 'i' in out2
print("isolation", True)
assert 'b' not in result
print("invalidate", True)
third = inline_functions(dict(dsk), [], fast_functions={inc})
assert third['c'] == result['c']
print("recompute", True)
"""


OBS_tg_02 = r"""
import ast
from typeguard._transformer import TypeguardTransformer
SRC = 'def foo() -> None:\n    x: int\n    if x := otherfunc():\n        pass\n'
SRC2 = 'def bar() -> None:\n    z: str\n    if z := otherfunc():\n        pass\n'
def walrus_call(tree):
    found = [n.value for n in ast.walk(tree) if isinstance(n, ast.NamedExpr)]
    assert len(found) == 1, ast.dump(tree)[:300]
    assert isinstance(found[0], ast.Call), ast.dump(found[0])[:300]
    return found[0]
t = TypeguardTransformer()
call = walrus_call(t.visit(ast.parse(SRC)))
assert len(call.args) == 4, f"flat check args expected, got {len(call.args)}"
assert isinstance(call.args[1], ast.Constant) and call.args[1].value == 'x'
print("get_hit", True)
t2 = TypeguardTransformer()
call2 = walrus_call(t2.visit(ast.parse(SRC)))
assert len(call2.args) == 4 and isinstance(call2.args[1], ast.Constant)
print("equivalence", True)
t3 = TypeguardTransformer()
call3 = walrus_call(t3.visit(ast.parse(SRC2)))
assert len(call3.args) == 4 and call3.args[1].value == 'z'
print("isolation", True)
fresh = TypeguardTransformer()
assert fresh._memo.ignored_names == set()
print("invalidate", True)
call4 = walrus_call(fresh.visit(ast.parse(SRC)))
assert len(call4.args) == 4 and call4.args[1].value == 'x'
print("recompute", True)
"""

OBS_tg_03 = r"""
import ast
from typeguard._transformer import TypeguardTransformer
t = TypeguardTransformer()
t.visit(ast.parse('from m import n as a\n'))
assert t._memo.local_names == {'a'}, sorted(t._memo.local_names)
assert t._memo.imported_names == {'a': 'm.n'}, dict(t._memo.imported_names)
print("get_hit", True)
t2 = TypeguardTransformer()
t2.visit(ast.parse('from m import n as a\n'))
assert t2._memo.imported_names == t._memo.imported_names
print("equivalence", True)
t3 = TypeguardTransformer()
t3.visit(ast.parse('from p import q as b\n'))
assert t3._memo.imported_names == {'b': 'p.q'}
assert 'a' not in t3._memo.local_names
print("isolation", True)
fresh = TypeguardTransformer()
assert fresh._memo.imported_names == {}
print("invalidate", True)
fresh.visit(ast.parse('from m import n as a\n'))
assert fresh._memo.imported_names == {'a': 'm.n'}
print("recompute", True)
"""

OBS_tg_04 = r"""
import ast
from typeguard._transformer import TypeguardTransformer
t = TypeguardTransformer()
tree = ast.parse('from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import json\n')
out = t.visit(tree)
assert out is not None, "TYPE_CHECKING branch must be preserved"
assert 'json' in t._memo.ignored_names, sorted(t._memo.ignored_names)
print("get_hit", True)
t2 = TypeguardTransformer()
out2 = t2.visit(ast.parse('from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import json\n'))
assert out2 is not None and t2._memo.ignored_names == t._memo.ignored_names
print("equivalence", True)
t3 = TypeguardTransformer()
out3 = t3.visit(ast.parse('if flag:\n    import yaml\n'))
assert out3 is not None
print("isolation", True)
fresh = TypeguardTransformer()
assert fresh._memo.ignored_names == set()
print("invalidate", True)
out4 = fresh.visit(ast.parse('from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import json\n'))
assert out4 is not None
assert 'json' in fresh._memo.ignored_names
print("recompute", True)
"""

MISC_OBS = {
    'f3r-cp-01': OBS_cp_01,
    'f3r-cp-02': OBS_cp_02,
    'f3r-cp-03': OBS_cp_03,
    'f3r-cp-04': OBS_cp_04,
    'f3r-cp-05': OBS_cp_05,
    'f3r-cp-06': OBS_cp_06,
    'f3r-cp-07': OBS_cp_07,
    'f3r-cp-08': OBS_cp_08,
    'f3r-tg-01': OBS_tg_01,
    'f3r-tg-04': OBS_tg_04,
    'f3r-tg-03': OBS_tg_03,
    'f3r-tg-02': OBS_tg_02,
    'f3r-dask-01': OBS_dask_01,
    'f3r-dask-02': OBS_dask_02,
    'f3r-dask-03': OBS_dask_03,
}
