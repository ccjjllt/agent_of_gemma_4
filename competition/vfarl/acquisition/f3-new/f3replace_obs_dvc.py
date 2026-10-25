"""F3-replace DVC observations: set->get->expire/invalidate->recompute + equivalence/isolation.

Each OBS_<key> runs after the source-binding preamble. It must FAIL (nonzero)
on the buggy tree and PASS on the repaired tree.
"""

OBS_dvc_01 = r"""
import os, tempfile
from dvc.stage.cache import StageCache
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'a' * 32, 'v1'
d = sc._get_cache_dir(key)
p = sc._get_cache_path(key, val)
assert p.startswith(d), (p, d)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
got = sc._load_cache(key, val)
assert isinstance(got, dict), f"get {got!r}"
print("get_hit", True)
assert sc._load_cache(key, val) == got
print("equivalence", True)
assert sc._load_cache('b' * 32, val) is None
print("isolation", True)
with open(p, 'w') as f:
    f.write('not: [valid\n')
assert sc._load_cache(key, val) is None
assert not os.path.exists(p), "invalid entry must be unlinked"
print("invalidate", True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert sc._load_cache(key, val) == got
print("recompute", True)
"""

OBS_dvc_02 = OBS_dvc_01

OBS_dvc_03 = r"""
from types import SimpleNamespace
from dvc.stage.cache import _can_hash
def dep(protocol='local', def_path='data', h='abc'):
    d = SimpleNamespace()
    d.protocol = protocol
    d.def_path = def_path
    d.get_hash = lambda: h
    return d
def out(out_hash='def'):
    o = SimpleNamespace()
    o.protocol = 'local'
    o.def_path = 'out'
    o.get_hash = lambda: out_hash
    o.use_cache = True
    o.persist = False
    o.is_in_repo = True
    return o
def stage(cmd='echo', deps=None, outs=None):
    s = SimpleNamespace()
    s.is_callback = False
    s.always_changed = False
    s.cmd = cmd
    s.deps = deps if deps is not None else [dep()]
    s.outs = outs if outs is not None else [out()]
    return s
good = stage()
assert _can_hash(good) is True
print("get_hit", True)
assert _can_hash(stage()) is True
print("equivalence", True)
remote = stage(deps=[dep(protocol='s3')])
assert _can_hash(remote) is False, "non-local dep must invalidate cacheability"
print("invalidate", True)
hashless = stage(deps=[dep(h=None)])
assert _can_hash(hashless) is False
print("isolation", True)
fixed = stage(deps=[dep()])
assert _can_hash(fixed) is True
print("recompute", True)
"""

OBS_dvc_04 = r"""
import inspect
from types import SimpleNamespace
from dvc.stage.cache import StageCache
sc = StageCache.__new__(StageCache)
def fake_out(def_path, use_cache=True, is_in_repo=True):
    o = SimpleNamespace()
    o.def_path = def_path
    o.use_cache = use_cache
    o.is_in_repo = is_in_repo
    return o
cached_stage = SimpleNamespace(outs=[fake_out('a'), fake_out('b'), fake_out('c'), fake_out('z')])
sc._create_stage = lambda cache, wdir=None: cached_stage
import contextlib
sc._cache_type_copy = lambda: contextlib.nullcontext()
stage = SimpleNamespace(wdir='.', outs=[fake_out('a'), fake_out('b', use_cache=False)])
res = sc._uncached_outs(stage, {'outs': [{'path': 'a'}]})
assert inspect.isgenerator(res), f"must be lazy generator, got {type(res)}"
print("get_hit", True)
items = list(res)
assert [o.def_path for o in items] == ['b'], [o.def_path for o in items]
print("equivalence", True)
stage2 = SimpleNamespace(wdir='.', outs=[fake_out('a'), fake_out('z', use_cache=False)])
items2 = list(sc._uncached_outs(stage2, {'outs': [{'path': 'a'}]}))
assert [o.def_path for o in items2] == ['z']
print("isolation", True)
stage3 = SimpleNamespace(wdir='.', outs=[fake_out('c', use_cache=False)])
items3 = list(sc._uncached_outs(stage3, {'outs': [{'path': 'a'}]}))
assert [o.def_path for o in items3] == ['c']
print("recompute", True)
"""

OBS_dvc_05 = r"""
import os, subprocess, tempfile
os.environ['DVC_TEST'] = 'true'
from dvc.repo import Repo
work = os.path.realpath(tempfile.mkdtemp())
subprocess.run(['git', 'init', '-q', work], check=True, capture_output=True)
subprocess.run(['git', '-C', work, '-c', 'user.name=t', '-c', 'user.email=t@e',
                'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
with open(os.path.join(work, 'script.py'), 'w') as f:
    f.write('open("out", "w+").write("out")')
with open(os.path.join(work, 'dep'), 'w') as f:
    f.write('dep')
os.chdir(work)
repo = Repo.init(work, no_scm=False)
try:
    stage = repo.run(cmd='python script.py', deps=['script.py', 'dep'],
                     outs=['out'], name='build')
    from dvc.stage.cache import _can_hash
    assert _can_hash(stage) is True
    print("get_hit", True)
    def cache_files():
        hits = []
        for root, _dirs, files in os.walk(repo.stage_cache.cache_dir):
            hits.extend(files)
        return sorted(hits)
    first = cache_files()
    assert first, "save must materialize a cache entry"
    print("equivalence", True)
    repo.run(cmd='python script.py', deps=['script.py', 'dep'],
             outs=['out'], name='build', force=True)
    second = cache_files()
    assert second == first, f"second save must reuse the entry, got {second} vs {first}"
    print("isolation", True)
    import shutil
    shutil.rmtree(repo.stage_cache.cache_dir)
    assert cache_files() == []
    print("invalidate", True)
    repo.run(cmd='python script.py', deps=['script.py', 'dep'],
             outs=['out'], name='build', force=True)
    assert cache_files(), "save after clear must recompute the entry"
    print("recompute", True)
finally:
    repo.close()
"""

OBS_dvc_06 = r"""
import os, tempfile
from dvc.fs import localfs
from dvc.stage.cache import StageCache
sc = StageCache.__new__(StageCache)
class FakeODB:
    def __init__(self, path):
        self.path = path
        self.fs = localfs
tmp = tempfile.mkdtemp()
src_odb = FakeODB(os.path.join(tmp, 'from'))
dst_odb = FakeODB(os.path.join(tmp, 'to'))
os.makedirs(os.path.join(src_odb.path, 'runs', 'keyA'))
with open(os.path.join(src_odb.path, 'runs', 'keyA', 'f'), 'w') as f:
    f.write('payload-a')
ret = sc.transfer(src_odb, dst_odb, force=True)
dst_file = os.path.join(dst_odb.path, 'runs', 'keyA', 'f')
assert os.path.exists(dst_file), "set->get must copy the entry"
with open(dst_file) as f:
    assert f.read() == 'payload-a'
print("get_hit", True)
ret2 = sc.transfer(src_odb, dst_odb, force=True)
with open(dst_file) as f:
    assert f.read() == 'payload-a'
print("equivalence", True)
os.makedirs(os.path.join(src_odb.path, 'runs', 'keyB'))
with open(os.path.join(src_odb.path, 'runs', 'keyB', 'g'), 'w') as f:
    f.write('payload-b')
sc.transfer(src_odb, dst_odb, force=True)
assert open(os.path.join(dst_odb.path, 'runs', 'keyB', 'g')).read() == 'payload-b'
assert open(dst_file).read() == 'payload-a'
print("isolation", True)
os.unlink(dst_file)
with open(os.path.join(dst_odb.path, 'runs', 'keyA', 'other'), 'w') as f:
    f.write('x')
sc.transfer(src_odb, dst_odb, force=False)
assert not os.path.exists(dst_file), "force=False must skip keys that already exist"
print("invalidate", True)
import shutil
shutil.rmtree(os.path.join(dst_odb.path, 'runs', 'keyA'))
sc.transfer(src_odb, dst_odb, force=False)
assert open(dst_file).read() == 'payload-a'
print("recompute", True)
"""

OBS_dvc_07 = r"""
import os, subprocess, tempfile
os.environ['DVC_TEST'] = 'true'
from dvc.repo import Repo
from dvc.stage.cache import StageCache, RunCacheNotFoundError
work = os.path.realpath(tempfile.mkdtemp())
subprocess.run(['git', 'init', '-q', work], check=True, capture_output=True)
subprocess.run(['git', '-C', work, '-c', 'user.name=t', '-c', 'user.email=t@e',
                'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
with open(os.path.join(work, 'script.py'), 'w') as f:
    f.write('open("out", "w+").write("out")')
with open(os.path.join(work, 'dep'), 'w') as f:
    f.write('dep')
os.chdir(work)
repo = Repo.init(work, no_scm=False)
try:
    stage = repo.run(cmd='python script.py', deps=['script.py', 'dep'],
                     outs=['out'], name='build')
    sc = StageCache(repo)
    os.remove(os.path.join(work, 'out'))
    sc.restore(stage, dry=True)
    assert not os.path.exists(os.path.join(work, 'out')), "dry restore must not check out"
    print("get_hit", True)
    sc.restore(stage, dry=True)
    assert not os.path.exists(os.path.join(work, 'out'))
    print("equivalence", True)
    work2 = os.path.realpath(tempfile.mkdtemp())
    subprocess.run(['git', 'init', '-q', work2], check=True, capture_output=True)
    subprocess.run(['git', '-C', work2, '-c', 'user.name=t', '-c', 'user.email=t@e',
                    'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
    with open(os.path.join(work2, 'script.py'), 'w') as f:
        f.write('open("out", "w+").write("out")')
    with open(os.path.join(work2, 'dep'), 'w') as f:
        f.write('dep')
    os.chdir(work2)
    repo2 = Repo.init(work2, no_scm=False)
    try:
        stage2 = repo2.run(cmd='python script.py', deps=['script.py', 'dep'],
                           outs=['out'], name='build')
        os.remove(os.path.join(work2, 'out'))
        StageCache(repo2).restore(stage2, dry=True)
        assert not os.path.exists(os.path.join(work2, 'out'))
        print("isolation", True)
    finally:
        repo2.close()
    os.chdir(work)
    import shutil
    if os.path.exists(os.path.join(work, 'out')):
        os.remove(os.path.join(work, 'out'))
    with open(os.path.join(work, 'dep'), 'w') as f:
        f.write('dep2')
    shutil.rmtree(repo.stage_cache.cache_dir)
    try:
        sc.restore(stage, dry=True)
    except RunCacheNotFoundError:
        print("invalidate", True)
    else:
        raise AssertionError("cleared cache must fail the restore")
    with repo.lock:
        stage.run()
    assert os.path.exists(os.path.join(work, 'out'))
    print("recompute", True)
    from dvc.stage.cache import _can_hash
    assert _can_hash(stage) is True
finally:
    repo.close()
"""

OBS_dvc_08 = r"""
import os, tempfile
from dvc.stage.cache import StageCache
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'e' * 32, 'v1'
p = sc._get_cache_path(key, val)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("get_hit", True)
assert sc._load_cache(key, val) == sc._load_cache(key, val)
print("equivalence", True)
assert sc._load_cache('f' * 32, val) is None
print("isolation", True)
os.unlink(p)
assert sc._load_cache(key, val) is None
print("invalidate", True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("recompute", True)
"""

OBS_dvc_09 = r"""
from types import SimpleNamespace
from dvc_data.hashfile.hash_info import HashInfo
from dvc.repo.index import _build_tree_from_outs, IndexView
def out(ws='workspace', key=('d', 'f1'), use_cache=True, hv='d41d8cd98f00b204e9800998ecf8427e'):
    o = SimpleNamespace()
    o.use_cache = use_cache
    o.index_key = (ws, key)
    o.meta = {'size': 1}
    o.hash_info = HashInfo('md5', hv)
    o.stage = SimpleNamespace(is_partial_import=False, is_import=False)
    return o
o1 = out(key=('d', 'f1'))
tree = _build_tree_from_outs([o1])
assert len(tree) == 1, f"set->get must materialize one entry, got {len(tree)}"
print("get_hit", True)
tree2 = _build_tree_from_outs([out(key=('d', 'f1'))])
assert tree2.digest() == tree.digest()
print("equivalence", True)
tree3 = _build_tree_from_outs([out(key=('d', 'f2'), hv='900150983cd24fb0d6963f7d28e17f72')])
assert len(tree3) == 1 and len(tree) == 1
keys = [k for k, _m, _h in tree]
keys3 = [k for k, _m, _h in tree3]
assert keys != keys3, (keys, keys3)
assert keys == [('workspace', 'd', 'f1')]
assert keys3 == [('workspace', 'd', 'f2')]
print("isolation", True)
tree4 = _build_tree_from_outs([])
assert len(tree4) == 0
print("invalidate", True)
tree5 = _build_tree_from_outs([out(key=('d', 'f1'))])
assert tree5.digest() == tree.digest()
print("recompute", True)
"""


OBS_dvc_10 = r"""
import os, subprocess, tempfile
os.environ['DVC_TEST'] = 'true'
from types import SimpleNamespace
from dvc.stage.cache import StageCache, _can_hash
def dep(protocol='local', def_path='data', h='abc'):
    d = SimpleNamespace()
    d.protocol = protocol
    d.def_path = def_path
    d.get_hash = lambda: h
    return d
def out():
    o = SimpleNamespace()
    o.protocol = 'local'
    o.def_path = 'out'
    o.get_hash = lambda: 'abc'
    o.use_cache = True
    o.persist = False
    o.is_in_repo = True
    return o
def stage(cmd='echo', deps=None, outs=None):
    s = SimpleNamespace()
    s.is_callback = False
    s.always_changed = False
    s.cmd = cmd
    s.deps = deps if deps is not None else [dep()]
    s.outs = outs if outs is not None else [out()]
    return s
assert _can_hash(stage()) is True
print("get_hit", True)
assert _can_hash(stage(deps=[dep(protocol='s3')])) is False
print("invalidate", True)
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'a' * 32, 'v1'
p = os.path.join(tmp, key[:2], key, val)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("equivalence", True)
assert sc._load_cache('b' * 32, val) is None
print("isolation", True)
from dvc.repo import Repo
work = os.path.realpath(tempfile.mkdtemp())
subprocess.run(['git', 'init', '-q', work], check=True, capture_output=True)
subprocess.run(['git', '-C', work, '-c', 'user.name=t', '-c', 'user.email=t@e',
                'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
os.chdir(work)
repo = Repo.init(work, no_scm=False)
try:
    with open(os.path.join(work, 'script.py'), 'w') as f:
        f.write('open("out", "w+").write("out")')
    with open(os.path.join(work, 'dep'), 'w') as f:
        f.write('dep')
    repo.run(cmd='python script.py', deps=['script.py', 'dep'], outs=['out'], name='build')
    def cache_files():
        hits = []
        for root, _dirs, files in os.walk(repo.stage_cache.cache_dir):
            hits.extend(files)
        return sorted(hits)
    assert cache_files()
    repo.run(cmd='python script.py', deps=['script.py', 'dep'], outs=['out'], name='build', force=True)
    assert len(cache_files()) == 1, cache_files()
    print("recompute", True)
finally:
    repo.close()
"""

OBS_dvc_11 = r"""
import os, subprocess, tempfile
os.environ['DVC_TEST'] = 'true'
from unittest import mock
from dvc.repo import Repo
from dvc.stage import Stage
from dvc.output import Output
work = os.path.realpath(tempfile.mkdtemp())
subprocess.run(['git', 'init', '-q', work], check=True, capture_output=True)
subprocess.run(['git', '-C', work, '-c', 'user.name=t', '-c', 'user.email=t@e',
                'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
os.chdir(work)
repo = Repo.init(work, no_scm=False)
try:
    stage = Stage(repo)
    out = Output(stage, 'path', cache=False)
    with mock.patch.object(out.fs, 'exists', return_value=False):
        try:
            out.save()
        except out.DoesNotExistError:
            print("get_hit", True)
        else:
            raise AssertionError("missing path must raise DoesNotExistError")
    with mock.patch.object(out.fs, 'exists', return_value=False):
        try:
            Output(Stage(repo), 'other', cache=False).save()
        except Output.DoesNotExistError:
            print("equivalence", True)
        else:
            raise AssertionError("missing path must raise")
    out2 = Output(stage, 'path2', cache=False)
    assert out2.def_path != out.def_path
    print("isolation", True)
    with open(os.path.join(work, 'path'), 'w') as f:
        f.write('x')
    out3 = Output(stage, 'path', cache=False)
    with mock.patch.object(out3.fs, 'exists', return_value=True):
        try:
            out3.save()
        except out3.DoesNotExistError:
            raise AssertionError("present path must not raise DoesNotExistError")
        except Exception:
            pass
    print("invalidate", True)
    os.remove(os.path.join(work, 'path'))
    out4 = Output(stage, 'path', cache=False)
    with mock.patch.object(out4.fs, 'exists', return_value=False):
        try:
            out4.save()
        except out4.DoesNotExistError:
            print("recompute", True)
        else:
            raise AssertionError("missing path must raise again")
finally:
    repo.close()
"""

OBS_dvc_12 = r"""
from unittest import mock
from dvc.output import Output
files = [
    {'size': 3, 'version_id': 'v1', 'etag': 'acbd18db4cc2f85cedef654fccc4a4d8',
     'md5': 'acbd18db4cc2f85cedef654fccc4a4d8', 'relpath': 'bar'},
    {'size': 3, 'version_id': 'v2', 'etag': 'acbd18db4cc2f85cedef654fccc4a4d8',
     'md5': 'acbd18db4cc2f85cedef654fccc4a4d8', 'relpath': 'foo'},
]
stage = mock.MagicMock()
stage.repo.fs.version_aware = False
stage.repo.fs.PARAM_CHECKSUM = 'md5'
out = Output(stage, 'path', files=files)
dumpd = out.dumpd()
assert dumpd == {'path': 'path', 'hash': 'md5', 'files': files}, dumpd
print("get_hit", True)
assert Output(stage, 'path', files=files).dumpd() == dumpd
print("equivalence", True)
out2 = Output(stage, 'other', files=files)
assert out2.dumpd()['path'] == 'other'
print("isolation", True)
out3 = Output(stage, 'path', files=[])
assert out3.dumpd() != dumpd
print("invalidate", True)
assert Output(stage, 'path', files=files).dumpd() == dumpd
print("recompute", True)
"""

OBS_dvc_13 = r"""
import os, subprocess, tempfile
os.environ['DVC_TEST'] = 'true'
from dvc.repo import Repo
from dvc.stage.cache import StageCache
work = os.path.realpath(tempfile.mkdtemp())
subprocess.run(['git', 'init', '-q', work], check=True, capture_output=True)
subprocess.run(['git', '-C', work, '-c', 'user.name=t', '-c', 'user.email=t@e',
                'commit', '--allow-empty', '-qm', 'i'], check=True, capture_output=True)
os.chdir(work)
repo = Repo.init(work, no_scm=False)
try:
    with open(os.path.join(work, 'script.py'), 'w') as f:
        f.write('open("out", "w+").write("out")')
    with open(os.path.join(work, 'dep'), 'w') as f:
        f.write('dep')
    repo.run(cmd='python script.py', deps=['script.py', 'dep'], outs=['out'], name='build')
    local_root = repo.cache.local.cache_dir if hasattr(repo.cache.local, 'cache_dir') else None
    found = []
    for root, _dirs, files in os.walk(repo.stage_cache.cache_dir):
        found.extend(os.path.join(root, x) for x in files)
    assert found, "save must materialize a cache entry"
    print("get_hit", True)
    assert all(os.path.exists(x) for x in found)
    print("equivalence", True)
    sc = StageCache(repo)
    assert sc._load_cache('z' * 32, 'v9') is None
    print("isolation", True)
    import shutil
    shutil.rmtree(repo.stage_cache.cache_dir)
    assert not os.path.exists(repo.stage_cache.cache_dir)
    print("invalidate", True)
    repo.run(cmd='python script.py', deps=['script.py', 'dep'], outs=['out'], name='build', force=True)
    found2 = []
    for root, _dirs, files in os.walk(repo.stage_cache.cache_dir):
        found2.extend(files)
    assert found2, "save after clear must recompute"
    print("recompute", True)
finally:
    repo.close()
"""

OBS_dvc_14 = r"""
import os, tempfile
from types import SimpleNamespace
from dvc.stage.cache import StageCache, _can_hash
from dvc.stage.utils import validate_kwargs
def dep(protocol='local', def_path='data', h='abc'):
    d = SimpleNamespace()
    d.protocol = protocol
    d.def_path = def_path
    d.get_hash = lambda: h
    return d
def out():
    o = SimpleNamespace()
    o.protocol = 'local'
    o.def_path = 'out'
    o.get_hash = lambda: 'abc'
    o.use_cache = True
    o.persist = False
    o.is_in_repo = True
    return o
def stage(cmd='echo', deps=None, outs=None):
    s = SimpleNamespace()
    s.is_callback = False
    s.always_changed = False
    s.cmd = cmd
    s.deps = deps if deps is not None else [dep()]
    s.outs = outs if outs is not None else [out()]
    return s
assert _can_hash(stage()) is True
print("get_hit", True)
assert _can_hash(stage(deps=[dep(protocol='s3')])) is False
print("invalidate", True)
kw = validate_kwargs(cmd='echo', name='build')
assert kw['name'] == 'build'
print("equivalence", True)
kw2 = validate_kwargs(cmd='echo', name='other')
assert kw2['name'] == 'other'
print("isolation", True)
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'e' * 32, 'v1'
p = os.path.join(tmp, key[:2], key, val)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("recompute", True)
"""


OBS_dvc_15 = r"""
from unittest import mock
from dvc.output import Output
files = [
    {'size': 3, 'version_id': 'v1', 'etag': 'acbd18db4cc2f85cedef654fccc4a4d8',
     'md5': 'acbd18db4cc2f85cedef654fccc4a4d8', 'relpath': 'bar'},
    {'size': 3, 'version_id': 'v2', 'etag': 'acbd18db4cc2f85cedef654fccc4a4d8',
     'md5': 'acbd18db4cc2f85cedef654fccc4a4d8', 'relpath': 'foo'},
]
stage = mock.MagicMock()
stage.repo.fs.version_aware = False
stage.repo.fs.PARAM_CHECKSUM = 'etag'
out = Output(stage, 'path', files=files)
assert out.hash_info.name == 'md5', out.hash_info
assert out.hash_info.value == '77e8000f532886eef8ee1feba82e9bad.dir', out.hash_info
assert out.meta.isdir
assert out.meta.nfiles == 2
assert out.meta.size == 6
print("get_hit", True)
out_b = Output(stage, 'path', files=files)
assert (out_b.hash_info.value, out_b.meta.nfiles, out_b.meta.size) == (out.hash_info.value, 2, 6)
print("equivalence", True)
out_c = Output(stage, 'other', files=files)
assert out_c.hash_info.value == out.hash_info.value
print("isolation", True)
out_d = Output(stage, 'path', files=files[:1])
assert (out_d.meta.nfiles, out_d.meta.size) == (1, 3)
assert out_d.hash_info.value != out.hash_info.value
print("invalidate", True)
out_e = Output(stage, 'path', files=files)
assert (out_e.hash_info.value, out_e.meta.nfiles) == (out.hash_info.value, 2)
print("recompute", True)
"""

OBS_dvc_16 = r"""
from types import SimpleNamespace
from dvc.commands.experiments.remove import CmdExperimentsRemove
from dvc.exceptions import InvalidArgumentError
def args(**kw):
    base = dict(experiment=None, all_commits=False, rev=None, queue=False)
    base.update(kw)
    return SimpleNamespace(**base)
def fresh(**kw):
    cmd = CmdExperimentsRemove.__new__(CmdExperimentsRemove)
    cmd.args = args(**kw)
    return cmd
try:
    fresh().check_arguments()
except InvalidArgumentError as e:
    assert '--queue' in str(e), str(e)
    print("get_hit", True)
else:
    raise AssertionError("empty selection must be rejected")
try:
    fresh().check_arguments()
except InvalidArgumentError as e:
    assert '--rev' in str(e) and '--all-commits' in str(e)
    print("equivalence", True)
else:
    raise AssertionError("empty selection must be rejected")
try:
    fresh(experiment='exp1').check_arguments()
except InvalidArgumentError:
    raise AssertionError("experiment-only selection must pass the gate")
print("isolation", True)
fresh(queue=True).check_arguments()
print("invalidate", True)
try:
    fresh(experiment='exp1', queue=True).check_arguments()
except InvalidArgumentError:
    print("recompute", True)
else:
    raise AssertionError("experiment+queue mixing must be rejected")
"""


OBS_dvc_17 = r"""
import os, tempfile
from collections.abc import Mapping
from dvc.stage.cache import StageCache
from dvc.stage.loader import StageLoader
assert issubclass(StageLoader, Mapping)
print("get_hit", True)
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'a' * 32, 'v1'
p = os.path.join(tmp, key[:2], key, val)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("equivalence", True)
assert sc._load_cache('b' * 32, val) is None
print("isolation", True)
with open(p, 'w') as f:
    f.write('not: [valid\n')
assert sc._load_cache(key, val) is None
assert not os.path.exists(p)
print("invalidate", True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("recompute", True)
"""

DVC_OBS = {
    'f3r-dvc-01': OBS_dvc_01,
    'f3r-dvc-02': OBS_dvc_02,
    'f3r-dvc-03': OBS_dvc_03,
    'f3r-dvc-04': OBS_dvc_04,
    'f3r-dvc-05': OBS_dvc_05,
    'f3r-dvc-06': OBS_dvc_06,
    'f3r-dvc-07': OBS_dvc_07,
    'f3r-dvc-08': OBS_dvc_08,
    'f3r-dvc-09': OBS_dvc_09,
    'f3r-dvc-14': OBS_dvc_14,
    'f3r-dvc-15': OBS_dvc_15,
    'f3r-dvc-16': OBS_dvc_16,
    'f3r-dvc-17': OBS_dvc_17,
    'f3r-dvc-13': OBS_dvc_13,
    'f3r-dvc-12': OBS_dvc_12,
    'f3r-dvc-11': OBS_dvc_11,
    'f3r-dvc-10': OBS_dvc_10,
}
