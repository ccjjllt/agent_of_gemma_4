"""F3-replace Conan observations: set->get->expire/invalidate->recompute + equivalence/isolation.

Each OBS_<key> runs after the source-binding preamble. It must FAIL (nonzero)
on the buggy tree and PASS on the repaired tree.
"""

OBS_conan_01 = r"""
import os, tempfile
from conan.api.model import RecipeReference
from conan.internal.cache.cache import PkgCache
real = tempfile.mkdtemp()
link = real + '_link'
os.symlink(real, link)
cache = PkgCache(link, {})
assert cache._base_folder == os.path.abspath(os.path.join(link, 'p')), cache._base_folder
print("get_hit", True)
ref = RecipeReference('pkg', '0.1')
layout = cache.create_export_recipe_layout(ref)
assert os.path.isdir(layout.base_folder), layout.base_folder
assert layout.base_folder.startswith(cache._base_folder)
print("equivalence", True)
layout2 = cache.create_export_recipe_layout(RecipeReference('pkg', '0.1'))
assert layout2.base_folder == layout.base_folder
print("isolation", True)
ref3 = RecipeReference('other', '0.1')
layout3 = cache.create_export_recipe_layout(ref3)
assert layout3.base_folder != layout.base_folder
print("recompute", True)
import shutil
shutil.rmtree(layout.base_folder)
assert not os.path.exists(layout.base_folder)
print("invalidate", True)
"""

OBS_conan_02 = r"""
import os, tempfile
from conan.api.model import RecipeReference
from conan.internal.cache.db.cache_database import CacheDatabase
tmp = tempfile.mkdtemp()
dbfile = os.path.join(tmp, 'cache.sqlite3')
db = CacheDatabase(filename=dbfile)
ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0)
from conan.internal.errors import ConanReferenceDoesNotExistInDB
db.create_recipe('t/abc', ref)
got = db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))
assert got['path'] == 't/abc', got
print("get_hit", True)
again = db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))
assert again['path'] == got['path']
print("equivalence", True)
try:
    db.get_recipe(RecipeReference('other', '0.1', revision='r1'))
except ConanReferenceDoesNotExistInDB:
    print("isolation", True)
else:
    raise AssertionError("unknown ref must miss")
db.remove_recipe(ref)
try:
    db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))
except ConanReferenceDoesNotExistInDB:
    print("invalidate", True)
else:
    raise AssertionError("removed row must miss")
db.create_recipe('t/abc', ref)
assert db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))['path'] == 't/abc'
print("recompute", True)
"""

OBS_conan_03 = r"""
import os, tempfile
from types import SimpleNamespace
from conans.client.downloaders.download_cache import DownloadCache
from conans.client.downloaders.caching_file_downloader import ConanInternalCacheDownloader
tmp = tempfile.mkdtemp()
dc = DownloadCache(tmp)
url = 'http://example.invalid/big.bin'
def cp(u):
    path, _h = dc.cached_path(u)
    return path
cpath = cp(url)
os.makedirs(os.path.dirname(cpath), exist_ok=True)
with open(cpath, 'wb') as f:
    f.write(b'z' * (10 * 1024 * 1024 + 1))
msgs = []
output = SimpleNamespace(info=msgs.append, warning=msgs.append)
dl = ConanInternalCacheDownloader(object(),
                                  config=SimpleNamespace(get=lambda *a, **k: tmp), scope='conan')
import io, contextlib
dest = os.path.join(tmp, 'out.bin')
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    dl.download(url, dest, auth=None, verify_ssl=True, retry=0, retry_wait=0)
assert os.path.getsize(dest) == 10 * 1024 * 1024 + 1
print("get_hit", True)
assert 'from download cache' in buf.getvalue(), buf.getvalue()[-300:]
print("equivalence", True)
buf2 = io.StringIO()
with contextlib.redirect_stderr(buf2):
    dl.download(url, dest, auth=None, verify_ssl=True, retry=0, retry_wait=0)
assert 'from download cache' in buf2.getvalue()
print("isolation", True)
os.unlink(cpath)
assert not os.path.exists(cp(url))
print("invalidate", True)
with open(cpath, 'wb') as f:
    f.write(b'q')
assert open(cpath, 'rb').read() == b'q'
print("recompute", True)
"""

OBS_conan_04 = r"""
import os
from conan.test.assets.genconanfile import GenConanfile
from conan.test.utils.tools import TestClient
c = TestClient()
c.save({'conanfile.py': GenConanfile('pkg', '0.1').with_exports('*').with_exports_sources('*'),
        'sorces/file.txt': ''})
c.run('create .')
ref_layout = c.exported_layout()
assert os.path.exists(ref_layout.source())
print("get_hit", True)
c.run('cache clean "*" -s -b')
assert not os.path.exists(ref_layout.source())
assert os.path.exists(ref_layout.download_export())
print("invalidate", True)
c.run('cache clean "*" -s -b')
print("equivalence", True)
c.run('create .')
assert os.path.exists(c.exported_layout().source())
print("recompute", True)
assert os.path.exists(c.exported_layout().download_export())
print("isolation", True)
"""

OBS_conan_05 = r"""
from conan.test.assets.genconanfile import GenConanfile
from conan.test.utils.tools import TestClient
c = TestClient()
c.save({'conanfile.py': GenConanfile('pkg', '0.1').with_exports('*').with_exports_sources('*'),
        'sorces/file.txt': ''})
c.run('create .')
ref_layout = c.exported_layout()
c.run('cache path pkg/0.1')
assert ref_layout.export() == str(c.out).rstrip()
print("get_hit", True)
c.run('cache path pkg/0.1 --folder=export_source')
assert ref_layout.export_sources() == str(c.out).rstrip()
print("equivalence", True)
c.run('cache path pkg/0.1 --folder=source')
assert ref_layout.source() == str(c.out).rstrip()
print("isolation", True)
c.run('remove pkg/0.1 -c')
c.run('cache path pkg/0.1 --folder=export_source', assert_error=True)
print("invalidate", True)
c.run('create .')
rl = c.exported_layout()
c.run('cache path pkg/0.1 --folder=export_source')
assert rl.export_sources() == str(c.out).rstrip()
print("recompute", True)
"""

OBS_conan_06 = r"""
import os, tempfile
from conan.api.model import RecipeReference, PkgReference
from conan.internal.cache.cache import PkgCache
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
def fresh_pref():
    return PkgReference(RecipeReference('pkg', '0.1', revision='r1'), 'pid123')
layout = cache.create_build_pkg_layout(fresh_pref())
assert os.path.isdir(layout.base_folder)
print("get_hit", True)
layout.reference.revision = 'prev1'
cache.assign_prev(layout)
print("equivalence", True)
layout2 = cache.create_build_pkg_layout(fresh_pref())
layout2.reference.revision = 'prev1'
cache.assign_prev(layout2)
assert os.path.isdir(layout2.base_folder)
print("isolation", True)
other = cache.create_build_pkg_layout(
    PkgReference(RecipeReference('other', '0.1', revision='r1'), 'pid999'))
assert other.base_folder != layout2.base_folder
print("recompute", True)
cache.remove_package_layout(layout2)
assert not os.path.exists(layout2.base_folder)
print("invalidate", True)
"""

OBS_conan_07 = r"""
from types import SimpleNamespace
from conans.client.graph.proxy import ConanProxy
calls = []
def fake_get(ref, remotes, update, check_update):
    calls.append(ref)
    return ('/layout', 'ok', 'remote1', ref)
app = SimpleNamespace(cache=SimpleNamespace(), remote_manager=SimpleNamespace(),
                      global_conf=SimpleNamespace(get=lambda *a, **k: None))
proxy = ConanProxy(app, editable_packages=[])
proxy._get_recipe = fake_get
r1 = proxy.get_recipe('pkg/0.1', [], False, False)
assert r1[0] == '/layout'
print("get_hit", True)
r2 = proxy.get_recipe('pkg/0.1', [], False, False)
assert r2 == r1
assert len(calls) == 1, f"second resolve must hit memo, got {len(calls)} fetches"
print("equivalence", True)
proxy.get_recipe('other/0.1', [], False, False)
assert len(calls) == 2
print("isolation", True)
proxy._resolved.clear()
proxy.get_recipe('pkg/0.1', [], False, False)
assert len(calls) == 3, "after invalidate the recipe must be recomputed"
print("invalidate", True)
r3 = proxy.get_recipe('pkg/0.1', [], False, False)
assert r3 == r1 and len(calls) == 3
print("recompute", True)
"""

OBS_conan_08 = r"""
import os, tempfile
from conan.api.model import RecipeReference
from conan.internal.cache.db.cache_database import CacheDatabase
tmp = tempfile.mkdtemp()
db = CacheDatabase(filename=os.path.join(tmp, 'c.sqlite3'))
ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0)
from conan.internal.errors import ConanReferenceDoesNotExistInDB
db.create_recipe('t/abc', ref)
assert db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))['path'] == 't/abc'
print("get_hit", True)
assert db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))['path'] == 't/abc'
print("equivalence", True)
try:
    db.get_recipe(RecipeReference('zzz', '9.9', revision='r1'))
except ConanReferenceDoesNotExistInDB:
    print("isolation", True)
else:
    raise AssertionError("unknown ref must miss")
db.remove_recipe(ref)
try:
    db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))
except ConanReferenceDoesNotExistInDB:
    print("invalidate", True)
else:
    raise AssertionError("removed row must miss")
db.create_recipe('t/abc', ref)
assert db.get_recipe(RecipeReference('pkg', '0.1', revision='r1'))['path'] == 't/abc'
print("recompute", True)
"""

OBS_conan_09 = OBS_conan_08

OBS_conan_10 = r"""
import os, tempfile
from conan.api.model import RecipeReference, PkgReference
from conan.internal.cache.db.cache_database import CacheDatabase
tmp = tempfile.mkdtemp()
db = CacheDatabase(filename=os.path.join(tmp, 'c.sqlite3'))
ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0)
db.create_recipe('t/abc', ref)
from conan.api.model import PkgReference as PR
from conan.internal.errors import ConanReferenceDoesNotExistInDB
pref = PR(ref, 'pid123', revision='prev1', timestamp=1.0)
db.create_package('p/xyz', pref, None)
assert db.get_package_lru(pref) is not None
print("get_hit", True)
db.update_package_timestamp(pref, path='p/xyz', build_id=None)
assert db.get_package_lru(pref) is not None
print("equivalence", True)
other = PR(RecipeReference('other', '0.1', revision='r1'), 'pid999', revision='prev9')
try:
    db.get_package_lru(other)
except ConanReferenceDoesNotExistInDB:
    print("isolation", True)
else:
    raise AssertionError("unknown pref must miss")
db.remove_package(pref)
try:
    db.get_package_lru(pref)
except ConanReferenceDoesNotExistInDB:
    print("invalidate", True)
else:
    raise AssertionError("removed package must miss")
db.create_package('p/xyz', pref, None)
assert db.get_package_lru(pref) is not None
print("recompute", True)
"""

OBS_conan_11 = r"""
from conan.test.utils.tools import TestClient
c = TestClient()
c.save({'conanfile.py': 'from conan import ConanFile\n'
                         'class Pkg(ConanFile):\n'
                         "    name = 'pkg'\n"
                         "    version = '0.1'\n"})
c.run('export .')
assert 'pkg/0.1' in c.out
print("get_hit", True)
c.run('export .')
print("equivalence", True)
c.run('cache clean pkg/0.1')
print("invalidate", True)
c.run('export .')
assert 'pkg/0.1' in c.out
print("recompute", True)
c.run('cache clean')
print("isolation", True)
"""

OBS_conan_12 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.api.model import RecipeReference
from conan.errors import ConanException
from conan.internal.cache.cache import PkgCache
from conans.client.graph.proxy import ConanProxy
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
ref = RecipeReference('pkg', '0.1')
layout = cache.create_export_recipe_layout(ref)
assert os.path.isdir(layout.base_folder)
print("get_hit", True)
assert cache.create_export_recipe_layout(RecipeReference('pkg', '0.1')).base_folder == layout.base_folder
print("equivalence", True)
assert cache.create_export_recipe_layout(RecipeReference('q', '2.0')).base_folder != layout.base_folder
print("isolation", True)
from conans.client.remote_manager import RemoteManager
store_p = os.path.join(tmp, 'p')
before = set(os.listdir(store_p))
mgr = RemoteManager(cache, auth_manager=SimpleNamespace(), home_folder=tmp)
def fake_call(remote, method, *a, **k):
    if method == 'get_recipe':
        raise ConanException('boom')
    raise AssertionError(method)
mgr._call_remote = fake_call
mgr._signer = SimpleNamespace(verify=lambda *a, **k: None)
rref = RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0)
remote = SimpleNamespace(remote_type='remote', name='myremote')
try:
    mgr.get_recipe(rref, remote)
except ConanException:
    pass
else:
    raise AssertionError("remote failure must propagate")
after = set(os.listdir(store_p))
assert after == before, f"failed fetch must clean the layout, leftover={after - before}"
print("invalidate", True)
layout2 = cache.create_export_recipe_layout(ref)
assert os.path.isdir(layout2.base_folder)
print("recompute", True)
"""

OBS_conan_13 = r"""
from conan.test.utils.tools import TestClient
c = TestClient()
c.save({'conanfile.py': 'from conan import ConanFile\n'
                         'class Pkg(ConanFile):\n'
                         "    name = 'pkg'\n"
                         "    version = '0.1'\n"})
c.run('export .')
assert 'pkg/0.1' in c.out
print("get_hit", True)
c.run('cache save pkg/0.1 --file backup.tgz')
import os
assert os.path.exists(os.path.join(c.current_folder, 'backup.tgz'))
print("equivalence", True)
c.run('remove pkg/0.1 -c')
from conan.api.model import RecipeReference
assert c.cache.get_recipe_revisions_references(RecipeReference('pkg', '0.1')) == []
print("invalidate", True)
c.run('cache restore backup.tgz')
c.run('list pkg/0.1:*')
assert 'pkg/0.1' in c.out
print("recompute", True)
c.run('cache save other/0.1 --file x.tgz', assert_error=True)
print("isolation", True)
"""

OBS_conan_14 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.errors import ConanException
from conans.client.downloaders.caching_file_downloader import SourcesCachingDownloader
tmp = tempfile.mkdtemp()
msgs = []
output = SimpleNamespace(info=msgs.append, warning=msgs.append, verbose=msgs.append,
                         debug=msgs.append)
def make_cf(backups):
    conf = {'core.sources:download_cache': tmp, 'core.sources:download_urls': backups}
    gconf = SimpleNamespace(get=lambda k, check_type=None: conf.get(k))
    helpers = SimpleNamespace(global_conf=gconf, requester=object(), home_folder=tmp)
    return SimpleNamespace(_conan_helpers=helpers, output=output, display_name='pkg/0.1')
sha = 'a' * 40
from conans.client.downloaders.download_cache import DownloadCache
seed = DownloadCache(tmp).source_path(sha)
os.makedirs(os.path.dirname(seed), exist_ok=True)
with open(seed, 'w') as f:
    f.write('payload-a')
dest = os.path.join(tmp, 'out.tgz')
dl = SourcesCachingDownloader(make_cf(['http://x/']))
dl._caching_download(['http://u/f'], dest, 0, 0, True, None, {}, None, None, sha, tmp, ['http://x/'])
assert open(dest).read() == 'payload-a'
print("get_hit", True)
dl._caching_download(['http://u/f'], dest, 0, 0, True, None, {}, None, None, sha, tmp, ['http://x/'])
assert open(dest).read() == 'payload-a'
print("equivalence", True)
sha2 = 'b' * 40
seed2 = DownloadCache(tmp).source_path(sha2)
with open(seed2, 'w') as f:
    f.write('payload-b')
dest2 = os.path.join(tmp, 'out2.tgz')
dl._caching_download(['http://u/g'], dest2, 0, 0, True, None, {}, None, None, sha2, tmp, ['http://x/'])
assert open(dest2).read() == 'payload-b'
print("isolation", True)
try:
    dl._caching_download(['http://u/f'], dest, 0, 0, True, None, {}, None, None, 'c' * 40, tmp, [None, 'http://x/'])
except ConanException as e:
    assert 'None backup remote' in str(e), e
    print("invalidate", True)
else:
    raise AssertionError("None backup remote must be rejected")
os.unlink(seed)
with open(seed, 'w') as f:
    f.write('payload-a')
dl._caching_download(['http://u/f'], dest, 0, 0, True, None, {}, None, None, sha, tmp, ['http://x/'])
assert open(dest).read() == 'payload-a'
print("recompute", True)
"""

OBS_conan_15 = r"""
import os, tempfile
from types import SimpleNamespace
from conans.client.downloaders.caching_file_downloader import SourcesCachingDownloader
tmp = tempfile.mkdtemp()
msgs = []
output = SimpleNamespace(info=msgs.append, warning=msgs.append)
conf = {'core.sources:download_cache': tmp, 'core.sources:download_urls': ['http://x/']}
gconf = SimpleNamespace(get=lambda k, check_type=None: conf.get(k))
helpers = SimpleNamespace(global_conf=gconf, requester=object(), home_folder=tmp)
cf = SimpleNamespace(_conan_helpers=helpers, output=output, display_name='pkg/0.1')
dl = SourcesCachingDownloader(cf)
seen = []
def fake_download(url, path, *a, **k):
    seen.append(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('x')
dl._file_downloader = SimpleNamespace(download=fake_download)
sha = 'd' * 40
cached = os.path.join(tmp, 'c.bin')
assert dl._backup_download('http://backup.local/bs', ['http://backup.local/bs'], sha, cached, ['http://u/f'], True) is True
assert seen[0] == 'http://backup.local/bs/' + sha, seen
print("get_hit", True)
seen.clear()
assert dl._backup_download('http://backup.local/bs', ['http://backup.local/bs'], sha, cached, ['http://u/f'], True) is True
assert seen[0] == 'http://backup.local/bs/' + sha
print("equivalence", True)
seen.clear()
assert dl._backup_download('http://other.local/o', ['http://other.local/o'], sha, cached, ['http://u/f'], True) is True
assert seen[0] == 'http://other.local/o/' + sha
print("isolation", True)
seen.clear()
assert dl._backup_download('http://backup.local/withslash/', ['http://backup.local/withslash/'], sha, cached, ['http://u/f'], True) is True
assert seen[0] == 'http://backup.local/withslash/' + sha, seen
print("invalidate", True)
assert open(cached).read() == 'x'
print("recompute", True)
"""


OBS_conan_16 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.internal.errors import NotFoundException
from conans.client.downloaders.download_cache import DownloadCache
from conans.client.downloaders.caching_file_downloader import SourcesCachingDownloader
tmp = tempfile.mkdtemp()
dc = DownloadCache(tmp)
path = dc.source_path('abc123')
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    f.write('src')
assert open(dc.source_path('abc123')).read() == 'src'
print("get_hit", True)
assert dc.source_path('abc123') == path
print("equivalence", True)
assert dc.source_path('def456') != path
print("isolation", True)
msgs = []
output = SimpleNamespace(info=msgs.append, warning=msgs.append)
conf = {'core.sources:download_cache': tmp, 'core.sources:download_urls': ['http://x/']}
gconf = SimpleNamespace(get=lambda k, check_type=None: conf.get(k))
helpers = SimpleNamespace(global_conf=gconf, requester=object(), home_folder=tmp)
cf = SimpleNamespace(_conan_helpers=helpers, output=output, display_name='pkg/0.1')
dl = SourcesCachingDownloader(cf)
def missing(url, path, *a, **k):
    raise NotFoundException('nope')
dl._file_downloader = SimpleNamespace(download=missing)
assert dl._backup_download('http://b/', ['http://b/'], 'e' * 40, os.path.join(tmp, 'c'), ['http://u/f'], False) is None
assert any("File ['http://u/f'] not found in http://b/" in m for m in msgs), msgs
print("miss_report", True)
with dc.lock('abc123'):
    pass
print("lock_contract", True)
os.unlink(path)
assert not os.path.exists(dc.source_path('abc123'))
print("invalidate", True)
with open(path, 'w') as f:
    f.write('src2')
assert open(dc.source_path('abc123')).read() == 'src2'
print("recompute", True)
"""

OBS_conan_17 = r"""
import json, os, tempfile
from conans.client.downloaders.download_cache import DownloadCache
tmp = tempfile.mkdtemp()
dc = DownloadCache(tmp)
assert dc.get_backup_sources_files([], None) == []
print("get_hit", True)
sdir = os.path.join(tmp, 's')
os.makedirs(sdir, exist_ok=True)
blob = os.path.join(sdir, 'blob1')
with open(blob, 'w') as f:
    f.write('payload')
with open(blob + '.json', 'w') as f:
    json.dump({'references': {'pkg/0.1': ['http://example.invalid/f']}, 'timestamp': 1}, f)
got = dc.get_backup_sources_files([], None)
assert blob in got, got
print("equivalence", True)
assert dc.get_backup_sources_files(['http://example.invalid'], None) == []
print("isolation", True)
class FakeList:
    def __init__(self, refs):
        self._refs = refs
    def refs(self):
        return self._refs
up = {'upload': False, 'revisions': {'r1': {'upload': False}}}
assert dc.get_backup_sources_files([], FakeList({'pkg/0.1': up}), only_upload=False) != []
print("upload_scope", True)
assert dc.get_backup_sources_files([], FakeList({'other/9.9': up}), only_upload=False) == []
print("ref_scope", True)
os.unlink(blob)
os.unlink(blob + '.json')
assert dc.get_backup_sources_files([], None) == []
print("invalidate", True)
"""


OBS_conan_18 = r"""
from conan.test.utils.tools import TestClient
c = TestClient()
c.save({'conanfile.py': 'from conan import ConanFile\n'
                         'class Pkg(ConanFile):\n'
                         "    name = 'pkg'\n"
                         "    version = '0.1'\n"})
c.run('export .')
assert 'pkg/0.1' in c.out
print("get_hit", True)
c.run('remove pkg/0.1 -c')
from conan.api.model import RecipeReference
assert c.cache.get_recipe_revisions_references(RecipeReference('pkg', '0.1')) == []
print("invalidate", True)
c.run('export .')
assert 'pkg/0.1' in c.out
print("recompute", True)
c.run('remove other/0.1 -c')
c.run('list pkg/0.1:*')
assert 'pkg/0.1' in c.out
print("isolation", True)
c.run('export .')
print("equivalence", True)
"""

OBS_conan_19 = r"""
import os, tempfile
from conan.api.model import RecipeReference
from conan.internal.cache.cache import PkgCache
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
ref = RecipeReference('pkg', '0.1')
layout = cache.create_export_recipe_layout(ref)
assert os.path.isdir(layout.base_folder)
print("get_hit", True)
assert cache.create_export_recipe_layout(RecipeReference('pkg', '0.1')).base_folder == layout.base_folder
print("equivalence", True)
assert cache.create_export_recipe_layout(RecipeReference('z', '9.9')).base_folder != layout.base_folder
print("isolation", True)
from conan.internal.errors import ConanReferenceDoesNotExistInDB
try:
    cache.get_recipe_lru(ref)
except ConanReferenceDoesNotExistInDB:
    print("invalidate", True)
else:
    raise AssertionError("unregistered recipe must miss lru")
import shutil
shutil.rmtree(layout.base_folder)
back = cache.create_export_recipe_layout(ref)
assert os.path.isdir(back.base_folder)
print("recompute", True)
"""


OBS_conan_20 = r"""
from types import SimpleNamespace
from conans.client.graph.python_requires import PyRequireLoader
fetches = []
class FakeLayout:
    def __init__(self, ref):
        self.reference = ref
    def conanfile(self):
        return '/tmp/fake-pyreq/conanfile.py'
class FakeProxy:
    def get_recipe(self, ref, remotes, update, check_update):
        fetches.append(str(ref))
        return (FakeLayout(ref), 'ok', SimpleNamespace(name='myremote'))
class FakeRangeResolver:
    def resolve(self, requirement, *args, **kwargs):
        pass
class FakeLoader:
    def load_basic_module(self, path, graph_lock, remotes=None, update=None, check_update=None):
        return (SimpleNamespace(), SimpleNamespace())
class FakeLock:
    def __init__(self):
        self.calls = []
    def __bool__(self):
        return True
    def resolve_locked_pyrequires(self, requirement, *args):
        self.calls.append(len(args))
app = SimpleNamespace(proxy=FakeProxy(), range_resolver=FakeRangeResolver())
conf = SimpleNamespace(get=lambda *a, **k: None)
loader = PyRequireLoader(app, conf)
assert loader._cached_py_requires == {}
assert hasattr(loader, '_resolve_prereleases'), "prerelease config state must exist"
print("get_hit", True)
def consumer(refstr):
    return SimpleNamespace(python_requires=refstr)
r1 = consumer('pkg/0.1')
loader.load_py_requires(r1, FakeLoader(), None, [], False, False)
assert len(fetches) == 1, fetches
first = r1.python_requires['pkg']
print("equivalence", True)
r1b = consumer('pkg/0.1')
loader.load_py_requires(r1b, FakeLoader(), None, [], False, False)
assert len(fetches) == 1, f"second load must hit cache, got {fetches}"
assert r1b.python_requires['pkg'] is first
print("cache_hit", True)
r2 = consumer('other/0.1')
loader.load_py_requires(r2, FakeLoader(), None, [], False, False)
assert len(fetches) == 2, fetches
print("isolation", True)
lock = FakeLock()
r3 = consumer('pkg/0.1')
loader.load_py_requires(r3, FakeLoader(), lock, [], False, False)
assert lock.calls == [1], f"locked resolve must thread prerelease config, got {lock.calls}"
print("locked", True)
loader._cached_py_requires.clear()
r4 = consumer('pkg/0.1')
loader.load_py_requires(r4, FakeLoader(), None, [], False, False)
assert len(fetches) == 3, f"after invalidate must refetch, got {fetches}"
print("invalidate", True)
r5 = consumer('pkg/0.1')
loader.load_py_requires(r5, FakeLoader(), None, [], False, False)
assert len(fetches) == 3
assert r5.python_requires['pkg'] is r4.python_requires['pkg']
print("recompute", True)
"""
OBS_conan_21 = r"""
import os, tempfile
from conan.test.assets.genconanfile import GenConanfile
from conan.test.utils.tools import TestClient
tmp = tempfile.mkdtemp()
pl1 = os.path.join(tmp, 'pkglist.json')
pl2 = os.path.join(tmp, 'pkglist2.json')
pl3 = os.path.join(tmp, 'pkglist3.json')
c = TestClient()
c.save({'conanfile.py': GenConanfile('pkg', '0.1')})
c.run('export .')
c.run('list *#*:* --format=json', redirect_stdout=pl1)
assert 'pkg/0.1' in open(pl1).read()
print("get_hit", True)
c.run(f'remove --list={pl1} -c')
assert "No binaries to remove for 'pkg/0.1" in c.out
print("equivalence", True)
c.run('list * --format=json', redirect_stdout=pl2)
c.run(f'remove --list={pl2} -c')
print("isolation", True)
c.run('export .')
c.run('list *#*:* --format=json', redirect_stdout=pl3)
c.run(f'remove --list={pl3} -c')
assert "No binaries to remove for 'pkg/0.1" in c.out
print("invalidate", True)
c.run('remove pkg/0.1 -c')
from conan.api.model import RecipeReference
assert c.cache.get_recipe_revisions_references(RecipeReference('pkg', '0.1')) == []
print("removed", True)
c.run('export .')
c.run('list *#*:* --format=json', redirect_stdout=pl1)
assert 'pkg/0.1' in open(pl1).read()
print("recompute", True)
"""

OBS_conan_22 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.api.model import RecipeReference
from conan.internal.cache.cache import PkgCache
import conan.api.subapi.download as dlmod
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
calls = []
server_ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=9.0)
def fake_get_recipe(ref, remote, metadata):
    calls.append('get_recipe')
    cache.create_ref_layout(ref)
rm = SimpleNamespace(
    get_recipe_revision_reference=lambda ref, remote: server_ref,
    get_recipe=fake_get_recipe,
    get_recipe_sources=lambda *a, **k: calls.append('get_recipe_sources'))
loader = SimpleNamespace(load_basic=lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not load')))
app = SimpleNamespace(cache=cache, remote_manager=rm, loader=loader)
dlmod.ConanBasicApp = lambda api: app
from conan.api.subapi.download import DownloadAPI
api = DownloadAPI(SimpleNamespace())
ref = RecipeReference('pkg', '0.1', revision='r1')
remote = SimpleNamespace(name='myremote')
assert api.recipe(ref, remote) is True
assert ref.timestamp == 9.0, ref.timestamp
print("get_hit", True)
assert 'get_recipe_sources' in calls, calls
print("equivalence", True)
calls.clear()
ref2 = RecipeReference('other', '0.2', revision='r2')
server_ref2 = RecipeReference('other', '0.2', revision='r2', timestamp=7.0)
rm.get_recipe_revision_reference = lambda ref, remote: server_ref2
assert api.recipe(ref2, remote) is True
assert 'get_recipe_sources' in calls
print("isolation", True)
from conan.errors import ConanException as _CE
cache._db.remove_recipe(ref2)
try:
    cache.recipe_layout(ref2)
except _CE:
    print("invalidate", True)
else:
    raise AssertionError("removed recipe must miss")
calls.clear()
assert api.recipe(ref2, remote) is True
assert 'get_recipe_sources' in calls
print("recompute", True)
"""

OBS_conan_23 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.api.model import RecipeReference
from conan.errors import ConanException
from conan.internal.cache.cache import PkgCache
from conans.client.remote_manager import RemoteManager
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
mgr = RemoteManager(cache, auth_manager=SimpleNamespace(), home_folder=tmp)
mgr._signer = SimpleNamespace(verify=lambda *a, **k: None)
remote = SimpleNamespace(remote_type='remote', name='myremote')
def corrupt(remote, method, *a, **k):
    if method == 'get_recipe':
        return {'other.txt': b'x'}
    raise AssertionError(method)
mgr._call_remote = corrupt
ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0)
try:
    mgr.get_recipe(ref, remote)
except ConanException as e:
    assert 'no conanfile.py' in str(e), e
    print("get_hit", True)
else:
    raise AssertionError("corrupt payload must be rejected")
def valid(remote, method, *a, **k):
    if method == 'get_recipe':
        d = a[1]
        os.makedirs(d, exist_ok=True)
        paths = {}
        for name in ('conanfile.py', 'conanmanifest.txt'):
            p = os.path.join(d, name)
            with open(p, 'w') as f:
                f.write(name)
            paths[name] = p
        return paths
    raise AssertionError(method)
mgr._call_remote = valid
layout = mgr.get_recipe(RecipeReference('q', '9.9', revision='r1', timestamp=1.0), remote)
assert os.path.isfile(os.path.join(layout.export(), 'conanfile.py'))
print("equivalence", True)
layout2 = mgr.get_recipe(RecipeReference('w', '8.8', revision='r1', timestamp=1.0), remote)
assert layout2.base_folder != layout.base_folder
print("isolation", True)
mgr._call_remote = corrupt
try:
    mgr.get_recipe(RecipeReference('e', '7.7', revision='r1', timestamp=1.0), remote)
except ConanException as e:
    assert 'no conanfile.py' in str(e)
    print("invalidate", True)
else:
    raise AssertionError("corrupt payload must be rejected")
mgr._call_remote = valid
layout3 = mgr.get_recipe(RecipeReference('pkg', '0.1', revision='r1', timestamp=1.0), remote)
assert os.path.isfile(os.path.join(layout3.export(), 'conanfile.py'))
print("recompute", True)
"""


OBS_conan_24 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.api.model import RecipeReference
from conan.errors import ConanException as _CE
from conan.internal.cache.cache import PkgCache
import conan.api.subapi.download as dlmod
tmp = tempfile.mkdtemp()
cache = PkgCache(tmp, {})
order = []
ts_updates = []
server_ref = RecipeReference('pkg', '0.1', revision='r1', timestamp=9.0)
def fake_revref(ref, remote):
    order.append('revref')
    return server_ref
from conan.internal.errors import ConanReferenceAlreadyExistsInDB as _Exists
def fake_get_recipe(ref, remote, metadata):
    order.append('get_recipe')
    try:
        cache.create_ref_layout(ref)
    except _Exists:
        pass
rm = SimpleNamespace(
    get_recipe_revision_reference=fake_revref,
    get_recipe=fake_get_recipe,
    get_recipe_sources=lambda *a, **k: order.append('get_recipe_sources'))
orig_update = cache._db.update_recipe_timestamp
def rec_update(ref):
    ts_updates.append(str(ref))
    return orig_update(ref)
cache._db.update_recipe_timestamp = rec_update
loader = SimpleNamespace(load_basic=lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not load')))
app = SimpleNamespace(cache=cache, remote_manager=rm, loader=loader)
dlmod.ConanBasicApp = lambda api: app
from conan.api.subapi.download import DownloadAPI
api = DownloadAPI(SimpleNamespace())
ref = RecipeReference('pkg', '0.1', revision='r1')
remote = SimpleNamespace(name='myremote')
assert api.recipe(ref, remote) is True
assert ref.timestamp == 9.0, ref.timestamp
print("get_hit", True)
assert order[:3] == ['revref', 'get_recipe', 'get_recipe_sources'], order
assert ts_updates == []
print("equivalence", True)
order.clear()
ref2 = RecipeReference('other', '0.2', revision='r2')
server_ref2 = RecipeReference('other', '0.2', revision='r2', timestamp=7.0)
rm.get_recipe_revision_reference = lambda r, m: (order.append('revref'), server_ref2)[1]
assert api.recipe(ref2, remote) is True
assert ref2.timestamp == 7.0
print("isolation", True)
cache._db.remove_recipe(ref2)
try:
    cache.recipe_layout(ref2)
except _CE:
    print("invalidate", True)
else:
    raise AssertionError("removed recipe must miss")
order.clear()
ref3 = RecipeReference('other', '0.2', revision='r2')
assert api.recipe(ref3, remote) is True
assert ref3.timestamp == 7.0
assert order[:3] == ['revref', 'get_recipe', 'get_recipe_sources'], order
print("recompute", True)
"""


OBS_conan_25 = r"""
import os, tempfile
from types import SimpleNamespace
from conan.api.model import RecipeReference
from conan.api.subapi.cache import CacheAPI
from conans.client.downloaders.download_cache import DownloadCache
tmp = tempfile.mkdtemp()
api = SimpleNamespace(cache_folder=tmp, config=SimpleNamespace(global_conf={}))
cache = CacheAPI(api)
assert cache.conan_api is api
print("get_hit", True)
dc = DownloadCache(tmp)
url = 'http://example.invalid/f.tgz'
def cp(u):
    path, _h = dc.cached_path(u)
    return path
cpath = cp(url)
os.makedirs(os.path.dirname(cpath), exist_ok=True)
with open(cpath, 'w') as f:
    f.write('data')
assert open(cp(url)).read() == 'data'
print("equivalence", True)
assert cp(url + '?x=1') != cpath
print("isolation", True)
os.unlink(cpath)
assert not os.path.exists(cp(url))
print("invalidate", True)
os.makedirs(os.path.dirname(cpath), exist_ok=True)
with open(cpath, 'w') as f:
    f.write('data2')
assert open(cp(url)).read() == 'data2'
print("recompute", True)
"""


OBS_conan_27 = r"""
import os
import pytest
from conan.tools.files import download
from conan.errors import ConanException
from conan.test.utils.file_server import TestFileServer
from conan.test.utils.mocks import ConanFileMock, RedirectedTestOutput
from conan.test.utils.test_files import temp_folder
from conan.test.utils.tools import redirect_output, TestRequester
from conans.util.files import chdir
work = temp_folder()
os.chdir(work)
conanfile = ConanFileMock()
conanfile._conan_helpers.requester = TestRequester({})
try:
    output = RedirectedTestOutput()
    with redirect_output(output):
        download(conanfile, "http://fakesomething", "path", retry=2, retry_wait=0.1)
except ConanException:
    pass
else:
    raise AssertionError("failing download must raise")
assert str(output).count("Waiting 0.1 seconds to retry...") == 2, str(output)
print("get_hit", True)
conanfile2 = ConanFileMock()
conanfile2._conan_helpers.requester = TestRequester({})
try:
    output2 = RedirectedTestOutput()
    with redirect_output(output2):
        download(conanfile2, "http://fakesomething", "path", retry=2, retry_wait=0.1)
except ConanException:
    pass
else:
    raise AssertionError("failing download must raise")
assert str(output2).count("Waiting 0.1 seconds to retry...") == 2
print("equivalence", True)
file_server = TestFileServer()
conanfile3 = ConanFileMock()
conanfile3._conan_helpers.requester = TestRequester({"file_server": file_server})
file_path = os.path.join(temp_folder(), "file.txt")
try:
    download(conanfile3, file_server.fake_url + "/manual.html", file_path, retry=2, retry_wait=0)
except ConanException:
    pass
else:
    raise AssertionError("not-found must raise")
assert "Waiting" not in str(conanfile3.output)
assert "retry" not in str(conanfile3.output)
print("isolation", True)
"""


OBS_conan_28 = r"""
import os, textwrap
from conan.test.utils.file_server import TestFileServer
from conan.test.utils.test_files import temp_folder
from conan.test.utils.tools import TestClient
from conans.util.files import save, rmdir, mkdir
c = TestClient(default_server_user=True, light=True)
file_server = TestFileServer()
c.servers["file_server"] = file_server
dl_cache = temp_folder()
http_internet = os.path.join(file_server.store, "internet")
save(os.path.join(http_internet, "myfile.txt"), "Hello, world!")
sha = "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
c.save_home({"global.conf": f"core.sources:download_cache={dl_cache}\n"
                            f"core.sources:download_urls=['origin', '{file_server.fake_url}/backups/']\n"
                            f"core.sources:upload_url={file_server.fake_url}/backups/"})
conanfile = ("from conan import ConanFile\n"
             "from conan.tools.files import download\n"
             "class Pkg2(ConanFile):\n"
             "    name = \"pkg\"\n"
             "    version = \"1.0\"\n"
             "    def source(self):\n"
             f"        download(self, \"{file_server.fake_url}/internet/myfile.txt\", \"myfile.txt\",\n"
             f"                 sha256=\"{sha}\")\n")
c.save({"conanfile.py": conanfile})
c.run("create .")
backups = os.path.join(file_server.store, "backups")
c.run("upload * -c -r=default")
assert sha in os.listdir(backups), os.listdir(backups)
assert sha + ".json" in os.listdir(backups)
print("get_hit", True)
c.run("upload * -c -r=default")
assert "already in server, skipping upload" in c.out
print("equivalence", True)
c.run("source .")
assert "retrieved from local download cache" in c.out or "found in origin" in c.out
print("isolation", True)
rmdir(dl_cache)
mkdir(dl_cache)
c.save_home({"global.conf": f"core.sources:download_cache={dl_cache}\n"
                            f"core.sources:download_urls=['{file_server.fake_url}/backups/', 'origin']\n"
                            f"core.sources:upload_url={file_server.fake_url}/backups/"})
c.run("source .")
assert "found in remote backup" in c.out, c.out[-500:]
print("invalidate", True)
c.run("source .")
assert "retrieved from local download cache" in c.out
print("recompute", True)
"""

CONAN_OBS = {
    'f3r-conan-01': OBS_conan_01,
    'f3r-conan-02': OBS_conan_02,
    'f3r-conan-03': OBS_conan_03,
    'f3r-conan-04': OBS_conan_04,
    'f3r-conan-05': OBS_conan_05,
    'f3r-conan-06': OBS_conan_06,
    'f3r-conan-07': OBS_conan_07,
    'f3r-conan-08': OBS_conan_08,
    'f3r-conan-09': OBS_conan_09,
    'f3r-conan-10': OBS_conan_10,
    'f3r-conan-11': OBS_conan_11,
    'f3r-conan-12': OBS_conan_12,
    'f3r-conan-13': OBS_conan_13,
    'f3r-conan-14': OBS_conan_14,
    'f3r-conan-15': OBS_conan_15,
    'f3r-conan-16': OBS_conan_16,
    'f3r-conan-17': OBS_conan_17,
    'f3r-conan-18': OBS_conan_18,
    'f3r-conan-19': OBS_conan_19,
    'f3r-conan-23': OBS_conan_23,
    'f3r-conan-24': OBS_conan_24,
    'f3r-conan-25': OBS_conan_25,
    'f3r-conan-28': OBS_conan_28,
    'f3r-conan-27': OBS_conan_27,
    'f3r-conan-22': OBS_conan_22,
    'f3r-conan-21': OBS_conan_21,
    'f3r-conan-20': OBS_conan_20,
}
