"""Collect explicitly reviewed cache and resource lifecycle train tasks."""
from pathlib import Path
import importlib.util
import sys
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / 'competition/vfarl/replay'
spec = importlib.util.spec_from_file_location('reviewed_boltons_collector', HERE.parent / 's600-specialist-a/collect.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.HERE = HERE
c.ART = REPLAY / 'artifacts/f600-review-120-boltons'

CACHE_BODY = """from boltons.cacheutils import LRI
c=LRI(max_size=2)
c['a']=10;c['b']=20
assert c['a']==10 and c['b']==20
c['c']=30
assert 'a' not in c and c['b']==20 and c['c']==30
c.pop('b')
assert 'b' not in c
c['b']=200
assert c['b']==200
c.clear()
assert len(c)==0
c['fresh']=99
assert c['fresh']==99
print('bounded_cache_eviction_invalidation_reuse')
"""
PLANS = [
 ('ptcijw82','F3','LRI._init_ll','cacheutils','cache_link_index_initialization_and_fresh_reset',CACHE_BODY),
 ('73363orf','F3','LRI.__getitem__','cacheutils','cache_hit_returns_value_and_tracks_miss_separately',CACHE_BODY),
 ('7yzbcjts','F3','LRI._set_key_and_add_to_front_of_ll','cacheutils','cache_insert_links_exact_key_value_before_eviction',CACHE_BODY),
 ('9k2u5lj0','F3','LRI._set_key_and_evict_last_in_ll','cacheutils','cache_eviction_replaces_oldest_key_only',CACHE_BODY),
 ('ci4vrwhc','F3','LRI.pop','cacheutils','cache_pop_invalidates_mapping_and_eviction_index',
  "from boltons.cacheutils import LRI\nc=LRI(max_size=2,values={'a':10,'b':20})\nassert c.pop('a',99)==10\nassert 'a' not in c and 'a' not in c._link_lookup\nc['c']=30;c['a']=100\nassert c['a']==100 and 'b' not in c\nassert c.pop('missing',99)==99\ntry:c.pop('missing')\nexcept KeyError:pass\nelse:raise AssertionError('missing cache key accepted')\nprint('pop_invalidate_recompute',dict(c))"),
 ('gtphibcc','F3','cachedmethod','cacheutils','shared_method_cache_keys_retain_function_scope',
  "from boltons.cacheutils import cachedmethod\nclass Box:\n def __init__(self):self.cache={};self.calls=[]\n @cachedmethod('cache')\n def first(self,value):self.calls.append(('first',value));return value+1\n @cachedmethod('cache')\n def second(self,value):self.calls.append(('second',value));return value+2\nb=Box()\nassert b.first(3)==4 and b.second(3)==5\nassert b.first(3)==4 and b.second(3)==5 and len(b.calls)==2\nb.cache.clear()\nassert b.second(3)==5 and len(b.calls)==3\nprint('same_args_distinct_method_scope',b.calls)"),
 ('1zqlbquo','F4','BufferedSocket.close','socketutils','socket_close_releases_handle_and_discards_pending_buffers',
  "from boltons.socketutils import BufferedSocket\nimport socket\na,b=socket.socketpair()\ns=BufferedSocket(a)\ns.rbuf=b'pending';s.rbuf_unconsumed=b'pending';s.buffer(b'queued')\ns.close()\nassert a.fileno()==-1 and s.rbuf==s.rbuf_unconsumed==b'' and not s.sbuf\nb.close()\nprint('socket_handle_closed_buffers_cleared')"),
 ('vo1adapj','F4','BufferedSocket.flush','socketutils','flush_drains_pending_output_without_inventing_payload',
  "from boltons.socketutils import BufferedSocket\nimport socket\na,b=socket.socketpair()\nb.settimeout(0.05)\ns=BufferedSocket(a)\ns.buffer(b'queued');result=s.flush()\nassert result is None and b.recv(99)==b'queued'\ntry:b.recv(1)\nexcept socket.timeout:pass\nelse:raise AssertionError('flush appended unexpected payload')\ns.close();b.close()\nprint('flush_exact_pending_output')"),
 ('7s8xmpcv','F4','SpooledIOBase._checkClosed','ioutils','stream_operations_reject_after_resource_close',
  "from boltons.ioutils import SpooledBytesIO\nf=SpooledBytesIO();f.write(b'abc');f.seek(0)\nassert f.read()==b'abc'\nf.close()\nfor operation in [f.read,lambda:f.write(b'x')]:\n try:operation()\n except ValueError:pass\n else:raise AssertionError('closed stream accepted I/O')\nprint('open_use_then_close_rejects_io')"),
 ('c7ls3siv','F4','SpooledIOBase.__enter__','ioutils','resource_context_entry_returns_same_live_owner',
  "from boltons.ioutils import SpooledBytesIO\nf=SpooledBytesIO()\nwith f as live:\n assert live is f\n live.write(b'abc')\nassert f.closed\ntry:\n with f:pass\nexcept ValueError:pass\nelse:raise AssertionError('closed resource entered')\nprint('context_live_identity_and_closed_exit')"),
]
PROMPTS = {
 'ptcijw82':'A bounded LRI cache must start empty, store values under their keys, evict the oldest inserted key when capacity is exceeded, and support reuse after clear().',
 '73363orf':'LRI cache lookups must return the stored value, preserve missing-key behavior, and maintain separate hit and miss counters. Reinsertions after invalidation must return the new value.',
 '7yzbcjts':'LRI cache insertion must preserve each key/value association and insertion order. Adding a third key to a capacity-two cache evicts the oldest key while preserving the two live values.',
 '9k2u5lj0':'When a bounded LRI cache reaches capacity, inserting a new key must evict exactly the oldest inserted key and retain correct values for the other keys. The cache must remain usable afterward.',
 'ci4vrwhc':'LRI.pop must remove an existing cache key and return its value even when a default is supplied. Missing keys use the supplied default or raise KeyError. Later insertion and eviction must continue to work.',
 'gtphibcc':'Two cached methods using the same object cache must retain independent results when called with the same arguments. Repeated calls reuse each method result, and clearing the cache permits recomputation.',
 '1zqlbquo':'BufferedSocket.close must release its underlying socket and discard its receive and send buffers. After close the socket handle must be closed and pending buffered data must no longer remain.',
 'vo1adapj':'BufferedSocket.flush must send the buffered output exactly once and add no bytes of its own. An empty flush does not create extra payload, and the method returns None.',
 '7s8xmpcv':'SpooledBytesIO must allow reads and writes while open and reject them with ValueError after close. This must work for ordinary use followed by resource release.',
 'c7ls3siv':'A SpooledBytesIO context must bind the same live stream object, support I/O inside the context, close it when exiting, and reject entering an already closed stream.',
}

if __name__ == '__main__':
    stage=sys.argv[1]
    chosen=PLANS if len(sys.argv)<3 else [p for p in PLANS if p[0] in sys.argv[2:]]
    if stage=='prepare':
        if (HERE/'reviewed-boltons-cache-resource.json').exists():
            raise ValueError('This source review is frozen; use a new batch identity to prepare a revision')
        c.prepare_boltons(chosen, review_filename='reviewed-boltons-cache-resource.json',
            curation_prompts=PROMPTS, primary_by_suffix={p[0]:'F' for p in chosen},course_membership='F600_specialist')
        import json
        path=HERE/'reviewed-boltons-cache-resource.json'
        prepared=json.loads(path.read_text())
        for task in prepared['tasks']:
            task['source_import_alias_group']='zz26'
        c.save(path,prepared)
    elif stage in {'replay','replay2'}:
        c.run(batch='boltons',attempt=2 if stage=='replay2' else 1,review_override=HERE/'reviewed-boltons-cache-resource.json')
