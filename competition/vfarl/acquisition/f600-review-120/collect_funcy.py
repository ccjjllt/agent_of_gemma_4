"""Collect source-reviewed cache/state contracts with the real official tools.

The plans are individually reviewed source edits and executable observations.
This does not train a model or change upstream verification requirements.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / 'competition/vfarl/replay'
sys.path.insert(0, str(REPLAY))
spec = importlib.util.spec_from_file_location('source_collector', ROOT / 'competition/vfarl/acquisition/s600-common-root/collect.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.HERE = HERE
c.ART = REPLAY / 'artifacts/f600-review-120'
c.REPORT_PREFIX = 'f600-review-'
c.REVIEW_SCOPE = 'F specialist source-reviewed cache lifecycle contracts; original train requirements and sources unchanged.'
c.DEFAULT_COURSE = 'F600_specialist'

PLANS = [
 ('d59n73ab', 'CacheMemory.__setitem__', 'inserted_value_and_expiration_metadata_keep_matching_order',
  'Store cache values with their expiration time and matching key queues, so fresh values are reused and expired values are invalidated.',
  "from funcy.calc import CacheMemory\nimport time\nc=CacheMemory(0.01)\nc['old']='value'\nassert c['old']=='value'\ntime.sleep(0.03)\ntry:c['old']\nexcept KeyError:pass\nelse:raise AssertionError('expired entry reused')\nc.timeout=10\nc['fresh']='new-value'\nassert c['fresh']=='new-value' and 'old' not in c\nassert list(c._keys)==['fresh']\nprint('value_metadata_expiry_and_recompute')"),
 ('8vv3z712', '_memory_decorator.decorator.invalidate_all', 'invalidate_all_clears_original_shared_memory',
  'Invalidate all entries in the original memoized function memory, so later same-key calls recompute independently.',
  "from funcy import memoize\ncalls=[]\n@memoize\ndef f(x):\n calls.append(x);return len(calls)\nassert f('a')==f('a')==1\nassert f('b')==2\nf.invalidate_all()\nassert f.memory=={}\nassert f('a')==3 and f('a')==3\nassert calls==['a','b','a']\nprint('invalidation_and_recompute',calls)"),
 ('ie8id75b', 'cache', 'timeout_controls_freshness_without_hidden_extension',
  'Cache timeout in seconds or timedelta defines actual expiration, including immediate expiry for negative timeout; no hidden extra second.',
  "from funcy import cache\nfrom datetime import timedelta\nimport time\nfor timeout in [0.01,timedelta(seconds=-1)]:\n calls=[]\n @cache(timeout)\n def f(x):\n  calls.append(x);return len(calls)\n first=f('a')\n time.sleep(0.03)\n second=f('a')\n assert (first,second)==(1,2),(timeout,first,second)\n assert calls==['a','a']\nprint('timeout_expiry_recomputes')"),
 ('thpitv9q', 'CacheMemory.clear', 'clear_resets_data_and_expiry_queue_together',
  'Clearing cache storage resets its entry and expiration queues together, then permits fresh insert and expire cycles.',
  "from funcy.calc import CacheMemory\nfrom collections import deque\nc=CacheMemory(0.01)\nc['a']='A'\nc.clear()\nassert not c and isinstance(c._keys,deque) and isinstance(c._expires,deque)\nassert len(c._keys)==len(c._expires)==0\nc['b']='B'\nassert c['b']=='B'\nimport time\ntime.sleep(0.03)\nc.expire()\nassert not c and not c._keys and not c._expires\nprint('clear_insert_expire_cycle')"),
 ('vrlfk4ah', 'memoize', 'memoized_function_binding_and_custom_key_equivalence',
  'Bare memoize binds the wrapped function, while configured key_func defines equivalent cache keys and invalidation.',
  "from funcy import memoize\ncalls=[]\n@memoize\ndef f(x):\n calls.append(x);return x+1\nassert f(2)==f(2)==3 and calls==[2]\ncount=[]\n@memoize(key_func=lambda word:word.lower())\ndef g(word):\n count.append(word);return len(count)\nassert g('A')==g('a')==1\ng.invalidate('a')\nassert g('A')==2 and count==['A','A']\nprint('key_equivalence_and_invalidate',calls,count)"),
 ('xyuc89kw', 'CacheMemory.expire', 'expire_evicts_matching_oldest_keys_only',
  'Expiration evicts only expired entries in matching FIFO order; live keys remain reusable and evicted keys can be recomputed.',
  "from funcy.calc import CacheMemory\nimport time\nc=CacheMemory(0.01)\nc['old']='old-value'\ntime.sleep(0.03)\nc.timeout=10\nc['live']='live-value'\nc.expire()\nassert 'old' not in c and c['live']=='live-value'\nc['old']='recomputed'\nassert c['old']=='recomputed' and c['live']=='live-value'\nassert len(c._keys)==len(c._expires)==2\nprint('expired_key_isolation_and_recompute')"),
 ('rmnnpv0q', '_make_lookuper.make_lookuper', 'lookup_table_populates_once_and_retains_missing_mode',
  'Populate the table once for repeated key lookups; parameterized lookup tables have independent argument scopes and missing-key mode stays strict or silent.',
  "from funcy import make_lookuper,silent_lookuper\ncalls=[]\n@make_lookuper\ndef lookup():\n calls.append(1);return {'a':3,'b':4}\nassert lookup('a')==3 and lookup('b')==4 and len(calls)==1\ntry:lookup('missing')\nexcept LookupError:pass\nelse:raise AssertionError('missing key accepted')\nassert len(calls)==1\n@silent_lookuper\ndef silent():return {'a':1}\nassert silent('missing') is None\n@make_lookuper\ndef scoped(prefix):return {prefix:prefix.upper()}\nassert scoped('x')('x')=='X' and scoped('y')('y')=='Y'\nprint('single_population_scope_and_missing_mode')"),
]

def configs():
    tasks = [dict(key='funcy-'+suffix, instance_id='Suor__funcy.207a7810.func_basic__'+suffix,
        canonical_repo='Suor/funcy', snapshot='207a7810c216c7408596d463d3f429686e83b871',
        relative_path='funcy/calc.py', symbol=symbol, family_contract=family, contract=contract,
        body=body, primary_adapter='F', submechanism='F3', source_license='BSD-3-Clause',
        task_profile='.venv-cache-task/lib/python3.12/site-packages')
        for suffix,symbol,family,contract,body in PLANS]
    for task in tasks:
        if task['key']=='funcy-d59n73ab':
            task['curated_prompt']='CacheMemory must return newly stored values while fresh, expire entries after their configured timeout, and remain usable for later insertions. Expiration must keep keys and expiration metadata aligned so that live entries are preserved.'
    return tasks

if __name__ == '__main__':
    (HERE / 'artifacts/pages').mkdir(parents=True, exist_ok=True)
    c.ART.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / 'competition/vfarl/acquisition/expand-cache/artifacts/pages').glob('Suor__funcy-*.json'):
        target = HERE / 'artifacts/pages' / p.name
        if not target.exists():
            target.symlink_to(p)
    stage = sys.argv[1]
    selected = configs()
    if len(sys.argv)>2:
        selected = [x for x in selected if x['key'] in sys.argv[2:]]
    for config in selected:
        if stage == 'prepare':
            c.prepare(config)
            print('prepared',config['key'],flush=True)
        elif stage == 'replay':
            c.replay(config)
            print('replayed',config['key'],flush=True)
    if stage == 'publish':
        initial = [x for x in selected if x['key'] != 'funcy-d59n73ab']
        storage = [x for x in selected if x['key'] == 'funcy-d59n73ab']
        if initial:
            c.publish(initial, review_name='reviewed-funcy-cache-candidates')
        if storage:
            c.publish(storage, review_name='reviewed-funcy-storage-candidate-v1')
