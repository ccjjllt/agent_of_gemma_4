"""Replace a duplicate DeferredValue family with one new cache-pop contract."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OBSERVATION="from boltons.cacheutils import LRI\nc=LRI(max_size=2,values={'a':1,'b':2});item=c.popitem();print('popped_cache_item',item,dict(c));assert item==('b',2) and dict(c)=={'a':1}\nc['c']=3;c['d']=4;assert dict(c)=={'c':3,'d':4}\n"
def helper():
    s=importlib.util.spec_from_file_location('a_boltons_common_replacement',HERE/'collect.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','run'));a=p.parse_args();m=helper();review=HERE/'reviewed-boltons-common-replacement.json'
    if a.stage=='prepare':
        m.prepare_boltons(plans=[('yj27psb2','F3','LRI.popitem','cacheutils','cache_pop_preserves_removed_pair_and_linked_eviction_state',OBSERVATION)],review_filename=review.name,curation_prompts={'yj27psb2':'Fix boltons.cacheutils.LRI.popitem. Remove and return one existing key/value pair, and remove the corresponding key from the cache eviction bookkeeping so later bounded insertions remain consistent. Preserve the existing empty-cache error contract.\n'},primary_by_suffix={'yj27psb2':'F'},course_membership='S600_common')
        d=json.loads(review.read_text());d['tasks'][0]['source_import_alias_group']='commonreplacement';d['replaces_duplicate_family']={'excluded_instance_id':'mahmoud__boltons.3bfcfdd0.func_basic__kvii3pan','existing_admitted_instance_id':'mahmoud__boltons.3bfcfdd0.func_pm_ctrl_invert_if__48b5m32l','family':'boltons.formatutils.DeferredValue.get_value','reason':'Same snapshot/production function and cache-enabled behavior; real duplicate run retained without adding a curriculum task'};review.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(hashlib.sha256(review.read_bytes()).hexdigest())
    else:m.run(batch='boltons_common',review_override=review)
