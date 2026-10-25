"""41 source-reviewed production contracts; one mutation per contract, common course."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import collect_furl
import collect as base
from collect_common import item

HOME=Path(__file__).resolve().parent
base.HERE=HOME/'common-furl-more-batch'
base.ART=base.REPLAY/'artifacts/s600-common-b'


def furl(suffix,symbol,sub,contract,body,path='furl/furl.py'):
    result=item('gruns/furl','da386f68',suffix,symbol,path,sub,contract,body,'Unlicense')
    result['task_python']='/Library/Developer/CommandLineTools/usr/bin/python3'
    result['task_site_packages']=str(base.REPLAY/'.venv-furl39-task/lib/python3.9/site-packages')
    result['task_runtime_profile_reason']='Task-only Python3.9.6 preserves fixed Furl upstream URL/IPv6 behavior; all77 original native cases pass. The official ADK parent stays Python3.12.13; no original verification requirements or source bytes are changed.'
    return result


base.CANDIDATES={
 'furl-query-encode':furl('0v9ni7yq','Query.encode','S1',
  'Query serialization must encode original keys and values in insertion order, use one key/value separator, preserve missing values, and honor quoting and pair delimiter options.',
  "q=m.Query([('a b','c d'),('flag',None),('a b','two')])\nr=q.encode(delimiter=';',quote_plus=True)\nprint('encoded_query',r)\nassert r=='a+b=c+d;flag;a+b=two'"),
 'furl-path-protocol':furl('1fhv9h7a','quacks_like_a_path_with_segments','R2',
  'Path protocol detection must require a segments attribute with a nonstring iterable value, without touching a missing attribute on ordinary strings.',
  "p=m.Path('/a/b')\nvalues=[m.quacks_like_a_path_with_segments(v) for v in ('/a/b',p,42)]\nprint('path_protocol',values)\nassert values==[False,True,False]"),
 'furl-quote-policy':furl('9upi6b6z','create_quote_fn','S1',
  'Quoting policy must intersect requested unquoted characters with the declared safe set and convert encoded spaces to plus only when requested.',
  "quote=m.create_quote_fn('/:',True)\na=quote('a b/c:d',False);b=quote('a b/c:d',True);c=quote('a b/c:d','/!')\nprint('quote_policies',a,b,c)\nassert (a,b,c)==('a+b%2Fc%3Ad','a+b/c:d','a+b/c%3Ad')\nassert m.create_quote_fn('/',False)('a b/c',False)=='a%20b%2Fc'"),
 'furl-multidict-add':furl('233u7o59','omdict1D.add','S3',
  'Multivalue insertion keeps each original key attached to its values and preserves the append order in both the global item sequence and per-key list.',
  "d=m.omdict1D();r=d.add('k',[1,2]);d.add('k',3)\nprint('inserted_values',d.allitems(),d.values('k'))\nassert r is d and d.allitems()==[('k',1),('k',2),('k',3)] and d.values('k')==[1,2,3]",'furl/omdict1D.py'),
 'furl-component-remove':furl('2mdytao1','furl.remove','F1',
  'URL component removal routes each supplied selector to its matching component, removes requested query entries and fragment suffixes, and preserves unrelated values.',
  "f=m.furl('http://u:p@h/a/b?x=1&y=2#c/d?z=3');r=f.remove(username=True,args='x',fragment_path='d')\nprint('removed_components',f.url)\nassert r is f and f.username is None and f.password=='p' and f.query.params.allitems()==[('y','2')] and str(f.fragment.path)=='c/'"),
 'furl-query-assignment':furl('2uqky8ob','QueryCompositionInterface.__setattr__','F1',
  'Assignment dispatch recognizes query and args aliases and loads the existing query object; an unrelated attribute is left for its owning object.',
  "f=m.furl('http://h/?x=1');original=f.query;f.args={'y':'2'}\nprint('query_assignment',f.query.params.allitems())\nassert f.query is original and f.query.params.allitems()==[('y','2')]\nassert m.QueryCompositionInterface.__setattr__(f,'ordinary',3) is False"),
 'furl-authority-path':furl('4341co1j','URLPathCompositionInterface._force_absolute','S2',
  'An authority forces an existing path to be absolute while an empty path remains empty and a path without an authority remains relative.',
  "f=m.furl('http://example.com');empty=m.Path('');nonempty=m.Path('a')\nassert not m.URLPathCompositionInterface._force_absolute(f,empty)\nassert m.URLPathCompositionInterface._force_absolute(f,nonempty)\ng=m.furl('a');assert not m.URLPathCompositionInterface._force_absolute(g,nonempty)\nprint('authority_path_boundary',f.url,g.url)"),
 'furl-url-load':furl('4rjl7yoc','furl.load','F1',
  'URL loading sends authority, path, query, and fragment fields to their original consumers and preserves credentials and the fluent return.',
  "f=m.furl();r=f.load('http://user:pass@example.com:81/a?x=1#part')\nprint('loaded_url_fields',f.asdict())\nassert r is f and (f.username,f.password,f.host,f.port)==('user','pass','example.com',81)\nassert str(f.path)=='/a' and str(f.query)=='x=1' and str(f.fragment)=='part'"),
 'furl-fragment-remove':furl('523p0dzk','Fragment.remove','F1',
  'Fragment removal acts only on explicitly supplied path or query selectors; removing the whole fragment clears it, and removing a query key preserves the path.',
  "f=m.Fragment('a/b?x=1&y=2');r=f.remove(args='x')\nprint('fragment_removed',str(f))\nassert r is f and str(f)=='a/b?y=2'\nf.remove(fragment=True);assert str(f)==''"),
 'furl-query-initialize':furl('6rc6jk4v','Query.__init__','F1',
  'Query construction forwards the declared strictness and complete query text to its real loader without dropping the initial character.',
  "q=m.Query('name=value',strict=False)\nprint('initial_query',q.strict,q.params.allitems())\nassert q.strict is False and q.params.allitems()==[('name','value')]\nassert m.Query(strict=True).strict is True"),
 'furl-multidict-update-bins':furl('8q7fz6wr','omdict1D._bin_update_items','S3',
  'Multivalue update separates replacements from appended leftovers, treats a scalar as one value, preserves existing per-key order, and honors an empty list as deletion.',
  "d=m.omdict1D([('a',1),('a',2),('b',3)]);d.updateall([('a',[4,5,6]),('b',[])])\nprint('updated_multivalues',d.allitems())\nassert d.allitems()==[('a',4),('a',5),('a',6)]\ne=m.omdict1D([('a',1)]);e.updateall([('a',7)]);assert e.values('a')==[7]",'furl/omdict1D.py'),
 'furl-query-load':furl('8wogoct3','Query.load','S2',
  'Replacing query contents must retain every parsed key/value pair including the terminal pair, remove old contents, and return the query object.',
  "q=m.Query('old=0');r=q.load('a=1&b=2')\nprint('loaded_query_pairs',q.params.allitems())\nassert r is q and q.params.allitems()==[('a','1'),('b','2')]"),
 'furl-path-append':furl('9363y9pt','Path.add','S2',
  'Path appending places new path segments after existing segments, preserves the leading root marker, and reconciles only the joining slash boundary.',
  "p=m.Path('/a/');r=p.add('b/c')\nprint('appended_path',str(p),p.segments)\nassert r is p and str(p)=='/a/b/c' and p.segments==['a','b','c']\np.add(['d']);assert str(p)=='/a/b/c/d'"),
 'furl-path-load':furl('a4e9j52o','Path.load','S2',
  'Path loading recognizes the leading absolute marker and preserves a trailing directory marker at its actual end, including empty input.',
  "p=m.Path();r=p.load('/a/b/')\nprint('loaded_path',str(p),p.segments,p.isabsolute)\nassert r is p and p.isabsolute and p.segments==['a','b',''] and str(p)=='/a/b/'\np.load('');assert str(p)=='' and not p.isabsolute"),
 'furl-fragment-grammar':furl('affh6me1','Fragment.load','S1',
  'Fragment decoding distinguishes a query from a plain path using key/value syntax and splits a path/query pair only when its suffix actually looks like a query.',
  "a=m.Fragment('x=1');b=m.Fragment('a/b?x=1');c=m.Fragment('a?plain?')\nprint('fragment_structures',a.asdict(),b.asdict(),c.asdict())\nassert str(a.path)=='' and a.query.params.allitems()==[('x','1')]\nassert str(b.path)=='a/b' and b.query.params.allitems()==[('x','1')]\nassert str(c.path)=='a%3Fplain%3F' and not c.query.params"),
 'furl-path-serialize':furl('b1g6smz1','Path._path_from_segments','S1',
  'Path serialization encodes each original segment and joins them in source order without reversing their positions.',
  "p=m.Path();r=p._path_from_segments(['first value','second','last'])\nprint('serialized_segments',r)\nassert r=='first%20value/second/last'"),
 'furl-query-remove':furl('c2puuoeb','Query.remove','S3',
  'Query removal distinguishes a whole key from a particular key/value pair and removes exactly the requested multivalue occurrences.',
  "q=m.Query('long=1&long=2&keep=3');r=q.remove([('long','1')])\nprint('remaining_query',q.params.allitems())\nassert r is q and q.params.allitems()==[('long','2'),('keep','3')]\nq.remove('long');assert q.params.allitems()==[('keep','3')]"),
 'furl-fragment-initialize':furl('ca5c0l1w','Fragment.__init__','F1',
  'Fragment construction retains the original fragment text, initializes a real Boolean separator, and forwards the caller strictness consistently to its path and query components.',
  "f=m.Fragment('abc?x=1',strict=False)\nprint('initial_fragment',str(f),f.strict,f.path.strict,f.query.strict)\nassert str(f)=='abc?x=1' and f.separator is True and not f.strict and not f.path.strict and not f.query.strict"),
 'furl-join-components':furl('dc54wz0a','furl.join','F1',
  'Sequential URL joins must resolve each relative operand against the current URL and load that result before processing the next operand.',
  "f=m.furl('http://example.com/base/');r=f.join('a/','b')\nprint('sequential_join',f.url)\nassert r is f and f.url=='http://example.com/base/a/b'"),
 'furl-path-remove':furl('eoi40b3a','Path.remove','S2',
  'Removing a terminal path suffix preserves the original root and the slash left by a relative suffix; removing the whole path clears its contents.',
  "p=m.Path('/a/b/c');r=p.remove('b/c')\nprint('removed_path_suffix',str(p))\nassert r is p and str(p)=='/a/'\np.remove(True);assert str(p)==''"),
 'furl-url-fields':furl('exsb78x2','furl.asdict','F1',
  'URL dictionary export forwards credentials and each path/query component to its matching field while exporting the encoded internationalized host.',
  "f=m.furl('http://user:pass@xn--eckwd4c7c.xn--zckzah/a?x=1#part');r=f.asdict()\nprint('url_field_channels',r)\nassert r['username']=='user' and r['password']=='pass' and r['host_encoded']=='xn--eckwd4c7c.xn--zckzah'\nassert r['path']['segments']==['a'] and r['query']['params']==[('x','1')]"),
 'furl-idna-decode':furl('g5xaaimb','idna_decode','S1',
  'IDNA decoding converts an ASCII domain representation to Unicode text and preserves nonstring absent values instead of producing bytes or reencoding the domain.',
  "r=m.idna_decode('xn--eckwd4c7c.xn--zckzah')\nprint('decoded_idna',r,type(r).__name__)\nassert r=='ドメイン.テスト' and isinstance(r,str)\nassert m.idna_decode(None) is None"),
 'furl-index-lookup':furl('h60jxgod','lget','S2',
  'Sequence lookup reads the exact supplied index including negative indices, and returns the supplied default only when that index lies outside the sequence.',
  "marker=object();a=m.lget(['a','b'],0);b=m.lget(['a','b'],-1);c=m.lget([],0,marker)\nprint('indexed_lookup',a,b,c is marker)\nassert (a,b,c is marker)==('a','b',True)"),
 'furl-multidict-replace':furl('huo0f0sq','omdict1D._set','S3',
  'Multivalue replacement treats a scalar string as one value and a list as its original flat sequence, replacing previous values rather than nesting that sequence.',
  "d=m.omdict1D();d['k']='abc';assert d.values('k')==['abc'];d['k']=[1,2]\nprint('replaced_multivalues',d.allitems())\nassert d.values('k')==[1,2]",'furl/omdict1D.py'),
 'furl-path-initialize':furl('hvxccjau','Path.__init__','F1',
  'Path construction forwards the actual path, strictness and forcing callback to initialized segment state, preserving a relative path when the callback does not force an absolute one.',
  "p=m.Path('a/b',strict=False);q=m.Path('a',force_absolute=lambda path:True)\nprint('initial_paths',str(p),str(q))\nassert str(p)=='a/b' and not p.strict and not p.isabsolute\nassert str(q)=='/a'"),
 'furl-url-add':furl('j78hdqob','furl.add','F1',
  'URL augmentation forwards query and fragment arguments to their matching path/query consumers while preserving prior entries and their append order.',
  "f=m.furl('http://h/a?x=1#c?y=2');r=f.add(path='b',args={'z':'3'},fragment_path='d',fragment_args={'w':'4'})\nprint('augmented_url',f.url)\nassert r is f and str(f.path)=='/a/b' and f.query.params.allitems()==[('x','1'),('z','3')]\nassert str(f.fragment.path)=='c/d' and f.fragment.query.params.allitems()==[('y','2'),('w','4')]"),
 'furl-normalize-path':furl('jvkgvrku','Path.normalize','S2',
  'Path normalization resolves dot segments without reversing the normalized character span, while retaining absolute root and trailing directory slash.',
  "p=m.Path('/a/../b/');r=p.normalize()\nprint('normalized_path',str(p))\nassert r is p and str(p)=='/b/'"),
 'furl-url-serialize':furl('m54x2vbj','furl.tostr','S1',
  'URL serialization preserves the scheme and authority channels and carries query quoting options to their matching serializer parameters.',
  "f=m.furl('http://h/a');f.args=[('x y','z w'),('k','v')];r=f.tostr(query_delimiter=';',query_quote_plus=False)\nprint('serialized_url',r)\nassert r=='http://h/a?x%20y=z%20w;k=v'"),
 'furl-url-split':furl('my23eydf','urlsplit','S1',
  'URL parsing retains the original custom scheme and authority field, preserving an absent authority separately from a present empty authority.',
  "a=m.urlsplit('custom://host/a?x=1#part');b=m.urlsplit('relative/path');c=m.urlsplit('http://')\nprint('split_url_fields',a,b,c)\nassert tuple(a)==('custom','host','/a','x=1','part')\nassert b.scheme is None and b.netloc is None and c.netloc==''"),
 'furl-remove-segment-boundary':furl('o4yy1i2h','remove_path_segments','S2',
  'Segment suffix deletion leaves an empty sequence for an exact match, preserves the source for an oversized removal, and retains the root and remaining separator at a relative suffix boundary.',
  "a=m.remove_path_segments(['','a','b','c'],['b','c']);b=m.remove_path_segments(['a'],['a']);c=m.remove_path_segments(['a'],['a','b'])\nprint('suffix_boundaries',a,b,c)\nassert (a,b,c)==(['','a',''],[],['a'])"),
 'furl-iterable-protocol':furl('ofw6qkh3','is_iterable_but_not_string','R2',
  'Nonstring iterable detection requires an actual callable iterator protocol and excludes string objects; merely possessing a noncallable __iter__ attribute does not implement iteration.',
  "class NonIterable:\n __iter__=None\nvalues=[m.is_iterable_but_not_string(v) for v in ('abc',[1],42,NonIterable())]\nprint('iterable_protocol',values)\nassert values==[False,True,False,False]",'furl/common.py'),
 'furl-path-owner-initialize':furl('ox6xmnb6','PathCompositionInterface.__init__','F1',
  'A path-owning interface forwards its real forcing callback and strictness into the owned Path, allowing later path parsing to apply that owner policy.',
  "f=m.furl('http://h/a',strict=True)\nprint('owned_path_policy',str(f.path),f.path.strict,callable(f.path._force_absolute))\nassert str(f.path)=='/a' and f.path.strict is True and callable(f.path._force_absolute)"),
 'furl-url-initialize':furl('oz7h9988','furl.__init__','F1',
  'URL construction retains its caller strictness and routes explicit query and path overrides to the corresponding components after the base URL has been loaded.',
  "f=m.furl('http://h/old',path='/new',args={'x':'1'},strict=True)\nprint('url_initial_overrides',f.url,f.strict,f.fragment.strict)\nassert f.url=='http://h/new?x=1' and f.strict and f.fragment.strict"),
 'furl-set-scheme':furl('q1bmw38d','set_scheme','S1',
  'Scheme replacement emits the new scheme before one colon and the original suffix, including HTTP URLs; an absent replacement removes only the old scheme.',
  "a=m.set_scheme('http://h/a','https');b=m.set_scheme('custom://h/a',None)\nprint('replaced_schemes',a,b)\nassert a=='https://h/a' and b=='//h/a'"),
 'furl-query-fields':furl('sd8pqrl4','Query.asdict','S3',
  'Query dictionary export preserves its encoded text and full duplicate-key item order rather than using an object repr or reversing the materialized multivalue sequence.',
  "q=m.Query('x=1&x=2&y=3');r=q.asdict()\nprint('query_field_representation',r)\nassert r=={'encoded':'x=1&x=2&y=3','params':[('x','1'),('x','2'),('y','3')]}"),
 'furl-idna-encode':furl('sxlizomj','idna_encode','S1',
  'IDNA encoding converts a Unicode domain into its ASCII domain text while preserving absent nonstring values.',
  "r=m.idna_encode('ドメイン.テスト')\nprint('encoded_idna',r,type(r).__name__)\nassert r=='xn--eckwd4c7c.xn--zckzah' and isinstance(r,str) and m.idna_encode(None) is None"),
 'furl-join-segment-boundary':furl('t1exlvha','join_path_segments','S2',
  'Segment joining removes only the redundant marker at a joining boundary, preserving every content segment and an intentional doubled slash.',
  "a=m.join_path_segments(['a',''],['b']);b=m.join_path_segments(['a',''],['','b']);c=m.join_path_segments(['a'],['','b'])\nprint('joined_segment_boundaries',a,b,c)\nassert (a,b,c)==(['a','b'],['a','','b'],['a','b'])"),
 'furl-query-decode':furl('tbrb4a4i','Query._extract_items_from_querystr','S1',
  'Query text decoding preserves each key/value channel, decodes percent escapes and plus-space syntax, and represents a bare key as a missing value.',
  "q=m.Query();r=q._extract_items_from_querystr('a+b=c%20d&flag&x=2')\nprint('decoded_query_pairs',r)\nassert r==[('a b','c d'),('flag',None),('x','2')]"),
 'furl-relative-join':furl('ywuqjpzc','urljoin','S2',
  'Relative URL resolution uses the declared base and operand in that order and restores the original custom scheme after normalizing the relative path.',
  "a=m.urljoin('custom://h/a/b','../c');b=m.urljoin('http://h/a/','b')\nprint('relative_resolution',a,b)\nassert a=='custom://h/c' and b=='http://h/a/b'"),
 'furl-query-update-order':furl('yxdesyrb','Query.set','S3',
  'Query update retains supplied duplicate-key value order while replacing existing values and preserving unrelated existing query entries.',
  "q=m.Query('x=old&keep=v');r=q.set([('x','first'),('x','second')])\nprint('updated_query_order',q.params.allitems())\nassert r is q and q.params.values('x')==['first','second'] and q.params['keep']=='v'"),
 'furl-string-protocol':furl('3nzkssvz','attemptstr','R2',
  'Best-effort string conversion invokes the object string protocol rather than repr and preserves the original object if its string conversion raises.',
  "class Display:\n def __str__(self):return 'display'\n def __repr__(self):return 'representation'\nclass Broken:\n def __str__(self):raise ValueError('cannot stringify')\na=Display();b=Broken();value=m.attemptstr(a)\nprint('string_protocol',value,m.attemptstr(b) is b)\nassert value=='display' and m.attemptstr(b) is b"),
}

ACTIVE_KEYS=list(base.CANDIDATES)[:21]
profile=HOME/'furl39-runtime-profile.json'
for config in base.CANDIDATES.values():
    config['task_runtime_file_profile']={'path':str(profile.relative_to(base.ROOT)),
                                         'sha256':base.sha(profile)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',nargs='+',choices=list(base.CANDIDATES),default=ACTIVE_KEYS)
    parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    if args.stage in {'prepare','all'}:
        for key in args.keys:base.prepare(key);print('prepared',key,flush=True)
    if args.stage in {'replay','all'}:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in pool.map(lambda key:(base.replay(key),key)[1],args.keys):print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:base.publish(args.keys)
