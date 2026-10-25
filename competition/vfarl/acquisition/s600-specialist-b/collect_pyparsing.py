"""Scale reviewed parser contracts after a successful complete native probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import collect as base

HOME=Path(__file__).resolve().parent
base.HERE=HOME/'pyparsing-batch'
base.ART=base.REPLAY/'artifacts/s600-pyparsing'
base.PAGE_ROOTS=(HOME/'third-pages',)


def item(suffix,symbol,path,sub,contract,body):
    return {'instance_id':'pyparsing__pyparsing.533adf47.func_basic__'+suffix,'canonical_repo':'pyparsing/pyparsing','snapshot':'533adf47',
        'symbol':symbol,'relative_path':path,'submechanism':sub,'contract':contract,'body':body,'declared_license':'MIT',
        'repair_family_id':'pyparsing::'+symbol+'::'+contract.split('.',1)[0],
        'task_site_packages':str(base.REPLAY/'.venv-pyparsing-task/lib/python3.12/site-packages'),
        'task_environment':{'MPLBACKEND':'Agg'},
        'source_dependency_basis':'Pinned original tox.ini requires pytest and matplotlib plus diagrams extra; original pyproject diagrams requires railroad-diagrams and jinja2. Actual task package versions recorded in each run. CPU Agg renderer.'}


base.CANDIDATES={
 'pp-quoted-string':item('0ad0lvth','QuotedString.parseImpl','pyparsing/core.py','S1',
    'Quoted-token decoding preserves escape values and the consumed closing boundary. Decode whitespace escapes only in unquoted results, while unquote_results=False preserves both quotes and original characters.',
    "import pyparsing as pp\ntext=chr(34)+'a'+chr(92)+'n'+'b'+chr(34)\na=pp.QuotedString(chr(34),esc_char=chr(92)).parse_string(text,parse_all=True).as_list()\nb=pp.QuotedString(chr(34),esc_char=chr(92),unquote_results=False).parse_string(text,parse_all=True).as_list()\nprint('quoted_results',a,b)\nassert a==['a'+chr(10)+'b'] and b==[text]"),
 'pp-word-bounds':item('bie0itjz','Word.parseImpl','pyparsing/core.py','S1',
    'Word lexer recognizes its declared minimum and maximum width and advances to the exclusive token end. The non-regex character-set implementation must retain the next token and accept its exact minimum.',
    "import pyparsing as pp\nword=pp.Word('abc ',min=2,max=3).leave_whitespace()\nassert word.parseImpl.__func__ is pp.Word.parseImpl\nexpression=word+pp.Literal('Z')\na=expression.parse_string('abcZ',parse_all=True).as_list();b=expression.parse_string('abZ',parse_all=True).as_list()\nprint('word_width',a,b)\nassert a==['abc','Z'] and b==['ab','Z']"),
 'pp-regex-match':item('75z01xe9','Regex.parseImplAsMatch','pyparsing/core.py','S1',
    'Regex match-object representation retains the actual re.Match and exclusive end. Matching a following token must use the true end rather than consume the final digit twice.',
    "import pyparsing as pp\nexpression=pp.Regex(r'\\d+',as_match=True)+pp.Literal('X')\nr=expression.parse_string('42X',parse_all=True)\nprint('regex_match_object',type(r[0]).__name__,r[0].group(),r[0].span(),r[1])\nassert r[0].group()=='42' and r[0].span()==(0,2) and r[1]=='X'"),
 'pp-white-token':item('9q865kfo','White.parseImpl','pyparsing/core.py','S1',
    'Whitespace lexer preserves the matched whitespace payload and width constraints. Return the contiguous declared whitespace before the next lexical token, with correct lower-bound acceptance.',
    "import pyparsing as pp\nexpression=pp.White(' ',min=2,max=3)+pp.Literal('x')\nr=expression.parse_string('  x',parse_all=True).as_list()\nprint('whitespace_token',r)\nassert r==['  ','x']"),
 'pp-combine':item('2b7zgvhy','Combine.postParse','pyparsing/core.py','S1',
    'Combined token representation preserves separator direction and flat result shape. Join the parsed component strings using the exact requested separator and retain named grouping only when declared.',
    "import pyparsing as pp\nexpression=pp.Combine(pp.Word(pp.alphas)+':'+pp.Word(pp.nums),join_string='->')\nr=expression.parse_string('abc:12',parse_all=True).as_list()\nprint('combined_token',r)\nassert r==['abc->:->12']"),
 'pp-date-decoder':item('a2bscwf4','pyparsing_common.convert_to_date','pyparsing/common.py','S1',
    'Date token decoding uses the declared year-month-day format and date-valued output. Invalid calendar dates raise the parser failure instead of silently returning an unconverted value.',
    "import pyparsing as pp\nfrom datetime import date\nexpression=pp.Regex(r'\\d{4}-\\d{2}-\\d{2}').set_parse_action(pp.pyparsing_common.convert_to_date())\nr=expression.parse_string('2025-10-25',parse_all=True)[0]\nprint('decoded_date',r,type(r).__name__)\nassert r==date(2025,10,25) and type(r) is date\ntry:\n expression.parse_string('2025-02-31',parse_all=True)\nexcept pp.ParseException:\n print('invalid_calendar_date_rejected')\nelse:\n raise AssertionError('invalid date accepted')"),
 'pp-go-column':item('30rt795x','GoToColumn.parseImpl','pyparsing/core.py','S2',
    'Column advancement retains the source span before the target column. Accept an already-aligned cursor, advance by target minus current column, and return exactly the skipped characters.',
    "import pyparsing as pp\na=(pp.Literal('a')+pp.GoToColumn(4)+pp.Literal('b')).parse_string('a  b',parse_all=True).as_list()\nb=(pp.Literal('a')+pp.GoToColumn(2)+pp.Literal('b')).parse_string('ab',parse_all=True).as_list()\nspan=pp.GoToColumn(4).parseImpl('a  b',1)\nprint('column_parser_results',a,b,'direct_source_span',span)\nassert a==['a','','b'] and b==['a','','b'] and span==(3,'  ')"),
 'pp-at-line-start':item('7ddzm4qw','AtLineStart.parseImpl','pyparsing/core.py','S2',
    'Wrapped line-start position uses one-based columns and preserves wrapped parsing. Match only actual line starts, without accepting an indented or interior occurrence.',
    "import pyparsing as pp\ntext='AAA first'+chr(10)+' AAA indented'+chr(10)+'AAA last'\nr=list(pp.AtLineStart(pp.Literal('AAA')).scan_string(text))\nstarts=[a for _,a,_ in r]\nprint('line_start_offsets',starts)\nassert starts==[0,text.rfind('AAA')]"),
 'pp-string-start':item('ayhm7yqb','StringStart.parseImpl','pyparsing/core.py','S2',
    'String-start assertion is zero-width and respects initial whitespace preprocessing. It must not advance the cursor or emit a synthetic token when the first real token follows permitted leading whitespace.',
    "import pyparsing as pp\nr=(pp.StringStart()+pp.Word(pp.nums)).parse_string('   42',parse_all=True).as_list()\nprint('zero_width_start',r)\nassert r==['42']"),
 'pp-at-string-start':item('2wvp3jps','AtStringStart.parseImpl','pyparsing/core.py','S2',
    'Wrapped absolute-string start accepts location zero and preserves original input. Its wrapped parser must see the same characters and retain the original action setting.',
    "import pyparsing as pp\nr=pp.AtStringStart(pp.Word(pp.nums)).parse_string('42 rest').as_list()\nprint('absolute_string_start',r)\nassert r==['42']"),
 'pp-tab-positions':item('7gwqaw8e','ParserElement.parse_with_tabs','pyparsing/core.py','S2',
    'Tab-preserving parser positions refer to original input characters. Enable matching literal tab characters without tab expansion and keep fluent configuration on the same parser.',
    "import pyparsing as pp\nexpression=pp.Literal('a')+pp.White(chr(9))+pp.Literal('b')\nassert expression.parse_with_tabs() is expression\nr=expression.parse_string('a'+chr(9)+'b',parse_all=True).as_list()\nprint('original_tab_tokens',r)\nassert r==['a',chr(9),'b']"),
 'pp-result-dict':item('5ezp2jjk','ParseResults.as_dict','pyparsing/results.py','S5',
    'Recursive parse-result export preserves mapping keys, nested list structure and leaf types. Named nested ParseResults become dictionaries, unnamed sequences become lists, and numeric leaves remain numeric.',
    "import pyparsing as pp\noriginal={'outer':{'number':7,'items':[1,2]},'name':'alpha'}\nr=pp.ParseResults.from_dict(original).as_dict()\nprint('nested_result_export',r)\nassert r==original and type(r['outer']['number']) is int"),
 'pp-result-from-dict':item('cjyrb40k','ParseResults.from_dict','pyparsing/results.py','S5',
    'Recursive parse-result construction preserves nested names and scalar-versus-sequence structure. Reconstruct all original dictionary entries and retain an explicitly requested outer result name.',
    "import pyparsing as pp\noriginal={'outer':{'number':7},'items':[1,2],'text':'ab'}\nr=pp.ParseResults.from_dict(original)\nnamed=pp.ParseResults.from_dict(original,name='root')\nprint('nested_result_import',r.as_dict(),named.as_dict())\nassert r.as_dict()==original and named.root.as_dict()==original"),
 'pp-flatten':item('00xusszu','_flatten','pyparsing/util.py','S5',
    'Nested iterable flattening preserves depth-first source order and treats strings as atomic leaves. Nested sequences are visited at their original position rather than appended behind later siblings.',
    "r=m._flatten([['a',['b']],'text',(1,2)])\nprint('nested_flatten_order',r)\nassert r==['a','b','text',1,2]"),
 'pp-compressed-regex':item('b7rm05ci','make_compressed_re','pyparsing/util.py','S5',
    'Recursive prefix-factored regex construction preserves every input word and optional suffix. Shared-prefix and whole-word alternatives must recognize the exact original word language.',
    "import re\nwords=sorted(['cat','car','dog','do'])\npattern=m.make_compressed_re(words)\ncompiled=re.compile('(?:'+pattern+')'+r'\\Z')\nprint('factored_regex',pattern)\nassert all(compiled.fullmatch(s) for s in words)\nassert not any(compiled.fullmatch(s) for s in ['ca','cart','dogs','acat'])"),
 'pp-infix-grammar':item('2yi790m6','infix_notation','pyparsing/helpers.py','S5',
    'Recursive infix grammar retains operator precedence and parentheses overrides. Multiplication binds more tightly than addition, while explicit grouping changes the recursive parse tree.',
    "import pyparsing as pp\nexpression=pp.infix_notation(pp.pyparsing_common.integer,[('*',2,pp.OpAssoc.LEFT),('+',2,pp.OpAssoc.LEFT)])\na=expression.parse_string('2+3*4',parse_all=True).as_list()\nb=expression.parse_string('(2+3)*4',parse_all=True).as_list()\nprint('operator_trees',a,b)\nassert a==[[2,'+',[3,'*',4]]] and b==[[[2,'+',3],'*',4]]"),
 'pp-expression-streamline':item('7sqbanzd','ParseExpression.streamline','pyparsing/core.py','S5',
    'Grammar-expression simplification preserves ordered child structure. Collapse eligible nested sequences without losing or reordering children, keeping fluent identity and parsing behavior.',
    "import pyparsing as pp\nexpression=pp.And([pp.And([pp.Literal('a'),pp.Literal('b')]),pp.Literal('c')])\nassert expression.streamline() is expression\nprint('simplified_child_count',len(expression.exprs))\nassert len(expression.exprs)==3\nassert expression.parse_string('abc',parse_all=True).as_list()==['a','b','c']"),
 'pp-and-streamline':item('1j284bob','And.streamline','pyparsing/core.py','S5',
    'Sequence grammar resolves pending skip nodes against the following expression. Its simplification must preserve start and end tokens and retain the intervening skipped source text.',
    "import pyparsing as pp\nexpression=pp.Literal('start')+...+pp.Literal('end')\nr=expression.parse_string('start middle end',parse_all=True).as_list()\nprint('resolved_skip_sequence',r)\nassert r[0]=='start' and r[-1]=='end' and r[1].strip()=='middle'"),
 'pp-grammar-children':item('bgqc7kxn','ParseExpression.recurse','pyparsing/core.py','S5',
    'Grammar-tree child enumeration retains the last child. Structural visitors must receive all ordered child expressions rather than a truncated prefix.',
    "import pyparsing as pp\nexpression=pp.And([pp.Literal('a'),pp.Literal('b'),pp.Literal('c')])\nchildren=expression.recurse()\nprint('grammar_children',[str(c) for c in children])\nassert len(children)==3 and children==expression.exprs\nassert expression.parse_string('abc',parse_all=True).as_list()==['a','b','c']"),
 'pp-indented-block':item('47xrbxni','IndentedBlock.__init__','pyparsing/core.py','S5',
    'Indented-block grammar preserves grouping and recursion options independently. A header followed by same-depth child statements forms one grouped block with the source parent anchor retained.',
    "import pyparsing as pp\nexpression=pp.Literal('head')+pp.IndentedBlock(pp.Word(pp.alphas))\ntext='head'+chr(10)+'  alpha'+chr(10)+'  beta'+chr(10)\nr=expression.parse_string(text,parse_all=True).as_list()\nprint('indented_tree',r)\nassert r==['head',['alpha','beta']]"),
}
base.CANDIDATES['pp-indented-block'].update({'primary_adapter':'F','submechanism':'F1','secondary_mechanisms':['S5','S2'],
    'course_membership':'S600_common','classification_basis':'Explicit root cross-review before publication: constructor routes recursive/grouped parameters into the wrong fields and changes inherited savelist/anchor initialization; this is configuration propagation F1, with grammar/coordinate consequences secondary.'})
base.CANDIDATES['pp-flatten'].update({'submechanism':'S4','secondary_mechanisms':['S5'],
    'classification_basis':'Explicit root cross-review before publication: the defect changes worklist insertion and leaf accumulation, so the repaired invariant is DFS leaf visitation order with strings atomic, not preservation or rebuilding of nested container structure.'})
base.CANDIDATES['pp-and-streamline'].update({'primary_adapter':'G','submechanism':'G-fluent_return_contract','secondary_mechanisms':['S5'],
    'course_membership':'S600_common','run_number':2,
    'contract':'Fluent grammar-configuration return contract. And.streamline must return the same parser object so the caller can immediately parse through the configured object; the diagnostic does not separately attribute background skip or indentation mutations.',
    'body':"import pyparsing as pp\nexpression=pp.And([pp.Word(pp.alphas),pp.Word(pp.alphas)])\nconfigured=expression.streamline()\nprint('fluent_parser_identity',configured is expression,configured)\nassert configured is expression\nr=configured.parse_string('alpha beta',parse_all=True).as_list()\nprint('chained_parser_result',r)\nassert r==['alpha','beta']",
    'classification_basis':'Root-approved common G classification: the demonstrated causal contract is the fluent return value used by the caller, not a proven isolated recursive grammar correction.',
    'collector_reproduction_correction':'Original run001 skip-sequence observation passes in both buggy and fixed implementations; its expected-error assumption was insufficient and the partial run is preserved. Run002 uses actual fluent return identity and chained parsing. Original source/task/oracle unchanged; this is collector correction, not Gemma policy error or a same-state preference pair.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',nargs='+',choices=list(base.CANDIDATES),default=list(base.CANDIDATES))
    parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    if args.stage in {'prepare','all'}:
        for key in args.keys:base.prepare(key);print('prepared',key,flush=True)
    if args.stage in {'replay','all'}:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in pool.map(lambda key:(base.replay(key),key)[1],args.keys):print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:base.publish(args.keys)
