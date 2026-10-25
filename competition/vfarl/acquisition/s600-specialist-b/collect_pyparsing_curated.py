"""Additional explicit behavior prompts for empty native train task descriptions.

Original task rows and empty adapted tasks remain intact. Every selected task is
actually replayed under its disclosed v2 prompt; no empty-prompt run is invented.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collect_pyparsing import item
import collect as base

HOME=Path(__file__).resolve().parent
base.HERE=HOME/'pyparsing-curated-batch'
base.ART=base.REPLAY/'artifacts/s600-pyparsing-curated'
base.PAGE_ROOTS=(HOME/'third-pages',)


def curated(suffix,symbol,path,sub,contract,body):
    result=item(suffix,symbol,path,sub,contract,body)
    result.update({'curated_prompt':'Fix pyparsing.'+symbol+'.\n\n'+contract+'\n',
        'curated_by':'Codex S600 fixed-source public API/defect and native behavior review 2026-10-25',
        'curation_source_basis':'Public API/docstrings at fixed upstream pyparsing533adf47, original source-bound defect diff, and direct observations of the unmodified original implementation. Prompt describes behavior only; no reference edit, required test IDs or future verifier outcomes.'})
    return result


base.CANDIDATES={
 'pp-literal':curated('mj5n6jib','Literal.parseImpl','pyparsing/core.py','S1',
    'A literal parser should recognize the complete declared text at the current input position, return that literal unchanged, and leave the following character for the next parser. Reject nonmatching text.',
    "import pyparsing as pp\nr=(pp.Literal('alpha')+pp.Literal('!')).parse_string('alpha!',parse_all=True).as_list()\nprint('literal_tokens',r)\nassert r==['alpha','!']"),
 'pp-caseless-literal':curated('l6tp84fe','CaselessLiteral.parseImpl','pyparsing/core.py','S1',
    'Case-insensitive literal matching should accept mixed-case input and return the declared literal spelling. Consume exactly that literal so an adjacent next token remains available.',
    "import pyparsing as pp\nr=(pp.CaselessLiteral('AbC')+pp.Literal('!')).parse_string('aBc!',parse_all=True).as_list()\nprint('caseless_literal',r)\nassert r==['AbC','!']"),
 'pp-keyword':curated('ofo4wuyq','Keyword.parseImpl','pyparsing/core.py','S1',
    'Keyword recognition should preserve the complete token and respect identifier boundaries on both sides. Case-insensitive mode should accept a matching keyword in another case without consuming an adjacent character.',
    "import pyparsing as pp\na=(pp.Keyword('if',caseless=True)+pp.Literal('!')).parse_string('IF!',parse_all=True).as_list()\nprint('keyword_token',a)\nassert a==['if','!']\ntry:\n pp.Keyword('if').parse_string('iffy',parse_all=True)\nexcept pp.ParseException:\n print('identifier_suffix_rejected')\nelse:\n raise AssertionError('keyword accepted identifier suffix')"),
 'pp-chars-not-in':curated('gx1b3twl','CharsNotIn.parseImpl','pyparsing/core.py','S1',
    'The excluded-character lexer should collect characters outside its exclusion set, accept its declared minimum length, and stop before an excluded character or its maximum width.',
    "import pyparsing as pp\nexpression=pp.CharsNotIn('x',min=2,max=3)+pp.Literal('x')\nr=expression.parse_string('abx',parse_all=True).as_list()\nprint('excluded_character_boundary',r)\nassert r==['ab','x']"),
 'pp-character-ranges':curated('o0e7wr5e','_collapse_string_to_ranges','pyparsing/util.py','S1',
    'Character-range compaction must retain every supplied character when escaping is disabled as well as when it is enabled. The resulting range expression should represent the same character set instead of dropping characters from its ranges.',
    "import re\nfor escape in (False,True):\n pattern=m._collapse_string_to_ranges('abcx',re_escape=escape)\n compiled=re.compile('['+pattern+']')\n print('character_range',escape,pattern)\n assert all(compiled.fullmatch(c) for c in 'abcx')\n assert not any(compiled.fullmatch(c) for c in 'dwz')"),
 'pp-expand-ranges':curated('m7w1gfss','srange','pyparsing/core.py','S1',
    'Character-range decoding should expand each inclusive range in its declared order and retain separate literal characters. Normal bracketed ranges must not reverse their endpoints in the returned character string.',
    "import pyparsing as pp\nr=pp.srange('[a-cx]')\nprint('expanded_character_range',r)\nassert r=='abcx'"),
 'pp-lookahead-position':curated('d1zp8zp9','FollowedBy.parseImpl','pyparsing/core.py','S2',
    'Positive lookahead should validate the next expression without consuming its characters. Preserve named results from the looked-ahead expression while leaving the same location for the following real parser.',
    "import pyparsing as pp\nexpression=pp.Literal('a')+pp.FollowedBy(pp.Literal('b')('peek'))+pp.Literal('b')\nr=expression.parse_string('ab',parse_all=True)\nprint('lookahead_tokens',r.as_list(),'named',r.peek)\nassert r.as_list()==['a','b'] and r.peek=='b'"),
 'pp-word-start-position':curated('eskqr3xa','WordStart.parseImpl','pyparsing/core.py','S2',
    'A word-start assertion should be zero-width and recognize the transition into the declared word-character set. It must leave the first character of the word available for the following token parser.',
    "import pyparsing as pp\nexpression=pp.WordStart(pp.alphas)+pp.Word(pp.alphas)\nr=expression.parse_string('word',parse_all=True).as_list()\nprint('word_start_token',r)\nassert r==['word']"),
 'pp-word-end-position':curated('fyy34whh','WordEnd.parseImpl','pyparsing/core.py','S2',
    'A word-end assertion should leave the following nonword character untouched and also accept the end of the input. It must not read beyond the input or add a character to the parse position.',
    "import pyparsing as pp\na=(pp.Word(pp.alphas)+pp.WordEnd(pp.alphas)+pp.Literal('!')).parse_string('word!',parse_all=True).as_list()\nb=(pp.Word(pp.alphas)+pp.WordEnd(pp.alphas)).parse_string('word',parse_all=True).as_list()\nprint('word_end_tokens',a,b)\nassert a==['word','!'] and b==['word']"),
 'pp-line-start-position':curated('pvqxemlq','LineStart.parseImpl','pyparsing/core.py','S2',
    'The line-start position token should accept column one without consuming text. Reject an interior position and preserve normal parsing of the first token on a line.',
    "import pyparsing as pp\nr=(pp.LineStart()+pp.Literal('start')).parse_string('start',parse_all=True).as_list()\nprint('line_start_token',r)\nassert r==['start']\ntry:\n pp.LineStart().parseImpl('xstart',1)\nexcept pp.ParseException:\n print('interior_column_rejected')\nelse:\n raise AssertionError('interior position accepted')"),
 'pp-line-end-position':curated('33lxz1nl','LineEnd.parseImpl','pyparsing/core.py','S2',
    'The line-end parser should consume a newline at the end of a line and also recognize an end-of-input line without a final newline. It must not index beyond the available input.',
    "import pyparsing as pp\na=(pp.Literal('a')+pp.LineEnd()).parse_string('a'+chr(10),parse_all=True).as_list()\nb=(pp.Literal('a')+pp.LineEnd()).parse_string('a',parse_all=True).as_list()\nprint('line_end_tokens',a,b)\nassert a==['a',chr(10)] and b==['a']"),
 'pp-string-end-position':curated('eyy1v04n','StringEnd.parseImpl','pyparsing/core.py','S2',
    'The string-end position token should accept the exact end of input without adding a synthetic result token. Complete parsing should reject extra trailing nonwhitespace text.',
    "import pyparsing as pp\nr=(pp.Word(pp.alphas)+pp.StringEnd()).parse_string('word',parse_all=True).as_list()\nprint('string_end_tokens',r)\nassert r==['word']\ntry:\n (pp.Word(pp.alphas)+pp.StringEnd()).parse_string('word!',parse_all=True)\nexcept pp.ParseException:\n print('trailing_character_rejected')\nelse:\n raise AssertionError('trailing character accepted')"),
 'pp-column-number':curated('fhftcrci','col','pyparsing/util.py','S2',
    'Column lookup should convert a source-character offset to its one-based column within the containing line. Positions at the beginning of the input or immediately following a newline belong to column one.',
    "text='ab'+chr(10)+'cde'\nr=[m.col(loc,text) for loc in (0,1,3,5)]\nprint('source_columns',r)\nassert r==[1,2,1,3]"),
 'pp-line-content':curated('rj8cjh6i','line','pyparsing/util.py','S2',
    'Source-line lookup should return the complete line containing a given character offset, excluding the preceding or terminating newline. This should work for both the first line and a final line without a newline.',
    "text='ab'+chr(10)+'cde'\nr=(m.line(1,text),m.line(4,text))\nprint('source_line_contents',r)\nassert r==('ab','cde')"),
 'pp-optional-grammar':curated('f1103l66','Opt.parseImpl','pyparsing/core.py','S5',
    'Optional grammar elements should emit their declared default only when absent and preserve their normal parsed value when present. Keep the following expression at the correct input position.',
    "import pyparsing as pp\nexpression=pp.Opt(pp.Literal('a'),default='missing')+pp.Literal('b')\na=expression.parse_string('b',parse_all=True).as_list();b=expression.parse_string('ab',parse_all=True).as_list()\nprint('optional_trees',a,b)\nassert a==['missing','b'] and b==['a','b']"),
 'pp-dictionary-grammar':curated('tnccs8tb','Dict.postParse','pyparsing/core.py','S5',
    'Dictionary grammar conversion should expose grouped key/value rows as their original named entries without destroying row structure. Preserve the declared dictionary versus ParseResults output mode and names.',
    "import pyparsing as pp\nrow=pp.Group(pp.Word(pp.alphas)+pp.Word(pp.nums))\nr=pp.Dict(pp.OneOrMore(row)).parse_string('red 1 blue 2',parse_all=True)\nprint('dictionary_tree',r.as_dict(),r.as_list())\nassert r.as_dict()=={'red':'1','blue':'2'} and r.as_list()==[['red','1'],['blue','2']]"),
 'pp-each-grammar':curated('u3ka8ix6','Each.parseImpl','pyparsing/core.py','S5',
    'An unordered Each grammar should recognize all its required expressions in whichever input order they appear. Required, optional, and repeated members must keep their respective roles instead of silently matching nothing.',
    "import pyparsing as pp\nexpression=pp.Each([pp.Literal('alpha'),pp.Literal('beta')])\na=expression.parse_string('beta alpha',parse_all=True).as_list();b=expression.parse_string('alpha beta',parse_all=True).as_list()\nprint('unordered_required_members',a,b)\nassert a==['beta','alpha'] and b==['alpha','beta']"),
 'pp-group-grammar':curated('hf3yt3pl','Group.postParse','pyparsing/core.py','S5',
    'Grouping should preserve a nested parse-result node by default. The explicit aslist option should instead create a nested ordinary list, retaining child order and the surrounding parse-tree boundary.',
    "import pyparsing as pp\nchild=pp.Word(pp.alphas)+pp.Word(pp.nums)\na=pp.Group(child).parse_string('alpha 12',parse_all=True)\nb=pp.Group(child,aslist=True).parse_string('alpha 12',parse_all=True)\nprint('grouped_node_types',type(a[0]).__name__,type(b[0]).__name__,a.as_list(),b.as_list())\nassert isinstance(a[0],pp.ParseResults) and type(b[0]) is list\nassert a.as_list()==b.as_list()==[['alpha','12']]"),
 'pp-or-grammar':curated('quswhrej','Or.parseImpl','pyparsing/core.py','S5',
    'Alternative grammar selection should return the longest result actually obtained when actions run. If the nominal longest candidate becomes shorter during action-aware parsing, another alternative can still provide the full longer match. Preserve the alternative tokens and consumed span.',
    "import pyparsing as pp\nclass ActionSensitive(pp.Token):\n def parseImpl(self,text,loc,do_actions=True):\n  if not text.startswith('ab',loc):\n   raise pp.ParseException(text,loc,'expected ab')\n  return (loc+1,['a']) if do_actions else (loc+2,['ab'])\nexpression=pp.Or([ActionSensitive(),pp.Literal('ab')])\nr=expression.parse_string('ab',parse_all=True).as_list()\nprint('action_aware_longest_alternative',r)\nassert r==['ab']"),
 'pp-nested-grammar':curated('qwryjxma','nested_expr','pyparsing/helpers.py','S5',
    'Balanced-delimiter parsing should produce the corresponding nested list structure. Quoted delimiters should remain content under the declared ignored-expression behavior, and explicit opener and closer options should continue to work.',
    "import pyparsing as pp\na=pp.nested_expr().parse_string('(a (b c) d)',parse_all=True).as_list()\nprint('nested_delimiter_tree',a)\nassert a==[['a',['b','c'],'d']]\nb=pp.nested_expr('[',']').parse_string('[a [b] c]',parse_all=True).as_list()\nassert b==[['a',['b'],'c']]"),
}


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
