"""Second S600 source batch; keep the first reviewed batch immutable."""
import argparse
import json
from pathlib import Path
import collect as base

HOME = Path(__file__).resolve().parent
base.HERE = HOME / "second-batch"
base.ART = base.REPLAY / "artifacts/s600-specialist-b-second"
base.PAGE_ROOTS = (HOME / "pages",)


def item(repo, snapshot, suffix, symbol, path, sub, contract, body, layout="."):
    return {"instance_id":repo.replace('/','__')+'.'+snapshot+'.func_basic__'+suffix,
            "canonical_repo":repo,"snapshot":snapshot,"symbol":symbol,"relative_path":path,
            "submechanism":sub,"contract":contract,"body":body,"source_layout":layout,
            "declared_license":"MIT","repair_family_id":repo+'::'+symbol+'::'+contract.split('.',1)[0]}


base.CANDIDATES = {
    "xml-namespace-binding":item('martinblech/xmltodict','0952f382','254qp8xw','_DictSAXHandler.startNamespaceDecl','xmltodict.py','S5',
        'XML namespace binding direction. Record declarations by prefix with URI values so the parsed tree retains xmlns bindings and can be emitted with the same qualified structure.',
        "import xmltodict\nsource='<root xmlns:ns=\"http://example.org/ns\"><ns:item>value</ns:item></root>'\nresult=xmltodict.parse(source,process_namespaces=True)\nprint('namespace_tree',result)\nassert result['root']['@xmlns']=={'ns':'http://example.org/ns'}\nassert result['root']['http://example.org/ns:item']=='value'"),
    "xml-emission-namespace":item('martinblech/xmltodict','0952f382','4ry5wj8i','_process_namespace','xmltodict.py','S5',
        'XML emission namespace-qualified names. Split the final separator from the namespace URI, map the namespace alias, preserve the separator and attribute marker in the emitted qualified name.',
        "namespaces={'http://example.org/ns':'ns'}\nvalues=(m._process_namespace('http://example.org/ns:item',namespaces),m._process_namespace('@http://example.org/ns:id',namespaces))\nprint('emission_names',values)\nassert values==('ns:item','@ns:id')\nimport xmltodict\nxml=xmltodict.unparse({'http://example.org/ns:root':{'@xmlns:ns':'http://example.org/ns','http://example.org/ns:item':'value'}},namespaces=namespaces)\nprint('emitted_xml',xml)\nassert '<ns:root' in xml and '<ns:item>value</ns:item>' in xml"),
    "xml-parsed-name":item('martinblech/xmltodict','0952f382','ioyywowe','_DictSAXHandler._build_name','xmltodict.py','S5',
        'XML parsed qualified-name resolution. Separate URI from local name at the final separator, resolve the URI alias, and retain the local name when a namespace is deliberately dropped.',
        "import xmltodict\nsource='<root xmlns:ns=\"http://example.org/ns\"><ns:item>value</ns:item></root>'\nalias=xmltodict.parse(source,process_namespaces=True,namespaces={'http://example.org/ns':'short'})\ndropped=xmltodict.parse(source,process_namespaces=True,namespaces={'http://example.org/ns':None})\nprint('resolved_names',alias,dropped)\nassert alias['root']['short:item']=='value'\nassert dropped['root']['item']=='value'"),
    "xml-comments":item('martinblech/xmltodict','0952f382','ydtcs2nb','_DictSAXHandler.comments','xmltodict.py','S1',
        'XML comment text preservation and whitespace policy. Preserve character order and trim comment whitespace only when strip_whitespace is enabled.',
        "import xmltodict\nsource='<root><!--  comment text  --><item>value</item></root>'\na=xmltodict.parse(source,process_comments=True)\nb=xmltodict.parse(source,process_comments=True,strip_whitespace=False)\nprint('comment_text',a['root']['#comment'],repr(b['root']['#comment']))\nassert a['root']['#comment']=='comment text'\nassert b['root']['#comment']=='  comment text  '"),
    "ini-lineof":item('pytest-dev/iniconfig','16793ead','tv0bgs07','IniConfig.lineof','src/iniconfig/__init__.py','S2',
        'INI source-coordinate lookup and one-based conversion. Resolve the section and key in their declared order and report their actual one-based source lines; unknown coordinates remain None.',
        "from iniconfig import IniConfig\nini=IniConfig('example.ini',data='[section]\\nkey=value\\n')\nvalues=(ini.lineof('section'),ini.lineof('section','key'),ini.lineof('missing'))\nprint('source_lines',values)\nassert values==(1,2,None)",'src'),
    "ini-error-position":item('pytest-dev/iniconfig','16793ead','vb9u5hga','ParseError.__str__','src/iniconfig/exceptions.py','S2',
        'INI diagnostic zero-based internal to one-based displayed source position. The exception keeps its original attributes and displays path, line plus one, and original message.',
        "from iniconfig.exceptions import ParseError\ne=ParseError('example.ini',0,'unexpected line')\nprint('diagnostic',str(e),'stored_line',e.lineno)\nassert str(e)=='example.ini:1: unexpected line'\nassert e.path=='example.ini' and e.lineno==0 and e.msg=='unexpected line'",'src'),
    "ini-comment-grammar":item('pytest-dev/iniconfig','16793ead','gh3fa9af','iscommentline','src/iniconfig/_parse.py','S5',
        'INI comment lexical grammar. Leading whitespace does not hide a comment marker, while ordinary key-value and section lines are not comments.',
        "values=[m.iscommentline(s) for s in ['  # comment','\\t; comment','key=value','[section]']]\nprint('comment_grammar',values)\nassert values==[True,True,False,False]",'src'),
    "parse-exact-result":item('r1chardj0n3s/parse','30da9e4f','1efyknco','Parser.parse','parse.py','S1',
        'Format parser exact-match result and lazy evaluation contract. Match the complete input, return None on mismatch, and preserve typed Result or explicitly unevaluated Match according to evaluate_result.',
        "import parse\np=parse.compile('{value:d}')\nresult=p.parse('42')\nprint('typed_result',result)\nassert result.named=={'value':42}\nassert p.parse('prefix42') is None\nmatch=p.parse('42',evaluate_result=False)\nassert isinstance(match,parse.Match) and match.evaluate_result().named=={'value':42}"),
}
base.CANDIDATES['xml-namespace-binding']['body'] = base.CANDIDATES['xml-namespace-binding']['body'].replace('<root xmlns:ns=', '<root id="root" xmlns:ns=')
base.CANDIDATES['xml-namespace-binding']['run_number'] = 2
base.CANDIDATES['xml-namespace-binding']['collector_reproduction_correction'] = 'Original run001 reproduction expected namespace attributes on an attribute-free element, while the pinned source retains declarations when an element has attributes. Actual source observation led to adding an ordinary root id attribute to exercise the intended declaration contract. Original publisher task unchanged; recipe-v1 and aborted run001 remain preserved. This is collector correction, not a model failure or verified action pair.'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',nargs='+',choices=list(base.CANDIDATES),default=list(base.CANDIDATES))
    parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all')
    args=parser.parse_args()
    for key in args.keys:
        if args.stage in {'prepare','all'}:
            base.prepare(key);print('prepared',key,flush=True)
        if args.stage in {'replay','all'}:
            base.replay(key);print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:
        base.publish(args.keys)
