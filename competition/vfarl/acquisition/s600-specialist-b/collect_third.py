"""Probe exact native requirements before scaling two further parser repositories."""
import argparse
from pathlib import Path
import collect as base

HOME=Path(__file__).resolve().parent
base.HERE=HOME/'third-batch'
base.ART=base.REPLAY/'artifacts/s600-specialist-b-third'
base.PAGE_ROOTS=(HOME/'third-pages',)


def item(repo,snapshot,suffix,symbol,path,sub,contract,body):
    return {'instance_id':repo.replace('/','__')+'.'+snapshot+'.func_basic__'+suffix,'canonical_repo':repo,'snapshot':snapshot,
            'symbol':symbol,'relative_path':path,'submechanism':sub,'contract':contract,'body':body,'declared_license':'MIT',
            'repair_family_id':repo+'::'+symbol+'::'+contract.split('.',1)[0]}


base.CANDIDATES={
    'pyparsing-located':item('pyparsing/pyparsing','533adf47','9fc57jao','Located.parseImpl','pyparsing/core.py','S2',
        'Parser location wrapper preserves token span and payload. Match start and end remain actual source offsets, the original token value is retained, and the next parser advances to the true end with named grouping intact.',
        "import pyparsing as pp\nexpression=pp.Located(pp.Word(pp.alphas))\ntext='  alpha beta'\nresult=expression.parse_string(text)\nprint('located',result.as_list(),result.dump())\nassert result.locn_start==2 and result.locn_end==7 and result.value.as_list()==['alpha']\nassert text[result.locn_start:result.locn_end]=='alpha'\nnamed=expression('group').parse_string('alpha')\nassert named.group.locn_start==0 and named.group.locn_end==5 and named.group.value.as_list()==['alpha']"),
    'parso-token-end':item('davidhalter/parso','338a5760','ty2loajw','Token.end_pos','parso/python/tokenize.py','S2',
        'Lexer token end position uses line-column coordinates and exclusive span boundaries. Single-line token ends add its width, and a terminal newline advances exactly one line to column zero.',
        "from parso.python.tokenize import PythonToken,NAME,NEWLINE\na=PythonToken(NAME,'alpha',(3,4),'');b=PythonToken(NEWLINE,'\\n',(3,9),'')\nprint('token_ends',a.end_pos,b.end_pos)\nassert a.end_pos==(3,9) and b.end_pos==(4,0)"),
}
base.CANDIDATES['pyparsing-located']['task_site_packages']=str(base.REPLAY/'.venv-pyparsing-task/lib/python3.12/site-packages')
base.CANDIDATES['pyparsing-located']['task_environment']={'MPLBACKEND':'Agg'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',nargs='+',choices=list(base.CANDIDATES),default=list(base.CANDIDATES))
    parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all')
    args=parser.parse_args()
    for key in args.keys:
        if args.stage in {'prepare','all'}:base.prepare(key);print('prepared',key,flush=True)
        if args.stage in {'replay','all'}:base.replay(key);print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:base.publish(args.keys)
