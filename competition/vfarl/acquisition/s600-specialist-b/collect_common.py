"""Acquire reviewed, independent common-course contracts and replay the real tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import collect as base

HOME = Path(__file__).resolve().parent
base.HERE = HOME / 'common-batch'
base.ART = base.REPLAY / 'artifacts/s600-common-b'
base.PAGE_ROOTS = (HOME / 'common-pages',)


def item(repo, snapshot, suffix, symbol, path, sub, contract, body, license):
    return {'instance_id': repo.replace('/', '__') + '.' + snapshot + '.func_basic__' + suffix,
            'canonical_repo': repo, 'snapshot': snapshot, 'symbol': symbol,
            'relative_path': path, 'submechanism': sub, 'primary_adapter': sub[0],
            'contract': contract, 'body': body, 'declared_license': license,
            'course_membership': 'S600_common',
            'task_site_packages': str(base.REPLAY / '.venv-common-b-task/lib/python3.12/site-packages'),
            'repair_family_id': repo + '::' + symbol + '::' + contract.split('.', 1)[0]}


base.CANDIDATES = {
    'furl-strip-scheme': item('gruns/furl', 'da386f68', 'agxy6lvw', 'strip_scheme', 'furl/furl.py', 'S2',
        'Scheme-prefix slicing must preserve every character after an optional scheme and colon. A URL without a scheme remains unchanged, and HTTP authority delimiters remain in the returned suffix.',
        "values=[m.strip_scheme(s) for s in ('http://example.com/a','example.com/a','//example.com/a','mailto:user@example.com')]\nprint('scheme_suffixes',values)\nassert values==['//example.com/a','example.com/a','//example.com/a','user@example.com']", 'Unlicense'),
    'bleach-nofollow': item('mozilla/bleach', '73871d76', '668s0dvv', 'nofollow', 'bleach/callbacks.py', 'S3',
        'Link relation augmentation preserves the existing relation list and adds nofollow once under case-insensitive membership. HTTP links receive the relation; mailto links retain their original attributes.',
        "href=(None,'href');rel=(None,'rel')\na={href:'http://example.com',rel:'external'};b={href:'https://example.com',rel:'NOFOLLOW'};c={href:'mailto:a@example.com'}\nassert m.nofollow(a) is a and a[rel]=='external nofollow'\nassert m.nofollow(b) is b and b[rel]=='NOFOLLOW'\nassert m.nofollow(c) is c and rel not in c\nprint('link_relations',a,b,c)", 'Apache-2.0'),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', choices=list(base.CANDIDATES), default=list(base.CANDIDATES))
    parser.add_argument('--stage', choices=('prepare', 'replay', 'publish', 'all'), default='all')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.stage in {'prepare', 'all'}:
        for key in args.keys:
            base.prepare(key)
            print('prepared', key, flush=True)
    if args.stage in {'replay', 'all'}:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in pool.map(lambda key: (base.replay(key), key)[1], args.keys):
                print('replayed', key, flush=True)
    if args.stage in {'publish', 'all'}:
        base.publish(args.keys)
