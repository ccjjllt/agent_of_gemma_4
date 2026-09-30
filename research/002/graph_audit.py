#!/usr/bin/env python3
"""1A CPU index audit (stdlib only): graph/embedding coverage vs source.

Usage:
  python3 research/002/graph_audit.py --graph <graph.json> --npz <emb.npz> \
      --source-root <snapshot dir> --out <result.json>

Inputs are official competition files + extracted snapshot (read-only).
Never reads task patch/test_patch. Prints JSON + writes --out.
Checks: node/edge counts, edge types, node text presence, short-name
ambiguity, neighbor-degree truncation risk (wrapper sorts+truncates 50),
embedding dim/finite/zero/dup, random-pair cosine baseline, top-1 margin
for a sample query, graph<->embedding key overlap, source AST
def/class/async coverage with proper qualnames, snapshot HEAD vs expected.
"""
import argparse
import ast
import json
import math
import random
import struct
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def load_npy(data: bytes):
    assert data[:6] == b'\x93NUMPY'
    hlen = struct.unpack('<H', data[8:10])[0]
    header = ast.literal_eval(data[10:10 + hlen].decode('latin1'))
    return header, data[10 + hlen:]


class Visitor(ast.NodeVisitor):
    def __init__(self, mod):
        self.stack = []
        self.mod = mod
        self.defs = {}

    def _q(self, name):
        return '.'.join([self.mod] + self.stack + [name])

    def visit_ClassDef(self, node):
        self.defs[self._q(node.name)] = 'class'
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.defs[self._q(node.name)] = 'def'
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.defs[self._q(node.name)] = 'async'
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--expected-commit', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = {}

    g = json.loads(Path(args.graph).read_text(encoding='utf-8'))
    nodes, edges = g['nodes'], g['edges']
    names = [n['name'] for n in nodes]
    out['graph'] = {
        'nodes': len(nodes), 'unique_names': len(set(names)),
        'edges': len(edges), 'edge_types': dict(Counter(e.get('type') for e in edges)),
        'empty_text': sum(1 for n in nodes if not n.get('text')),
    }
    short = defaultdict(list)
    for n in names:
        short[n.split('.')[-1]].append(n)
    amb = {s: len(v) for s, v in short.items() if len(v) > 1}
    out['graph']['ambiguous_short'] = len(amb)
    deg = defaultdict(set)
    for e in edges:
        deg[e['source']].add(e['target'])
        deg[e['target']].add(e['source'])
    degs = [len(deg[n]) for n in names]
    out['graph']['degree_max'] = max(degs)
    out['graph']['degree_gt50_truncated_by_wrapper'] = sum(1 for d in degs if d > 50)

    z = zipfile.ZipFile(args.npz)
    vecs = {}
    for n in z.namelist():
        h, d = load_npy(z.read(n))
        assert h['descr'] == '<f4' and h['shape'] == (256,) and not h['fortran_order'], (n, h)
        vecs[n[:-4]] = struct.unpack('<256f', d[:1024])
    out['embeddings'] = {'vectors': len(vecs), 'dim': 256}
    out['embeddings']['nonfinite'] = sum(1 for v in vecs.values() for x in v if x != x or x in (float('inf'), float('-inf')))
    out['embeddings']['zero_vectors'] = sum(1 for v in vecs.values() if all(x == 0.0 for x in v))
    seen = set()
    dup = 0
    for v in vecs.values():
        b = struct.pack('<256f', *v)
        if b in seen:
            dup += 1
        seen.add(b)
    out['embeddings']['exact_duplicates'] = dup

    def cos(a, b):
        d = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return sum(x * y for x, y in zip(a, b)) / d if d else 0.0

    random.seed(20260930)
    keys = list(vecs)
    pairs = sorted(cos(vecs[a], vecs[b]) for a, b in (random.sample(keys, 2) for _ in range(2000)))
    out['embeddings']['random_pair_cosine'] = {'p50': pairs[1000], 'p95': pairs[1900], 'max': pairs[-1]}
    q = 'rich.console.Console.print' if 'rich.console.Console.print' in vecs else keys[len(keys) // 2]
    sims = sorted(((cos(vecs[q], v), k) for k, v in vecs.items() if k != q), reverse=True)
    out['embeddings']['sample_query'] = {'query': q, 'top1': [sims[0][1], sims[0][0]],
                                         'top1_margin_over_median': sims[0][0] - pairs[1000]}

    gnames = set(names)
    enames = set(vecs)
    out['key_overlap'] = {'graph': len(gnames), 'emb': len(enames),
                          'graph_not_emb': len(gnames - enames), 'emb_not_graph': len(enames - gnames)}

    root = Path(args.source_root)
    defs = {}
    nfiles = 0
    for fp in sorted(root.rglob('*.py')):
        if any(p.startswith('.') for p in fp.relative_to(root).parts) or '__pycache__' in fp.parts:
            continue
        try:
            tree = ast.parse(fp.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            continue
        nfiles += 1
        v = Visitor(fp.relative_to(root).as_posix()[:-3].replace('/', '.'))
        v.visit(tree)
        defs.update(v.defs)
    kinds = Counter(defs.values())
    missing = [d for d in defs if d not in gnames]
    out['source'] = {'files': nfiles, 'symbols': len(defs), 'kinds': dict(kinds),
                     'covered': sum(1 for d in defs if d in gnames),
                     'missing_by_kind': dict(Counter(defs[d] for d in missing)),
                     'missing_async': [d for d in missing if defs[d] == 'async'],
                     'graph_not_in_source': len(gnames - set(defs))}
    r = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    head = r.stdout.strip() if r.returncode == 0 else None
    out['snapshot_head'] = head
    out['expected_commit'] = args.expected_commit
    out['commit_match'] = (head == args.expected_commit) if args.expected_commit else None

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
