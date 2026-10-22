#!/usr/bin/env python3
"""Prepare a pinned, train-only public-trajectory sample and token census.

This is a data producer. It does not replay commands or claim local task success.
Metadata projection and train assignment precede every new body read.
"""
import argparse
import collections
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE.parent / 'data'
SEED = 'p016-source-profile-20261022'
SMITH_REV = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'
REAL_REV = '879c96b91dfa321b4d1c621d149a9200f8a1d118'
SOURCES = {'lego-real': 'Lego-X/SWE-Lego-Real-Data',
           'lego-synthetic': 'Lego-X/SWE-Lego-Synthetic-Data',
           'swe-smith': 'SWE-bench/SWE-smith-trajectories'}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def get_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception as exc:
            if attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                delay = min(30, int(exc.headers.get('Retry-After', '30')))
            time.sleep(delay)


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.bytes_read = url, size, 0, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.pos + size)
        if stop <= self.pos: return b''
        request = urllib.request.Request(self.url + '?p016_range=%d-%d' % (self.pos, stop - 1),
                                         headers={'Range': 'bytes=%d-%d' % (self.pos, stop - 1)})
        cache = DATA / 'metadata-ranges' / (digest(self.url) + '-%d-%d.bin' % (self.pos, stop - 1))
        if cache.exists(): body = cache.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if response.status != 206:
                            raise RuntimeError('Server ignored metadata range: refusing full body download')
                        body = response.read()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(body)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2 ** (attempt + 1))
        if len(body) != stop - self.pos: raise RuntimeError('Invalid range length')
        self.pos = stop
        self.bytes_read += len(body)
        return body


def repo_from_id(iid):
    return re.split(r'\.[0-9a-f]{8}\.', iid)[0].replace('__', '/', 1).lower()


def metadata():
    import pyarrow.parquet as pq
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / 'metadata-lego-real.json').exists():
        old = read(ROOT / 'research/015/data/source-metadata.json')
        dump(DATA / 'metadata-lego-real.json', old)
    if not (DATA / 'metadata-lego-synthetic.json').exists():
        dataset = SOURCES['lego-synthetic']
        rev = get_json('https://huggingface.co/api/datasets/' + dataset)['sha']
        files = get_json('https://huggingface.co/api/datasets/' + dataset + '/tree/' + rev + '/data')
        file = next(f for f in files if f['path'] == 'data/resolved-00000-of-00001.parquet')
        url = 'https://huggingface.co/datasets/%s/resolve/%s/%s' % (dataset, rev, file['path'])
        stream = RangeReader(url, file['size'])
        columns = ['instance_id', 'repo', 'resolved', 'image_name', 'difficulty']
        rows = pq.ParquetFile(stream).read(columns=columns, use_threads=False).to_pylist()
        # Original repo identity is encoded in instance_id; repo is a synthetic mirror.
        projected = []
        for i, r in enumerate(rows):
            original = repo_from_id(r['instance_id'])
            projected.append(dict(r, row_index=i, mirror_repo=r['repo'], repo=original))
        dump(DATA / 'metadata-lego-synthetic.json', {
            'source_dataset': dataset, 'revision': rev, 'columns_read': columns,
            'rows': projected, 'projected_bytes_read': stream.bytes_read,
            'source_file_bytes': file['size'], 'source_file': file,
            'trajectory_content_read': False})
        print('synthetic metadata', len(rows), stream.bytes_read, flush=True)
    if not (DATA / 'metadata-swe-smith.json').exists():
        rows = []
        shards = read(ROOT / 'research/011/data-scaling-20261018/shard-manifest.json')
        for shard in shards:
            path = Path(shard['path'])
            if not path.exists(): raise RuntimeError('Missing existing shard: ' + str(path))
            projection = pq.ParquetFile(path).read(columns=['instance_id','resolved','model','traj_id']).to_pylist()
            for index, row in enumerate(projection):
                iid, resolved, model, traj = [row[k] for k in ('instance_id','resolved','model','traj_id')]
                rows.append(dict(row_index=index, instance_id=iid, repo=repo_from_id(iid),
                                 resolved=resolved, teacher_source=model, traj_id=traj,
                                 source_file=str(path.relative_to(ROOT))))
        dump(DATA / 'metadata-swe-smith.json', {'source_dataset': SOURCES['swe-smith'],
             'revision': SMITH_REV, 'columns_read': ['file_row_number', 'instance_id', 'resolved', 'model', 'traj_id'],
             'rows': rows, 'shards': shards, 'trajectory_content_read': False})
        print('smith metadata', len(rows), flush=True)


def balanced(rows, count):
    groups = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: digest(SEED + r['instance_id'] + r.get('traj_id', ''))):
        groups[row['repo']].append(row)
    selected = []
    while groups and len(selected) < count:
        for repo in sorted(groups, key=lambda r: digest(SEED + r)):
            if groups[repo] and len(selected) < count: selected.append(groups[repo].pop(0))
        groups = {k: v for k, v in groups.items() if v}
    return selected


def task_key(row, source):
    iid = row['instance_id']
    match = re.search(r'\.pr_(\d+)$', iid)
    if source == 'lego-real': match = re.search(r'-(\d+)$', iid)
    return row['repo'] + '::pr:' + match.group(1) if match else iid.lower()


def freeze():
    path = HERE / 'source-sample-freeze.json'
    if path.exists(): raise RuntimeError('Sample already frozen; preserve this version')
    policy = read(HERE / 'train-family-policy.json')
    allowed, denied = set(policy['allowed_train_repositories']), set(policy['excluded_repositories'])
    old_real = {r['instance_id'] for r in read(ROOT / 'research/015/data-preparation/train-identity-freeze.json')['selected']}
    selected, summaries, admitted = [], {}, set()
    for source in SOURCES:
        meta = read(DATA / ('metadata-' + source + '.json'))
        candidates = [r for r in meta['rows'] if r['repo'] in allowed and r['repo'] not in denied and r['resolved']]
        unique = {}
        for r in sorted(candidates, key=lambda r: digest(SEED + r['instance_id'] + r.get('traj_id', ''))):
            unique.setdefault(r['instance_id'], r)
        candidates = [r for r in unique.values() if task_key(r, source) not in admitted]
        if source == 'lego-real':
            cached = [r for r in candidates if r['instance_id'] in old_real]
            chosen = cached + balanced([r for r in candidates if r['instance_id'] not in old_real], 48 - len(cached))
        elif source == 'lego-synthetic': chosen = balanced(candidates, 48)
        else:
            chosen = balanced([r for r in candidates if re.search(r'\.pr_\d+$', r['instance_id'])], 32)
            chosen += balanced([r for r in candidates if not re.search(r'\.pr_\d+$', r['instance_id'])], 32)
        for row in chosen:
            selected.append(dict(row, source=source, dataset=SOURCES[source], revision=meta['revision'],
                                 split='train', canonical_task_key=task_key(row, source),
                                 source_split='tool' if source == 'swe-smith' else 'resolved'))
            admitted.add(task_key(row, source))
        summaries[source] = {'eligible_unique_issues': len(candidates), 'selected': len(chosen),
                             'selected_repositories': len({r['repo'] for r in chosen}),
                             'revision': meta['revision']}
    dump(path, {'frozen_before_new_content_read': True, 'seed': SEED,
                'policy_sha256': digest((HERE / 'train-family-policy.json').read_bytes()),
                'selection': 'Historical train families only; deterministic repository round robin; Lego Real reuses 32 previously frozen train rows then adds 16; Synthetic 48; Smith PR32 + synthetic32. Unique issue within source and canonical repo/PR or synthetic identity across sources. No body length selection or outcome replacement.',
                'source_summaries': summaries, 'selected': selected})
    print(json.dumps(summaries), flush=True)


def acquire():
    policy = read(HERE / 'train-family-policy.json')
    freeze = read(HERE / 'source-sample-freeze.json')
    if freeze['policy_sha256'] != digest((HERE / 'train-family-policy.json').read_bytes()):
        raise RuntimeError('Policy changed after freeze')
    con = None
    records = []
    live = {}
    for item in freeze['selected']:
        if args.source and item['source'] not in args.source: continue
        if item['split'] != 'train' or item['repo'] not in policy['allowed_train_repositories'] or item['repo'] in policy['excluded_repositories']:
            raise RuntimeError('Refusing nontrain body')
        source, iid = item['source'], item['instance_id']
        path = DATA / 'source-rows' / source / (iid + '.json')
        if path.exists():
            row = read(path)
        elif source == 'lego-real' and (ROOT / 'research/015/data/source-rows' / (iid + '.json')).exists():
            original = ROOT / 'research/015/data/source-rows' / (iid + '.json')
            row = read(original)['rows'][0]['row']
            dump(path, row)
        elif source == 'swe-smith':
            if con is None:
                import duckdb
                con = duckdb.connect()
            got = con.execute('SELECT instance_id,traj_id,messages,model,resolved FROM read_parquet(?, file_row_number=true) WHERE file_row_number=? AND instance_id=? AND traj_id=?',
                [str(ROOT / item['source_file']), item['row_index'], iid, item['traj_id']]).fetchone()
            if not got: raise RuntimeError('Atomic Smith row mismatch')
            row = dict(instance_id=got[0], traj_id=got[1], messages=json.loads(got[2]), model=got[3], resolved=got[4])
            dump(path, row)
        else:
            if source not in live:
                live[source] = get_json('https://huggingface.co/api/datasets/' + item['dataset'])['sha']
            if live[source] != item['revision']: raise RuntimeError('Dataset server no longer matches pin')
            query = urllib.parse.urlencode({'dataset': item['dataset'], 'config': 'default',
                    'split': 'resolved', 'offset': item['row_index'], 'length': 1})
            response = get_json('https://datasets-server.huggingface.co/rows?' + query)
            rows = response.get('rows', [])
            if len(rows) != 1 or rows[0]['row']['instance_id'] != iid:
                raise RuntimeError('Dataset server row differs from pinned metadata')
            row = rows[0]['row']
            if item.get('base_commit') and row.get('base_commit') != item['base_commit']:
                raise RuntimeError('Base commit mismatch')
            if 'truncated_cells' not in rows[0] or rows[0]['truncated_cells']:
                raise RuntimeError('Refusing truncated or completeness-unknown HTTP row')
            dump(path, row)
            dump(DATA / 'source-receipts' / source / (iid + '.json'), {
                'instance_id':iid,'revision':item['revision'],'row_index':item['row_index'],
                'truncated_cells':rows[0]['truncated_cells'],'server_partial':response.get('partial'),
                'body_sha256':digest(path.read_bytes()),'method':'original HTTP row response'})
            time.sleep(.35)
        if row['instance_id'] != iid or not row['resolved']: raise RuntimeError('Source identity/status mismatch')
        if item.get('base_commit') and row.get('base_commit') != item['base_commit']:
            raise RuntimeError('Cached base commit mismatch')
        if item.get('traj_id') and row.get('traj_id') != item['traj_id']:
            raise RuntimeError('Cached trajectory identity mismatch')
        records.append(dict(item, local_path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                            sha256=digest(path.read_bytes()), evidence='upstream resolved label; not locally revalidated'))
        print('acquired', len(records), source, iid, flush=True)
    for source in live:
        if get_json('https://huggingface.co/api/datasets/' + SOURCES[source])['sha'] != live[source]:
            raise RuntimeError('Dataset revision changed during acquisition')
    dump(HERE / 'source-acquisition.json', {'rows': records,
         'frozen_selection_sha256': digest((HERE / 'source-sample-freeze.json').read_bytes()),
         'body_scope': 'Only exactly frozen train rows. Smith final patch column deliberately not read.',
         'network_bytes': 'Row payload sizes recorded; upstream HTTP overhead not measured.'})


def calls(message):
    out = []
    for call in message.get('tool_calls', []) or []:
        fn = call.get('function', call)
        args = fn.get('arguments', {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except ValueError: args = {'_malformed_json': args}
        out.append((fn.get('name', 'unknown'), args))
    content = message.get('content', '')
    if isinstance(content, str):
        for name, body in re.findall(r'<function=([^>]+)>(.*?)</function>', content, re.S):
            out.append((name, {k: v.strip('\n') for k, v in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S)}))
    return out


def receipts():
    """Retain HTTP row completeness, matching the already frozen body exactly."""
    manifest = read(HERE / 'source-acquisition.json')
    source_revisions = {}
    for item in manifest['rows']:
        if item['source'] == 'swe-smith':
            item['transport'] = {'method':'atomic local Parquet row query','truncated':False}
            continue
        target = DATA / 'source-receipts' / item['source'] / (item['instance_id'] + '.json')
        if target.exists():
            receipt = read(target)
        else:
            old = ROOT / 'research/015/data/source-rows' / (item['instance_id'] + '.json')
            if item['source'] == 'lego-real' and old.exists():
                response = read(old)
                method = 'reused P015 exact HTTP response envelope'
            else:
                if item['dataset'] not in source_revisions:
                    current = get_json('https://huggingface.co/api/datasets/' + item['dataset'])['sha']
                    if current != item['revision']: raise RuntimeError('Receipt revision differs from frozen body')
                    source_revisions[item['dataset']] = current
                query = urllib.parse.urlencode({'dataset':item['dataset'],'config':'default',
                    'split':'resolved','offset':item['row_index'],'length':1})
                response = get_json('https://datasets-server.huggingface.co/rows?' + query)
                method = 're-read identical frozen train row; body equality confirmed'
                time.sleep(2)
            rows = response.get('rows', [])
            if len(rows) != 1 or rows[0].get('row_idx') != item['row_index']:
                raise RuntimeError('Receipt row index mismatch')
            if rows[0]['row'] != read(ROOT / item['local_path']):
                raise RuntimeError('Receipt body differs from frozen body')
            if 'truncated_cells' not in rows[0] or rows[0]['truncated_cells']:
                raise RuntimeError('Missing or nonempty server truncated_cells')
            receipt = {'instance_id':item['instance_id'],'revision':item['revision'],
                'row_index':item['row_index'],'truncated_cells':rows[0]['truncated_cells'],
                'server_partial':response.get('partial'), 'body_sha256':item['sha256'],
                'method':method}
            dump(target, receipt)
        if receipt['body_sha256'] != item['sha256']: raise RuntimeError('Receipt hash mismatch')
        item['transport'] = receipt
        print('complete HTTP row',item['source'],item['instance_id'],flush=True)
    for dataset, rev in source_revisions.items():
        if get_json('https://huggingface.co/api/datasets/' + dataset)['sha'] != rev:
            raise RuntimeError('Source changed during receipt acquisition')
    dump(HERE / 'source-acquisition.json',manifest)


def content_text(value):
    if isinstance(value, str): return value
    if value is None: return ''
    if isinstance(value, list): return '\n'.join(v.get('text', '') for v in value if isinstance(v, dict))
    return json.dumps(value, ensure_ascii=False)


def census():
    sys.path.insert(0, str(ROOT))
    from training.encoding.gemma_tool_chat import load_tokenizer
    from training.data.sample_behaviors import _iter_invoked_argv, _prog_name
    tokpath = ROOT / 'research/015/tokenizer-assets/official-31b-ct-v2-tokenizer'
    tok = load_tokenizer(str(tokpath))
    template = (tokpath / 'chat_template.jinja').read_text()
    tools = read(ROOT / 'training/configs/harness_tools.json')
    records = []
    for item in read(HERE / 'source-acquisition.json')['rows']:
        path = ROOT / item['local_path']
        if digest(path.read_bytes()) != item['sha256']: raise RuntimeError('Source body changed')
        row = read(path)
        messages = row['messages']
        if isinstance(messages, str): messages = json.loads(messages)
        # Diagnostic representation only: preserve native source calls as text.
        rendered = []
        functions = collections.Counter()
        edits = verifies = finishes = read_actions = malformed = 0
        obstacles = set()
        for m in messages:
            body = content_text(m.get('content'))
            tool_calls = calls(m) if m['role'] == 'assistant' else []
            for name, args in tool_calls:
                functions[name] += 1
                if '_malformed_json' in args: malformed += 1
                command = args.get('command', '')
                if name in ('str_replace_editor', 'str_replace_based_edit_tool'):
                    if command == 'view': read_actions += 1
                    elif command in ('str_replace', 'create', 'insert', 'undo_edit'): edits += 1
                    if command in ('insert', 'undo_edit'): obstacles.add('stateful_editor_' + command)
                elif name in ('bash', 'execute_bash', 'terminal', 'run_command'):
                    obstacles.add('shell_cwd_timeout_and_output_semantics')
                    for argv in _iter_invoked_argv(str(command)):
                        names = [_prog_name(a) for a in argv]
                        if names and (names[0] in ('pytest', 'py.test', 'tox', 'nosetests') or
                            (names[0].startswith('python') and '-m' in names and any(x in names for x in ('pytest','unittest','tox')))):
                            verifies += 1; break
                elif name in ('finish', 'submit', 'submit_patch'): finishes += 1
                elif name in ('think', 'task_tracker'): obstacles.add('source_only_' + name)
                else: obstacles.add('unknown_tool_' + name)
            if m.get('tool_calls'):
                body += '\n' + json.dumps({'tool_calls': m['tool_calls']}, ensure_ascii=False, sort_keys=True)
                obstacles.add('native_tool_json_not_target_schema')
            role = m['role']
            if role == 'tool':
                role = 'user'
                body = '[source tool response %s]\n%s' % (m.get('name', m.get('tool_call_id', '')), body)
                obstacles.add('source_tool_response_requires_target_semantic_mapping')
            rendered.append({'role': role, 'content': body})
        ids = tok.apply_chat_template(rendered, tools=tools, chat_template=template,
                                      tokenize=True, add_generation_prompt=False,
                                      enable_thinking=True, preserve_thinking=True)
        length = len(ids['input_ids']) if hasattr(ids, 'keys') else len(ids)
        assistant_tokens = sum(len(tok.encode(m['content'], add_special_tokens=False)) for m in rendered if m['role'] == 'assistant')
        entry = {k: item[k] for k in ('source','dataset','revision','instance_id','repo','sha256','local_path')}
        entry.update(source_full_template_tokens=length, assistant_content_tokens=assistant_tokens,
                     source_messages=len(messages), source_assistant_turns=sum(m['role']=='assistant' for m in messages),
                     native_tool_calls=dict(functions), editor_actions=edits, explicit_view_actions=read_actions,
                     invoked_test_runner_actions=verifies, finish_actions=finishes,
                     malformed_argument_calls=malformed, conversion_obstacles=sorted(obstacles),
                     source_resolved_label=True, task_replayed=False, target_protocol_training_ready=False)
        records.append(entry)
        print('profiled',len(records),item['source'],length,flush=True)
    summary = {}
    for source in SOURCES:
        rows = [r for r in records if r['source'] == source]
        if not rows: continue
        lengths = sorted(r['source_full_template_tokens'] for r in rows)
        summary[source] = {'issues': len(rows), 'repositories': len({r['repo'] for r in rows}),
          'min': min(lengths), 'median': lengths[len(lengths)//2], 'p90': lengths[min(len(lengths)-1,int(len(lengths)*.9))],
          'max': max(lengths), 'fit_counts': {str(n):sum(x<=n for x in lengths) for n in (8192,16384,32768,65536,131072)},
          'source_total_tokens':sum(lengths), 'assistant_content_tokens':sum(r['assistant_content_tokens'] for r in rows),
          'with_editor_action':sum(r['editor_actions']>0 for r in rows),
          'with_test_runner_action':sum(r['invoked_test_runner_actions']>0 for r in rows),
          'with_finish_action':sum(r['finish_actions']>0 for r in rows)}
    dump(HERE / 'source-profile.json', {'summary':summary,'rows':records,
       'frozen_sample_issues':len(read(HERE / 'source-sample-freeze.json')['selected']),
       'profiled_issues':len(records),
       'measurement':'Complete source-native content in official Gemma31B CTv2 template, with existing six-tool schema overhead. Native structured source calls serialized losslessly as JSON text; tool observations labeled as source responses. This is comparable diagnostic token length, NOT semantically converted competition protocol or trainable label encoding.',
       'tokenizer_sha256':digest((tokpath/'tokenizer.json').read_bytes()),
       'template_sha256':digest(template),
       'assistant_content_tokens_are_not_accepted_training_targets':True,
       'behavior_counts_are_syntactic_not_execution_success':True})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['metadata','freeze','acquire','receipts','census'])
    parser.add_argument('--source', action='append', choices=list(SOURCES), help='Optional acquisition source subset')
    args = parser.parse_args()
    globals()[args.mode]()
