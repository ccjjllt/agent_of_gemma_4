#!/usr/bin/env python3
"""Produce full source-native SWE-smith/Gemma offline SFT candidate shards.

This does not execute repositories or convert to the competition's six tools.
Only identities frozen as train may be opened. Source observations stay verbatim;
terminal submit needs no invented response. Licenses and transfer remain admission
work, so produced tensor records are deliberately not marked training_eligible.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time
from collections import Counter
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from training.encoding.gemma_tool_chat import (EncodingError, encode_transaction, count_template_tokens,
    load_tokenizer, normalize_tools, sha256_json, tokenizer_fingerprint)

OUT = ROOT / 'research/016/data/encoded-native'
UPSTREAM = Path(__file__).parent / 'upstream'
DATA = ROOT / 'research/016/data-preparation'
PROTOCOL_ID = 'p016-swesmith-native-bash-editor-submit-v2'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def source_schema():
    """Reproduce official Command.get_function_calling_tool, without execution."""
    bash = {'type': 'function', 'function': {'name': 'bash',
        'description': 'runs the given command directly in bash',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string',
            'description': 'The bash command to execute.'}}, 'required': ['command']}}}
    tools = [bash]
    for filename in ('sweagent-editor.yaml', 'sweagent-submit.yaml'):
        source = yaml.safe_load((UPSTREAM / filename).read_text())
        for name, spec in source['tools'].items():
            properties, required = {}, []
            for arg in spec.get('arguments', []):
                properties[arg['name']] = {k: copy.deepcopy(arg[k])
                    for k in ('type', 'description', 'items', 'enum') if k in arg}
                if arg['required']:
                    required.append(arg['name'])
            parameters = {'type': 'object'}
            if properties:
                parameters.update(properties=properties, required=required)
            tools.append({'type': 'function', 'function': {'name': name,
                'description': spec.get('docstring', ''), 'parameters': parameters}})
    return normalize_tools(tools)[0]


def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, dict) and x.get('type') == 'text'
                                     and isinstance(x.get('text'), str) for x in value):
        return ''.join(x['text'] for x in value)
    raise EncodingError('source content is not lossless text')


def typed_arguments(name, args, schema):
    parameters = schema[name]['parameters']
    properties = parameters.get('properties', {})
    if not isinstance(args, dict) or set(args) - set(properties):
        raise EncodingError('unknown source tool arguments: ' + name)
    if set(parameters.get('required', [])) - set(args):
        raise EncodingError('missing source tool arguments: ' + name)
    for key, value in args.items():
        spec = properties[key]
        valid = {'string': lambda x: isinstance(x, str),
                 'integer': lambda x: type(x) is int,
                 'array': lambda x: isinstance(x, list) and all(type(v) is int for v in x)}
        if not valid[spec['type']](value):
            raise EncodingError('wrong source argument type: ' + name + '.' + key)
        if spec.get('enum') and value not in spec['enum']:
            raise EncodingError('unknown source editor command')
    return args


def xml_call(content, schema, source_index):
    matches = list(re.finditer(r'<function=([A-Za-z_][A-Za-z_0-9]*)>', content))
    if len(matches) != 1 or not content.rstrip().endswith('</function>'):
        raise EncodingError('XML assistant must contain exactly one complete terminal function block')
    found = matches[0]
    name = found[1]
    if name not in schema:
        raise EncodingError('unknown XML function: ' + name)
    body = content[found.end():content.rfind('</function>')]
    args, position = {}, 0
    for param in re.finditer(r'<parameter=([A-Za-z_][A-Za-z_0-9]*)>(.*?)</parameter>', body, re.S):
        if body[position:param.start()].strip():
            raise EncodingError('unparsed XML parameter bytes')
        key, value = param[1], param[2]
        if key in args or key not in schema[name]['parameters'].get('properties', {}):
            raise EncodingError('duplicate or unknown XML parameter')
        if '<parameter=' in value or '</function>' in value:
            raise EncodingError('ambiguous XML parameter boundary')
        if schema[name]['parameters']['properties'][key]['type'] != 'string':
            value = json.loads(value)
        args[key] = value
        position = param.end()
    if body[position:].strip():
        raise EncodingError('unparsed XML function body')
    return content[:found.start()], {'id': 'source-xml-%d' % source_index, 'type': 'function',
        'function': {'name': name, 'arguments': typed_arguments(name, args, schema)}}


def adapt_system_serialization(content):
    """Replace only the source XML-output contract, retaining tool/task semantics."""
    start_marker = 'If you choose to call a function ONLY reply in the following format with NO suffix:'
    if start_marker not in content:
        if '<function=' in content:
            raise EncodingError('unrecognized XML serialization instruction; needs explicit adaptation')
        return content, None
    if content.count(start_marker) != 1:
        raise EncodingError('ambiguous XML serialization instruction block')
    start = content.index(start_marker)
    tail = content[start:]
    reminder = '<IMPORTANT>\nReminder:\n'
    if tail.count(reminder) != 1:
        raise EncodingError('unknown source XML reminder structure')
    _, rules = tail.split(reminder, 1)
    format_rule = '- Function calls MUST follow the specified format, start with <function= and end with </function>'
    if rules.count(format_rule) != 1 or not rules.rstrip().endswith('</IMPORTANT>'):
        raise EncodingError('unknown source XML format rule')
    rules = rules.replace(format_rule,
        '- Function calls MUST use the declared Gemma native tool-call format.')
    replacement = ('When calling a function, use the declared Gemma native tool-call format '
        'with typed arguments. Put any reasoning in the thought channel before the tool call. '
        'Do not append user-facing content after a tool call.\n\n' + reminder + rules)
    adapted = content[:start] + replacement
    sha = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    return adapted, {'version': 'xml-output-contract-to-gemma-native-v2',
        'source_system_sha256': sha(content), 'adapted_system_sha256': sha(adapted),
        'original_character_span': [start, len(content)], 'original_block_sha256': sha(tail),
        'replacement': replacement, 'replacement_sha256': sha(replacement),
        'preserved_prefix_sha256': sha(content[:start])}


def convert(row, identity, tools):
    schema = {x['function']['name']: x['function'] for x in tools}
    messages, call_positions, formats, masked = [], {}, Counter(), {}
    pending = None
    system_rewrites = []
    for source_index, message in enumerate(row['messages']):
        role = message['role']
        if role in ('system', 'user') and pending is None:
            if role == 'user' and any(x['role'] == 'assistant' for x in messages):
                raise EncodingError('unexpected non-observation user after actions')
            content = text(message['content'])
            if role == 'system':
                content, rewrite = adapt_system_serialization(content)
                if rewrite:
                    system_rewrites.append(dict(rewrite, source_message_index=source_index))
            messages.append({'role': role, 'content': content})
            continue
        if role == 'assistant':
            if pending:
                raise EncodingError('missing source observation before next action')
            content = text(message['content'])
            calls = message.get('tool_calls')
            if calls:
                if len(calls) != 1 or message.get('thought') != content:
                    raise EncodingError('ambiguous native assistant thought/action structure')
                call = copy.deepcopy(calls[0]); call.pop('index', None)
                fn = call['function']; name = fn['name']
                if name not in schema:
                    raise EncodingError('unknown native function')
                args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
                fn['arguments'] = typed_arguments(name, args, schema)
                thought = content
                formats['native_function_calling'] += 1
            else:
                thought, call = xml_call(content, schema, source_index)
                formats['xml_function_calling'] += 1
            if not isinstance(call.get('id'), str) or call['id'] in call_positions:
                raise EncodingError('duplicate or absent source action id')
            ordinal = len(call_positions)
            call_positions[call['id']] = ordinal
            messages.append({'role': 'assistant', 'content': '',
                             'reasoning_content': thought, 'tool_calls': [call]})
            pending = call
            if message.get('turn_mask') is True:
                masked[ordinal] = 'source turn_mask=true'
            continue
        if role not in ('tool', 'user') or pending is None:
            raise EncodingError('orphan or unsupported source observation')
        observation = text(message['content'])
        if role == 'tool' and message.get('tool_call_ids') != [pending['id']]:
            raise EncodingError('source tool response identity mismatch')
        if role == 'user' and not (re.match(r'observation:', observation, re.I) or
                observation.startswith('Your command ran successfully and did not produce any output.')):
            raise EncodingError('XML observation prefix is not an environment response')
        ordinal = call_positions[pending['id']]
        # Only clearly diagnosed editor-operation failures are masked here.
        # Reproduction/test failures are retained; nonzero exits are not errors
        # by themselves. Broader quality admission is intentionally separate.
        if pending['function']['name'] == 'str_replace_editor' and any(s in observation for s in (
            'No replacement was performed', 'Invalid `command`', 'Invalid `path`',
            'Invalid `view_range`', 'does not exist. Please provide a valid path',
            'Multiple occurrences of string', 'file already exists')):
            masked[ordinal] = 'observed editor operation error'
        messages.append({'role': 'tool', 'name': pending['function']['name'],
                         'tool_call_id': pending['id'], 'content': observation})
        pending = None
    if not pending or pending['function']['name'] != 'submit' or pending['function']['arguments']:
        raise EncodingError('source trajectory must finish with complete submit action')
    selected = [i for i in range(len(call_positions)) if i not in masked]
    if len(call_positions) - 1 not in selected:
        raise EncodingError('source terminal submit is masked')
    tx = {'sample_id': row['traj_id'], 'transaction_id': row['traj_id'],
          'origin_kind': 'offline-source-native-expert-trajectory',
          'protocol_id': PROTOCOL_ID,
          'group': {'issue': identity['instance_id'], 'repo': identity['repo'], 'split': 'train'},
          'source': {'uri': identity['local_path'], 'sha256': identity['sha256'],
              'dataset': identity['dataset'], 'dataset_revision': identity['revision'],
              'license': 'MIT dataset declaration; upstream repository admission pending',
              'format': 'parsed-native-or-xml-to-gemma-native-tools'},
          'messages': messages, 'target_assistant_steps': selected,
          'allow_terminal_tool_call': True}
    return tx, {'conversion_version': PROTOCOL_ID, 'system_serialization_rewrites': system_rewrites,
                'source_formats': dict(formats), 'masked_assistant_steps': masked,
                'assistant_steps': len(call_positions), 'terminal_tool_response_absent': True}


def quality_review_flags(tx):
    """Specific operations needing review; a test filename alone is not a flag."""
    flags = []
    created_paths = set()
    for index, message in enumerate(tx['messages']):
        if message['role'] != 'assistant':
            continue
        function = message['tool_calls'][0]['function']
        args, name = function['arguments'], function['name']
        reason = None
        if name == 'str_replace_editor' and args.get('command') == 'create':
            created_paths.add(args['path'])
        if name == 'str_replace_editor' and args.get('command') == 'str_replace':
            path = args['path']
            is_test = bool(re.search(r'/(?:tests?|test_[^/]*)(?:/|\.py$)', path))
            before, after = args.get('old_str', ''), args.get('new_str', '')
            if is_test:
                if any(s in after and s not in before for s in (
                        'pytest.skip(', 'pytest.mark.skip', 'unittest.skip', '.skipTest(', 'assert True')):
                    reason = 'adds explicit skip or unconditional assertion to validation code'
                elif ('assert' in before and 'assert' not in after) or (
                        'skip' in after.lower() and 'test' in after.lower() and len(after) < len(before) / 2):
                    reason = 'removes validation assertions or replaces a check with a skip statement'
        elif name == 'bash':
            command = args['command']
            urls = re.findall(r'https?://[^\s\'\"]+', command)
            local_http_only = bool(urls) and all(urlsplit(url).hostname in
                ('localhost', '127.0.0.1', '::1') for url in urls)
            if re.search(r'\bgit\s+(?:fetch|pull|clone)\b', command) or (
                    re.search(r'\b(?:curl|wget)\b', command) and not local_http_only):
                reason = 'remote material retrieval requires provenance review; not presumed future-free'
            elif re.search(r'\brm\b[^\n;]*\s(?:\S*/)?tests?(?:/|\s|$)', command):
                try:
                    parts = shlex.split(command)
                    test_paths = [part for part in parts if re.search(r'(?:^|/)tests?(?:/|$)', part)]
                except ValueError:
                    test_paths = []
                if not test_paths or not all(path in created_paths for path in test_paths):
                    reason = 'removes a test directory or its contents; requires contextual review'
            elif re.search(r'\bgit\s+show\b|\bgit\s+log\b[^\n;]*(?:\s-p\b|--patch\b)', command):
                reason = 'reads non-working-tree source or patch history; needs revision provenance review'
            elif re.search(r'\bgit\s+diff\b[^\n;]*(?:\b[0-9a-f]{7,40}\b|HEAD[~^]|\b(?:origin|upstream)/)', command):
                reason = 'compares non-current Git revisions; needs revision provenance review'
            elif re.search(r'\bgit\s+(?:checkout|reset|restore)\b', command):
                match = re.search(r'\bgit\s+(checkout|reset|restore)\b([^\n;&|]*)', command)
                try:
                    operands = shlex.split(match[2]) if match else []
                except ValueError:
                    operands = []
                # Restoring named regression-test files to undo this trajectory's
                # own test edits does not expose an alternative production fix.
                paths = operands[operands.index('--') + 1:] if '--' in operands else []
                test_restore = bool(paths) and all(re.search(r'(?:^|/)(?:tests?|test_[^/]*)/', p)
                    or Path(p).name.startswith('test_') for p in paths)
                if not test_restore:
                    reason = 'restores or resets production state; needs task-specific provenance review'
        if reason:
            flags.append({'message_index': index, 'tool': name, 'reason': reason})
    return flags


def main():
    started = time.monotonic()
    producer_source = Path(__file__).read_bytes()
    producer_sha256 = hashlib.sha256(producer_source).hexdigest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', '--out', type=Path, default=OUT)
    ap.add_argument('--selection', type=Path, default=DATA / 'source-sample-freeze.json')
    ap.add_argument('--acquisition', type=Path, default=DATA / 'source-acquisition.json')
    ap.add_argument('--admission', type=Path, help='Explicit source-hash-bound native-SFT admission decisions')
    ap.add_argument('--admitted-only', action='store_true', help='Write only admitted rows; retain all decisions in the manifest')
    ap.add_argument('--thought-loss', nargs='+', choices=('include', 'mask'), default=['include', 'mask'])
    args = ap.parse_args()
    if args.admitted_only and not args.admission:
        ap.error('--admitted-only requires --admission')
    freeze_path = args.selection
    acquisition_path = args.acquisition
    freeze, acquisition = read(freeze_path), read(acquisition_path)
    admission = read(args.admission) if args.admission else {'rows': []}
    admitted = {x['instance_id']: x for x in admission['rows']}
    frozen = {x['instance_id']: x for x in freeze['selected'] if x['source'] == 'swe-smith'}
    rows = [x for x in acquisition['rows'] if x['source'] == 'swe-smith']
    if len(rows) != len(frozen) or len({x['canonical_task_key'] for x in rows}) != len(rows):
        raise ValueError('frozen source selection is incomplete or duplicated')
    if acquisition['frozen_selection_sha256'] != digest(freeze_path):
        raise ValueError('acquisition does not match frozen identities')
    tokenizer_path = ROOT / 'research/015/tokenizer-assets/official-31b-ct-v2-tokenizer'
    tokenizer = load_tokenizer(str(tokenizer_path))
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    template_path = tokenizer_path / 'chat_template.jinja'
    template = template_path.read_text()
    tools = source_schema()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'producer-source.py').write_bytes(producer_source)
    tools_path = args.output / 'swe-smith-native-tools.json'
    dump(tools_path, {'protocol_id': PROTOCOL_ID, 'tools': tools,
                     'schema_provenance': read(UPSTREAM / 'sources.json'),
                     'exact_historical_teacher_schema_claim': False})
    summary = {'scope': 'complete source-native offline SFT candidates, not competition six-tool data',
        'frozen_selection_sha256': digest(freeze_path), 'acquisition_sha256': digest(acquisition_path),
        'producer_sha256': producer_sha256, 'protocol_id': PROTOCOL_ID,
        'template_sha256': digest(template_path),
        'tokenizer_hash': tokenizer_hash, 'tool_schema_hash': sha256_json(tools),
        'source_rows': len(rows), 'rows': [], 'shards': {}}
    writers, outputs = {}, {}
    for bucket in ('le16k', '16k-to-32k'):
        for thought_loss in args.thought_loss:
            name = 'swe-smith-native-%s-thought-%s.jsonl' % (bucket, thought_loss)
            outputs[name] = {'records': 0, 'input_tokens': 0, 'supervised_tokens': 0}
            writers[bucket, thought_loss] = (args.output / name).open('w')
    try:
        for identity in rows:
            frozen_row = frozen[identity['instance_id']]
            if identity['split'] != 'train' or frozen_row['split'] != 'train' or any(
                    identity[k] != frozen_row[k] for k in ('repo', 'revision', 'canonical_task_key')):
                raise ValueError('non-train or changed source identity')
            path = ROOT / identity['local_path']
            if digest(path) != identity['sha256']:
                raise ValueError('source bytes changed: ' + str(path))
            row = read(path)
            if row['instance_id'] != identity['instance_id'] or row['resolved'] is not True:
                raise ValueError('source outcome or identity mismatch')
            evidence = {'instance_id': identity['instance_id'], 'repo': identity['repo'],
                        'source_sha256': identity['sha256'], 'upstream_resolved': True,
                        'local_replay': False, 'competition_six_tool_ready': False}
            try:
                tx, conversion = convert(row, identity, tools)
                review_flags = quality_review_flags(tx)
                conversion['quality_review_flags'] = review_flags
                evidence.update(conversion)
                decision = admitted.get(identity['instance_id'], {})
                eligible = decision.get('eligible') is True and (not review_flags or
                    decision.get('reviewed_quality_flags') == review_flags)
                if decision and decision.get('source_sha256') != identity['sha256']:
                    raise ValueError('admission decision has a different source hash')
                if eligible:
                    commit = decision.get('base_commit')
                    if commit is not None:
                        if not re.fullmatch(r'[0-9a-f]{40}', commit):
                            raise ValueError('provided base_commit is not a full Git SHA')
                        tx['group']['commit'] = commit
                    if not decision.get('license'):
                        raise ValueError('eligible native sample lacks data license evidence')
                    tx['source']['license'] = decision['license']
                evidence['native_sft_admitted'] = eligible
                evidence['admission_reason'] = ('specific source action flags need contextual review'
                    if review_flags and not eligible else decision.get('reason', 'not reviewed for native-SFT admission'))
                n = count_template_tokens(tx, tokenizer, tools, chat_template=template)
                encoded = {policy: encode_transaction(tx, tokenizer, tools, 32768,
                    thought_loss=policy, tokenizer_hash=tokenizer_hash, chat_template=template,
                    tokenizer_revision='official-mounted-31b-ct-v2') for policy in args.thought_loss
                    if n <= 32768 and (eligible or not args.admitted_only)}
                evidence['input_tokens'] = n
                evidence['supervised_tokens'] = {p: r['meta']['supervised_tokens'] for p, r in encoded.items()}
                bucket = 'le16k' if n <= 16384 else '16k-to-32k' if n <= 32768 else None
                evidence['bucket'] = bucket or 'over32k-kept-as-original'
                if bucket and (eligible or not args.admitted_only):
                    for policy, record in encoded.items():
                        record['meta'].update(training_eligible=eligible,
                            admission=evidence['admission_reason'],
                            admission_sha256=digest(args.admission) if args.admission else None,
                            target_competition_protocol_ready=False, source_native_conversion=conversion,
                            expert_source_model=row['model'], source_success_evidence='upstream resolved label')
                        writers[bucket, policy].write(json.dumps(record, ensure_ascii=False) + '\n')
                        name = Path(writers[bucket, policy].name).name
                        outputs[name]['records'] += 1
                        outputs[name]['input_tokens'] += n
                        outputs[name]['supervised_tokens'] += record['meta']['supervised_tokens']
            except (EncodingError, ValueError, KeyError) as error:
                evidence['conversion_failure'] = str(error)
            summary['rows'].append(evidence)
            print(json.dumps({k: evidence[k] for k in ('instance_id', 'input_tokens', 'bucket', 'conversion_failure') if k in evidence}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for name, evidence in outputs.items():
        evidence.update(sha256=digest(args.output / name), bytes=(args.output / name).stat().st_size)
    summary['shards'] = outputs
    summary['completed_full_trajectories'] = sum('input_tokens' in x for x in summary['rows'])
    summary['complete_fit_16k'] = sum(x.get('input_tokens', 10**9) <= 16384 for x in summary['rows'])
    summary['complete_fit_32k'] = sum(x.get('input_tokens', 10**9) <= 32768 for x in summary['rows'])
    summary['admitted_fit_32k'] = sum(x.get('native_sft_admitted', False)
        and x.get('input_tokens', 10**9) <= 32768 for x in summary['rows'])
    summary['cpu_wall_seconds'] = time.monotonic() - started
    dump(args.output / 'swe-smith-native-manifest.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'shards')}, indent=2))


if __name__ == '__main__':
    main()
