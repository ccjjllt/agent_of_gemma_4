"""Export verified fixed100 action views into the existing SFT trainer schema.

This is a data production entry point, not a replay, tokenizer, or training run.
Exact source IDs and masks are retained; only the trainer representation changes.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile

from training.common import load_encoded_jsonl, sha256_file, validate_encoded_record
from training.encoding.gemma_tool_chat import (
    LOSS_IGNORE, normalize_tools, sha256_json, tokenizer_fingerprint,
)
from competition.vfarl.format_replays import schema_for_template


ROOT = Path(__file__).resolve().parents[2]
VERSION = 'vfarl-action-view-labels-v1'
PROTOCOL_ID = 'vfarl-fixed100-codex-action-view-v1'
RECOVERY_VERSION = 'vfarl-controlled-prefix-recovery-labels-v1'
RECOVERY_PROTOCOL_ID = 'vfarl-controlled-prefix-recovery-v1'


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    path = resolve(path)
    require(path.is_file() and sha256_file(path) == expected,
            'Missing or changed pinned input: ' + str(path))
    return path


def task_origin(task):
    """Task generation and real demonstration execution are distinct concepts."""
    iid, dataset = task['instance_id'], task['source_dataset']
    if dataset == 'SWE-bench/SWE-smith-py':
        if re.search(r'\.(?:func_|lm_)', iid):
            return dict(kind='swesmith-synthetic-mutation', synthetic_mutation=True,
                basis='Pinned SWE-smith train source and mutation-type instance ID namespace (.func_/.lm_); not a publisher boolean field')
        require('.pr_' in iid, 'Unclassified SWE-smith task origin: ' + iid)
        return dict(kind='swesmith-pull-request-reproduction', synthetic_mutation=False,
            basis='Pinned SWE-smith train source with .pr_ instance ID namespace')
    require(dataset == 'SWE-Gym/SWE-Gym', 'Unclassified external task source: ' + iid)
    return dict(kind='external-pull-request-repair', synthetic_mutation=False,
        basis='Pinned SWE-Gym train source')


def export(supervision_manifest, tokenizer_path, declarations_path, output, max_length=8192):
    from transformers import AutoTokenizer

    supervision_manifest = resolve(supervision_manifest)
    supervision_manifest_bytes = supervision_manifest.read_bytes()
    manifest = json.loads(supervision_manifest_bytes)
    require(manifest.get('format') == 'vfarl-fixed100-current-supervision-v2'
            and manifest.get('status') == 'current-source-template-parser-aligned',
            'Expected the explicit current, source-aligned supervision version')
    action_path = pinned(manifest['recommended_action_view'], manifest['action_sha256'])
    selection_path = pinned(manifest['selection'], manifest['selection_sha256'])
    index_path = pinned(manifest['consumption_index'], manifest['consumption_index_sha256'])
    selection, index = read_json(selection_path), read_json(index_path)
    tasks = {r['instance_id']: r for r in selection['tasks']}
    indexed = {r['instance_id']: r for r in index['tasks']}
    require(len(tasks) == len(selection['tasks']) == len(indexed) == 100
            and set(tasks) == set(indexed), 'Expected exactly the same fixed100 task IDs')
    require(type(max_length) is int and max_length >= 2, 'Invalid full-sequence limit')
    tokenizer_path = resolve(tokenizer_path)
    pinned(tokenizer_path / 'tokenizer.json', manifest['tokenizer_sha256'])
    template_path = pinned(tokenizer_path / 'chat_template.jinja', manifest['template_sha256'])
    template = template_path.read_bytes().decode('utf-8')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    declarations_path = resolve(declarations_path)
    declarations_hash = sha256_file(declarations_path)
    declarations = read_json(declarations_path)
    tools, _ = normalize_tools([dict(type='function', function=dict(
        name=d['name'], description=d.get('description', ''),
        parameters=schema_for_template(d.get('parameters',
            {'type': 'object', 'properties': {}, 'required': []})))) for d in declarations])
    require(len(tools) == 6, 'Expected the actual six-tool declaration set')
    tools_hash = sha256_json(tools)
    thought_config = dict(enable_thinking=False, include_thoughts=False,
                          forced_empty_thought_control='prompt-only')
    protocol = dict(encoding_version=VERSION, protocol_id=PROTOCOL_ID,
        tools_hash=tools_hash, thought_config=thought_config, content_format='preserve',
        mask_policy='Copy verified action-view mask exactly: prompt/tool observations ignored; prescribed current tool action or complete terminal template fragment selected',
        origin='Codex-scripted actual-tool off-policy demonstration; not sampled model-stream capture')
    provenance = dict(tokenizer_hash=tokenizer_hash,
        chat_template_hash=manifest['template_sha256'], tools_hash=tools_hash,
        protocol_id=PROTOCOL_ID, protocol_hash=sha256_json(protocol))
    current_rows, contexts, checked_sources = {}, {}, set()
    original_lines = {}
    origin_counts = Counter()
    for iid, task in tasks.items():
        require(task['physical_split'] == 'train', 'Non-training task: ' + iid)
        info, teacher = indexed[iid], indexed[iid]['teacher']
        require(info['current_task_input_sha256'] == task['task_input_sha256']
            and info['current_prompt_sha256'] == task['official_initial_user_prompt_sha256']
            and info['publisher_source_sha256'] == task['source_task_sha256'],
            'Selection/index task version mismatch: ' + iid)
        pinned(task['source_task_path'], task['source_task_sha256'])
        adapted = read_json(pinned(task['task_input'], task['task_input_sha256']))
        canonical_path = pinned(teacher['canonical'], teacher['canonical_sha256'])
        canonical = read_json(canonical_path)
        require(canonical.get('eligible_for_human_off_policy_SFT') is True
            and not canonical.get('ineligibility_reasons')
            and canonical.get('strict_verbatim_required_ids_passed') is True
            and canonical.get('source_task_sha256') == task['task_input_sha256']
            and canonical.get('original_task_sha256') == task['source_task_sha256']
            and canonical.get('initial_prompt_sha256') == task['official_initial_user_prompt_sha256'],
            'Canonical evidence does not admit this current task: ' + iid)
        messages_path = pinned(teacher['messages'], teacher['messages_sha256'])
        messages = read_json(messages_path)
        require(hashlib.sha256(messages[0]['content'].encode('utf-8')).hexdigest()
                == task['official_initial_user_prompt_sha256'], 'Actual prompt mismatch: ' + iid)
        call_positions, text_positions = {}, []
        for position, message in enumerate(messages):
            calls = message.get('tool_calls') or []
            for call in calls:
                require(call['id'] not in call_positions, 'Duplicate actual tool-call ID: ' + iid)
                call_positions[call['id']] = position
                arguments = call['function']['arguments']
                if isinstance(arguments, str):
                    call['function']['arguments'] = json.loads(arguments)
            if message['role'] == 'assistant' and not calls and message.get('content'):
                text_positions.append(position)
        actual_declarations = resolve(canonical['actual_run_directory']) / 'adk-six-tool-declarations.json'
        pinned(actual_declarations, declarations_hash)
        source = teacher['source_action_file']
        if source not in checked_sources:
            pinned(source, teacher['source_action_file_sha256'])
            checked_sources.add(source)
        origin = task_origin(task)
        origin_counts[origin['kind']] += 1
        contexts[iid] = dict(task=task, teacher=teacher, origin=origin,
            base_commit=adapted['base_commit'], messages=messages,
            call_positions=call_positions, text_positions=text_positions)
        for row in info['source_rows']:
            key = iid, row['action_index']
            require(key not in current_rows, 'Duplicate source decision: ' + repr(key))
            current_rows[key] = row
            original_lines.setdefault(row['file'], {})[row['line']] = row['raw_sha256']
    for source, needed in original_lines.items():
        require(source in checked_sources, 'Unbound original action file: ' + source)
        remaining = dict(needed)
        with resolve(source).open('rb') as handle:
            for number, raw in enumerate(handle, 1):
                if number in remaining:
                    require(hashlib.sha256(raw).hexdigest() == remaining.pop(number),
                            'Original source action line changed: ' + source + ':' + str(number))
                if not remaining:
                    break
        require(not remaining, 'Missing original action lines: ' + source)
    output = resolve(output)
    require(not output.exists(), 'Refusing to overwrite trainer export: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count, selected_tokens, maximum, records = 0, 0, 0, []
    seen = set()
    with tempfile.TemporaryDirectory(prefix='vfarl-trainer-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'export'
        stage.mkdir()
        target = stage / 'encoded.jsonl'
        with action_path.open('rb') as source, target.open('w', encoding='utf-8', newline='\n') as destination:
            for line_number, raw in enumerate(source, 1):
                row = json.loads(raw)
                iid, action_index = row['instance_id'], row['action_index']
                key = iid, action_index
                require(key in current_rows and key not in seen, 'Unknown/duplicate action: ' + repr(key))
                seen.add(key)
                context = contexts[iid]
                task, teacher = context['task'], context['teacher']
                raw_hash = hashlib.sha256(raw).hexdigest()
                original_row = current_rows[key]
                require(raw_hash == original_row['raw_sha256'], 'Action-view source row changed')
                require(row['source_task_sha256'] == task['source_task_sha256']
                    and row['official_initial_user_prompt_sha256'] == task['official_initial_user_prompt_sha256']
                    and row['source_episode'] == teacher['messages']
                    and row['source_episode_sha256'] == teacher['messages_sha256']
                    and row.get('on_policy') is False and row.get('old_logprobs') is None
                    and row.get('generated_reasoning') is None,
                    'Action provenance/behavior semantics differ from the approved source')
                if row['tool'] is not None:
                    position = context['call_positions'].get(row['tool_call_id'])
                    require(position is not None, 'Missing actual tool-call position')
                else:
                    require(len(context['text_positions']) == 1, 'Ambiguous actual terminal text')
                    position = context['text_positions'][0]
                reconstructed_prefix = tokenizer.apply_chat_template(
                    context['messages'][:position], tools=tools,
                    chat_template=template, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                require(reconstructed_prefix == row['prompt'],
                        'Deployment tools/template do not reproduce the actual generation prefix')
                ids, mask = row['input_ids'], row['assistant_loss_mask']
                require(isinstance(ids, list) and isinstance(mask, list)
                    and len(ids) == len(mask) and 2 <= len(ids) <= max_length
                    and all(type(v) is int and v >= 0 for v in ids)
                    and all(type(v) is int and v in (0, 1) for v in mask),
                    'Invalid/oversize exact source tokens; no truncation is allowed')
                labels = [token if keep else LOSS_IGNORE for token, keep in zip(ids, mask)]
                meta = dict(provenance, encoding_version=VERSION,
                    transaction_id=iid + '#' + str(action_index),
                    origin_kind='codex-scripted-real-tool-off-policy-demonstration',
                    synthetic=False, training_eligible=True,
                    synthetic_semantics='False means real executed trace, not SyntheticTokenizer/smoke fixture; task mutation is recorded independently below',
                    task_bug_origin=context['origin'],
                    group=dict(issue=iid, repo=task['canonical_repo'],
                               commit=context['base_commit'], split='train'),
                    source=dict(uri=str(action_path), sha256=manifest['action_sha256'],
                        raw_sha256=raw_hash, format=row['schema_version'],
                        source_id=iid + '#' + str(action_index),
                        trace_sha256=teacher['messages_sha256'],
                        dataset=task['source_dataset'], dataset_revision=task['source_revision']),
                    behavior=dict(authorship='Codex prescribed and reviewed the actually executed tool recipe',
                        model_sampled=False, on_policy=False, behavior_logprobs_recorded=False),
                    source_evidence_kind=row['evidence_kind'],
                    task_version=dict(selection_sha256=manifest['selection_sha256'],
                        publisher_blob_sha256=task['source_task_sha256'],
                        adapted_task_sha256=task['task_input_sha256'],
                        actual_prompt_sha256=task['official_initial_user_prompt_sha256']),
                    actual_trace=dict(messages=teacher['messages'], messages_sha256=teacher['messages_sha256'],
                        canonical=teacher['canonical'], canonical_sha256=teacher['canonical_sha256'],
                        strict_original_verifier_preserved=True),
                    original_action_row=dict(file=original_row['file'], line=original_row['line'],
                        raw_sha256=original_row['raw_sha256'], action_view_line=line_number),
                    action_index=action_index, role=task['role'], primary_adapter=task['primary_adapter'],
                    submechanism=task['submechanism'], repair_family_id=task['repair_family_id'],
                    snapshot_group_id=task['snapshot_group_id'], thought_config=thought_config,
                    sequence_length=len(ids), raw_token_length=len(ids), max_seq_len=max_length,
                    truncated=False, alignment='exact-existing-token-and-mask-map; no retemplate or retokenization',
                    supervised_tokens=sum(v != LOSS_IGNORE for v in labels[1:]))
                encoded = dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels, meta=meta)
                validate_encoded_record(encoded, max_length)
                destination.write(json.dumps(encoded, ensure_ascii=False, separators=(',', ':')) + '\n')
                records.append(dict(instance_id=iid, action_index=action_index,
                    action_view_line=line_number, action_view_raw_sha256=raw_hash,
                    exact_source_ids_sha256=sha256_json(ids),
                    exact_source_mask_sha256=sha256_json(mask), encoded_line=row_count + 1))
                row_count += 1
                selected_tokens += meta['supervised_tokens']
                maximum = max(maximum, len(ids))
        require(seen == set(current_rows) and row_count == manifest['stats']['decisions'],
                'Export must retain every approved action exactly once')
        loaded, consumed = load_encoded_jsonl(target, max_length)
        require(len(loaded) == row_count and consumed['supervised_tokens_after_shift'] == selected_tokens,
                'Existing trainer consumption differs from the exact mapping')
        (stage / 'deployment-tools.json').write_text(json.dumps(
            dict(protocol_id=PROTOCOL_ID, tools=tools), ensure_ascii=False, indent=2) + '\n')
        (stage / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
        (stage / 'trainer-overrides.json').write_text(json.dumps(dict(
            protocol_id=PROTOCOL_ID, tokenizer_name_or_path=str(tokenizer_path),
            chat_template_path=str(template_path), tool_schema_path=str(output / 'deployment-tools.json'),
            max_sequence_length=max_length), ensure_ascii=False, indent=2) + '\n')
        (stage / 'source-supervision-manifest.json').write_bytes(supervision_manifest_bytes)
        result = dict(format=VERSION, input=str(action_path), input_sha256=manifest['action_sha256'],
            supervision_manifest=str(output / 'source-supervision-manifest.json'),
            supervision_manifest_sha256=hashlib.sha256(supervision_manifest_bytes).hexdigest(),
            source_supervision_manifest_path=str(supervision_manifest),
            source_frozen_copy=dict(path=str(output / 'source-supervision-manifest.json'),
                sha256=hashlib.sha256(supervision_manifest_bytes).hexdigest()),
            selection=str(selection_path), selection_sha256=manifest['selection_sha256'],
            consumption_index=str(index_path), consumption_index_sha256=manifest['consumption_index_sha256'],
            recommended_training_input=str(output / 'encoded.jsonl'), encoded_sha256=sha256_file(target),
            rows=row_count, tasks=len(tasks), supervised_tokens_after_shift=selected_tokens,
            maximum_context_tokens=maximum, max_seq_len=max_length, task_origin_counts=dict(origin_counts),
            deployment_provenance=provenance, protocol=protocol,
            actual_declarations=str(declarations_path), actual_declarations_sha256=declarations_hash,
            source_tokenizer_json_sha256=manifest['tokenizer_sha256'],
            tokenizer_class=type(tokenizer).__name__,
            exporter_source_sha256=sha256_file(__file__),
            fingerprint_implementation_sha256=sha256_file(ROOT / 'training/encoding/gemma_tool_chat.py'),
            runtime_packages={n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers')},
            actual_trainer_load=consumed,
            no_ids_or_masks_reencoded=True, no_row_dropped=True, no_truncation=True,
            source_template_prefixes_reconstructed=row_count,
            model_inference=False, optimizer_steps=0, gpu_used=False,
            authorship='Codex-scripted real-tool off-policy demonstrations; legacy human_reviewed source labels are provenance only',
            synthetic_semantics='Real executed trace eligibility is independent of the task-level synthetic mutation origin',
            rows_source_index=records)
        (stage / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        stage.rename(output)
    return result


def export_recovery_only(recovery_manifest, selection_path, tokenizer_path,
                         declarations_path, output, max_length=8192):
    """Map only explicitly admitted recovery spans; never admit the wrong prefix."""
    from transformers import AutoTokenizer
    from competition.vfarl.format_replays import action_examples

    recovery_manifest, selection_path = resolve(recovery_manifest), resolve(selection_path)
    frozen_manifest = recovery_manifest.read_bytes()
    manifest, selection = json.loads(frozen_manifest), read_json(selection_path)
    require(manifest.get('schema_version') == 'controlled-prefix-recovery-only-sft-v1'
        and manifest.get('eligible_for_recovery_only_SFT') is True
        and manifest.get('eligible_for_full_episode_SFT') is False
        and manifest.get('included_in_current100_main_SFT') is False,
        'Expected independently admitted recovery-only input, never main/full-episode SFT')
    require(type(max_length) is int and max_length >= 2, 'Invalid full-sequence limit')
    tasks = {t['instance_id']: t for t in selection['tasks']}
    require(len(tasks) == len(selection['tasks']) == 100, 'Explicit fixed100 selection required')
    boundary_path = pinned(manifest['source_boundary'], manifest['source_boundary_sha256'])
    boundary = read_json(boundary_path)
    require(boundary.get('eligible_for_full_episode_SFT') is False
        and boundary.get('partial_supervision_candidates_only') is True,
        'Controlled prefix boundary must remain partial-only historical evidence')
    traces = {t['instance_id']: t for t in boundary['traces']}
    action_path = pinned(manifest['output_actions'], manifest['output_actions_sha256'])
    records_path = pinned(manifest['context_records'], manifest['context_records_sha256'])
    records = [json.loads(raw) for raw in records_path.read_bytes().splitlines()]
    contexts = {r['instance_id']: r for r in records}
    episodes = {e['instance_id']: e for e in manifest['episodes']}
    require(len(episodes) == len(manifest['episodes']) == len(records) == len(contexts)
        == manifest['source_cases'] and set(episodes) == set(contexts)
        and set(episodes) <= tasks.keys(), 'Recovery case identities are not unique/current')
    pinned(manifest['formatter'], manifest['formatter_sha256'])
    tokenizer_path = resolve(tokenizer_path)
    pinned(tokenizer_path/'tokenizer.json', manifest['tokenizer_sha256'])
    template_path = pinned(tokenizer_path/'chat_template.jinja', manifest['template_sha256'])
    pinned(tokenizer_path/'tokenizer_config.json', manifest['tokenizer_config_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    declarations_path = resolve(declarations_path)
    declaration_sha = sha256_file(declarations_path)
    tools, _ = normalize_tools([dict(type='function', function=dict(name=d['name'],
        description=d.get('description', ''), parameters=schema_for_template(d.get(
            'parameters', {'type': 'object', 'properties': {}, 'required': []}))))
        for d in read_json(declarations_path)])
    require(len(tools) == 6, 'Actual six-tool declarations required')
    template = template_path.read_bytes().decode('utf-8')

    def render(messages, declarations=tools, generation=False):
        return tokenizer.apply_chat_template(messages, tools=declarations,
            chat_template=template, tokenize=False, add_generation_prompt=generation,
            enable_thinking=False)

    selected, sources, origin_counts = {}, [], Counter()
    for iid, ep in episodes.items():
        task, trace, context = tasks[iid], traces[iid], contexts[iid]
        require(selection['tasks'][ep['index']]['instance_id'] == iid
            and task['physical_split'] == 'train'
            and ep['task_input_sha256'] == task['task_input_sha256']
            and ep['original_source_task_sha256'] == task['source_task_sha256']
            and ep['actual_initial_prompt_sha256'] == task['official_initial_user_prompt_sha256'],
            'Recovery evidence differs from the explicitly selected task version: ' + iid)
        adapted = read_json(pinned(task['task_input'], task['task_input_sha256']))
        pinned(task['source_task_path'], task['source_task_sha256'])
        pinned(ep['task_input'], task['task_input_sha256'])
        pinned(ep['original_source_task'], ep['original_source_task_sha256'])
        run = resolve(ep['source_actual_run'])
        message_path = pinned(ep['source_raw_messages'], ep['source_raw_messages_sha256'])
        recipe_path = pinned(run/'recipe.json', ep['source_recipe_sha256'])
        event_path = pinned(run/'tool-events.jsonl', ep['source_tool_events_sha256'])
        summary_path = pinned(run/'run-summary.json', ep['source_run_summary_sha256'])
        pinned(ep['tools_declarations'], declaration_sha)
        require(ep['tools_declarations_sha256'] == declaration_sha
            and trace['actual_run'] == ep['source_actual_run']
            and trace['raw_messages_sha256'] == ep['source_raw_messages_sha256']
            and trace['actual_recipe_sha256'] == ep['source_recipe_sha256']
            and trace['tool_events_sha256'] == ep['source_tool_events_sha256']
            and trace['actual_run_summary_sha256'] == ep['source_run_summary_sha256']
            and trace['official_original_scope'] == ep['source_official_original_scope']
            and ep['source_intervention_official_result'] is None
            and trace['intervention']['official_intervention_state_result'] is None,
            'Recovery source/boundary scope differs: ' + iid)
        recipe, summary = read_json(recipe_path), read_json(summary_path)
        require(recipe.get('reviewed_by') and recipe.get('review_notes')
            and recipe.get('eligible_for_full_episode_SFT') is False
            and recipe.get('eligible_for_human_off_policy_SFT') is False
            and summary['source_task_sha256'] == task['task_input_sha256']
            and summary['initial_prompt_sha256'] == task['official_initial_user_prompt_sha256']
            and summary['source_tree_sha256'] == ep['source_tree_sha256']
            and summary.get('on_policy') is False and summary.get('behavior_policy') is None,
            'Actual controlled trace provenance differs: ' + iid)
        oracle_pins = copy.deepcopy(ep['source_official_original_scope'])
        for phase, resolved in (('baseline', False), ('final', True)):
            pin = oracle_pins[phase]
            actual = read_json(pinned(pin['result_file'], pin['result_sha256']))
            pinned(pin['junit'], pin['junit_sha256'])
            require(actual.get('resolved') is resolved and pin['resolved'] is resolved,
                    'Original-scope oracle evidence differs: ' + iid)
        messages = read_json(message_path)
        require(messages[0]['role'] == 'user' and hashlib.sha256(
            messages[0]['content'].encode()).hexdigest() == task['official_initial_user_prompt_sha256'],
            'Actual raw prompt differs from explicit selection: ' + iid)
        events = [json.loads(raw) for raw in event_path.read_bytes().splitlines()]
        calls = [(mi, c) for mi, m in enumerate(messages) for c in m.get('tool_calls') or []]
        require(len(calls) == len(recipe['actions']) == len(events), 'Actual event/recipe counts differ')
        for index, ((mi, call), event, prescribed) in enumerate(zip(calls, events, recipe['actions'])):
            args = call['function']['arguments']
            args = json.loads(args) if isinstance(args, str) else args
            call['function']['arguments'] = args
            require(isinstance(args, dict) and event['index'] == index
                and event['tool_call_id'] == call['id'] == messages[mi+1].get('tool_call_id')
                and event['tool'] == call['function']['name'] == prescribed['tool']
                and args == event['arguments'] == prescribed['arguments']
                and json.loads(messages[mi+1]['content']) == event['response'],
                'Actual raw call/response/recipe identity differs: ' + iid)
        bound = trace['proposed_supervision_boundary']
        failure, after = bound['after_tool_response_event'], bound['after_raw_message_index']
        require(events[failure]['response'].get('status') == 'error'
            and messages[after].get('tool_call_id') == events[failure]['tool_call_id']
            and bound['context_only_tool_event_indices'] == list(range(failure+1)),
            'Recovery must begin after the real failed feedback')
        wanted_actions = [c['tool_event_index'] for c in bound['candidate_recovery_tool_actions']]
        wanted_messages = [c['assistant_raw_message_index'] for c in bound['candidate_recovery_tool_actions']]
        wanted_actions += [bound['candidate_future_formatter_terminal_action_index']]
        wanted_messages += [bound['candidate_actual_terminal_raw_message_index']]
        require(ep['selected_source_action_indices'] == wanted_actions
            and ep['selected_source_raw_message_indices'] == wanted_messages
            and context['selected_recovery_raw_message_indices'] == wanted_messages
            and all(pos > after for pos in wanted_messages), 'Recovery selection crosses its context-only boundary')
        archive = pinned(ep['context_archive'], ep['context_archive_sha256'])
        rendered = archive.read_bytes().decode('utf-8')
        require(render(messages) == rendered and context['messages_source'] == ep['source_raw_messages']
            and context['messages_source_sha256'] == ep['source_raw_messages_sha256']
            and context['rendered_text_sha256'] == ep['context_archive_sha256']
            and context['source_boundary_sha256'] == manifest['source_boundary_sha256'],
            'Unlabelled context archive is not the actual message rendering')
        prepared = copy.deepcopy(context)
        prepared.update(messages_source=str(message_path), rendered_text=str(archive))
        regenerated = action_examples(prepared, tokenizer, tools)
        require(len(regenerated) == len(wanted_actions) == len(ep['actions']), 'Selected span count differs')
        origin = task_origin(task)
        origin_counts[origin['kind']] += 1
        for view_index, (source_index, mi, span, expected, encoded) in enumerate(zip(
                wanted_actions, wanted_messages, context['action_spans'], ep['actions'], regenerated)):
            require(span['source_message_index'] == mi and span['original_action_index'] == source_index
                and expected['source_action_index'] == source_index
                and expected['source_raw_message_index'] == mi, 'Selected raw source indices differ')
            if span['tool'] is not None:
                msg = messages[mi]
                require(len(msg['tool_calls']) == 1 and msg['tool_calls'][0]['id'] == span['tool_call_id'],
                        'Selected actual call is not unique')
                isolated = render([msg], [])
                fragment = isolated[isolated.index('<|tool_call>'):isolated.rindex('<tool_call|>')+len('<tool_call|>')]
            else:
                require(mi == len(messages)-1 and messages[mi]['content'] == recipe['final_response'],
                        'Terminal text was not actually recorded in the reviewed recipe')
                fragment = rendered[len(render(messages[:mi])):]
            require(fragment == rendered[span['start_char']:span['end_char']] == encoded['completion']
                and encoded['prompt'] == render(messages[:mi], generation=True),
                'Actual selected prefix/completion does not match its source message')
            key = iid, source_index
            require(key not in selected, 'Duplicate selected recovery decision')
            selected[key] = dict(task=task, base_commit=adapted['base_commit'], episode=ep,
                boundary=bound, origin=origin, expected=expected, regenerated=encoded,
                view_index=view_index, raw_message_index=mi, oracle_pins=oracle_pins)
        sources.append(dict(instance_id=iid, raw_messages=str(message_path),
            raw_messages_sha256=ep['source_raw_messages_sha256'], recipe=str(recipe_path),
            recipe_sha256=ep['source_recipe_sha256'], tool_events=str(event_path),
            tool_events_sha256=ep['source_tool_events_sha256'], run_summary=str(summary_path),
            run_summary_sha256=ep['source_run_summary_sha256'], original_scope_oracle=oracle_pins,
            intervened_state_official_result=None, selected_action_indices=wanted_actions,
            selected_raw_message_indices=wanted_messages, boundary_after_raw_message_index=after))
    thought = dict(enable_thinking=False, include_thoughts=False, forced_empty_thought_control='prompt-only')
    protocol = dict(encoding_version=RECOVERY_VERSION, protocol_id=RECOVERY_PROTOCOL_ID,
        tools_hash=sha256_json(tools), thought_config=thought,
        mask_policy='Only selected actual recovery completion; all prior controlled-prefix assistant actions and tool observations have zero loss',
        origin='Codex-controlled wrong prefix and actual Codex off-policy recovery; not full-episode canonical or Gemma behavior')
    provenance = dict(tokenizer_hash=tokenizer_fingerprint(tokenizer),
        chat_template_hash=manifest['template_sha256'], tools_hash=sha256_json(tools),
        protocol_id=RECOVERY_PROTOCOL_ID, protocol_hash=sha256_json(protocol))
    output = resolve(output)
    require(not output.exists(), 'Refusing to overwrite recovery export: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, row_index, token_count, maximum = set(), [], 0, 0
    with tempfile.TemporaryDirectory(prefix='vfarl-recovery-trainer-', dir=output.parent) as temporary:
        stage = Path(temporary)/'export'
        stage.mkdir()
        target = stage/'encoded.jsonl'
        with action_path.open('rb') as source, target.open('w', encoding='utf-8', newline='\n') as dest:
            for number, raw in enumerate(source, 1):
                row = json.loads(raw)
                key = row['instance_id'], row['action_index']
                require(key in selected and key not in seen, 'Unknown or duplicate recovery action')
                seen.add(key)
                c, task = selected[key], selected[key]['task']
                ep, generated = c['episode'], c['regenerated']
                require(row.get('schema_version') == 'gemma-controlled-prefix-recovery-action-sft-v1'
                    and row.get('eligible_for_recovery_only_SFT') is True
                    and row.get('eligible_for_full_episode_SFT') is False
                    and row.get('included_in_current100_main_SFT') is False
                    and row.get('intervened_state_official_result') is None
                    and row.get('on_policy') is False and row.get('old_logprobs') is None
                    and row.get('generated_reasoning') is None
                    and row['task_input_sha256'] == task['task_input_sha256']
                    and row['source_task_sha256'] == task['source_task_sha256']
                    and row['official_initial_user_prompt_sha256'] == task['official_initial_user_prompt_sha256']
                    and row['source_episode_sha256'] == ep['source_raw_messages_sha256']
                    and row['source_raw_message_index'] == c['raw_message_index']
                    and row['recovery_view_action_index'] == c['view_index']
                    and row['source_boundary_sha256'] == manifest['source_boundary_sha256']
                    and row['source_boundary_after_raw_message_index'] == c['boundary']['after_raw_message_index']
                    and row['source_failure_response_tool_event_index'] == c['boundary']['after_tool_response_event'],
                    'Recovery action provenance/scope changed')
                for field in ('prompt', 'completion', 'input_ids', 'assistant_loss_mask', 'tool', 'tool_call_id'):
                    require(row[field] == generated[field], 'Actual recovery template/token/mask mismatch: ' + field)
                for field in ('primary_adapter', 'submechanism', 'canonical_repo', 'repair_family_id',
                              'snapshot_group_id', 'source_dataset', 'source_revision', 'physical_split'):
                    require(row[field] == task[field], 'Current selection label/source differs: ' + field)
                ids, mask = row['input_ids'], row['assistant_loss_mask']
                require(len(ids) == len(mask) and 2 <= len(ids) <= max_length
                    and all(type(x) is int and x >= 0 for x in ids)
                    and all(type(x) is int and x in (0, 1) for x in mask), 'Invalid or oversize untruncated source tokens')
                for field in ('prompt', 'completion', 'input_ids', 'assistant_loss_mask'):
                    actual = hashlib.sha256(row[field].encode()).hexdigest() if isinstance(row[field], str) else sha256_json(row[field])
                    require(actual == row[field+'_sha256'] == c['expected'][field+'_sha256'], 'Source action hash differs')
                labels = [token if keep else LOSS_IGNORE for token, keep in zip(ids, mask)]
                raw_sha = hashlib.sha256(raw).hexdigest()
                meta = dict(provenance, encoding_version=RECOVERY_VERSION,
                    transaction_id='recovery-only:'+key[0]+'#'+str(key[1]), synthetic=False,
                    training_eligible=True, task_bug_origin=c['origin'],
                    group=dict(issue=key[0], repo=task['canonical_repo'], commit=c['base_commit'], split='train'),
                    origin_kind='codex-controlled-prefix-real-recovery-only',
                    eligible_for_full_episode_SFT=False, included_in_current100_main_SFT=False,
                    behavior=dict(authorship='Codex controlled prefix and reviewed actual recovery recipe',
                                  model_sampled=False, on_policy=False, behavior_logprobs_recorded=False),
                    source=dict(uri=str(action_path), sha256=manifest['output_actions_sha256'],
                        raw_sha256=raw_sha, line=number, source_id=key[0]+'#'+str(key[1]),
                        trace_sha256=ep['source_raw_messages_sha256']),
                    selection=dict(path=str(selection_path), sha256=sha256_file(selection_path),
                        publisher_blob_sha256=task['source_task_sha256'], adapted_task_sha256=task['task_input_sha256'],
                        actual_prompt_sha256=task['official_initial_user_prompt_sha256']),
                    recovery_boundary=dict(path=str(boundary_path), sha256=manifest['source_boundary_sha256'],
                        after_tool_response_event=c['boundary']['after_tool_response_event'],
                        after_raw_message_index=c['boundary']['after_raw_message_index'],
                        source_action_index=key[1], source_raw_message_index=c['raw_message_index'],
                        recovery_view_action_index=c['view_index'], wrong_prefix_loss='zero context only',
                        intervened_state_official_result=None, same_state_counterfactual_pair=False),
                    actual_trace=dict(messages=ep['source_raw_messages'], messages_sha256=ep['source_raw_messages_sha256'],
                        recipe_sha256=ep['source_recipe_sha256'], tool_events_sha256=ep['source_tool_events_sha256'],
                        original_scope_oracle_pins=c['oracle_pins']),
                    role=task['role'], primary_adapter=task['primary_adapter'],
                    repair_family_id=task['repair_family_id'], snapshot_group_id=task['snapshot_group_id'],
                    thought_config=thought, sequence_length=len(ids), raw_token_length=len(ids), max_seq_len=max_length,
                    truncated=False, alignment='Exact existing recovery IDs/mask mapping; no retemplate output',
                    supervised_tokens=sum(x != LOSS_IGNORE for x in labels[1:]))
                encoded = dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels, meta=meta)
                validate_encoded_record(encoded, max_length)
                dest.write(json.dumps(encoded, ensure_ascii=False, separators=(',', ':'))+'\n')
                row_index.append(dict(instance_id=key[0], source_action_index=key[1],
                    raw_message_index=c['raw_message_index'], input_line=number,
                    input_raw_sha256=raw_sha, encoded_line=number))
                token_count += meta['supervised_tokens']
                maximum = max(maximum, len(ids))
        require(seen == set(selected) and len(seen) == manifest['recovery_assistant_decisions']
            and token_count == manifest['assistant_loss_tokens'], 'Recovery-only decisions/tokens changed')
        loaded, consumption = load_encoded_jsonl(target, max_length)
        require(len(loaded) == len(seen) and consumption['supervised_tokens_after_shift'] == token_count,
                'Actual trainer reader differs from exact recovery mapping')
        (stage/'source-recovery-manifest.json').write_bytes(frozen_manifest)
        for name, value in [('deployment-tools.json', dict(protocol_id=RECOVERY_PROTOCOL_ID, tools=tools)),
                            ('protocol.json', protocol), ('trainer-overrides.json', dict(
                            protocol_id=RECOVERY_PROTOCOL_ID, tokenizer_name_or_path=str(tokenizer_path),
                            chat_template_path=str(template_path), tool_schema_path=str(output/'deployment-tools.json'),
                            max_sequence_length=max_length))]:
            (stage/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        result = dict(format=RECOVERY_VERSION, input=str(action_path), input_sha256=manifest['output_actions_sha256'],
            source_recovery_manifest_path=str(recovery_manifest),
            source_frozen_copy=dict(path=str(output/'source-recovery-manifest.json'), sha256=hashlib.sha256(frozen_manifest).hexdigest()),
            selection=str(selection_path), selection_sha256=sha256_file(selection_path),
            selection_scope='Explicit version evidence for these task IDs; never automatic latest selection lookup',
            source_boundary=str(boundary_path), source_boundary_sha256=manifest['source_boundary_sha256'],
            context_records=str(records_path), context_records_sha256=manifest['context_records_sha256'],
            recommended_training_input=str(output/'encoded.jsonl'), encoded_sha256=sha256_file(target),
            rows=len(seen), tasks=len(episodes), supervised_tokens_after_shift=token_count,
            maximum_context_tokens=maximum, max_seq_len=max_length, task_origin_counts=dict(origin_counts),
            deployment_provenance=provenance, protocol=protocol, source_episodes=sources, rows_source_index=row_index,
            actual_declarations=str(declarations_path), actual_declarations_sha256=declaration_sha,
            exporter_source_sha256=sha256_file(__file__), tokenizer_class=type(tokenizer).__name__,
            runtime_packages={n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers')},
            actual_trainer_load=consumption, eligible_for_full_episode_SFT=False,
            included_in_current100_main_SFT=False, model_inference=False, optimizer_steps=0, gpu_used=False,
            oracle_rerun=False, full_episode_normalization_invoked=False, no_row_dropped=True,
            no_ids_or_masks_reencoded=True, no_truncation=True,
            source_template_completions_reconstructed=len(seen),
            interpretation='Independent recovery-only trainer input. Wrong prefix is context only; original-scope oracle pins are evidence, not an intervened-state reward or model gain.')
        (stage/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        stage.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--supervision-manifest', type=Path)
    source.add_argument('--recovery-manifest', type=Path,
        help='Independently admitted recovery-only action view; never main100/full-episode SFT')
    parser.add_argument('--selection', type=Path,
        help='Explicit fixed100 version required only with --recovery-manifest')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--declarations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=8192)
    args = parser.parse_args()
    if args.recovery_manifest:
        if args.selection is None:
            parser.error('--recovery-manifest requires an explicit --selection')
        result = export_recovery_only(args.recovery_manifest, args.selection,
            args.tokenizer, args.declarations, args.out, args.max_length)
    else:
        if args.selection is not None:
            parser.error('--selection belongs to the separate recovery-only export')
        result = export(args.supervision_manifest, args.tokenizer, args.declarations, args.out, args.max_length)
    print(json.dumps({k: result[k] for k in ('recommended_training_input', 'encoded_sha256',
        'rows', 'tasks', 'supervised_tokens_after_shift', 'maximum_context_tokens', 'task_origin_counts')}, indent=2))


if __name__ == '__main__':
    main()
