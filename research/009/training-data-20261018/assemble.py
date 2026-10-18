"""Assemble the explicitly admitted replay transactions and encode full CT conversations."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADMITTED = [
    'apispec/attempt-01', 'pydantic/attempt-01', 'tenacity/attempt-03',
    'mypy/attempt-01', 'dataset/attempt-01', 'deepdiff/attempt-01',
    'yamllint/attempt-01', 'apispec-pr950/attempt-01',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    usage_path = HERE / 'data-usage.json'
    usage = json.loads(usage_path.read_text())
    transactions, entries = [], []
    for rel in ADMITTED:
        path = HERE / 'outputs' / rel / 'transaction.json'
        tx = json.loads(path.read_text())
        if tx['attempt_status'] != 'local_behavior_repair':
            raise ValueError('A failed attempt is not a positive training transaction')
        if tx['group']['repo'].lower() in usage['excluded_repositories']:
            raise ValueError('Reserved repository cannot be used for training')
        transactions.append(tx)
        entries.append({'sample_id': tx['sample_id'], 'group': tx['group'], 'role': 'train',
                        'allow_parameter_update': True, 'path': str(path.relative_to(HERE)),
                        'sha256': sha(path), 'target_assistant_steps': tx['target_assistant_steps'],
                        'task_resolved': None, 'local_outcome': tx['observed_outcome']})
    dataset = HERE / 'transactions-train.jsonl'
    dataset.write_text(''.join(json.dumps(tx, ensure_ascii=False) + '\n' for tx in transactions))
    output = HERE / 'encoded-train-12b.jsonl'
    command = [
        sys.executable, str(ROOT / 'training/encoding/gemma_tool_chat.py'), 'encode',
        '--input', str(dataset), '--output', str(output),
        '--tokenizer', str(ROOT / 'training/artifacts/tokenizers/gemma4-12b-ct-v2'),
        '--tools', str(ROOT / 'training/configs/harness_tools.json'),
        '--chat-template', str(ROOT / 'training/configs/chat_templates/gemma4_12b_ct_v2.jinja'),
        '--thinking', 'on', '--thought-loss', 'mask', '--max-seq-len', '8192', '--on-oversize', 'error',
    ]
    subprocess.run(command, check=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    lengths = [{'issue': row['meta']['group']['issue'], 'repo': row['meta']['group']['repo'],
                'input_tokens': len(row['input_ids']),
                'supervised_tokens': sum(v != -100 for v in row['labels'])} for row in rows]
    usage.update(admitted_transactions=entries, admitted_independent_issues=len({t['group']['issue'] for t in transactions}),
                 admitted_repo_count=len({t['group']['repo'] for t in transactions}),
                 candidates=[{'sample_id': 'p009-alive-progress', 'path': 'outputs/alive-progress/attempt-01/transaction.json',
                              'role': 'candidate_component_only', 'allow_parameter_update': False,
                              'reason': 'Actual source AST factories executed, but module/public API initialization bypassed; excluded from first SFT.'}],
                 excluded_attempts=[{'path': 'outputs/tenacity/attempt-01/transaction.json', 'reason': 'Runtime import failed before and after'},
                                    {'path': 'outputs/tenacity/attempt-02/transaction.json', 'reason': 'Timeout and repository mypy config conflict'}],
                 admitted_file={'path': output.name, 'sha256': sha(output), 'rows': len(rows),
                                'input_tokens': sum(x['input_tokens'] for x in lengths),
                                'supervised_tokens': sum(x['supervised_tokens'] for x in lengths),
                                'max_input_tokens': max(x['input_tokens'] for x in lengths), 'truncated': False},
                 dataset_policy='Only encoded-train-12b.jsonl with this fingerprint is first-SFT input. Source trajectories, candidates, failed attempts and provisional files are not training inputs. Local success does not imply whole-issue success.')
    usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2) + '\n')
    summary = {'rows': len(rows), 'independent_issues': usage['admitted_independent_issues'],
               'independent_repos': usage['admitted_repo_count'], 'lengths': lengths,
               'input_tokens': sum(x['input_tokens'] for x in lengths),
               'supervised_tokens': sum(x['supervised_tokens'] for x in lengths),
               'max_input_tokens': max(x['input_tokens'] for x in lengths),
               'whole_benchmark_evaluations': 0, 'task_resolved_labels': None,
               'gpu_runs_in_data_production': 0, 'paid_teacher_calls': 0,
               'no_history_truncation': True, 'tool_responses_newly_executed': True}
    (HERE / 'training-data-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    assets = [HERE / name for name in ['transactions-train.jsonl', 'encoded-train-12b.jsonl',
                                      'encoded-train-12b.jsonl.report.json', 'source-manifest.json',
                                      'data-usage.json', 'training-data-summary.json', 'transaction-plans.json',
                                      'produce.py', 'download_sources.py', 'assemble.py']]
    (HERE / 'output-manifest.json').write_text(json.dumps(
        [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in assets],
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()

