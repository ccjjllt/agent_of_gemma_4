"""Freeze a pure-weight dev A/B only after a real RFT adapter exists. No launch."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent
ROOT = HERE.parents[3]
NAME = 'p015_rft_lora'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-kernel', required=True, help='Real completed RFT source kernel, owner/slug')
    parser.add_argument('--training-manifest', type=Path, required=True, help='Downloaded real final training_manifest.json')
    parser.add_argument('--selection', type=Path, default=ROOT/'research/015/evaluation/final-selection.json')
    parser.add_argument('--material-kernel', default='zhuating/p015-natural-pr-panel-cpu')
    parser.add_argument('--output-dir', type=Path, default=HERE/'kernel')
    parser.add_argument('--manifest-path', type=Path, default=HERE/'carrier-manifest.json')
    parser.add_argument('--kernel-id', default='zhuating/p015-31b-weight-ab')
    parser.add_argument('--title', default='P015 31B Weight AB')
    parser.add_argument('--fast-thinking-budget', type=int, choices=(1024,),
                        help='Optionally run Stage C base-only1024 after all24 A/B attempts end')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', args.adapter_kernel):
        raise ValueError('Supply one explicit owner/slug RFT kernel')
    training = json.loads(args.training_manifest.read_text())
    if (training.get('status') != 'training_completed'
            or training.get('metrics', {}).get('optimizer_steps', 0) < 1
            or training.get('config', {}).get('base_family') != 'gemma-4-31b-it-qat-q4_0-unquantized'):
        raise ValueError('A completed real31B training manifest is required')
    adapter = training['adapter']; cfg = adapter['adapter_config']
    if (adapter['rank'] != 8 or cfg['r'] != 8 or cfg['lora_alpha'] != 16
            or cfg['peft_type'] != 'LORA' or cfg['bias'] != 'none'
            or len(adapter['tensors']) != 600 or adapter['tensor_parameters'] != 53043200):
        raise ValueError('Expected the actual independent rank8 text q/o/MLP adapter')
    capacity = ROOT/'research/015/capacity/output-fsdp-metadata/p015-fsdp/run/training_manifest.json'
    if capacity.is_file() and training['signature'] == json.loads(capacity.read_text())['signature']:
        raise ValueError('The capacity diagnostic adapter is not a task RFT candidate')
    files = adapter['files']
    if set(files) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Expected exactly the two independent final PEFT files')
    for name, pin in files.items():
        if (type(pin['bytes']) is not int or pin['bytes'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', pin['sha256'])):
            raise ValueError('Missing real final file identity: '+name)
    selection = json.loads(args.selection.read_text())
    ids = selection['dev_task_ids']
    frozen = json.loads((ROOT/'research/015/evaluation/final-selection.json').read_text())
    if (ids != frozen['dev_task_ids'] or len(ids) != 12 or len(set(ids)) != 12
            or set(ids) & set(selection['train_task_ids'])):
        raise ValueError('Use the frozen twelve independent dev tasks only')
    protocol = json.loads((ROOT/'research/015/protocol.json').read_text())
    base_dir = INFERENCE/'base-agent'
    base = {str(p.relative_to(base_dir)): p.read_text() for p in sorted(base_dir.rglob('*')) if p.is_file()}
    if len(base) != 4 or {k: hashlib.sha256(v.encode()).hexdigest() for k,v in base.items()} != protocol['baseline_agent_members']:
        raise ValueError('Base four files differ from the frozen official Agent')
    if re.search(r'^adapter\s*:', base['agent.yaml'], re.M):
        raise ValueError('Base Agent unexpectedly selects an adapter')
    adapted = dict(base)
    model_line = 'model: '+protocol['base_model']+'\n'
    if base['agent.yaml'].count(model_line) != 1:
        raise ValueError('Expected one actual base model alias in Agent YAML')
    adapted['agent.yaml'] = base['agent.yaml'].replace(model_line, model_line+'adapter: '+NAME+'\n')
    # The official schema's adapter field resolves through adapter:<name>;
    # worker registration already binds that alias to openai/<name>.
    bundle = {'base': base, 'adapter': adapted}
    if args.fast_thinking_budget is not None:
        fast = dict(base)
        fast['configs/sampling.yaml'], count = re.subn(
            r'(?m)^(\s*thinking_budget:\s*)4096\s*$',
            lambda match: match.group(1)+str(args.fast_thinking_budget), base['configs/sampling.yaml'])
        if count != 1:
            raise ValueError('Expected exactly one actual4096 thinking_budget in frozen sampling config')
        bundle['fast_base'] = fast
    settings = {'study': '31b-real-rft-weight-ab', 'task_ids': ids, 'task_count': 12,
        'model_alias': protocol['base_model'], 'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2',
        'budgets': protocol['budgets'], 'concurrency': 4, 'wall_seconds': 3600,
        'attempt_seconds': 720, 'material_archive_name': selection.get('material_archive_name', 'prepared-panel-material.tar.gz'),
        'material_sha256': selection['material_sha256'], 'material_bytes': selection['material_bytes'],
        'purpose': 'simultaneous same-service weight A/B on twelve frozen dev issues only; old5/12 remains historical',
        'selection_sha256': sha(args.selection),
        'fast_thinking_budget': args.fast_thinking_budget,
        'stage_c_scope': 'optional mechanism exploration only, after all24 A/B attempts finish; no weight-causal or historical-control gain claim',
        'adapter': {'name': NAME, 'source_kernel': args.adapter_kernel,
            'manifest_sha256': sha(args.training_manifest), 'signature': training['signature'],
            'files': files, 'rank': 8, 'parameters': adapter['tensor_parameters']},
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    original_builder = importlib.util.spec_from_file_location('p015_base_builder', INFERENCE/'build_carrier.py')
    module = importlib.util.module_from_spec(original_builder); original_builder.loader.exec_module(module)
    worker = (INFERENCE/'attempt_worker.py').read_text()
    coordinator = (HERE/'coordinator.py').read_text()
    source = ('SETTINGS = '+repr(settings)+'\n'+module.BOOTSTRAP+'\nBUNDLE = '+repr(bundle)
              +'\nWORKER_SOURCE = '+repr(worker)+'\n'+coordinator)
    ast.parse(source)
    # Never permit this new builder to overwrite a v1/next carrier or artifacts.
    if (args.output_dir.resolve() == INFERENCE.resolve()
            or (INFERENCE.resolve() in args.output_dir.resolve().parents and HERE.resolve() not in args.output_dir.resolve().parents)):
        raise ValueError('Weight A/B outputs must be inside weight-ab or an independent external directory')
    if args.manifest_path.resolve().parent == INFERENCE.resolve():
        raise ValueError('Do not overwrite the baseline carrier manifest')
    if args.manifest_path.exists() or args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('A/B output and manifest must be new; frozen artifacts are immutable')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'run.py').write_text(source)
    metadata = {'id':args.kernel_id, 'title':args.title, 'code_file':'run.py', 'language':'python',
        'kernel_type':'script', 'is_private':True, 'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
        'dataset_sources':['metric/gemma-4-developer-agent-wheelhouse'],
        'competition_sources':['gemma-4-developer-agent'],
        'kernel_sources':[args.material_kernel,args.adapter_kernel],
        'model_sources':[protocol['source_model_version']],
        'docker_image':module.IMAGE,'docker_image_pinning_type':'original'}
    (args.output_dir/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    result = {'settings':settings,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),
        'coordinator_sha256':hashlib.sha256(coordinator.encode()).hexdigest(),
        'source_bytes':len(source.encode()), 'platform_seconds':4200,
        'planned_attempts':24,'independent_dev_issues':12,
        'optional_stage_c_attempts':12 if args.fast_thinking_budget is not None else 0,
        'expected_routes':{'base':'official CT base path','adapter':NAME,
            **({'fast_base':'official CT base path'} if args.fast_thinking_budget is not None else {})},
        'only_AB_agent_difference':'official adapter YAML field; prompts/tools/sampling/budgets identical',
        'optional_C_difference':'base without adapter, only thinking_budget4096 to1024; all other fields identical',
        'capacity_adapter_mounted':False,'historical5of12_used_as_control':False,
        'official_schema_source':'P006 adk_submission/schema.py adapter; resolvers/models.py resolve_model; P014 eval12b prove_role_routing/register_worker_aliases'}
    args.manifest_path.parent.mkdir(parents=True,exist_ok=True)
    args.manifest_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
