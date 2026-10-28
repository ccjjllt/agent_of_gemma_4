# Frozen R3 to current-four CPU R4

Only task-scope/input-version binding is added. Embedded SOURCE_BUNDLE remains byte-identical.

```diff
--- frozen-carrier-r3/run.py
+++ current-four-carrier-r4/run.py
@@ -19,7 +19,7 @@
 RUNTIME_SHA256 = '0a8e277010c63701b1a163869bd6998732c18307770d7297a5cea524391879a0'
 CONTEXT_LENGTH = 4096
 TRAINING_CONFIG = 'training/configs/vfarl_l4x4.yaml'
-ONLY_TASKS = ['getmoto__moto-5348', 'getmoto__moto-6159', 'iterative__dvc.1d6ea681.pr_7628', 'matthewwithanm__python-markdownify.6258f5c3.pr_161', 'marshmallow-code__marshmallow.9716fc62.lm_rewrite__nqk1hyjp']
+ONLY_TASKS = ['getmoto__moto-5348', 'getmoto__moto-6159', 'iterative__dvc.1d6ea681.pr_7628', 'matthewwithanm__python-markdownify.6258f5c3.pr_161']
 EXPORT_RUNTIME = True
 COLLECT_ONLY = False
 FOCAL_STATES = 5
@@ -697,6 +697,76 @@
     (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
 
 
+CURRENT_CPU_INPUTS = <exact compressed frozen-input bytes; see run.py>
+
+
+def bind_current_four_cpu_inputs(material, out):
+    payload = json.loads(zlib.decompress(base64.b64decode(CURRENT_CPU_INPUTS)))
+    parent_raw = base64.b64decode(payload['parent_selection_base64'])
+    if hashlib.sha256(parent_raw).hexdigest() != payload['parent_selection_sha256']:
+        raise RuntimeError('Current parent selection identity changed')
+    parent = json.loads(parent_raw)
+    by_id = {row['instance_id']: row for row in parent['tasks']}
+    if len(by_id) != 100 or payload['only_tasks'] != ONLY_TASKS:
+        raise RuntimeError('Expected the authorized four-task current scope')
+    subset = dict(format='fixed100-current-four-cpu-selection-v1',
+        parent_selection_sha256=payload['parent_selection_sha256'],
+        tasks=[by_id[iid] for iid in ONLY_TASKS])
+    for binding in payload['bindings']:
+        iid = binding['instance_id']
+        asset = material / 'assets' / iid
+        task_path = asset / 'task.json'
+        if sha(task_path) != binding['old_task_sha256']:
+            raise RuntimeError('Original material task changed: ' + iid)
+        original = json.loads(task_path.read_bytes())
+        override = payload['task_overrides_base64'].get(iid)
+        current_raw = base64.b64decode(override) if override else task_path.read_bytes()
+        current = json.loads(current_raw)
+        if (hashlib.sha256(current_raw).hexdigest() != binding['current_task_sha256']
+                or {k:v for k,v in original.items() if k != 'problem_statement'}
+                   != {k:v for k,v in current.items() if k != 'problem_statement'}
+                or sha(asset / 'source-snapshot.tgz') != binding['source_snapshot_sha256']
+                or sha(asset / 'submitted.patch') != binding['reviewed_patch_sha256']):
+            raise RuntimeError('Current task/source/patch/oracle binding changed: ' + iid)
+        if override:
+            task_path.write_bytes(current_raw)
+    (material / 'current-four-selection.json').write_text(json.dumps(subset, ensure_ascii=False, indent=2) + '\n')
+    (material / 'current-four-questions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in payload['questions']))
+    evidence = dict(parent_selection_sha256=payload['parent_selection_sha256'],
+        actual_four_selection_sha256=sha(material / 'current-four-selection.json'),
+        actual_four_questions_sha256=sha(material / 'current-four-questions.jsonl'),
+        tasks=payload['bindings'], source_bytecode_policy='legacy-default',
+        original_material_selection_unchanged=True,
+        boundary='Only these four current task versions are evaluated; other material assets are not relabeled current. No agent/model prompt execution.')
+    (out / 'current-four-input-binding.json').write_text(json.dumps(evidence, indent=2) + '\n')
+
+
+def bind_current_four_public_prompt(material, out, source, env):
+    # Real pinned harness function; render identity only, without starting an agent/model.
+    program = """import json,hashlib,inspect,sys
+from pathlib import Path
+from types import SimpleNamespace
+from swegemma.harness.agent_runner import build_agent_prompt
+material,out=map(Path,sys.argv[1:])
+selection=json.loads((material/'current-four-selection.json').read_text())
+records=[]
+for row in selection['tasks']:
+ asset=material/'assets'/row['instance_id']
+ task=SimpleNamespace(**json.loads((asset/'task.json').read_text()))
+ old=json.loads((asset/'run-summary.json').read_text())
+ config=SimpleNamespace(budget=SimpleNamespace(**old['evaluation_budget']),harness=SimpleNamespace(**old['harness_limits']),enable_sandbox_testing=True)
+ prompt=build_agent_prompt(task,config,declared_tools=['run_command','read_file','edit_file','write_file','get_status','submit_patch'])
+ digest=hashlib.sha256(prompt.encode()).hexdigest()
+ if digest!=row['official_initial_user_prompt_sha256']:raise RuntimeError('Actual current public prompt mismatch: '+row['instance_id'])
+ records.append(dict(instance_id=row['instance_id'],actual_rendered_prompt_sha256=digest,evaluation_budget=old['evaluation_budget'],harness_limits=old['harness_limits']))
+path=Path(inspect.getsourcefile(build_agent_prompt))
+record=dict(actual_harness_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tasks=records,model_or_agent_execution=False,scope='Render-only current public prompt identity; CPU oracle keeps frozen prevalidation budget/harness')
+(out/'current-four-public-prompt-binding.json').write_text(json.dumps(record,indent=2)+chr(10))
+"""
+    subprocess.run([sys.executable, '-c', program, str(material), str(out)],
+                   env=env, cwd=source, check=True)
+
+
 def task_scopes(selection):
     """Keep independent research comparison tasks outside reward collection."""
     task_ids = [row['instance_id'] for row in selection['tasks']]
@@ -865,6 +935,7 @@
                                          or name.startswith(prefixes))
     unpack(material_archive, Path('/kaggle/working'), include=material_filter)
     material = Path('/kaggle/working/vfarl-input')
+    bind_current_four_cpu_inputs(material, out)
     selection = json.loads((material / 'selection.json').read_text())
     task_ids, collection_ids, validation_ids = task_scopes(selection)
     if not set(ONLY_TASKS) <= set(task_ids):
@@ -946,10 +1017,11 @@
         env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
     # Kaggle denied all namespace probes. CPU task acceptance and GPU rollout
     # must use the same production chroot executor and its live capability probe.
+    bind_current_four_public_prompt(material, out, source, env)
     isolation = 'chroot'
     common = [sys.executable, '-m', 'competition.vfarl.l4x4.run',
-              '--selection', str(material / 'selection.json'),
-              '--questions', str(material / 'questions.jsonl'),
+              '--selection', str(material / 'current-four-selection.json'),
+              '--questions', str(material / 'current-four-questions.jsonl'),
               '--assets', str(material / 'assets'),
               '--tools', str(source / 'competition/vfarl/six-tools.json'),
               '--isolation', isolation]
```
