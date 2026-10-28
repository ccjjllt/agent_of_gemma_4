# UV-only r5 to final CPU r5: one non-gating owned-process observation

```diff
--- preserved-uv-only-r5/run.py
+++ final-r5-budget-observation/run.py
@@ -430,6 +430,49 @@
             problems[key] = {'expected_profile': label, 'observed_problems': failures,
                              'reason': 'No valid packaged dependency profile; prepare/export the CPU runtime before collection'}
     return bindings, problems
+
+
+def observe_linux_owned_process_budget(out):
+    # One actual <=3s process observation, not a CPU4 readiness/eligibility gate.
+    folder = out / 'owned-process-budget-observation'
+    folder.mkdir()
+    program = <exact inline program; inspect run.py and budget-observation-inline.txt>
+    started = time.monotonic()
+    summary = dict(control_source_sha256='fd25b6732986acd99ecfe8cb3d0d5f192a994362d8e8739297e258eb8c9412d4', outer_timeout_seconds=3,
+        execution_seconds=.75, cleanup_seconds=1.5, source_or_oracle_changed=False,
+        CPU4_is_gated=False, model_payload_present=False)
+    try:
+        with (folder / 'stdout-stderr.log').open('w') as log:
+            result = subprocess.run([sys.executable, '-c', program, str(folder)],
+                stdout=log, stderr=subprocess.STDOUT, timeout=3, check=False)
+        summary['actual_supervisor_returncode'] = result.returncode
+    except Exception as error:
+        summary['failure'] = f'{type(error).__name__}: {error}'
+    finally:
+        summary['elapsed_seconds'] = time.monotonic() - started
+        for name, key in [('primitive-result.json', 'actual_primitive_result'),
+                          ('owned-process-budget-receipt.json', 'actual_guard_receipt')]:
+            path = folder / name
+            if path.is_file():
+                try:
+                    summary[key] = json.loads(path.read_text())
+                except Exception as error:
+                    summary[key + '_read_failure'] = f'{type(error).__name__}: {error}'
+        events = []
+        if (folder / 'stdout-stderr.log').is_file():
+            for line in (folder / 'stdout-stderr.log').read_text().splitlines():
+                try:
+                    event = json.loads(line)
+                    if isinstance(event, dict) and str(event.get('event', '')).startswith('budget_probe_'):
+                        events.append(event)
+                except ValueError:
+                    pass
+        summary['actual_process_events'] = events
+        summary['actual_setsid_descendant_observed'] = any(
+            e.get('event') == 'budget_probe_grandchild' and e.get('sid') == e.get('pid') for e in events)
+        (folder / 'observation.json').write_text(json.dumps(summary, indent=2) + '\n')
+        print(json.dumps(dict(phase='actual_owned_process_budget_observation', **summary)), flush=True)
+    # Unsupported/failure remains actual evidence; do not stop the original CPU four tasks.
 
 
 def ensure_cpu_uv(stage, out, prep):
@@ -1010,6 +1053,12 @@
     out = Path('/kaggle/working/vfarl-' + MODE)
     out.mkdir(exist_ok=False)
     _OUTPUT = out
+    if MODE == 'prevalidate':
+        try:
+            observe_linux_owned_process_budget(out)
+        except Exception as error:
+            print(json.dumps(dict(phase='actual_owned_process_budget_observation',
+                CPU4_is_gated=False, failure=f'{type(error).__name__}: {error}')), flush=True)
     if MODE == 'capabilities':
         record_capabilities(out)
         return
```
