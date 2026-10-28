# CPU r4 to r5: existing uv runtime executable repair only

```diff
--- frozen-carrier-r4/run.py
+++ cpu-uv-repair-carrier-r5/run.py
@@ -430,6 +430,99 @@
             problems[key] = {'expected_profile': label, 'observed_problems': failures,
                              'reason': 'No valid packaged dependency profile; prepare/export the CPU runtime before collection'}
     return bindings, problems
+
+
+def ensure_cpu_uv(stage, out, prep):
+    # Repair this runtime input only; never install/download a new uv package.
+    target = stage / 'runtime/bin/uv'
+    before = dict(path=str(target), exists=target.exists(), is_symlink=target.is_symlink())
+    try:
+        info = target.lstat()
+        before.update(mode=oct(info.st_mode), size=info.st_size,
+            symlink_target=os.readlink(target) if target.is_symlink() else None,
+            resolved_path=str(target.resolve()),
+            original_sha256=sha(target) if target.is_file() else None)
+    except OSError as error:
+        before['lstat_error'] = f'{type(error).__name__}: {error}'
+    candidates = [target]
+    path_uv = shutil.which('uv')
+    if path_uv:
+        candidates.append(Path(path_uv).resolve())
+    selected, version = None, None
+    attempts, seen = [], set()
+    for candidate in candidates:
+        resolved = candidate.resolve()
+        if resolved in seen:
+            attempts.append(dict(path=str(candidate), reason='same already-attempted resolved path'))
+            continue
+        seen.add(resolved)
+        try:
+            if not candidate.is_file() or not os.access(candidate, os.X_OK):
+                attempts.append(dict(path=str(candidate), reason='not an executable file'))
+                continue
+            with candidate.open('rb') as stream:
+                if stream.read(4) != b'\x7fELF':
+                    attempts.append(dict(path=str(candidate), reason='not a native Linux executable; wrapper not run/copied'))
+                    continue
+            result = subprocess.run([str(candidate), '--version'], capture_output=True,
+                                    text=True, timeout=15, check=True)
+            if not result.stdout.startswith('uv '):
+                raise RuntimeError('Unexpected uv executable version output')
+            selected, version = candidate.resolve(), result.stdout.strip()
+            break
+        except Exception as error:
+            attempts.append(dict(path=str(candidate), error=f'{type(error).__name__}: {error}',
+                returncode=getattr(error, 'returncode', None),
+                stdout=getattr(error, 'stdout', None), stderr=getattr(error, 'stderr', None)))
+    origin = 'existing-packaged-or-PATH-executable'
+    wheel_record = None
+    if selected is None:
+        import zipfile
+        wheels = [row for row in prep['runtime_wheels'] if row['name'].startswith('uv-')]
+        if len(wheels) != 1:
+            raise RuntimeError('No unique already-declared uv wheel; do not download a replacement')
+        wheel_record = wheels[0]
+        wheel = stage / 'runtime-wheels' / wheel_record['name']
+        if not wheel.is_file() or sha(wheel) != wheel_record['sha256']:
+            raise RuntimeError('Existing uv wheel is unavailable or its input pin differs')
+        with zipfile.ZipFile(wheel) as archive:
+            names = [name for name in archive.namelist() if name.endswith('/uv') and not name.endswith('/')]
+            if len(names) != 1:
+                raise RuntimeError('Existing uv wheel has no unique binary member')
+            data = archive.read(names[0])
+        if not data.startswith(b'\x7fELF'):
+            raise RuntimeError('Expected the existing Linux uv native binary')
+        target.parent.mkdir(parents=True, exist_ok=True)
+        if target.is_symlink():
+            target.unlink()
+        target.write_bytes(data)
+        target.chmod(0o755)
+        selected = target.resolve()
+        origin = 'sha-bound-wheel-already-inside-runtime-input'
+    if target.is_symlink() or selected != target.resolve() or not target.exists():
+        # A regular copied ELF remains usable when the repaired runtime is exported;
+        # an absolute symlink to the base image would not be a portable runtime input.
+        with selected.open('rb') as stream:
+            if stream.read(4) != b'\x7fELF':
+                raise RuntimeError('PATH uv is not a native executable; no wrapper is silently repackaged')
+        target.parent.mkdir(parents=True, exist_ok=True)
+        if target.is_symlink():
+            target.unlink()
+        shutil.copy2(selected, target)
+        target.chmod(0o755)
+    actual = subprocess.run([str(target), '--version'], capture_output=True,
+                            text=True, timeout=15, check=True)
+    if not actual.stdout.startswith('uv '):
+        raise RuntimeError('Restored uv executable did not identify itself')
+    # Existing install commands retain their exact version pins and CLI arguments.
+    # Explicitly prohibit index downloads, including for an absent cached dependency.
+    os.environ['UV_OFFLINE'] = '1'
+    record = dict(before=before, origin=origin, source_path=str(selected),
+        source_sha256=sha(selected), target_path=str(target), target_sha256=sha(target),
+        actual_version=actual.stdout.strip(), executable_attempts=attempts,
+        input_wheel=wheel_record, UV_OFFLINE='1', new_downloads_permitted=False)
+    (out / 'uv-runtime-repair.json').write_text(json.dumps(record, indent=2) + '\n')
+    print(json.dumps(dict(phase='actual_cpu_uv_runtime_repair', **record)), flush=True)
 
 
 def prepare_cpu_repairs(stage, source, out, prep):
@@ -968,6 +1061,7 @@
         target.parent.mkdir(parents=True, exist_ok=True)
         target.write_text(content)
     if MODE == 'prevalidate':
+        ensure_cpu_uv(stage, out, prep)
         prepare_cpu_repairs(stage, source, out, prep)
     else:
         # A later pre-policy selection can reuse an immutable dependency
```
