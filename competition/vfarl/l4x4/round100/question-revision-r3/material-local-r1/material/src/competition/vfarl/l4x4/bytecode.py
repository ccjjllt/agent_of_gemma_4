"""Explicit task-Python bytecode semantics for prospective CPU/rollout runs.

The legacy manager remains the default. The opt-in profile redirects normal
CPython cache lookup to a task-local empty prefix and disables cache writes.
It neither deletes source/dependency caches nor changes the parent model's env.
"""
from __future__ import annotations

LEGACY = 'legacy-default'
FRESH_PREFIX = 'isolated-prefix-no-write-v1'
POLICIES = (LEGACY, FRESH_PREFIX)


def policy_name(runtime):
    name = runtime.get('source_bytecode_policy', LEGACY)
    if name not in POLICIES:
        raise ValueError('Unknown source bytecode policy: ' + str(name))
    return name


def configure_manager(manager, name=LEGACY):
    """Wrap this manager only; retain official translation and result handling.

The shell exports occur after the official environment sanitizer. TMPDIR is
already the sandbox's own tmp directory, including under chroot/bwrap. The
prefix starts absent in a fresh sandbox and is included in focal snapshots.
Python -E/-I and commands explicitly overriding the variables are outside this
profile's guarantee; already imported modules still require a new process.
"""
    policy_name({'source_bytecode_policy': name})
    evidence = {'name': name, 'default_changed': False,
                'cache_files_deleted': False,
                'model_or_shared_dependency_cache_modified': False}
    if getattr(manager, '_vfarl_source_bytecode_policy', None):
        raise ValueError('A source bytecode policy is already installed')
    if name == LEGACY:
        return evidence
    original_exec = manager.exec

    def execute(sandbox_id, command, *, timeout=None):
        # A newline preserves the original command as one complete shell body.
        # -B alone still reads source-adjacent timestamp/size bytecode caches.
        prefix = ('export PYTHONDONTWRITEBYTECODE=1\n'
                  'export PYTHONPYCACHEPREFIX="$TMPDIR/.vfarl-empty-bytecode-v1"\n')
        return original_exec(sandbox_id, prefix + command, timeout=timeout)

    manager.exec = execute
    evidence.update(environment={
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPYCACHEPREFIX': '$TMPDIR/.vfarl-empty-bytecode-v1'},
        injection='manager-local shell exports after official sanitizer',
        boundary='Normal new CPython processes honoring environment; -E/-I, explicit overrides and live imported modules remain separate')
    manager._vfarl_source_bytecode_policy = dict(evidence)
    return evidence
