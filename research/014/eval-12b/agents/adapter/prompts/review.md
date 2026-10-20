You review the actual production fix in /workspace, check the issue contract, and submit.

# Original issue
{problem_description}

# Repair handoff
{repair_handoff}

Read git status --short and the actual git diff. Check the exact behaviours, names, exception types, messages and edge cases requested by the issue. The handoff is evidence to examine, not a substitute for the diff. If a concrete requirement is missing, make one targeted correction.

Run one focused acceptance command using existing relevant tests or a short inline check. Use PYTHONPATH=/workspace/src:/workspace and confirm module.__file__ points to the working tree before interpreting Python results. Use timeout 10 for inline reproductions, or timeout 60 (up to 120 if needed) for a named existing test file. A timeout means verification incomplete, not proof the patch is wrong. Do not rerun an equivalent check already sufficient for acceptance unless there is a specific concern.

Do not create, modify or delete test files, and do not run full suites. Remove only temporary products demonstrably created during this attempt; preserve pre-existing tracked and untracked files. Check the final git diff and status. Call submit_patch with a nonempty production patch. If it is empty, inspect actual source state and make a real correction before any retry. End with a factual sentence describing the fix and verification result.
