#!/bin/bash
# F4 second-wheel runner: complements the live first-round worker.
# Reads inputs read-only from replay/artifacts/f4-new + replay/*.json,
# writes ONLY under acquisition/f4-new/f4-cont/. Never overwrites f4-new old files.
# Usage: bash run_second_wheel.sh <key>  (e.g. f4-new-checker-var)
# Requires: competition/vfarl/replay/.venv
set -u
ROOT=/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4
VF=$ROOT/competition/vfarl
KEY=${1:?usage: run_second_wheel.sh '<key>'}
OUT=$VF/acquisition/f4-new/f4-cont/$KEY-run-002
if [ -e "$OUT" ]; then echo "refuse: output exists: $OUT"; exit 1; fi
PREP=$VF/replay/$KEY-source-preparation.json
TASK_REL=$(python3 -c "import json;print(json.load(open('$PREP'))['adapted_task'])")
WS_REL=$(python3 -c "import json;print(json.load(open('$PREP'))['workspace'])")
RECIPE_REL=$(python3 -c "import json;print(json.load(open('$PREP'))['recipe'])")
PROFILE=$(python3 -c "
import json,glob
ss=json.load(open('$VF/replay/artifacts/f4-new/$KEY-run-001/run-summary.json')).get('task_site_packages_profile')
print(ss if ss else '')")
ARGS=(--task "$ROOT/$TASK_REL" --workspace "$ROOT/$WS_REL" --recipe "$ROOT/$RECIPE_REL" --output "$OUT" --verify --verify-baseline)
if [ -n "$PROFILE" ]; then ARGS+=(--task-site-packages "$PROFILE"); fi
exec "$VF/replay/.venv/bin/python" "$VF/replay/run_official.py" "${ARGS[@]}"
