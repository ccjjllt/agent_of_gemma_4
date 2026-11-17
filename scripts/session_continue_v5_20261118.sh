#!/bin/bash
# New run from published v5; fresh optimizer and 64-step LR horizon.
# The whole session, including vLLM load and shutdown, is limited to eight hours.
# Deploy this file as /root/autodl-tmp/gemma4/logs/session-20261118-cont-v5.sh.
set -u

PROJECT=/root/gemma4/project
LOGS=/root/autodl-tmp/gemma4/logs
RUN=/root/autodl-tmp/gemma4/runs/spo-v3-cont-v5-r128-20261118
CKPT=/root/autodl-fs/spo-checkpoints/spo-v3-cont-v5-r128-20261118

if [ -e "$RUN" ]; then
  echo "refusing: run dir already exists; use the documented resume procedure: $RUN"
  exit 2
fi
mkdir -p "$LOGS" "$CKPT" /root/autodl-tmp/gemma4/tmp

if [ "${1:-}" != "--supervised" ]; then
  exec /root/gemma4/envs/learner/bin/python "$PROJECT/scripts/session_supervisor.py" \
    --budget-seconds 28800 --grace-seconds 180 \
    --session-file "$LOGS/session-20261118-cont-v5.json" \
    --log "$LOGS/session-20261118-cont-v5.log" \
    -- /bin/bash "$0" --supervised
fi

export PATH=/root/gemma4/envs/inference-vllm/bin:/usr/local/bin:/usr/bin:/bin
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/root/autodl-tmp/gemma4/tmp
command -v ninja >/dev/null || { echo "ninja not on PATH"; exit 2; }
if curl -sf http://127.0.0.1:8000/health >/dev/null; then
  echo "refusing: a vLLM service is already using port 8000"
  exit 2
fi

/root/gemma4/envs/inference-vllm/bin/vllm serve /autodl-fs/data \
  --served-model-name google/gemma-4-31B-it-qat-q4_0-unquantized \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-lora --max-loras 2 --max-cpu-loras 2 --max-lora-rank 128 \
  --enable-sleep-mode --enable-prefix-caching \
  --host 127.0.0.1 --port 8000 > "$LOGS/vllm-cont-v5-20261118.log" 2>&1 &
VLLM=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 "$VLLM" 2>/dev/null || break
  sleep 5
done
if ! curl -sf http://127.0.0.1:8000/health >/dev/null; then
  echo "vLLM failed to start"
  kill "$VLLM" 2>/dev/null || true
  wait "$VLLM" 2>/dev/null || true
  exit 2
fi

cd "$PROJECT" || exit 2
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_v3_continue_v5_rank128_20261118.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$RUN" \
  --backend-url http://127.0.0.1:8000
code=$?
kill "$VLLM" 2>/dev/null || true
wait "$VLLM" 2>/dev/null || true
exit "$code"
