#!/usr/bin/env bash
# Install isolated environments on the Linux server. Never starts training or a GPU job.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_server.sh {data|learner|runtime} [options]
  --base PATH          Persistent data-disk root (default /root/autodl-tmp/gemma4)
  --wheelhouse PATH    Downloaded official metric wheelhouse v31
  --bootstrap-python P Python executable with venv support (default python3)

data:    Optional CLI environment: Python 3.12, Kaggle 2.2.4, Hugging Face Hub 0.36.2.
learner: Python 3.12, repository requirements/learner.txt (candidate versions).
runtime: Python 3.13 in /opt/gemma-runtime, CPU PyTorch and official runtime wheels.

Download official data, model files, and runtime wheels manually on the server.
The data phase only installs CLI tools; it does not download any assets.
No credentials, model weights, or training datasets are copied from this computer.
EOF
}

phase="${1:-}"
if [[ "$phase" == "--help" || "$phase" == "-h" || -z "$phase" ]]; then
  usage
  exit 0
fi
shift
case "$phase" in data|learner|runtime) ;; *) usage >&2; exit 2 ;; esac
base=/root/autodl-tmp/gemma4
wheelhouse=""
bootstrap_python=python3
while (($#)); do
  case "$1" in
    --base|--wheelhouse|--bootstrap-python)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --base) base="$2" ;;
        --wheelhouse) wheelhouse="$2" ;;
        --bootstrap-python) bootstrap_python="$2" ;;
      esac
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux ]] || { printf 'Run this installer on the Linux training server.\n' >&2; exit 2; }
[[ "$base" == /* && "$base" != / ]] || { printf '--base must be an absolute data-disk directory.\n' >&2; exit 2; }
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$base" != "$project_root" && "$base" != "$project_root/"* ]] || {
  printf 'Keep environments and official assets outside the source checkout.\n' >&2; exit 2;
}
if [[ "$phase" == learner && ! -f "$project_root/requirements/learner.txt" ]]; then
  printf 'Missing requirements/learner.txt; do not install unpinned latest packages.\n' >&2
  exit 2
fi
if [[ "$phase" == runtime ]]; then
  [[ "$EUID" -eq 0 ]] || { printf 'runtime installation needs root for /opt and low-privilege task execution.\n' >&2; exit 2; }
  [[ -n "$wheelhouse" ]] || wheelhouse="$base/official/wheelhouse-v31"
  runtime_wheels=(
    adk_eval_core-0.1.0-py3-none-any.whl
    adk_submission-0.2.13-py3-none-any.whl
    swegemma-0.2.11-py3-none-any.whl
    google_adk-1.36.1-py3-none-any.whl
    google_genai-2.11.0-py3-none-any.whl
    transformers-5.13.1-py3-none-any.whl
  )
  for name in "${runtime_wheels[@]}"; do
    [[ -f "$wheelhouse/$name" ]] || { printf 'Missing official wheel: %s\n' "$wheelhouse/$name" >&2; exit 2; }
  done
fi

umask 077
mkdir -p "$base/envs" "$base/install-records" "$base/cache/uv"
bootstrap_env="$base/envs/bootstrap"
if [[ ! -x "$bootstrap_env/bin/python" ]]; then
  "$bootstrap_python" -m venv "$bootstrap_env"
fi
"$bootstrap_env/bin/python" -m pip install --disable-pip-version-check 'uv==0.9.26'
uv_bin="$bootstrap_env/bin/uv"
export UV_CACHE_DIR="$base/cache/uv"
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=""

create_env() {
  local environment="$1" python_version="$2"
  if [[ ! -x "$environment/bin/python" ]]; then
    "$uv_bin" venv --python "$python_version" --seed "$environment"
  fi
  "$environment/bin/python" - "$python_version" <<'PY'
import sys
expected = tuple(map(int, sys.argv[1].split('.')))
if sys.version_info[:2] != expected:
    raise SystemExit(f'Existing environment Python {sys.version_info[:2]} does not match {expected}')
PY
}

case "$phase" in
  data)
    environment="$base/envs/data"
    create_env "$environment" 3.12
    "$uv_bin" pip install --python "$environment/bin/python" \
      'kaggle==2.2.4' 'huggingface_hub==0.36.2'
    ;;
  learner)
    environment="$base/envs/learner"
    create_env "$environment" 3.12
    "$uv_bin" pip install --python "$environment/bin/python" \
      --index-url https://download.pytorch.org/whl/cu128 \
      'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' 'torchao==0.14.1+cu128'
    "$uv_bin" pip install --python "$environment/bin/python" \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
      -r "$project_root/requirements/learner.lock.txt"
    ;;
  runtime)
    environment=/opt/gemma-runtime
    # The tool worker's unprivileged uid must be able to read both the venv and
    # its managed Python executable; a Python installed under /root will fail.
    export UV_PYTHON_INSTALL_DIR=/opt/gemma-python
    export UV_PYTHON_PREFERENCE=only-managed
    mkdir -p "$UV_PYTHON_INSTALL_DIR"
    "$uv_bin" python install 3.13
    create_env "$environment" 3.13
    "$uv_bin" pip install --python "$environment/bin/python" \
      --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.9.1+cpu' 'torchvision==0.24.1+cpu'
    official_paths=()
    for name in "${runtime_wheels[@]}"; do official_paths+=("$wheelhouse/$name"); done
    "$uv_bin" pip install --python "$environment/bin/python" "${official_paths[@]}"
    chmod -R a+rX /opt/gemma-runtime /opt/gemma-python
    ;;
esac
"$uv_bin" pip freeze --python "$environment/bin/python" > "$base/install-records/$phase.freeze.txt"
printf 'Prepared %s at %s\n' "$phase" "$environment"
printf 'Installed dependency record: %s/install-records/%s.freeze.txt\n' "$base" "$phase"
printf 'No model was loaded and no training was started.\n'
