#!/usr/bin/env bash
# Environment only: no datasets, checkpoints, credentials or training jobs.
set -Eeuo pipefail
trap 'printf "Environment setup failed at line %s.\n" "$LINENO" >&2' ERR

DOPD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOPD_PREFIX="$DOPD_ROOT/.envs/d-opd"
DOPD_CHECK_ONLY=0
while (($#)); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo '--prefix requires a directory' >&2; exit 2; }
            DOPD_PREFIX="$2"; shift 2 ;;
        --check-only) DOPD_CHECK_ONLY=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: bash setup_env.sh [--prefix DIRECTORY] [--check-only]

Create an isolated Python 3.10 / CUDA 12.4 environment and verify d-OPD.
Requires Linux x86_64, Conda, an NVIDIA driver and an Ampere/Ada/Hopper GPU.
Default environment: .envs/d-opd beside this script.
Downloads software packages only; no models or datasets.
MAX_JOBS controls compilation parallelism (default: 4).
EOF
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'This installer targets Linux x86_64.' >&2; exit 1;
}
DOPD_CONDA="${CONDA_EXE:-}"
if [[ ! -x "$DOPD_CONDA" ]]; then DOPD_CONDA="$(command -v conda || true)"; fi
[[ -n "$DOPD_CONDA" ]] || { echo 'Install Conda/Miniforge first, then rerun this script.' >&2; exit 1; }
DOPD_CONDA_BASE="$("$DOPD_CONDA" --no-plugins info --base)"
# shellcheck source=/dev/null
source "$DOPD_CONDA_BASE/etc/profile.d/conda.sh"
DOPD_PREFIX="$(realpath -m -- "$DOPD_PREFIX")"
export MAX_JOBS="${MAX_JOBS:-4}"
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_JOBS must be a positive integer.' >&2; exit 2; }
export CONDA_REPODATA_USE_SHARDS=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
unset PYTHONPATH

if [[ "$DOPD_CHECK_ONLY" == 0 ]]; then
    if [[ -e "$DOPD_PREFIX" && ! -f "$DOPD_PREFIX/.dopd-managed" ]]; then
        echo "Refusing to modify an unrelated existing directory: $DOPD_PREFIX" >&2
        echo 'Choose a new --prefix, or use --check-only.' >&2
        exit 1
    fi
    if [[ ! -f "$DOPD_PREFIX/conda-meta/history" ]]; then
        conda create --yes --prefix "$DOPD_PREFIX" --override-channels \
            -c nvidia/label/cuda-12.4.1 -c conda-forge \
            python=3.10 pip 'cuda-toolkit=12.4.1' 'gxx_linux-64=12'
        touch "$DOPD_PREFIX/.dopd-managed"
    fi
    # Conda compiler activation hooks may read unset variables.
    set +u
    conda activate "$DOPD_PREFIX"
    set -u
    # CUDA packages may use the target-specific header/library layout.
    if [[ -f "$DOPD_PREFIX/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
        mkdir -p "$DOPD_PREFIX/.dopd-cuda"
        ln -sfn "$DOPD_PREFIX/bin" "$DOPD_PREFIX/.dopd-cuda/bin"
        ln -sfn "$DOPD_PREFIX/targets/x86_64-linux/include" "$DOPD_PREFIX/.dopd-cuda/include"
        ln -sfn "$DOPD_PREFIX/targets/x86_64-linux/lib" "$DOPD_PREFIX/.dopd-cuda/lib64"
        export CUDA_HOME="$DOPD_PREFIX/.dopd-cuda"
    else
        export CUDA_HOME="$DOPD_PREFIX"
    fi
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA compiler installation is incomplete.' >&2; exit 1; }
    export TORCH_EXTENSIONS_DIR="$DOPD_PREFIX/.cache/torch_extensions"
    export TRITON_CACHE_DIR="$DOPD_PREFIX/.cache/triton"
    conda env config vars set --prefix "$DOPD_PREFIX" CUDA_HOME="$CUDA_HOME" MAX_JOBS="$MAX_JOBS" \
        TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    python -m pip install 'pip==25.3' 'setuptools==75.8.0' wheel ninja packaging 'numpy==1.26.4'
    python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
    DS_BUILD_OPS=0 DS_BUILD_CPU_ADAM=1 python -m pip install --no-build-isolation \
        -r "$DOPD_ROOT/requirements.txt"
    # FlashAttention renames its downloaded wheel; keep build/wheel temporaries on the same filesystem.
    python -m pip install --no-cache-dir --no-build-isolation 'flash-attn==2.7.4.post1'
else
    [[ -f "$DOPD_PREFIX/conda-meta/history" ]] || { echo "No Conda environment at $DOPD_PREFIX" >&2; exit 1; }
    # Conda compiler activation hooks may read unset variables.
    set +u
    conda activate "$DOPD_PREFIX"
    set -u
fi

cd "$DOPD_ROOT"
python -m pip check
python check_env.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
printf '\nEnvironment checks passed. Activate with:\n  conda activate %q\n' "$DOPD_PREFIX"
