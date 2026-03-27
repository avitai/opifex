#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGED_ENV_FILE=".opifex.env"
REQUESTED_BACKEND="auto"
PYTHON_VERSION=""
RECREATE=false
FORCE_CLEAN=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Opifex development environment setup

Usage:
  ./setup.sh [options]

Options:
  --backend <auto|cpu|cuda12|metal>  Choose the backend policy. Default: auto
  --python <version>                 Create the environment with a specific Python version
  --recreate                         Remove the existing .venv before syncing
  --force-clean                      Remove .venv, the generated .opifex.env, and repo-local test artifacts
  --dry-run                          Print the resolved backend and uv commands without changing files
  --help, -h                         Show this help

Notes:
  - Opifex uses uv for all repo-maintained setup and test workflows.
  - Linux CUDA development uses JAX's locally-bundled CUDA runtime via the `gpu` extra.
  - The setup does not rely on a system CUDA toolkit or custom LD_LIBRARY_PATH injection.
  - The generated backend file is .opifex.env. User-owned .env is not modified.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

have_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

resolve_backend() {
    if [[ "$REQUESTED_BACKEND" != "auto" ]]; then
        printf '%s\n' "$REQUESTED_BACKEND"
        return
    fi

    case "$(uname -s)" in
        Linux)
            if have_nvidia_gpu; then
                printf 'cuda12\n'
            else
                printf 'cpu\n'
            fi
            ;;
        Darwin)
            if [[ "$(uname -m)" == "arm64" ]]; then
                printf 'metal\n'
            else
                printf 'cpu\n'
            fi
            ;;
        *)
            printf 'cpu\n'
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || die "--backend requires a value"
            REQUESTED_BACKEND="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        --force-clean)
            FORCE_CLEAN=true
            RECREATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$REQUESTED_BACKEND" in
    auto|cpu|cuda12|metal) ;;
    *)
        die "unsupported backend '$REQUESTED_BACKEND'"
        ;;
esac

command -v uv >/dev/null 2>&1 || die "uv is required but not installed"
command -v python3 >/dev/null 2>&1 || die "python3 is required but not installed"

BACKEND="$(resolve_backend)"
SYNC_ARGS=(sync --extra dev --extra test)

case "$BACKEND" in
    cpu) ;;
    cuda12)
        SYNC_ARGS+=(--extra gpu)
        ;;
    metal)
        SYNC_ARGS+=(--extra metal)
        ;;
    *)
        die "resolved unsupported backend '$BACKEND'"
        ;;
esac

if [[ "$DRY_RUN" == true ]]; then
    echo "project root: $PROJECT_ROOT"
    echo "requested backend: $REQUESTED_BACKEND"
    echo "resolved backend: $BACKEND"
    if [[ -n "$PYTHON_VERSION" ]]; then
        echo "uv venv --python $PYTHON_VERSION"
    fi
    echo "uv ${SYNC_ARGS[*]}"
    echo "python3 scripts/setup_env.py write --backend $BACKEND --output $MANAGED_ENV_FILE"
    exit 0
fi

cd "$PROJECT_ROOT"

if [[ "$RECREATE" == true ]]; then
    if [[ -d ".venv" ]]; then
        rm -rf .venv
    fi
fi

if [[ "$FORCE_CLEAN" == true ]]; then
    rm -f "$MANAGED_ENV_FILE"
    rm -rf .pytest_cache htmlcov
    rm -f temp/test-results.json temp/coverage.json
fi

if [[ -n "$PYTHON_VERSION" ]]; then
    uv venv --python "$PYTHON_VERSION"
fi

uv "${SYNC_ARGS[@]}"
python3 scripts/setup_env.py write --backend "$BACKEND" --output "$MANAGED_ENV_FILE"

if [[ -f ".env" ]]; then
    cat <<EOF
Note: .env already exists and is treated as a user-owned override layer.
It will be loaded after $MANAGED_ENV_FILE and can override the generated backend settings.
EOF
fi

cat <<EOF
Opifex environment synced.
Resolved backend: $BACKEND
Managed backend file: $MANAGED_ENV_FILE

Next steps:
  source ./activate.sh
  uv run python scripts/verify_opifex_gpu.py
EOF
