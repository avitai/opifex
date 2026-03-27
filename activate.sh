#!/usr/bin/env bash

# shellcheck disable=SC2317
_opifex_activate_die() {
    echo "error: $*" >&2
    return 1 2>/dev/null || exit 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _opifex_activate_die "use 'source ./activate.sh' so the environment stays active"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$PROJECT_ROOT/.venv/bin/activate"
MANAGED_ENV_FILE="${OPIFEX_MANAGED_ENV_FILE:-$PROJECT_ROOT/.opifex.env}"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    _opifex_activate_die "virtual environment not found; run ./setup.sh first"
fi

_opifex_reset_previous_managed_env() {
    local variable
    for variable in ${OPIFEX_MANAGED_ENV_VARS:-}; do
        unset "$variable"
    done
    unset OPIFEX_MANAGED_ENV_VARS
}

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

_opifex_reset_previous_managed_env

if [[ -f "$MANAGED_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$MANAGED_ENV_FILE"
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env"
fi

if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env.local"
fi

echo "Opifex environment active (${OPIFEX_BACKEND:-auto})."
echo "Use 'uv run python scripts/verify_opifex_gpu.py' to inspect the active JAX backend."
