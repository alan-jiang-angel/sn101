#!/usr/bin/env bash
set -euo pipefail

# Ubuntu runner for the Tag101 OpenRouter miner.
# Usage:
#   ./run_miner.sh                                  # run in foreground
#   ./run_miner.sh --pm2_name=tag101-miner          # run under pm2
#   ./run_miner.sh --wallet_name=abc --axon_port=10101
#
# Any argument not recognised below is forwarded verbatim to the miner.

# ---------------------------------------------------------------------------
# Resolve repo root FIRST so every relative path below is unambiguous.
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Defaults (env vars already set in the shell win, so this file stays reusable)
# ---------------------------------------------------------------------------

# OpenRouter credentials
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-your_openrouter_api_key}"
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
# NOTE: OpenRouter model IDs are namespaced -- "openai/gpt-4o-mini", not "gpt-4o-mini".
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-4o-mini}"

# Bittensor / Tag101 wallet values
export BT_WALLET_NAME="${BT_WALLET_NAME:-default}"
export BT_WALLET_HOTKEY="${BT_WALLET_HOTKEY:-default}"
export SUBTENSOR_NETWORK="${SUBTENSOR_NETWORK:-finney}"

# Optional task server URL if not using defaults
# export TASK_SERVER_URL="https://crawler.tag101.ai"

# Miner axon settings
export AXON_PORT="${AXON_PORT:-8091}"
export AXON_IP="${AXON_IP:-0.0.0.0}"
export AXON_EXTERNAL_IP="${AXON_EXTERNAL_IP:-}"
export AXON_EXTERNAL_PORT="${AXON_EXTERNAL_PORT:-}"

PM2_NAME=""

# ---------------------------------------------------------------------------
# Parse overrides from command-line arguments.
# ---------------------------------------------------------------------------
MINER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --wallet_name=*)         BT_WALLET_NAME="${arg#*=}" ;;
    --wallet_hotkey=*)       BT_WALLET_HOTKEY="${arg#*=}" ;;
    --subtensor_network=*)   SUBTENSOR_NETWORK="${arg#*=}" ;;
    --axon_port=*)           AXON_PORT="${arg#*=}" ;;
    --axon_ip=*)             AXON_IP="${arg#*=}" ;;
    --axon_external_ip=*)    AXON_EXTERNAL_IP="${arg#*=}" ;;
    --axon_external_port=*)  AXON_EXTERNAL_PORT="${arg#*=}" ;;
    --openrouter_api_key=*)  OPENROUTER_API_KEY="${arg#*=}" ;;
    --openrouter_base_url=*) OPENROUTER_BASE_URL="${arg#*=}" ;;
    --openrouter_model=*)    OPENROUTER_MODEL="${arg#*=}" ;;
    --pm2_name=*|--pm2-name=*) PM2_NAME="${arg#*=}" ;;
    *)                       MINER_ARGS+=("$arg") ;;
  esac
done

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/.venv/bin/activate"
fi

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: no python interpreter found (looked for .venv/bin/python, python3, python)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ "$OPENROUTER_API_KEY" = "your_openrouter_api_key" ] || [ -z "$OPENROUTER_API_KEY" ]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  echo "       Edit this script, export it in your shell, or pass --openrouter_api_key=sk-or-..." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build the python argument vector (NOTE: no leading "python" here -- pm2 and
# direct execution both need the interpreter kept separate from its arguments).
# -u keeps stdout unbuffered so pm2 logs appear immediately.
# ---------------------------------------------------------------------------
PY_ARGS=(
  -u
  -m tag101.miner
  --task.miner_module tag101.tasks.openrouter_miner
  --wallet.name "$BT_WALLET_NAME"
  --wallet.hotkey "$BT_WALLET_HOTKEY"
  --subtensor.network "$SUBTENSOR_NETWORK"
  --axon.port "$AXON_PORT"
  --axon.ip "$AXON_IP"
)

if [ -n "$AXON_EXTERNAL_IP" ]; then
  PY_ARGS+=(--axon.external_ip "$AXON_EXTERNAL_IP")
fi
if [ -n "$AXON_EXTERNAL_PORT" ]; then
  PY_ARGS+=(--axon.external_port "$AXON_EXTERNAL_PORT")
fi

# Safe expansion of a possibly-empty array under `set -u`.
if [ "${#MINER_ARGS[@]}" -gt 0 ]; then
  PY_ARGS+=("${MINER_ARGS[@]}")
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if [ -n "$PM2_NAME" ]; then
  if ! command -v pm2 >/dev/null 2>&1; then
    echo "ERROR: pm2 not found. Install it with: npm install -g pm2" >&2
    exit 1
  fi

  # Replace any existing process with the same name instead of erroring out.
  if pm2 describe "$PM2_NAME" >/dev/null 2>&1; then
    echo "Removing existing pm2 process '$PM2_NAME'..."
    pm2 delete "$PM2_NAME" >/dev/null
  fi

  # --interpreter none => pm2 execs the python binary directly and passes
  # everything after `--` as its argv. This is the fix for the double-`python` bug.
  pm2 start "$PYTHON_BIN" \
    --name "$PM2_NAME" \
    --cwd "$REPO_DIR" \
    --interpreter none \
    --update-env \
    --time \
    -- "${PY_ARGS[@]}"

  pm2 save >/dev/null 2>&1 || true
  echo
  echo "Started under pm2 as '$PM2_NAME'."
  echo "  logs:    pm2 logs $PM2_NAME"
  echo "  restart: pm2 restart $PM2_NAME --update-env"
  echo "  stop:    pm2 stop $PM2_NAME"
else
  exec "$PYTHON_BIN" "${PY_ARGS[@]}"
fi