#!/usr/bin/env bash
set -euo pipefail

# Small Ubuntu runner for the Tag101 OpenRouter miner.
# Update the default values below before running.

# OpenRouter credentials
export OPENROUTER_API_KEY="your_openrouter_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="gpt-4o-mini"

# Bittensor / Tag101 wallet values
export BT_WALLET_NAME="default"
export BT_WALLET_HOTKEY="default"
export SUBTENSOR_NETWORK="finney"

# Optional task server URL if not using defaults
# export TASK_SERVER_URL="https://crawler.tag101.ai"

# Miner axon settings
export AXON_PORT="8091"
export AXON_IP="0.0.0.0"
# export AXON_EXTERNAL_IP="your_public_ip"
# export AXON_EXTERNAL_PORT="8091"
export PM2_NAME=""

# Parse overrides from command-line arguments.
# Supported forms: --wallet_name=abc --axon_port=10101 --pm2_name=myminer
# Any unsupported args are forwarded to the miner command.
MINER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --wallet_name=*)
      BT_WALLET_NAME="${arg#*=}" ;;
    --wallet_hotkey=*)
      BT_WALLET_HOTKEY="${arg#*=}" ;;
    --subtensor_network=*)
      SUBTENSOR_NETWORK="${arg#*=}" ;;
    --axon_port=*)
      AXON_PORT="${arg#*=}" ;;
    --axon_ip=*)
      AXON_IP="${arg#*=}" ;;
    --axon_external_ip=*)
      AXON_EXTERNAL_IP="${arg#*=}" ;;
    --axon_external_port=*)
      AXON_EXTERNAL_PORT="${arg#*=}" ;;
    --openrouter_api_key=*)
      OPENROUTER_API_KEY="${arg#*=}" ;;
    --openrouter_base_url=*)
      OPENROUTER_BASE_URL="${arg#*=}" ;;
    --openrouter_model=*)
      OPENROUTER_MODEL="${arg#*=}" ;;
    --pm2_name=*|--pm2-name=*)
      PM2_NAME="${arg#*=}" ;;
    *)
      MINER_ARGS+=("$arg") ;;
  esac
 done

# Python environment setup (assumes .venv exists in repo root)
if [ -x "./.venv/bin/activate" ]; then
  source "./.venv/bin/activate"
fi

cd "$(dirname "$0")"

CMD=(python -m tag101.miner
  --task.miner_module tag101.tasks.openrouter_miner
  --wallet.name "$BT_WALLET_NAME"
  --wallet.hotkey "$BT_WALLET_HOTKEY"
  --subtensor.network "$SUBTENSOR_NETWORK"
  --axon.port "$AXON_PORT"
  --axon.ip "$AXON_IP"
  "${MINER_ARGS[@]}"
)

if [ -n "$PM2_NAME" ]; then
  pm2 start python --name "$PM2_NAME" --cwd "$(pwd)" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi
