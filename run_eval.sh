#!/usr/bin/env bash
# Single evaluation run.
#
#   cp .env.example .env      # then fill in DEEPSEEK_API_KEY
#   ./run_eval.sh --ratio 0.1 --seed 42 --output result_seed42.json
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Missing .env: cp .env.example .env and fill in DEEPSEEK_API_KEY" >&2
  exit 1
fi

set -a
source .env
set +a

# A-Mem's RobustOpenAIController only reads OPENAI_API_KEY, but the openai SDK
# itself honours OPENAI_BASE_URL — so pointing the repo at DeepSeek needs no
# code change.
export OPENAI_API_KEY="${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY not set}"
export OPENAI_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com/v1}"

PY="${PYTHON:-python}"
echo "== endpoint: ${OPENAI_BASE_URL}  model: ${LLM_MODEL:-deepseek-chat} =="
exec "$PY" run_metered.py --model "${LLM_MODEL:-deepseek-chat}" "$@"
