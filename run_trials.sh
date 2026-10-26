#!/usr/bin/env bash
# Run N trials at different seeds and aggregate.
#
#   ./run_trials.sh 3 0.1          # 3 trials, one conversation
#   ./run_trials.sh 3 1.0 legacy   # 3 trials, full benchmark, paper's c5 reference
#
# The memory cache is keyed by backend+model, NOT by seed, so trial 1 builds the
# memories and trials 2..N only re-run the QA phase (~400 calls instead of ~1,700).
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-3}
RATIO=${2:-0.1}
TAG=${3:-new}

if [ ! -f .env ]; then
  echo "Missing .env: cp .env.example .env and fill in DEEPSEEK_API_KEY" >&2
  exit 1
fi
set -a; source .env; set +a
export OPENAI_API_KEY="${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY not set}"
export OPENAI_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com/v1}"

PY="${PYTHON:-python}"
EXTRA=""
[ "$TAG" = "legacy" ] && EXTRA="--legacy_c5_reference"

FILES=()
for i in $(seq 1 "$N"); do
  SEED=$((42 + i - 1))
  OUT="result_${TAG}_seed${SEED}.json"
  echo ""
  echo "===== trial $i/$N  seed=$SEED  ratio=$RATIO  $TAG ====="
  "$PY" run_metered.py --ratio "$RATIO" --backend openai \
      --model "${LLM_MODEL:-deepseek-chat}" --retrieve_k 10 \
      --seed "$SEED" --output "$OUT" $EXTRA
  FILES+=("$OUT")
done

echo ""
echo "===== aggregate ====="
"$PY" aggregate_trials.py "${FILES[@]}"
