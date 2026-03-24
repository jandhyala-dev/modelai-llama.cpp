#!/usr/bin/env bash
# Phase D orchestrator — runs all registered models through Phase D testing.
# Reads models.json, skips models not on disk, runs smallest-first.
#
# Usage:
#   ./scripts/run-phase-d-all.sh [--model NAME] [--dry-run]
#
# Options:
#   --model NAME   Run only this model (incremental testing)
#   --dry-run      Show what would run without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELAI_MODELS_DIR:?Set MODELAI_MODELS_DIR to your models directory}"
REGISTRY="${REPO_DIR}/bench-results/models.json"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="${REPO_DIR}/bench-results/phase-d-${TIMESTAMP}"

# Parse args
MODEL_FILTER=""
DRY_RUN=false
SERVER_URL=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --server-url) SERVER_URL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Validate registry
if [[ ! -f "$REGISTRY" ]]; then
  echo "ERROR: Model registry not found: $REGISTRY"
  exit 1
fi

# Validate Python + jq
command -v python3 >/dev/null || { echo "ERROR: python3 required"; exit 1; }
command -v jq >/dev/null || { echo "ERROR: jq required"; exit 1; }

# Read and validate registry
MODELS=$(jq -c '.models[]' "$REGISTRY")
TOTAL=$(echo "$MODELS" | wc -l | tr -d ' ')

# Schema validation (F2 from adversarial review)
# Check required fields exist (note: compaction_supported can be false, so check for null not falsy)
INVALID_ENTRIES=$(jq -r '.models[] | select(.name == null or .filename == null or .context_size == null or (.compaction_supported | type) != "boolean" or .use_cases == null) | .name // "unnamed"' "$REGISTRY" 2>/dev/null)
if [[ -n "$INVALID_ENTRIES" ]]; then
  echo "ERROR: Registry has entries with missing required fields:"
  echo "$INVALID_ENTRIES" | while read -r entry; do echo "  - $entry"; done
  echo "Required: name (string), filename (string), context_size (number), compaction_supported (boolean), use_cases (array)"
  exit 1
fi
echo "=========================================="
echo "Phase D Orchestrator"
echo "=========================================="
echo "Registry:   $REGISTRY ($TOTAL models)"
echo "Models dir: $MODELS_DIR"
echo "Output:     $OUT_DIR"
echo "Filter:     ${MODEL_FILTER:-all}"
echo "Server:     ${SERVER_URL:-managed (auto-start/stop)}"
echo "Dry run:    $DRY_RUN"
echo "=========================================="

# Copy use-cases.json to output dir
mkdir -p "$OUT_DIR"
USE_CASES=$(find "${REPO_DIR}/bench-results" -name "use-cases.json" -maxdepth 2 | head -1)
if [[ -n "$USE_CASES" ]]; then
  cp "$USE_CASES" "$OUT_DIR/use-cases.json"
  echo "Use cases:  $(jq '.use_cases | length' "$OUT_DIR/use-cases.json") use cases loaded"
else
  echo "ERROR: No use-cases.json found in bench-results/"
  exit 1
fi

# Track results
TESTED=0
SKIPPED=0
FAILED=0
ORDER=0

echo ""
while IFS= read -r model; do
  NAME=$(echo "$model" | jq -r '.name')
  FILENAME=$(echo "$model" | jq -r '.filename')
  CTX=$(echo "$model" | jq -r '.context_size')
  RATIOS=$(echo "$model" | jq -r '.compaction_ratios | join(",")')
  USE_CASES_LIST=$(echo "$model" | jq -r '.use_cases | join(",")')
  MODEL_PATH="${MODELS_DIR}/${FILENAME}"
  ORDER=$((ORDER + 1))

  # Filter
  if [[ -n "$MODEL_FILTER" && "$NAME" != "$MODEL_FILTER" ]]; then
    continue
  fi

  # Check model exists on disk
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[SKIP] $NAME — not on disk: $FILENAME"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
  echo "[RUN]  $NAME ($SIZE, ctx=${CTX}, ratios=${RATIOS:-none})"

  if $DRY_RUN; then
    echo "       [DRY RUN] Would run: python3 scripts/run-phase-d-test.py ..."
    continue
  fi

  # Server management
  if [[ -z "$SERVER_URL" ]]; then
    pkill -f llama-server 2>/dev/null || true
    sleep 2
  fi

  # Build command
  CMD=(python3 "${SCRIPT_DIR}/run-phase-d-test.py"
    --model-name "$NAME"
    --model-order "$ORDER"
    --context-size "$CTX"
    --use-cases "$USE_CASES_LIST"
    --out-dir "$OUT_DIR"
    --compaction-ratios "${RATIOS:-2}")

  if [[ -n "$SERVER_URL" ]]; then
    CMD+=(--server-url "$SERVER_URL")
  else
    CMD+=(--model-path "$MODEL_PATH")
  fi

  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/${NAME}.log"

  EXIT_CODE=${PIPESTATUS[0]}
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[FAIL] $NAME — exit code $EXIT_CODE"
    FAILED=$((FAILED + 1))
  else
    echo "[PASS] $NAME"
    TESTED=$((TESTED + 1))
  fi

  # Kill server between models (only if we manage it)
  if [[ -z "$SERVER_URL" ]]; then
    pkill -f llama-server 2>/dev/null || true
    sleep 3
  fi
done <<< "$MODELS"

echo ""
echo "=========================================="
echo "Phase D Complete"
echo "Tested: $TESTED / $TOTAL"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo "Results: $OUT_DIR"
echo "=========================================="
