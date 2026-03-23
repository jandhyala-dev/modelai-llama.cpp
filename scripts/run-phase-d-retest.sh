#!/usr/bin/env bash
set -euo pipefail

# Phase D Retest — rebuilt from HEAD (be2491cb7) with all 5 compaction fixes
# Date: 2026-03-23

MODELS_DIR="$HOME/dev/whippet/models"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="$HOME/dev/whippet/modelai-llama.cpp/bench-results/phase-d-retest-${TIMESTAMP}"

mkdir -p "$OUT_DIR/results"

# Copy use-cases from previous run
cp "$HOME/dev/whippet/modelai-llama.cpp/bench-results/phase-d-20260316-112022/use-cases.json" "$OUT_DIR/"

# Use cases: 4 representative ones (one per major category)
USE_CASES="D-1-1,D-2-1,D-3-1,D-5-1"
RATIOS="2,10"

# Write metadata
cat > "$OUT_DIR/metadata.json" << 'METAEOF'
{
  "schema_version": 1,
  "timestamp": "TIMESTAMP_PLACEHOLDER",
  "phase": "D-retest",
  "description": "Phase D retest after 5 compaction bug fixes (be2491cb7)",
  "engine_commit": "be2491cb7",
  "previous_run": "phase-d-20260316-112022",
  "previous_engine_commit": "8c316954",
  "fixes_included": [
    "4b6af8b02 — fix post-compaction crash when slot reuses stale prompt cache",
    "da87e6397 — fix ablation flags",
    "d3a5759ab — fix is_imrope not serialized in state save/restore",
    "b44d4cd1e — fix seq_cp not copying is_imrope + clear() reset",
    "ae691cb20 — V5 Sprint 1 safety + measurement foundation"
  ],
  "hardware": {
    "cpu": "Apple M2 Pro",
    "memory_gb": 32
  },
  "compaction_ratios": [2, 10],
  "use_cases": ["D-1-1", "D-2-1", "D-3-1", "D-5-1"]
}
METAEOF
# Fix timestamp
sed -i '' "s/TIMESTAMP_PLACEHOLDER/$(date -u +%Y-%m-%dT%H:%M:%SZ)/" "$OUT_DIR/metadata.json"

echo "============================================================"
echo "Phase D Retest — $(date)"
echo "Output: $OUT_DIR"
echo "============================================================"
echo ""

run_model() {
  local order=$1
  local name=$2
  local path=$3
  local ctx=$4

  echo ""
  echo ">>> [$order] $name @ ${ctx} ctx"
  python3 "$SCRIPTS_DIR/run-phase-d-test.py" \
    --model-path "$path" \
    --model-name "$name" \
    --model-order "$order" \
    --context-size "$ctx" \
    --use-cases "$USE_CASES" \
    --compaction-ratios "$RATIOS" \
    --out-dir "$OUT_DIR"
  echo "<<< [$order] $name @ ${ctx} DONE"
}

# Start with the two previously-failed models, then the rest
# Order matches severity: largest first (most likely to stress memory)

# 1. Qwen3.5-35B-A3B — previously HTTP 500
run_model 1 "Qwen3.5-35B-A3B" "$MODELS_DIR/Qwen3.5-35B-A3B-Q4_K_M.gguf" 32768

# 2. Qwen3-30B-A3B-Instruct — previously HTTP 400
run_model 2 "Qwen3-30B-A3B-Instruct" "$MODELS_DIR/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf" 32768

# 3. Qwen3-30B-A3B-Thinking
run_model 3 "Qwen3-30B-A3B-Thinking" "$MODELS_DIR/Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL.gguf" 32768

# 4. Qwen3-Coder-30B-A3B
run_model 4 "Qwen3-Coder-30B-A3B" "$MODELS_DIR/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf" 32768

# 5. Qwen3-14B — 32K and 64K
run_model 5 "Qwen3-14B" "$MODELS_DIR/Qwen3-14B-128K-UD-Q4_K_XL.gguf" 32768
run_model 5 "Qwen3-14B" "$MODELS_DIR/Qwen3-14B-128K-UD-Q4_K_XL.gguf" 65536

# 6. Qwen3-8B — 32K and 64K
run_model 6 "Qwen3-8B" "$MODELS_DIR/Qwen3-8B-128K-UD-Q4_K_XL.gguf" 32768
run_model 6 "Qwen3-8B" "$MODELS_DIR/Qwen3-8B-128K-UD-Q4_K_XL.gguf" 65536

# 7. Gemma-3-4B — 32K and 64K
run_model 7 "Gemma-3-4B" "$MODELS_DIR/gemma-3-4b-it-UD-Q4_K_XL.gguf" 32768
run_model 7 "Gemma-3-4B" "$MODELS_DIR/gemma-3-4b-it-UD-Q4_K_XL.gguf" 65536

echo ""
echo "============================================================"
echo "Phase D Retest COMPLETE — $(date)"
echo "Results: $OUT_DIR"
echo "============================================================"
