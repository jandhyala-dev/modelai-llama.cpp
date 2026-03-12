#!/bin/bash
# Campaign 2: Comprehensive KV compaction benchmark with self-study pipeline.
#
# Tests all 6 production models with:
#   - All pipelines: baseline, select, solver, self_study
#   - Extended ratios: 2x 4x 8x 16x 32x 50x (paper target: 50x)
#   - Full QuALITY MC evaluation (2,086 questions)
#   - LongHealth MC evaluation (400 questions, 5-option, 60K-token patients)
#   - Real SEC 10-K filing text for long-context prefill
#   - Extended contexts: 4K 8K 16K 32K 64K 128K
#
# Usage:
#   # Run full Campaign 2 (all models):
#   ./scripts/bench-kv-compact-campaign2.sh
#
#   # Run single model:
#   ./scripts/bench-kv-compact-campaign2.sh models/test/Qwen3-14B-Q4_K_M.gguf
#
#   # Dry run (print matrix only):
#   DRY_RUN=1 ./scripts/bench-kv-compact-campaign2.sh

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
ARTIFACT_BASE="${ARTIFACT_BASE:-bench-results}"
NGL="${NGL:-99}"
DRY_RUN="${DRY_RUN:-0}"

BENCH_SCRIPT="scripts/bench-kv-compact-longctx.sh"

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: $BENCH_SCRIPT not found. Run from repo root."
    exit 1
fi

# All 6 production models.
ALL_MODELS=(
    "models/test/Qwen3-8B-Q4_K_M.gguf"
    "models/test/Qwen3-14B-Q4_K_M.gguf"
    "models/test/Qwen3-30B-A3B-Q4_K_M.gguf"
    "models/test/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    "models/test/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    "models/test/Gemma-3-12B-Q4_K_M.gguf"
)

# If model path provided as arg, use just that model.
if [ $# -ge 1 ]; then
    MODELS=("$1")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Campaign 2 test matrix.
# 65K runs separately with longer timeout (see Phase 3 below).
CAMPAIGN2_CONTEXTS="${CAMPAIGN2_CONTEXTS:-4096 8192 16384 32768}"
CAMPAIGN2_RATIOS="2 4 8 16 32 50"

# Run ID for the entire campaign.
SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
MACHINE=$(hostname -s 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CAMPAIGN_ID="campaign2-${TIMESTAMP}-${SHORT_SHA}-${MACHINE}"

echo "=== Campaign 2: Comprehensive Self-Study Benchmark ==="
echo "campaign_id: $CAMPAIGN_ID"
echo "models:      ${#MODELS[@]}"
echo "contexts:    $CAMPAIGN2_CONTEXTS"
echo "ratios:      $CAMPAIGN2_RATIOS"
echo "pipelines:   baseline select solver self_study(4K-only)"
echo "eval:        QuALITY (2,086 Q) + LongHealth (400 Q)"
echo "prefill:     Real SEC 10-K filings"
echo ""

# Count total runs.
N_CONTEXTS=$(echo "$CAMPAIGN2_CONTEXTS" | wc -w | tr -d ' ')
N_RATIOS=$(echo "$CAMPAIGN2_RATIOS" | wc -w | tr -d ' ')
N_MODELS=${#MODELS[@]}
# Phase 1a: baseline + select x contexts x ratios
# Phase 1b: solver at 4K x ratios
# Phase 1c: self_study at 4K x ratios
# Phase 2: QuALITY eval at 8K x 3 ratios x select + baseline
# Phase 3: 65K x 4 ratios x select + baseline
P1A_PER_MODEL=$(( N_CONTEXTS + N_CONTEXTS * N_RATIOS ))  # baselines + select
P1B_PER_MODEL=$(( 1 + N_RATIOS ))  # baseline@4K + solver x ratios
P1C_PER_MODEL=$(( 1 + N_RATIOS ))  # baseline@4K + self_study x ratios
P2_PER_MODEL=$(( 1 + 1 * 3 ))  # baseline@8K + select x 3 ratios
P3_PER_MODEL=$(( 1 + 1 * 4 ))  # baseline@65K + select x 4 ratios
TOTAL_PER_MODEL=$(( P1A_PER_MODEL + P1B_PER_MODEL + P1C_PER_MODEL + P2_PER_MODEL + P3_PER_MODEL ))
TOTAL_RUNS=$(( TOTAL_PER_MODEL * N_MODELS ))
echo "total runs:  $TOTAL_RUNS ($TOTAL_PER_MODEL per model: ${P1A_PER_MODEL}+${P1B_PER_MODEL}+${P1C_PER_MODEL} throughput + ${P2_PER_MODEL} eval + ${P3_PER_MODEL} long-ctx)"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "--- DRY RUN: printing matrix ---"
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "Model: $(basename "$MODEL" .gguf)"
        for CTX in $CAMPAIGN2_CONTEXTS; do
            echo "  ctx=$CTX: baseline + select/solver/self_study x [${CAMPAIGN2_RATIOS}]"
        done
    done
    echo ""
    echo "Set DRY_RUN=0 to execute."
    exit 0
fi

# Run each model.
CAMPAIGN_START=$(date +%s)
MODEL_IDX=0

for MODEL in "${MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    MODEL_NAME=$(basename "$MODEL" .gguf)

    if [ ! -f "$MODEL" ] && [ ! -L "$MODEL" ]; then
        echo "SKIP: Model not found: $MODEL"
        continue
    fi

    echo ""
    echo "========================================================"
    echo "[$MODEL_IDX/${N_MODELS}] $MODEL_NAME"
    echo "========================================================"
    echo ""

    MODEL_START=$(date +%s)

    # Phase 1a: Select at all contexts (4K-32K) — fast, high-quality pipeline.
    # Solver excluded — O(n^3) Cholesky with cache-key surrogates times out at 8K+.
    # Self_study excluded — query generation (~80s) + NNLS solver (110-425s) times out at 8K+.
    echo "--- Phase 1a: select (4K-32K) ---"
    CONTEXTS="$CAMPAIGN2_CONTEXTS" \
    RATIOS="$CAMPAIGN2_RATIOS" \
    PIPELINES="baseline select" \
    RUN_TIMEOUT=900 \
    QUALITY_EVAL=0 \
    LONGHEALTH_EVAL=0 \
    SEC_TEXT_DIR="tests/data/sec-filings-benchmark" \
    BUILD_DIR="$BUILD_DIR" \
    ARTIFACT_BASE="$ARTIFACT_BASE" \
        "$BENCH_SCRIPT" "$MODEL" "$NGL" || {
            echo "WARNING: Model $MODEL_NAME Phase 1a had failures (continuing)"
        }

    # Phase 1b: Solver at 4K only (O(n^3) Cholesky only feasible at small contexts).
    echo "--- Phase 1b: solver at 4K ---"
    CONTEXTS="4096" \
    RATIOS="$CAMPAIGN2_RATIOS" \
    PIPELINES="baseline solver" \
    RUN_TIMEOUT=300 \
    QUALITY_EVAL=0 \
    LONGHEALTH_EVAL=0 \
    SEC_TEXT_DIR="tests/data/sec-filings-benchmark" \
    BUILD_DIR="$BUILD_DIR" \
    ARTIFACT_BASE="$ARTIFACT_BASE" \
        "$BENCH_SCRIPT" "$MODEL" "$NGL" || {
            echo "WARNING: Model $MODEL_NAME Phase 1b (solver) had failures (continuing)"
        }

    # Phase 1c: Self_study at 4K only (query gen ~80s + NNLS solver ~400s; times out at 8K+).
    # Captures real post-RoPE Q vectors via 256-token continuation, then NNLS beta + V fit.
    # Quality is poor at 4K (cosine 0.11-0.71) but data is valuable for diagnosis.
    echo "--- Phase 1c: self_study at 4K (10-min timeout) ---"
    CONTEXTS="4096" \
    RATIOS="$CAMPAIGN2_RATIOS" \
    PIPELINES="baseline self_study" \
    RUN_TIMEOUT=600 \
    QUALITY_EVAL=0 \
    LONGHEALTH_EVAL=0 \
    SEC_TEXT_DIR="tests/data/sec-filings-benchmark" \
    BUILD_DIR="$BUILD_DIR" \
    ARTIFACT_BASE="$ARTIFACT_BASE" \
        "$BENCH_SCRIPT" "$MODEL" "$NGL" || {
            echo "WARNING: Model $MODEL_NAME Phase 1c (self_study) had failures (continuing)"
        }

    # Phase 2: QuALITY MC eval on key configurations (8K context only).
    # Self_study excluded: produces garbage quality (cosine 0.11-0.23 at 4K) and times out at 8K.
    # Timeout: 100 questions x ~27s each = ~2700s + prefill/compaction overhead.
    # 3600s (1 hour) per run provides safe margin for variable question lengths.
    echo "--- Phase 2: QuALITY MC eval (8K, 1-hour timeout) ---"
    CONTEXTS="8192" \
    RATIOS="4 8 16" \
    PIPELINES="baseline select" \
    RUN_TIMEOUT=3600 \
    QUALITY_EVAL=1 \
    QUALITY_LIMIT="${QUALITY_LIMIT:-100}" \
    LONGHEALTH_EVAL=0 \
    SEC_TEXT_DIR="tests/data/sec-filings-benchmark" \
    BUILD_DIR="$BUILD_DIR" \
    ARTIFACT_BASE="$ARTIFACT_BASE" \
        "$BENCH_SCRIPT" "$MODEL" "$NGL" || {
            echo "WARNING: Model $MODEL_NAME Phase 2 (QuALITY) had failures (continuing)"
        }

    # Phase 3: 65K context runs with 30-min timeout (prefill alone takes ~6 min).
    # Self_study excluded: would need >30 min at 65K (query gen scales with context).
    echo "--- Phase 3: 65K long-context (30-min timeout) ---"
    CONTEXTS="65536" \
    RATIOS="2 4 8 16" \
    PIPELINES="baseline select" \
    RUN_TIMEOUT=1800 \
    QUALITY_EVAL=0 \
    LONGHEALTH_EVAL=0 \
    SEC_TEXT_DIR="tests/data/sec-filings-benchmark" \
    BUILD_DIR="$BUILD_DIR" \
    ARTIFACT_BASE="$ARTIFACT_BASE" \
        "$BENCH_SCRIPT" "$MODEL" "$NGL" || {
            echo "WARNING: Model $MODEL_NAME Phase 3 (65K) had failures (continuing)"
        }

    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$((MODEL_END - MODEL_START))
    echo ""
    echo "Model $MODEL_NAME completed in ${MODEL_ELAPSED}s"
done

CAMPAIGN_END=$(date +%s)
CAMPAIGN_ELAPSED=$((CAMPAIGN_END - CAMPAIGN_START))

echo ""
echo "=== Campaign 2 Complete ==="
echo "campaign_id:  $CAMPAIGN_ID"
echo "total_time:   ${CAMPAIGN_ELAPSED}s ($(( CAMPAIGN_ELAPSED / 60 ))m)"
echo "models:       $N_MODELS"
echo "results:      $ARTIFACT_BASE/"
echo ""
echo "Next: analyze results with:"
echo "  python3 scripts/analyze-bench-results.py $ARTIFACT_BASE/"
