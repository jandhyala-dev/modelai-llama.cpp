#!/bin/bash
# Long-context benchmark driver for KV cache compaction (6b-15 / 6b-15b).
#
# Iterates: context sizes x compression ratios x pipelines.
# Produces a self-describing artifact directory with manifest.json + results.csv.
#
# Usage:
#   # Full benchmark on one model:
#   ./scripts/bench-kv-compact-longctx.sh models/test/Qwen3-14B-Q4_K_M.gguf
#
#   # With QuALITY evaluation (4K-8K contexts):
#   QUALITY_EVAL=1 ./scripts/bench-kv-compact-longctx.sh models/test/Qwen3-14B-Q4_K_M.gguf
#
#   # Include self_study pipeline (6b-15b):
#   SELF_STUDY=1 ./scripts/bench-kv-compact-longctx.sh models/test/Qwen3-14B-Q4_K_M.gguf
#
#   # Smoke test (small model, quick):
#   ./scripts/bench-kv-compact-longctx.sh models/test/stories15M-q4_0.gguf

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [ngl]}"
NGL="${2:-99}"
BUILD_DIR="${BUILD_DIR:-build}"
ARTIFACT_BASE="${ARTIFACT_BASE:-bench-results}"
QUALITY_EVAL="${QUALITY_EVAL:-0}"
SELF_STUDY="${SELF_STUDY:-0}"

BIN="${BUILD_DIR}/bin/test-kv-compact-longctx"

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first:"
    echo "  cmake --build $BUILD_DIR --target test-kv-compact-longctx"
    exit 1
fi

if [ ! -f "$MODEL" ] && [ ! -L "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# Model name for artifacts.
MODEL_NAME=$(basename "$MODEL" .gguf)

# Run ID: YYYYMMDD-HHMMSS-<shortsha>-<machine>
SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
MACHINE=$(hostname -s 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_ID="${TIMESTAMP}-${SHORT_SHA}-${MACHINE}"

# Artifact directory.
RUN_DIR="${ARTIFACT_BASE}/${RUN_ID}"
mkdir -p "$RUN_DIR"
RESULTS_CSV="${RUN_DIR}/results.csv"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
ENV_TXT="${RUN_DIR}/env.txt"

# Capture environment snapshot.
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model: $MODEL"
    echo "model_name: $MODEL_NAME"
    echo "ngl: $NGL"
    echo "run_id: $RUN_ID"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "os: $(uname -s) $(uname -r)"
    echo "arch: $(uname -m)"
    echo "hostname: $(hostname 2>/dev/null || echo unknown)"
    echo "quality_eval: $QUALITY_EVAL"
    echo "self_study: $SELF_STUDY"
} > "$ENV_TXT"

echo "=== KV Compaction Long-Context Benchmark ==="
echo "model:     $MODEL_NAME"
echo "run_id:    $RUN_ID"
echo "output:    $RUN_DIR/"
echo ""

# Context sizes and ratios.
CONTEXTS="${CONTEXTS:-4096 8192 16384 32768}"
RATIOS="${RATIOS:-2 4 8}"

# Pipelines: baseline + compacted.
PIPELINES="baseline select solver"
# OMP: W1 smoke only — add if model is small enough.
MODEL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL" 2>/dev/null || echo 0)
if [ "$MODEL_SIZE" -lt 2000000000 ]; then
    PIPELINES="$PIPELINES omp"
fi
# 6b-15b: self_study if enabled.
if [ "$SELF_STUDY" = "1" ]; then
    PIPELINES="$PIPELINES self_study"
fi

N_PASS=0
N_FAIL=0
N_CRASH=0
N_SKIP=0

run_one() {
    local pipeline="$1"
    local ctx="$2"
    local ratio="$3"
    local quality_flag="$4"

    echo "--- $pipeline | ctx=$ctx | ratio=${ratio}x ---"

    PIPELINE="$pipeline" RATIO="$ratio" RUN_ID="$RUN_ID" \
        ARTIFACT="$RESULTS_CSV" QUALITY_EVAL="$quality_flag" \
        QUALITY_DATA="tests/data/quality-validation.jsonl" \
        "$BIN" -m "$MODEL" -ngl "$NGL" -c "$ctx" 2>>"$STDERR_LOG" \
        | tee -a "$STDOUT_LOG" || {
            local rc=$?
            if [ $rc -eq 139 ] || [ $rc -eq 137 ] || [ $rc -eq 134 ]; then
                echo "  CRASH (signal $rc)"
                N_CRASH=$((N_CRASH + 1))
            else
                echo "  FAIL (exit $rc — threshold miss or error)"
                N_FAIL=$((N_FAIL + 1))
            fi
            return 0  # don't abort the script
        }
    N_PASS=$((N_PASS + 1))
    echo ""
}

# Run baseline for each context size (no ratio variation).
for CTX in $CONTEXTS; do
    run_one "baseline" "$CTX" "1" "$QUALITY_EVAL"
done

# Run compacted pipelines for each context x ratio.
for PIPELINE in $PIPELINES; do
    [ "$PIPELINE" = "baseline" ] && continue
    for CTX in $CONTEXTS; do
        # QuALITY eval only at 4K-8K (articles fit in context).
        local_quality="0"
        if [ "$QUALITY_EVAL" = "1" ] && [ "$CTX" -le 8192 ]; then
            local_quality="1"
        fi
        for RATIO in $RATIOS; do
            run_one "$PIPELINE" "$CTX" "$RATIO" "$local_quality"
        done
    done
done

# Write manifest.json.
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
UPSTREAM_BASE=$(git merge-base upstream-master HEAD 2>/dev/null || echo "unknown")
OS_NAME=$(uname -s)
HW_DESC="${MACHINE}"

cat > "${RUN_DIR}/manifest.json" <<MANIFEST
{
  "schema_version": 1,
  "run_id": "${RUN_ID}",
  "repo": "modelai-llama.cpp",
  "branch": "${BRANCH}",
  "commit_sha": "${COMMIT}",
  "upstream_base_sha": "${UPSTREAM_BASE}",
  "hardware": "${HW_DESC}",
  "os": "${OS_NAME}",
  "backend_build": "metal-release",
  "models": ["${MODEL_NAME}"],
  "pipelines": [$(echo "$PIPELINES" | sed 's/ /", "/g' | sed 's/^/"/' | sed 's/$/"/')],
  "contexts": [$(echo "$CONTEXTS" | sed 's/ /, /g')],
  "ratios": [$(echo "$RATIOS" | sed 's/ /, /g')],
  "workloads": ["W1", "W2", "W3"],
  "artifact_files": {
    "results_csv": "results.csv",
    "stdout_log": "stdout.log",
    "stderr_log": "stderr.log",
    "env_txt": "env.txt"
  }
}
MANIFEST

# Summary.
TOTAL=$((N_PASS + N_FAIL + N_CRASH))
echo ""
echo "=== Summary ==="
echo "model:   $MODEL_NAME"
echo "run_id:  $RUN_ID"
echo "total:   $TOTAL runs"
echo "  pass:  $N_PASS"
echo "  fail:  $N_FAIL (threshold miss)"
echo "  crash: $N_CRASH"
echo ""

# Display results table if available.
if [ -f "$RESULTS_CSV" ]; then
    echo "--- Results CSV (first 5 cols + key metrics) ---"
    head -1 "$RESULTS_CSV"
    echo "..."
    ROWS=$(wc -l < "$RESULTS_CSV")
    echo "($((ROWS - 1)) data rows)"
fi

echo ""
echo "Artifacts saved: $RUN_DIR/"
echo "  manifest.json"
echo "  results.csv"
echo "  stdout.log"
echo "  stderr.log"
echo "  env.txt"
