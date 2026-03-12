#!/bin/bash
# Benchmark script for KV cache compaction workload tests (6b-13).
#
# Usage:
#   # Run all pipelines on a 14B model:
#   ./scripts/bench-kv-compact-workload.sh models/test/Qwen3-14B-Q4_K_M.gguf
#
#   # Smoke test:
#   ./scripts/bench-kv-compact-workload.sh models/test/stories15M-q4_0.gguf
#
#   # Custom context size:
#   N_CTX=8192 ./scripts/bench-kv-compact-workload.sh models/test/Qwen3-14B-Q4_K_M.gguf

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [ngl]}"
NGL="${2:-99}"
N_CTX="${N_CTX:-4096}"
BUILD_DIR="${BUILD_DIR:-build}"
ARTIFACT_DIR="${ARTIFACT_DIR:-bench-results}"

BIN="${BUILD_DIR}/bin/test-kv-compact-workload"

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first:"
    echo "  cmake --build $BUILD_DIR --target test-kv-compact-workload"
    exit 1
fi

if [ ! -f "$MODEL" ] && [ ! -L "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# Extract model name for file naming.
MODEL_NAME=$(basename "$MODEL" .gguf)

mkdir -p "$ARTIFACT_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COMBINED_CSV="${ARTIFACT_DIR}/${MODEL_NAME}-${TIMESTAMP}.csv"

echo "=== KV Compaction Benchmark ==="
echo "model:    $MODEL"
echo "ngl:      $NGL"
echo "n_ctx:    $N_CTX"
echo "output:   $COMBINED_CSV"
echo ""

SINGLE_CSV=""
cleanup() { [ -n "$SINGLE_CSV" ] && rm -f "$SINGLE_CSV"; }
trap cleanup EXIT INT TERM

FIRST=1
for PIPELINE in select solver omp; do
    echo "--- Pipeline: $PIPELINE ---"
    SINGLE_CSV=$(mktemp)

    PIPELINE="$PIPELINE" ARTIFACT="$SINGLE_CSV" \
        "$BIN" -m "$MODEL" -ngl "$NGL" -c "$N_CTX" 2>&1 \
        || echo "  (pipeline '$PIPELINE' exited with status $? — threshold miss, not a crash)"

    echo ""

    # Merge CSVs: write header only from first file.
    if [ "$FIRST" -eq 1 ]; then
        cat "$SINGLE_CSV" > "$COMBINED_CSV"
        FIRST=0
    else
        tail -n +2 "$SINGLE_CSV" >> "$COMBINED_CSV"
    fi
    rm -f "$SINGLE_CSV"
done

echo "=== Combined results ==="
column -t -s, "$COMBINED_CSV" 2>/dev/null || cat "$COMBINED_CSV"
echo ""
echo "Artifact saved: $COMBINED_CSV"
