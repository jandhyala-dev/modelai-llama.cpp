#!/bin/bash
# Full benchmark suite: modelai-llama.cpp vs upstream llama.cpp
# Runs sequentially to avoid memory contention on Apple Silicon
#
# Metrics collected:
#   1. Inference speed (llama-bench): pp512 tok/s, tg128 tok/s
#   2. KV compaction quality (test-kv-compact-workload): cosine similarity at 2x-50x
#   3. Compaction timing, baseline vs compacted tok/s
#
# Output: bench-results/full-YYYYMMDD-HHMMSS/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODELAI_BENCH="./build/bin/llama-bench"
UPSTREAM_BENCH="/tmp/llama-upstream/build/bin/llama-bench"
WORKLOAD_TEST="./build/bin/test-kv-compact-workload"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="bench-results/full-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

BLOB_DIR="$HOME/.ollama/models/blobs"
MANIFEST_DIR="$HOME/.ollama/models/manifests/registry.ollama.ai/library"

# Hardware info
HW_CPU=$(sysctl -n machdep.cpu.brand_string)
HW_MEM=$(sysctl hw.memsize | awk '{printf "%.0fGB", $2/1073741824}')
HW_GPU="Apple Metal (unified)"
GIT_SHA=$(git rev-parse --short HEAD)

echo "============================================="
echo "ModelAI Full Benchmark Suite"
echo "Date: $(date)"
echo "Hardware: $HW_CPU / $HW_MEM RAM"
echo "Commit: $GIT_SHA"
echo "Results: $RESULTS_DIR"
echo "============================================="

# Write metadata
cat > "$RESULTS_DIR/metadata.json" <<METAEOF
{
  "timestamp": "$TIMESTAMP",
  "hardware": {
    "cpu": "$HW_CPU",
    "memory": "$HW_MEM",
    "gpu": "$HW_GPU"
  },
  "modelai_commit": "$GIT_SHA",
  "upstream_commit": "$(cd /tmp/llama-upstream && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
}
METAEOF

# CSV headers for master results
SPEED_CSV="$RESULTS_DIR/speed-comparison.csv"
echo "model,size_gb,modelai_pp512_tok_s,modelai_tg128_tok_s,upstream_pp512_tok_s,upstream_tg128_tok_s,pp_delta_pct,tg_delta_pct" > "$SPEED_CSV"

COMPACT_CSV="$RESULTS_DIR/compaction-quality.csv"
echo "model,size_gb,pipeline,ratio,cosine_similarity,threshold,pass_fail,compaction_time_ms,baseline_tok_s,compacted_tok_s,prefix_tokens" > "$COMPACT_CSV"

# Helper: resolve ollama manifest to GGUF blob path
resolve_model() {
    local model_name="$1"
    local tag="$2"
    local manifest="$MANIFEST_DIR/$model_name/$tag"
    if [ ! -f "$manifest" ]; then
        echo ""
        return
    fi
    python3 -c "
import json, sys
with open('$manifest') as f:
    m = json.load(f)
for layer in (m.get('layers') or []):
    if layer.get('mediaType') == 'application/vnd.ollama.image.model':
        print('$BLOB_DIR/' + layer['digest'].replace('sha256:', 'sha256-'))
        sys.exit(0)
print('')
" 2>/dev/null
}

# Helper: get model file size in GB
model_size_gb() {
    python3 -c "import os; print(f'{os.path.getsize(\"$1\")/1073741824:.1f}')"
}

# Helper: run llama-bench and extract tok/s
run_bench() {
    local bench_bin="$1"
    local model_path="$2"
    local label="$3"
    local output_file="$4"

    echo "  Running $label bench..." >&2
    if "$bench_bin" -m "$model_path" -p 512 -n 128 -r 2 -o csv > "$output_file" 2>/dev/null; then
        local pp_tok_s tg_tok_s
        # CSV has n_prompt,n_gen,n_depth columns: PP row has "512","0", TG row has "0","128"
        pp_tok_s=$(grep '"512","0","0"' "$output_file" | tail -1 | awk -F',' '{print $(NF-1)}' | tr -d '"')
        tg_tok_s=$(grep '"0","128","0"' "$output_file" | tail -1 | awk -F',' '{print $(NF-1)}' | tr -d '"')
        echo "$pp_tok_s,$tg_tok_s"
    else
        echo ","
    fi
}

# Helper: parse workload test output and append to COMPACT_CSV
parse_workload_output() {
    local raw_file="$1"
    local model_name="$2"
    local size_gb="$3"

    # Extract prefix_tokens from "prefix_tokens: NNNN" line
    local prefix
    prefix=$(grep -m1 "prefix_tokens:" "$raw_file" | awk '{print $2}')

    # Parse summary lines like:
    #   2x: cos=0.9993 (>= 0.95) PASS | compact=70.0ms | decode=89.1->50.8 tok/s | n_kv=768
    grep -E '^\s+[0-9]+x:' "$raw_file" | while read -r line; do
        local ratio cosine threshold pf compact_ms baseline_tok compacted_tok
        ratio=$(echo "$line" | grep -oE '[0-9]+x' | head -1 | tr -d 'x')
        cosine=$(echo "$line" | grep -oE 'cos=[0-9.]+' | cut -d= -f2)
        threshold=$(echo "$line" | grep -oE '>= [0-9.]+' | awk '{print $2}')
        pf=$(echo "$line" | grep -oE '(PASS|FAIL)')
        compact_ms=$(echo "$line" | grep -oE 'compact=[0-9.]+ms' | grep -oE '[0-9.]+')
        baseline_tok=$(echo "$line" | grep -oE 'decode=[0-9.]+' | cut -d= -f2)
        compacted_tok=$(echo "$line" | grep -oE '>[0-9.]+' | head -1 | tr -d '>')

        echo "$model_name,$size_gb,select,$ratio,$cosine,$threshold,$pf,$compact_ms,$baseline_tok,$compacted_tok,$prefix" >> "$COMPACT_CSV"
    done
}

# ====================================================================
# Model list: text-only models compatible with KV compaction
# Excludes: vision/multimodal, embedding, iSWA (gemma3), duplicates
# ====================================================================
declare -a MODELS=(
    # "display_name|ollama_model|ollama_tag"
    "TinyLlama-1.1B|tinyllama|1.1b"
    "Llama3.2-3B|llama3.2|3b"
    "CodeGemma-7B|codegemma|7b"
    "Mistral-7B|mistral|7b"
    "Llama3.1-8B|llama3.1|8b"
    "Granite3.1-Dense-8B|granite3.1-dense|8b"
    "Aya-8B|aya|8b"
    "Qwen2.5-7B|qwen2.5|7b-instruct"
    "Qwen3-8B|qwen3|8b"
    "DeepSeek-R1-8B|deepseek-r1|8b"
    "Gemma2-9B|gemma2|9b"
    "Phi4-14B|phi4|latest"
    "Qwen2.5-14B|qwen2.5|14b"
    "Qwen2.5-Coder-14B|qwen2.5-coder|14b"
    "Qwen3-14B|qwen3|14b"
    "DeepSeek-R1-14B|deepseek-r1|14b"
    "Qwen3-30B-A3B|qwen3|30b"
)

TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=0
SKIPPED=0

echo ""
echo "Models to benchmark: $TOTAL"
echo ""

for entry in "${MODELS[@]}"; do
    IFS='|' read -r display_name ollama_model ollama_tag <<< "$entry"

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "===== [$CURRENT/$TOTAL] $display_name ($ollama_model:$ollama_tag) ====="

    # Resolve GGUF path
    model_path=$(resolve_model "$ollama_model" "$ollama_tag")
    if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
        echo "  SKIP: model file not found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    size_gb=$(model_size_gb "$model_path")
    echo "  Path: ...$(echo "$model_path" | tail -c 20)"
    echo "  Size: ${size_gb} GB"

    # Create model results directory
    model_dir="$RESULTS_DIR/$display_name"
    mkdir -p "$model_dir"

    # 1. Speed benchmark: modelai-llama.cpp
    modelai_result=$(run_bench "$MODELAI_BENCH" "$model_path" "modelai" "$model_dir/bench-modelai.csv")
    modelai_pp=$(echo "$modelai_result" | cut -d',' -f1)
    modelai_tg=$(echo "$modelai_result" | cut -d',' -f2)
    echo "  modelai: pp512=${modelai_pp:-N/A} tok/s | tg128=${modelai_tg:-N/A} tok/s"

    # 2. Speed benchmark: upstream llama.cpp
    upstream_result=$(run_bench "$UPSTREAM_BENCH" "$model_path" "upstream" "$model_dir/bench-upstream.csv")
    upstream_pp=$(echo "$upstream_result" | cut -d',' -f1)
    upstream_tg=$(echo "$upstream_result" | cut -d',' -f2)
    echo "  upstream: pp512=${upstream_pp:-N/A} tok/s | tg128=${upstream_tg:-N/A} tok/s"

    # Calculate delta
    pp_delta=""
    tg_delta=""
    if [ -n "$modelai_pp" ] && [ -n "$upstream_pp" ]; then
        pp_delta=$(python3 -c "m=$modelai_pp; u=$upstream_pp; print(f'{((m-u)/u)*100:.1f}')" 2>/dev/null || echo "")
    fi
    if [ -n "$modelai_tg" ] && [ -n "$upstream_tg" ]; then
        tg_delta=$(python3 -c "m=$modelai_tg; u=$upstream_tg; print(f'{((m-u)/u)*100:.1f}')" 2>/dev/null || echo "")
    fi

    echo "$display_name,$size_gb,$modelai_pp,$modelai_tg,$upstream_pp,$upstream_tg,$pp_delta,$tg_delta" >> "$SPEED_CSV"

    # 3. Compaction quality workload test
    echo "  Running compaction workload test..."
    raw_output="$model_dir/workload-raw.txt"
    if "$WORKLOAD_TEST" -m "$model_path" -c 4096 -n 16 > "$raw_output" 2>&1; then
        parse_workload_output "$raw_output" "$display_name" "$size_gb"
        # Show summary
        grep -E '^\s+[0-9]+x' "$raw_output" | head -6
    else
        echo "  COMPACTION FAILED (see $raw_output)"
        FAILED=$((FAILED + 1))
    fi

    # Let GPU memory settle
    sleep 2
    echo "  Done."
done

echo ""
echo "============================================="
echo "Benchmark Complete"
echo "Date: $(date)"
echo "Models: $TOTAL total, $((TOTAL - SKIPPED - FAILED)) passed, $FAILED failed, $SKIPPED skipped"
echo "Results: $RESULTS_DIR/"
echo "============================================="

# Print summary tables
echo ""
echo "=== Speed Comparison ==="
column -t -s',' "$SPEED_CSV" 2>/dev/null || cat "$SPEED_CSV"

echo ""
echo "=== Compaction Quality ==="
column -t -s',' "$COMPACT_CSV" 2>/dev/null || cat "$COMPACT_CSV"

echo ""
echo "Results directory: $RESULTS_DIR"
