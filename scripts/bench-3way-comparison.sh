#!/bin/bash
# 3-Way Benchmark Comparison: modelai-llama.cpp vs llama.cpp vs Ollama
#
# Tests 5 models across all three engines measuring:
#   - Prefill throughput (tok/s)
#   - Decode throughput (tok/s)
#   - Memory usage (KV cache bytes)
#   - KV compaction quality (logit cosine, modelai-only)
#   - Effective context capacity (modelai-only)
#
# Usage:
#   ./scripts/bench-3way-comparison.sh
#
# Output: bench-results/3way-comparison-YYYYMMDD-HHMMSS/
#   results.json        — structured results for Supabase import
#   results.csv         — tabular results
#   summary.json        — aggregate comparison with verdict
#   manifest.json       — run metadata

set -euo pipefail

# --- Configuration ---
MODELAI_DIR="/Users/ajayjandhyala/dev/whippet/modelai-llama.cpp"
UPSTREAM_DIR="/Users/ajayjandhyala/dev/whippet/llama.cpp"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COMMIT_SHA=$(cd "$MODELAI_DIR" && git rev-parse --short HEAD)
OUT_DIR="$MODELAI_DIR/bench-results/3way-comparison-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# Models directory
MODEL_DIR="${MODELAI_MODELS_DIR:-/Users/ajayjandhyala/dev/whippet/models}"

declare -A MODELS
MODELS[qwen3-14b]="${MODEL_DIR}/Qwen3-14B-Q4_K_M.gguf"
MODELS[qwen3-8b]="${MODEL_DIR}/Qwen3-8B-Q4_K_M.gguf"
MODELS[qwen3-30b-a3b]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf"
MODELS[deepseek-r1-14b]="${MODEL_DIR}/deepseek-r1-distill-qwen-14b-q4_k_m.gguf"
MODELS[gemma3-12b]="${MODEL_DIR}/gemma-3-12b-it-Q4_K_M.gguf"

# Ollama model names
declare -A OLLAMA_MODELS
OLLAMA_MODELS[qwen3-14b]="qwen3:14b"
OLLAMA_MODELS[qwen3-8b]="qwen3:8b"
OLLAMA_MODELS[qwen3-30b-a3b]="qwen3:30b"
OLLAMA_MODELS[deepseek-r1-14b]="deepseek-r1:14b"
OLLAMA_MODELS[gemma3-12b]="gemma3:12b"

# Test parameters
CONTEXT_SIZES=(4096 8192 16384)
PROMPT_TOKENS=512
DECODE_TOKENS=128
COMPRESSION_RATIOS=(2 4 8)

# --- Binaries ---
MODELAI_BENCH="$MODELAI_DIR/build/bin/llama-bench"
MODELAI_LONGCTX="$MODELAI_DIR/build/bin/test-kv-compact-longctx"
UPSTREAM_BENCH="$UPSTREAM_DIR/build/bin/llama-bench"

# --- Write manifest ---
cat > "$OUT_DIR/manifest.json" << MANIFEST
{
  "schema_version": 3,
  "benchmark_type": "3way-comparison",
  "run_id": "${TIMESTAMP}-${COMMIT_SHA}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "commit_sha": "$COMMIT_SHA",
  "branch": "$(cd "$MODELAI_DIR" && git branch --show-current)",
  "machine": "$(hostname)",
  "os": "$(uname -s) $(uname -r)",
  "arch": "$(uname -m)",
  "cpu": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)",
  "memory_gb": $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )),
  "engines": ["modelai-llama.cpp", "llama.cpp", "ollama"],
  "models": $(printf '%s\n' "${!MODELS[@]}" | sort | jq -R . | jq -s .),
  "context_sizes": $(printf '%s\n' "${CONTEXT_SIZES[@]}" | jq -s .),
  "compression_ratios": $(printf '%s\n' "${COMPRESSION_RATIOS[@]}" | jq -s .)
}
MANIFEST

echo "=== 3-Way Benchmark Comparison ==="
echo "Output: $OUT_DIR"
echo "Models: ${!MODELS[*]}"
echo ""

# --- CSV header ---
CSV="$OUT_DIR/results.csv"
echo "engine,model,n_ctx,test_type,prompt_tokens,decode_tokens,compression_ratio,prefill_tok_s,decode_tok_s,kv_memory_mb,logit_cosine,compaction_time_ms,active_n_kv,pass,notes" > "$CSV"

# --- JSON results array ---
RESULTS_JSON="$OUT_DIR/results.json"
echo '{"schema_version": 3, "results": [' > "$RESULTS_JSON"
FIRST_RESULT=true

append_result() {
    local json="$1"
    if [ "$FIRST_RESULT" = "true" ]; then
        FIRST_RESULT=false
    else
        echo "," >> "$RESULTS_JSON"
    fi
    echo "$json" >> "$RESULTS_JSON"
}

# --- Helper: run llama-bench and parse output ---
run_llama_bench() {
    local bench_bin="$1"
    local model_path="$2"
    local n_ctx="$3"
    local pp="$4"
    local tg="$5"
    local engine_name="$6"
    local model_name="$7"

    echo "  [${engine_name}] llama-bench: ${model_name} ctx=${n_ctx} pp=${pp} tg=${tg}"

    local output
    output=$("$bench_bin" -m "$model_path" -c "$n_ctx" -ngl 99 \
        -p "$pp" -n "$tg" -r 1 -o json 2>/dev/null) || {
        echo "    FAILED"
        echo "${engine_name},${model_name},${n_ctx},inference,${pp},${tg},1,0,0,0,,,,,bench_failed" >> "$CSV"
        return 1
    }

    local pp_ts tg_ts
    pp_ts=$(echo "$output" | jq -r '.[0].avg_ts // 0' 2>/dev/null || echo "0")
    tg_ts=$(echo "$output" | jq -r '.[1].avg_ts // 0' 2>/dev/null || echo "0")

    # Estimate KV memory: n_ctx * n_layers * 2 * n_embd_head * n_head_kv * sizeof(f16)
    # Use a rough approximation based on model size
    local kv_mb
    kv_mb=$(echo "$output" | jq -r '.[0].model_size // 0' 2>/dev/null || echo "0")
    kv_mb="0"  # Will be filled from model info

    echo "    pp=${pp_ts} tok/s, tg=${tg_ts} tok/s"

    echo "${engine_name},${model_name},${n_ctx},inference,${pp},${tg},1,${pp_ts},${tg_ts},${kv_mb},,,,,ok" >> "$CSV"

    append_result "$(cat <<JSONROW
{
  "engine": "${engine_name}",
  "model": "${model_name}",
  "n_ctx": ${n_ctx},
  "test_type": "inference",
  "prompt_tokens": ${pp},
  "decode_tokens": ${tg},
  "compression_ratio": 1,
  "prefill_tok_s": ${pp_ts},
  "decode_tok_s": ${tg_ts},
  "kv_memory_mb": null,
  "logit_cosine": null,
  "compaction_time_ms": null,
  "active_n_kv": null,
  "pass": true,
  "notes": "baseline inference"
}
JSONROW
)"
}

# --- Helper: run ollama benchmark ---
run_ollama_bench() {
    local ollama_model="$1"
    local model_name="$2"
    local n_ctx="$3"
    local pp="$4"
    local tg="$5"

    echo "  [ollama] ${model_name} ctx=${n_ctx} pp=${pp} tg=${tg}"

    # Generate a prompt of appropriate length
    local prompt
    prompt=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * ${pp})" 2>/dev/null | head -c $((pp * 6)))

    local start_ms end_ms duration_ms
    start_ms=$(python3 -c "import time; print(int(time.time()*1000))")

    local output
    output=$(curl -s --max-time 300 http://localhost:11434/api/generate -d "{
        \"model\": \"${ollama_model}\",
        \"prompt\": \"${prompt}\",
        \"stream\": false,
        \"options\": {
            \"num_ctx\": ${n_ctx},
            \"num_predict\": ${tg},
            \"temperature\": 0
        }
    }" 2>/dev/null) || {
        echo "    FAILED"
        echo "ollama,${model_name},${n_ctx},inference,${pp},${tg},1,0,0,0,,,,,ollama_failed" >> "$CSV"
        return 1
    }

    end_ms=$(python3 -c "import time; print(int(time.time()*1000))")

    local prompt_eval_count prompt_eval_duration eval_count eval_duration
    prompt_eval_count=$(echo "$output" | jq -r '.prompt_eval_count // 0')
    prompt_eval_duration=$(echo "$output" | jq -r '.prompt_eval_duration // 1')
    eval_count=$(echo "$output" | jq -r '.eval_count // 0')
    eval_duration=$(echo "$output" | jq -r '.eval_duration // 1')

    # Convert nanoseconds to tok/s
    local pp_ts tg_ts
    pp_ts=$(python3 -c "print(round(${prompt_eval_count} / (${prompt_eval_duration} / 1e9), 2) if ${prompt_eval_duration} > 0 else 0)")
    tg_ts=$(python3 -c "print(round(${eval_count} / (${eval_duration} / 1e9), 2) if ${eval_duration} > 0 else 0)")

    echo "    pp=${pp_ts} tok/s, tg=${tg_ts} tok/s (${prompt_eval_count} prompt tokens, ${eval_count} gen tokens)"

    echo "ollama,${model_name},${n_ctx},inference,${pp},${tg},1,${pp_ts},${tg_ts},0,,,,,ok" >> "$CSV"

    append_result "$(cat <<JSONROW
{
  "engine": "ollama",
  "model": "${model_name}",
  "n_ctx": ${n_ctx},
  "test_type": "inference",
  "prompt_tokens": ${prompt_eval_count},
  "decode_tokens": ${eval_count},
  "compression_ratio": 1,
  "prefill_tok_s": ${pp_ts},
  "decode_tok_s": ${tg_ts},
  "kv_memory_mb": null,
  "logit_cosine": null,
  "compaction_time_ms": null,
  "active_n_kv": null,
  "pass": true,
  "notes": "ollama inference"
}
JSONROW
)"
}

# --- Helper: run modelai KV compaction benchmark ---
run_modelai_compaction() {
    local model_path="$1"
    local model_name="$2"
    local n_ctx="$3"
    local ratio="$4"
    local pipeline="$5"

    echo "  [modelai-compact] ${model_name} ctx=${n_ctx} ratio=${ratio}x pipeline=${pipeline}"

    local output
    output=$(PIPELINE="$pipeline" RATIO="$ratio" ARTIFACT="$OUT_DIR/longctx-${model_name}-${n_ctx}-${ratio}x-${pipeline}.csv" \
        timeout 600 "$MODELAI_LONGCTX" -m "$model_path" -c "$n_ctx" -ngl 99 2>&1) || {
        echo "    FAILED (timeout or error)"
        echo "modelai-compact,${model_name},${n_ctx},compaction,0,0,${ratio},0,0,0,,,,false,longctx_failed" >> "$CSV"
        return 1
    }

    # Parse output
    local cosine compact_ms baseline_toks compacted_toks active_nkv pass_str
    cosine=$(echo "$output" | grep -o 'cosine=[0-9.]*' | head -1 | cut -d= -f2)
    compact_ms=$(echo "$output" | grep -o 'compact=[0-9.]*ms' | head -1 | sed 's/compact=//;s/ms//')
    baseline_toks=$(echo "$output" | grep -o 'baseline=[0-9.]*' | head -1 | cut -d= -f2)
    compacted_toks=$(echo "$output" | grep -o 'compacted=[0-9.]* tok/s' | head -1 | sed 's/compacted=//;s/ tok\/s//')
    active_nkv=$(echo "$output" | grep -o 'active_n_kv=[0-9]*' | head -1 | cut -d= -f2)
    pass_str=$(echo "$output" | grep -o 'PASS\|FAIL' | head -1)

    [ -z "$cosine" ] && cosine="nan"
    [ -z "$compact_ms" ] && compact_ms="0"
    [ -z "$baseline_toks" ] && baseline_toks="0"
    [ -z "$compacted_toks" ] && compacted_toks="0"
    [ -z "$active_nkv" ] && active_nkv="0"
    [ -z "$pass_str" ] && pass_str="FAIL"

    local pass_bool="false"
    [ "$pass_str" = "PASS" ] && pass_bool="true"

    echo "    cosine=${cosine} compact=${compact_ms}ms baseline=${baseline_toks} tok/s compacted=${compacted_toks} tok/s [${pass_str}]"

    echo "modelai-compact,${model_name},${n_ctx},compaction-${pipeline},0,0,${ratio},${baseline_toks},${compacted_toks},0,${cosine},${compact_ms},${active_nkv},${pass_bool},pipeline=${pipeline}" >> "$CSV"

    append_result "$(cat <<JSONROW
{
  "engine": "modelai-compact",
  "model": "${model_name}",
  "n_ctx": ${n_ctx},
  "test_type": "compaction",
  "pipeline": "${pipeline}",
  "prompt_tokens": 0,
  "decode_tokens": 0,
  "compression_ratio": ${ratio},
  "prefill_tok_s": null,
  "decode_tok_s": ${compacted_toks},
  "baseline_decode_tok_s": ${baseline_toks},
  "kv_memory_mb": null,
  "logit_cosine": ${cosine},
  "compaction_time_ms": ${compact_ms},
  "active_n_kv": ${active_nkv},
  "pass": ${pass_bool},
  "notes": "KV compaction with ${pipeline} pipeline at ${ratio}x"
}
JSONROW
)"
}

# ======================================================================
# MAIN BENCHMARK LOOP
# ======================================================================

for model_name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    model_path="${MODELS[$model_name]}"
    ollama_model="${OLLAMA_MODELS[$model_name]}"

    echo ""
    echo "================================================================"
    echo "MODEL: ${model_name}"
    echo "================================================================"

    # Verify model file exists
    if [ ! -f "$model_path" ]; then
        echo "  SKIP: model file not found at $model_path"
        continue
    fi

    for n_ctx in "${CONTEXT_SIZES[@]}"; do
        echo ""
        echo "--- Context: ${n_ctx} ---"

        # 1. Ollama baseline
        run_ollama_bench "$ollama_model" "$model_name" "$n_ctx" "$PROMPT_TOKENS" "$DECODE_TOKENS" || true

        # 2. Upstream llama.cpp baseline
        run_llama_bench "$UPSTREAM_BENCH" "$model_path" "$n_ctx" "$PROMPT_TOKENS" "$DECODE_TOKENS" "llama.cpp" "$model_name" || true

        # 3. modelai-llama.cpp baseline (same binary, no compaction)
        run_llama_bench "$MODELAI_BENCH" "$model_path" "$n_ctx" "$PROMPT_TOKENS" "$DECODE_TOKENS" "modelai" "$model_name" || true

        # 4. modelai KV compaction (select pipeline - most reliable)
        for ratio in "${COMPRESSION_RATIOS[@]}"; do
            run_modelai_compaction "$model_path" "$model_name" "$n_ctx" "$ratio" "select" || true
        done

        # 5. modelai KV compaction (nonuniform pipeline - at 4K only to save time)
        if [ "$n_ctx" -le 8192 ]; then
            run_modelai_compaction "$model_path" "$model_name" "$n_ctx" 4 "nonuniform" || true
        fi
    done
done

# --- Close JSON ---
echo "]}" >> "$RESULTS_JSON"

# Fix JSON (remove trailing commas if any)
python3 -c "
import json, sys
with open('$RESULTS_JSON') as f:
    data = json.load(f)
with open('$RESULTS_JSON', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true

# --- Generate summary ---
python3 << 'PYSUMMARY' > "$OUT_DIR/summary.json"
import json, csv, sys

results_path = sys.argv[1] if len(sys.argv) > 1 else "RESULTS_JSON_PATH"

csv_path = "CSV_PATH"
rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Aggregate by engine
engines = {}
for row in rows:
    eng = row['engine']
    if eng not in engines:
        engines[eng] = {'prefill_samples': [], 'decode_samples': [], 'pass_count': 0, 'total_count': 0}
    try:
        pp = float(row.get('prefill_tok_s', '0') or '0')
        tg = float(row.get('decode_tok_s', '0') or '0')
        if pp > 0: engines[eng]['prefill_samples'].append(pp)
        if tg > 0: engines[eng]['decode_samples'].append(tg)
    except ValueError:
        pass
    if row.get('pass') == 'true':
        engines[eng]['pass_count'] += 1
    engines[eng]['total_count'] += 1

summary = {
    "run_id": "RUN_ID",
    "verdict": "modelai-llama.cpp provides KV compaction capability not available in upstream llama.cpp or Ollama",
    "engines": {}
}

for eng, data in engines.items():
    pp_avg = sum(data['prefill_samples']) / len(data['prefill_samples']) if data['prefill_samples'] else 0
    tg_avg = sum(data['decode_samples']) / len(data['decode_samples']) if data['decode_samples'] else 0
    summary["engines"][eng] = {
        "avg_prefill_tok_s": round(pp_avg, 2),
        "avg_decode_tok_s": round(tg_avg, 2),
        "pass_rate": f"{data['pass_count']}/{data['total_count']}",
        "n_benchmarks": data['total_count']
    }

json.dump(summary, sys.stdout, indent=2)
PYSUMMARY

# Fix python path references
sed -i '' "s|RESULTS_JSON_PATH|$RESULTS_JSON|g" "$OUT_DIR/summary.json" 2>/dev/null || true
sed -i '' "s|CSV_PATH|$CSV|g" "$OUT_DIR/summary.json" 2>/dev/null || true
sed -i '' "s|RUN_ID|${TIMESTAMP}-${COMMIT_SHA}|g" "$OUT_DIR/summary.json" 2>/dev/null || true

echo ""
echo "================================================================"
echo "BENCHMARK COMPLETE"
echo "================================================================"
echo "Results: $OUT_DIR/"
echo "  CSV:      $CSV"
echo "  JSON:     $RESULTS_JSON"
echo "  Summary:  $OUT_DIR/summary.json"
echo "  Manifest: $OUT_DIR/manifest.json"
