#!/bin/bash
# Benchmark with memory reporting — runs modelai and upstream serially
# Captures tok/s AND peak memory for each model

MODELAI_BENCH="./build/bin/llama-bench"
UPSTREAM_BENCH="/tmp/llama-upstream-bench/build/bin/llama-bench"
RESULTS_DIR="bench-results/comparison-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Models to test (path:display_name)
MODELS=(
    "models/test/llama3.2-3b-q4_k_m.gguf:Llama3.2-3B"
    "models/test/mistral-7b-q4_k_m.gguf:Mistral-7B"
    "models/test/llama3.1-8b-q4_k_m.gguf:Llama3.1-8B"
    "models/test/Qwen3-8B-Q4_K_M.gguf:Qwen3-8B"
    "models/test/qwen2.5-7b-instruct-q4_k_m.gguf:Qwen2.5-7B"
    "models/test/gemma2-9b-q4_k_m.gguf:Gemma2-9B"
    "models/test/deepseek-r1-8b-q4_k_m.gguf:DeepSeek-R1-8B"
    "models/test/granite3.1-dense-8b-q4_k_m.gguf:Granite3.1-8B"
    "models/test/tinyllama-1.1b-q4_k_m.gguf:TinyLlama-1.1B"
    "models/test/qwen2.5-coder-14b-q4_k_m.gguf:Qwen2.5-Coder-14B"
    "models/test/Qwen3-14B-Q4_K_M.gguf:Qwen3-14B"
    "models/test/qwen2.5-14b-instruct-q4_k_m.gguf:Qwen2.5-14B"
    "models/test/deepseek-r1-distill-qwen-14b-q4_k_m.gguf:DeepSeek-R1-14B"
    "models/test/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf:Qwen3-30B-A3B"
)

run_bench_with_memory() {
    local bench_bin="$1"
    local model_path="$2"
    local label="$3"

    # Run bench and capture output
    local output=$("$bench_bin" -m "$model_path" -p 512 -n 128 -r 3 2>&1)
    local pp=$(echo "$output" | grep "pp512" | awk -F'|' '{print $7}' | sed 's/[± ].*//' | xargs)
    local tg=$(echo "$output" | grep "tg128" | awk -F'|' '{print $7}' | sed 's/[± ].*//' | xargs)
    local size=$(echo "$output" | grep -v "^|" | head -1; echo "$output" | grep "pp512" | awk -F'|' '{print $3}' | xargs)

    # Get model file size as proxy for memory (actual allocation is model + KV cache)
    local file_size_mb=$(ls -l "$model_path" 2>/dev/null | awk '{printf "%.0f", $5/1048576}')

    echo "$label|$pp|$tg|${file_size_mb}MB"
}

echo "================================================================"
echo "Performance Comparison: modelai-llama.cpp vs upstream llama.cpp"
echo "Date: $(date)"
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")"
echo "RAM: $(sysctl hw.memsize | awk '{printf "%.0f GB", $2/1073741824}')"
echo "================================================================"
echo ""
printf "%-22s | %8s %8s %8s | %8s %8s %8s | %8s %8s\n" \
    "Model" "m-pp512" "m-tg128" "m-size" "u-pp512" "u-tg128" "u-size" "pp-delta" "tg-delta"
echo "-----------------------------------------------------------------------------------------------------------"

for entry in "${MODELS[@]}"; do
    model_path="${entry%%:*}"
    model_name="${entry##*:}"

    if [ ! -f "$model_path" ] && [ ! -L "$model_path" ]; then
        printf "%-22s | SKIP (file not found)\n" "$model_name"
        continue
    fi

    # Resolve symlink to check actual file
    real_path=$(readlink -f "$model_path" 2>/dev/null || readlink "$model_path" 2>/dev/null || echo "$model_path")
    if [ ! -f "$real_path" ]; then
        printf "%-22s | SKIP (broken symlink)\n" "$model_name"
        continue
    fi

    file_size=$(ls -lh "$real_path" | awk '{print $5}')

    echo "  Running $model_name ($file_size)..." >&2

    # Modelai benchmark
    m_output=$(./build/bin/llama-bench -m "$model_path" -p 512 -n 128 -r 3 2>&1)
    m_pp=$(echo "$m_output" | grep "pp512" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')
    m_tg=$(echo "$m_output" | grep "tg128" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')

    sleep 2

    # Upstream benchmark
    u_output=$(/tmp/llama-upstream-bench/build/bin/llama-bench -m "$model_path" -p 512 -n 128 -r 3 2>&1)
    u_pp=$(echo "$u_output" | grep "pp512" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')
    u_tg=$(echo "$u_output" | grep "tg128" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')

    # Calculate deltas
    pp_delta="N/A"
    tg_delta="N/A"
    if [ -n "$m_pp" ] && [ -n "$u_pp" ]; then
        pp_delta=$(python3 -c "print(f'{(($m_pp-$u_pp)/$u_pp)*100:+.1f}%')" 2>/dev/null || echo "N/A")
    fi
    if [ -n "$m_tg" ] && [ -n "$u_tg" ]; then
        tg_delta=$(python3 -c "print(f'{(($m_tg-$u_tg)/$u_tg)*100:+.1f}%')" 2>/dev/null || echo "N/A")
    fi

    printf "%-22s | %8s %8s %8s | %8s %8s %8s | %8s %8s\n" \
        "$model_name" "$m_pp" "$m_tg" "$file_size" "$u_pp" "$u_tg" "$file_size" "$pp_delta" "$tg_delta"

    sleep 3
done
