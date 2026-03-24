#!/bin/bash
# Compare modelai-llama.cpp vs upstream llama.cpp tok/s performance
# Runs serially to avoid memory contention

MODELAI_BENCH="./build/bin/llama-bench"
UPSTREAM_BENCH="/tmp/llama-upstream-bench/build/bin/llama-bench"
MODEL_DIR="${MODELAI_MODELS_DIR:?Set MODELAI_MODELS_DIR to your models directory}"
RESULTS_DIR="bench-results/comparison-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Test matrix: representative models across architectures
# Using pp512 (prompt processing) and tg128 (token generation)
MODELS=(
    "${MODEL_DIR}/llama3.2-3b-q4_k_m.gguf:Llama3.2-3B"
    "${MODEL_DIR}/mistral-7b-q4_k_m.gguf:Mistral-7B"
    "${MODEL_DIR}/llama3.1-8b-q4_k_m.gguf:Llama3.1-8B"
    "${MODEL_DIR}/Qwen3-8B-Q4_K_M.gguf:Qwen3-8B"
    "${MODEL_DIR}/qwen2.5-7b-instruct-q4_k_m.gguf:Qwen2.5-7B"
    "${MODEL_DIR}/gemma2-9b-q4_k_m.gguf:Gemma2-9B"
)

echo "Performance Comparison: modelai-llama.cpp vs upstream llama.cpp" > "$RESULTS_DIR/comparison.txt"
echo "Date: $(date)" >> "$RESULTS_DIR/comparison.txt"
echo "Hardware: $(sysctl -n machdep.cpu.brand_string) / $(sysctl hw.memsize | awk '{printf "%.0fGB RAM", $2/1073741824}')" >> "$RESULTS_DIR/comparison.txt"
echo "" >> "$RESULTS_DIR/comparison.txt"

for entry in "${MODELS[@]}"; do
    model_path="${entry%%:*}"
    model_name="${entry##*:}"

    if [ ! -f "$model_path" ] && [ ! -L "$model_path" ]; then
        echo "SKIP $model_name: file not found"
        continue
    fi

    echo ""
    echo "=== $model_name ==="
    echo "--- modelai-llama.cpp ---"

    # Run modelai bench (pp512, tg128, 3 reps)
    modelai_result=$("$MODELAI_BENCH" -m "$model_path" -p 512 -n 128 -r 3 -o csv 2>/dev/null)
    echo "$modelai_result" > "$RESULTS_DIR/${model_name}-modelai.csv"

    modelai_pp=$(echo "$modelai_result" | grep "pp512" | tail -1 | awk -F',' '{print $(NF-1)}')
    modelai_tg=$(echo "$modelai_result" | grep "tg128" | tail -1 | awk -F',' '{print $(NF-1)}')
    echo "  modelai pp512: ${modelai_pp:-N/A} tok/s | tg128: ${modelai_tg:-N/A} tok/s"

    echo "--- upstream llama.cpp ---"
    # Run upstream bench (same params)
    upstream_result=$("$UPSTREAM_BENCH" -m "$model_path" -p 512 -n 128 -r 3 -o csv 2>/dev/null)
    echo "$upstream_result" > "$RESULTS_DIR/${model_name}-upstream.csv"

    upstream_pp=$(echo "$upstream_result" | grep "pp512" | tail -1 | awk -F',' '{print $(NF-1)}')
    upstream_tg=$(echo "$upstream_result" | grep "tg128" | tail -1 | awk -F',' '{print $(NF-1)}')
    echo "  upstream pp512: ${upstream_pp:-N/A} tok/s | tg128: ${upstream_tg:-N/A} tok/s"

    # Calculate delta
    if [ -n "$modelai_pp" ] && [ -n "$upstream_pp" ]; then
        pp_delta=$(python3 -c "m=$modelai_pp; u=$upstream_pp; print(f'{((m-u)/u)*100:+.1f}%')" 2>/dev/null)
    else
        pp_delta="N/A"
    fi
    if [ -n "$modelai_tg" ] && [ -n "$upstream_tg" ]; then
        tg_delta=$(python3 -c "m=$modelai_tg; u=$upstream_tg; print(f'{((m-u)/u)*100:+.1f}%')" 2>/dev/null)
    else
        tg_delta="N/A"
    fi

    echo "  Delta: pp512=${pp_delta} | tg128=${tg_delta}"

    echo "$model_name | modelai pp=$modelai_pp tg=$modelai_tg | upstream pp=$upstream_pp tg=$upstream_tg | delta pp=$pp_delta tg=$tg_delta" >> "$RESULTS_DIR/comparison.txt"

    # Memory settle between models
    sleep 3
done

echo ""
echo "Results saved to $RESULTS_DIR/"
cat "$RESULTS_DIR/comparison.txt"
