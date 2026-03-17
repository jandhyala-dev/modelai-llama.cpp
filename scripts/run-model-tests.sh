#!/bin/bash
# Serial model integration test runner — one model at a time to avoid memory pressure
# Usage: ./scripts/run-model-tests.sh [model1.gguf model2.gguf ...]

set -e

TEST_BIN="./build/bin/test-kv-compact-pipeline-integration"
RESULTS_DIR="bench-results/model-tests-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [ $# -eq 0 ]; then
    # Default: test all models in models/test/ (skip stories15M which is CI fixture)
    MODELS=(
        models/test/mistral-7b-q4_k_m.gguf
        models/test/phi4-14b-q4_k_m.gguf
        models/test/llama3.1-8b-q4_k_m.gguf
        models/test/llama3.2-3b-q4_k_m.gguf
        models/test/gemma2-9b-q4_k_m.gguf
        models/test/granite3.1-dense-8b-q4_k_m.gguf
        models/test/deepseek-r1-8b-q4_k_m.gguf
        models/test/qwen2.5-coder-14b-q4_k_m.gguf
        models/test/tinyllama-1.1b-q4_k_m.gguf
    )
else
    MODELS=("$@")
fi

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Model Integration Test Results — $(date)" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

total_pass=0
total_fail=0
total_skip=0

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model" .gguf)
    echo ""
    echo "================================================================"
    echo "Testing: $model_name"
    echo "================================================================"

    # Check model file exists
    if [ ! -f "$model" ] && [ ! -L "$model" ]; then
        echo "  SKIP: model file not found"
        echo "$model_name: SKIP (file not found)" >> "$SUMMARY_FILE"
        ((total_skip++))
        continue
    fi

    # Check symlink target exists
    if [ -L "$model" ]; then
        target=$(readlink "$model")
        if [ ! -f "$target" ]; then
            echo "  SKIP: symlink target not found: $target"
            echo "$model_name: SKIP (symlink broken)" >> "$SUMMARY_FILE"
            ((total_skip++))
            continue
        fi
    fi

    output_file="$RESULTS_DIR/${model_name}.txt"

    # Run integration test with timeout (5 minutes per model)
    echo "  Running integration tests..."
    if timeout 300 "$TEST_BIN" "$model" > "$output_file" 2>&1; then
        # Extract results from output
        passed=$(grep -c "PASS" "$output_file" 2>/dev/null || echo "0")
        failed=$(grep -c "FAIL" "$output_file" 2>/dev/null || echo "0")
        select_cos=$(grep -oP "select 2x cosine = \K[0-9.]+" "$output_file" 2>/dev/null || echo "N/A")
        solver_cos=$(grep -oP "solver 2x cosine = \K[0-9.]+" "$output_file" 2>/dev/null || echo "N/A")

        # Get test summary line
        summary_line=$(tail -5 "$output_file" | grep -E "passed|failed" | head -1)

        echo "  Result: $summary_line"
        echo "  Select cosine: $select_cos | Solver cosine: $solver_cos"
        echo "$model_name: $summary_line | select=$select_cos solver=$solver_cos" >> "$SUMMARY_FILE"
        ((total_pass++))
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "  TIMEOUT (5 min limit exceeded)"
            echo "$model_name: TIMEOUT" >> "$SUMMARY_FILE"
        else
            # Extract last few lines for error context
            tail -5 "$output_file" 2>/dev/null
            echo "  FAILED (exit code $exit_code)"
            echo "$model_name: FAILED (exit $exit_code)" >> "$SUMMARY_FILE"
        fi
        ((total_fail++))
    fi

    # Brief pause to let memory settle
    sleep 2
done

echo "" >> "$SUMMARY_FILE"
echo "Total: $total_pass passed, $total_fail failed, $total_skip skipped" >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
cat "$SUMMARY_FILE"
echo ""
echo "Detailed results in: $RESULTS_DIR/"
