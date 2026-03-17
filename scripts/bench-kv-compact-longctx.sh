#!/bin/bash
# Long-context benchmark driver for KV cache compaction (6b-15 / 6b-15b).
#
# Iterates: context sizes x compression ratios x pipelines.
# Produces a self-describing artifact directory with manifest.json + results.csv.
#
# Usage:
#   # Full benchmark on one model:
#   ./scripts/bench-kv-compact-longctx.sh /path/to/Qwen3-14B-Q4_K_M.gguf
#
#   # With QuALITY evaluation (4K-8K contexts):
#   QUALITY_EVAL=1 ./scripts/bench-kv-compact-longctx.sh /path/to/Qwen3-14B-Q4_K_M.gguf
#
#   # Include self_study pipeline (6b-15b):
#   SELF_STUDY=1 ./scripts/bench-kv-compact-longctx.sh /path/to/Qwen3-14B-Q4_K_M.gguf
#
#   # Smoke test (small model, quick):
#   ./scripts/bench-kv-compact-longctx.sh /path/to/stories15M-q4_0.gguf

set -euo pipefail

# --- Process cleanup (prevents orphaned PIDs) ---
CHILD_PIDS=()

cleanup() {
    local sig="${1:-TERM}"
    # Kill all tracked child PIDs (test binaries + watchdog sleeps).
    for pid in "${CHILD_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        kill -"$sig" "$pid" 2>/dev/null || true
    done
    # Release GPU lock.
    release_gpu_lock
}

trap 'cleanup TERM; exit 130' INT
trap 'cleanup TERM; exit 143' TERM
trap 'cleanup TERM' EXIT

# --- GPU lockfile (prevents concurrent bench runs) ---
LOCK_DIR="/tmp/bench-kv-compact.lock"
SKIP_GPU_LOCK="${SKIP_GPU_LOCK:-0}"

acquire_gpu_lock() {
    [ "$SKIP_GPU_LOCK" = "1" ] && return 0
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        echo "$(date +%Y%m%d-%H%M%S)" > "$LOCK_DIR/started"
        return 0
    fi
    # Lock exists — check if holder is still alive.
    local holder_pid
    holder_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$holder_pid" ] && kill -0 "$holder_pid" 2>/dev/null; then
        local started
        started=$(cat "$LOCK_DIR/started" 2>/dev/null || echo "unknown")
        echo "ERROR: Another benchmark is running (pid $holder_pid, started $started)."
        echo "  Wait for it to finish, or remove the lock: rm -rf $LOCK_DIR"
        exit 1
    fi
    # Stale lock — reclaim.
    echo "WARNING: Removing stale lock (pid ${holder_pid:-unknown} is dead)"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
    echo "$$" > "$LOCK_DIR/pid"
    echo "$(date +%Y%m%d-%H%M%S)" > "$LOCK_DIR/started"
}

release_gpu_lock() {
    [ "$SKIP_GPU_LOCK" = "1" ] && return 0
    # Only release if we own the lock.
    local holder_pid
    holder_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ "$holder_pid" = "$$" ]; then
        rm -rf "$LOCK_DIR"
    fi
}

# --- Pre-flight GPU check ---
# Finds GPU-heavy processes by matching binary names (not shell wrappers).
# Uses pgrep without -f to match process names, avoiding false positives
# from parent shells that happen to contain binary names in their argv.
preflight_gpu_check() {
    local dominated=0
    local warnings=""

    # Check for ollama serve (loads models into GPU memory).
    local ollama_pids
    ollama_pids=$(pgrep -x "ollama" 2>/dev/null || true)
    if [ -n "$ollama_pids" ]; then
        warnings="${warnings}\n  - ollama is running (pids: $ollama_pids). Stop with: brew services stop ollama"
        dominated=1
    fi

    # Check for llama-cli / llama-server binaries (exact process name match).
    local llama_pids
    llama_pids=$(pgrep -x "llama-cli" 2>/dev/null || true)
    llama_pids="${llama_pids}$(pgrep -x "llama-server" 2>/dev/null || true)"
    if [ -n "$llama_pids" ]; then
        warnings="${warnings}\n  - llama-cli/server running (pids: $llama_pids)"
        dominated=1
    fi

    # Check for other test-kv-compact binaries (exact name, exclude our own PID).
    local test_pids
    test_pids=$(pgrep -x "test-kv-compact-longctx" 2>/dev/null | grep -v "^$$\$" || true)
    if [ -n "$test_pids" ]; then
        warnings="${warnings}\n  - Other test-kv-compact processes (pids: $test_pids)"
        dominated=1
    fi

    if [ "$dominated" -eq 1 ]; then
        echo "WARNING: GPU may be contested by other processes:"
        echo -e "$warnings"
        if [ "${FORCE_RUN:-0}" = "1" ]; then
            echo "  FORCE_RUN=1 — proceeding anyway (throughput numbers may be unreliable)"
        else
            echo "  Set FORCE_RUN=1 to proceed, or stop the competing processes first."
            exit 1
        fi
    fi
}

# --- Preflight-only mode: test GPU checks without running benchmarks ---
if [ "${1:-}" = "--preflight" ]; then
    echo "=== Preflight GPU Check ==="
    SKIP_GPU_LOCK=1  # Don't acquire lock in check-only mode.
    FORCE_RUN=1      # Don't exit — just report.
    preflight_gpu_check
    echo "Lock status:"
    if [ -d "$LOCK_DIR" ]; then
        echo "  LOCKED by pid $(cat "$LOCK_DIR/pid" 2>/dev/null || echo unknown), started $(cat "$LOCK_DIR/started" 2>/dev/null || echo unknown)"
        local_holder=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
        if [ -n "$local_holder" ] && kill -0 "$local_holder" 2>/dev/null; then
            echo "  Holder is ALIVE"
        else
            echo "  Holder is DEAD (stale lock)"
        fi
    else
        echo "  No lock held"
    fi
    echo "Preflight complete."
    exit 0
fi

MODEL="${1:?Usage: $0 <model.gguf> [ngl]  (or: $0 --preflight)}"
NGL="${2:-99}"
BUILD_DIR="${BUILD_DIR:-build}"
ARTIFACT_BASE="${ARTIFACT_BASE:-bench-results}"
QUALITY_EVAL="${QUALITY_EVAL:-0}"
LONGHEALTH_EVAL="${LONGHEALTH_EVAL:-0}"
SELF_STUDY="${SELF_STUDY:-0}"
SEC_TEXT_DIR="${SEC_TEXT_DIR:-}"

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

# Acquire GPU lock and run pre-flight checks.
acquire_gpu_lock
preflight_gpu_check

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
    echo "longhealth_eval: $LONGHEALTH_EVAL"
    echo "self_study: $SELF_STUDY"
    echo "sec_text_dir: ${SEC_TEXT_DIR:-none}"
} > "$ENV_TXT"

echo "=== KV Compaction Long-Context Benchmark ==="
echo "model:     $MODEL_NAME"
echo "run_id:    $RUN_ID"
echo "output:    $RUN_DIR/"
echo ""

# Context sizes and ratios.
CONTEXTS="${CONTEXTS:-4096 8192 16384 32768}"
RATIOS="${RATIOS:-2 4 8}"

# Pipelines: baseline + compacted. Override with PIPELINES env var.
if [ -n "${PIPELINES:-}" ]; then
    # Use caller's pipeline list directly.
    true
else
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
fi

N_PASS=0
N_FAIL=0
N_CRASH=0
N_SKIP=0
N_TIMEOUT=0

# Per-run timeout in seconds. Default 600s (10 min). Override with RUN_TIMEOUT env var.
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"

run_one() {
    local pipeline="$1"
    local ctx="$2"
    local ratio="$3"
    local quality_flag="$4"
    local longhealth_flag="$5"

    local start_ts=$(date +%s)
    echo "--- $pipeline | ctx=$ctx | ratio=${ratio}x (timeout=${RUN_TIMEOUT}s) ---"

    # Run test with watchdog timeout. Use a PID file to track the actual binary.
    local tmp_out=$(mktemp)
    local pid_file=$(mktemp)

    # Launch in background, capture stdout, stderr to log.
    PIPELINE="$pipeline" RATIO="$ratio" RUN_ID="$RUN_ID" \
        ARTIFACT="$RESULTS_CSV" QUALITY_EVAL="$quality_flag" \
        QUALITY_DATA="tests/data/quality-validation-full.jsonl" \
        QUALITY_LIMIT="${QUALITY_LIMIT:-100}" \
        LONGHEALTH_EVAL="$longhealth_flag" \
        LONGHEALTH_DATA="tests/data/longhealth-benchmark-v5.json" \
        LONGHEALTH_LIMIT="${LONGHEALTH_LIMIT:-50}" \
        SEC_TEXT_DIR="${SEC_TEXT_DIR}" \
        "$BIN" -m "$MODEL" -ngl "$NGL" -c "$ctx" \
        > "$tmp_out" 2>>"$STDERR_LOG" &
    local bin_pid=$!
    echo "$bin_pid" > "$pid_file"
    CHILD_PIDS+=("$bin_pid")

    # Watchdog: kill if exceeds timeout.
    {
        sleep "$RUN_TIMEOUT" 2>/dev/null || true
        if kill -0 "$bin_pid" 2>/dev/null; then
            echo "  TIMEOUT after ${RUN_TIMEOUT}s — killing pid $bin_pid"
            kill "$bin_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$bin_pid" 2>/dev/null || true
        fi
    } &
    local watchdog_pid=$!
    CHILD_PIDS+=("$watchdog_pid")

    # Wait for the test process. Capture real exit code (|| true masks it).
    local rc=0
    wait "$bin_pid" 2>/dev/null || rc=$?

    # Kill the watchdog (|| true to prevent set -e abort on already-dead process).
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    # Remove finished PIDs from tracking.
    CHILD_PIDS=("${CHILD_PIDS[@]/$bin_pid}" "${CHILD_PIDS[@]/$watchdog_pid}")

    # Display and log captured stdout.
    if [ -s "$tmp_out" ]; then
        cat "$tmp_out" | tee -a "$STDOUT_LOG"
    fi
    rm -f "$tmp_out" "$pid_file"

    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))

    if [ $rc -eq 137 ] || [ $rc -eq 143 ]; then
        if [ $elapsed -ge $((RUN_TIMEOUT - 5)) ]; then
            echo "  TIMEOUT ($pipeline ctx=$ctx ratio=${ratio}x after ${elapsed}s)"
            N_TIMEOUT=$((N_TIMEOUT + 1))
        else
            echo "  CRASH (signal $rc after ${elapsed}s)"
            N_CRASH=$((N_CRASH + 1))
        fi
    elif [ $rc -eq 139 ] || [ $rc -eq 134 ]; then
        echo "  CRASH (signal $rc after ${elapsed}s)"
        N_CRASH=$((N_CRASH + 1))
    elif [ $rc -ne 0 ]; then
        echo "  FAIL (exit $rc after ${elapsed}s — threshold miss or error)"
        N_FAIL=$((N_FAIL + 1))
    else
        echo "  PASS (${elapsed}s)"
        N_PASS=$((N_PASS + 1))
    fi
    echo ""
}

# Run baseline for each context size (no ratio variation).
for CTX in $CONTEXTS; do
    # QuALITY: only at contexts <= 8K (articles fit).
    local_quality_bl="0"
    if [ "$QUALITY_EVAL" = "1" ] && [ "$CTX" -le 8192 ]; then
        local_quality_bl="1"
    fi
    # LongHealth: only at contexts >= 32K (patient records are ~10K tokens each, need 60K+).
    local_lh_bl="0"
    if [ "$LONGHEALTH_EVAL" = "1" ] && [ "$CTX" -ge 32768 ]; then
        local_lh_bl="1"
    fi
    run_one "baseline" "$CTX" "1" "$local_quality_bl" "$local_lh_bl"
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
        # LongHealth eval only at >= 32K (patient records need ~60K tokens).
        local_lh="0"
        if [ "$LONGHEALTH_EVAL" = "1" ] && [ "$CTX" -ge 32768 ]; then
            local_lh="1"
        fi
        for RATIO in $RATIOS; do
            run_one "$PIPELINE" "$CTX" "$RATIO" "$local_quality" "$local_lh"
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
TOTAL=$((N_PASS + N_FAIL + N_CRASH + N_TIMEOUT))
echo ""
echo "=== Summary ==="
echo "model:   $MODEL_NAME"
echo "run_id:  $RUN_ID"
echo "total:   $TOTAL runs"
echo "  pass:    $N_PASS"
echo "  fail:    $N_FAIL (threshold miss)"
echo "  crash:   $N_CRASH"
echo "  timeout: $N_TIMEOUT (>${RUN_TIMEOUT}s)"
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
