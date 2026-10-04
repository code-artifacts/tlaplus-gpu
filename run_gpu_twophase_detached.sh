#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

JAVA_BIN="${JAVA_BIN:-/usr/local/jdk/jdk-12/bin/java}"
JAR="${JAR:-tlatools/org.lamport.tlatools/dist/tla2tools.jar}"
MODEL="${MODEL:-TwoPhase.tla}"
METADIR="${METADIR:-/data/tlc_states_gpu}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
GPU_LIB_DIR="${GPU_LIB_DIR:-$ROOT_DIR}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.4/lib64}"
WORKERS="${WORKERS:-16}"
GPU_FRONTIER_BATCH="${GPU_FRONTIER_BATCH:-2048}"
GPU_INCREMENTAL_DECODE="${GPU_INCREMENTAL_DECODE:-true}"
GPU_PIPELINE="${GPU_PIPELINE:-true}"
GPU_RESIDENT="${GPU_RESIDENT:-false}"
GPU_RESIDENT_MAX_STATES="${GPU_RESIDENT_MAX_STATES:-380000000}"
GPU_RESIDENT_HASH_CAPACITY="${GPU_RESIDENT_HASH_CAPACITY:-536870912}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-$LOG_DIR/twophase_gpu_${RUN_ID}.log}"
SUMMARY_LOG="${SUMMARY_LOG:-$LOG_DIR/twophase_gpu_runs.tsv}"
PID_FILE="${PID_FILE:-$LOG_DIR/twophase_gpu_${RUN_ID}.pid}"

mkdir -p "$LOG_DIR" "$(dirname "$METADIR")"

if [[ "${RUN_IN_BACKGROUND:-0}" != "1" ]]; then
  export RUN_IN_BACKGROUND=1 RUN_ID RUN_LOG SUMMARY_LOG PID_FILE JAVA_BIN JAR MODEL METADIR LOG_DIR GPU_LIB_DIR CUDA_LIB_DIR WORKERS GPU_FRONTIER_BATCH GPU_INCREMENTAL_DECODE GPU_PIPELINE GPU_RESIDENT GPU_RESIDENT_MAX_STATES GPU_RESIDENT_HASH_CAPACITY
  nohup setsid "$0" "$@" >>"$RUN_LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "GPU TwoPhase started in background."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "Summary: $SUMMARY_LOG"
  echo "PID file: $PID_FILE"
  exit 0
fi

start_epoch=$(date +%s)
start_iso=$(date -Is)
status="success"

{
  echo "===== GPU TwoPhase run ====="
  echo "run_id=$RUN_ID"
  echo "start=$start_iso"
  echo "host=$(hostname)"
  echo "cwd=$ROOT_DIR"
  echo "jar=$JAR"
  echo "model=$MODEL"
  echo "metadir=$METADIR"
  echo "gpu_lib_dir=$GPU_LIB_DIR"
  echo "cuda_lib_dir=$CUDA_LIB_DIR"
  echo "workers=$WORKERS"
  echo "gpu_frontier_batch=$GPU_FRONTIER_BATCH"
  echo "gpu_incremental_decode=$GPU_INCREMENTAL_DECODE"
  echo "gpu_pipeline=$GPU_PIPELINE"
  echo "gpu_resident=$GPU_RESIDENT"
  echo "gpu_resident_max_states=$GPU_RESIDENT_MAX_STATES"
  echo "gpu_resident_hash_capacity=$GPU_RESIDENT_HASH_CAPACITY"
  echo
  echo "Command:"
  if [[ "$GPU_RESIDENT" == "true" ]]; then
    echo "LD_LIBRARY_PATH=$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-} /usr/bin/time -p $JAVA_BIN -XX:+UseParallelGC -Djava.library.path=$GPU_LIB_DIR -Dtlc2.TLCGlobals.useGPU=true -Dtlc.gpu.resident.max.states=$GPU_RESIDENT_MAX_STATES -Dtlc.gpu.resident.hash.capacity=$GPU_RESIDENT_HASH_CAPACITY -jar $JAR -workers 1 -metadir $METADIR -modelcheck $MODEL"
  else
    echo "LD_LIBRARY_PATH=$CUDA_LIB_DIR:\${LD_LIBRARY_PATH:-} /usr/bin/time -p $JAVA_BIN -XX:+UseParallelGC -Djava.library.path=$GPU_LIB_DIR -Dtlc.gpu.enabled=true -Dtlc.gpu.legacy.expand=false -Dtlc.gpu.incremental.decode=$GPU_INCREMENTAL_DECODE -Dtlc.gpu.pipeline=$GPU_PIPELINE -Dtlc.gpu.frontier.batch=$GPU_FRONTIER_BATCH -jar $JAR -workers $WORKERS -metadir $METADIR -modelcheck $MODEL"
  fi
  echo
} >>"$RUN_LOG"

set +e
if [[ "$GPU_RESIDENT" == "true" ]]; then
  LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
  /usr/bin/time -p "$JAVA_BIN" \
    -XX:+UseParallelGC \
    -Djava.library.path="$GPU_LIB_DIR" \
    -Dtlc2.TLCGlobals.useGPU=true \
    -Dtlc.gpu.resident.max.states="$GPU_RESIDENT_MAX_STATES" \
    -Dtlc.gpu.resident.hash.capacity="$GPU_RESIDENT_HASH_CAPACITY" \
    -jar "$JAR" \
    -workers 1 \
    -metadir "$METADIR" \
    -modelcheck "$MODEL" \
    >>"$RUN_LOG" 2>&1
else
  LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
  /usr/bin/time -p "$JAVA_BIN" \
    -XX:+UseParallelGC \
    -Djava.library.path="$GPU_LIB_DIR" \
    -Dtlc.gpu.enabled=true \
    -Dtlc.gpu.legacy.expand=false \
    -Dtlc.gpu.incremental.decode="$GPU_INCREMENTAL_DECODE" \
    -Dtlc.gpu.pipeline="$GPU_PIPELINE" \
    -Dtlc.gpu.frontier.batch="$GPU_FRONTIER_BATCH" \
    -jar "$JAR" \
    -workers "$WORKERS" \
    -metadir "$METADIR" \
    -modelcheck "$MODEL" \
    >>"$RUN_LOG" 2>&1
fi
exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
  status="failed"
fi

end_epoch=$(date +%s)
end_iso=$(date -Is)
elapsed_sec=$((end_epoch - start_epoch))

states_generated=$(grep -Eo '[0-9,]+ states generated' "$RUN_LOG" | tail -n 1 | awk '{print $1}' | tr -d ',' || true)
distinct_states=$(grep -Eo '[0-9,]+ distinct states found' "$RUN_LOG" | tail -n 1 | awk '{print $1}' | tr -d ',' || true)
if [[ -z "${states_generated:-}" ]]; then
  states_generated=$(grep -Eo 'generated states: [0-9,]+' "$RUN_LOG" | tail -n 1 | awk '{print $3}' | tr -d ',' || true)
fi
if [[ -z "${distinct_states:-}" ]]; then
  distinct_states=$(grep -Eo 'distinct states: [0-9,]+' "$RUN_LOG" | tail -n 1 | awk '{print $3}' | tr -d ',' || true)
fi
states_generated="${states_generated:-NA}"
distinct_states="${distinct_states:-NA}"

{
  echo
  echo "===== GPU TwoPhase summary ====="
  echo "end=$end_iso"
  echo "status=$status"
  echo "exit_code=$exit_code"
  echo "elapsed_seconds=$elapsed_sec"
  echo "states_generated=$states_generated"
  echo "distinct_states=$distinct_states"
} >>"$RUN_LOG"

if [[ ! -f "$SUMMARY_LOG" ]]; then
  printf 'run_id\tstart\tend\tstatus\texit_code\telapsed_seconds\tstates_generated\tdistinct_states\tlog\n' >"$SUMMARY_LOG"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$RUN_ID" "$start_iso" "$end_iso" "$status" "$exit_code" "$elapsed_sec" \
  "$states_generated" "$distinct_states" "$RUN_LOG" >>"$SUMMARY_LOG"

exit "$exit_code"
