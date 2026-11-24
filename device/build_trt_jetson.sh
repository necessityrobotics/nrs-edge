#!/usr/bin/env bash
# nrs-edge: build a TensorRT FP16 engine ON THE JETSON with trtexec, time it, and run the device self-test.
#
#   bash build_trt_jetson.sh nrs-edge_w640h400_fp32.onnx             # build + 60 s timing + self-test
#   DURATION=30 bash build_trt_jetson.sh model.onnx                  # shorter timing runs
#   SKIP_TIMING=1 SKIP_SELFTEST=1 bash build_trt_jetson.sh model.onnx
#   TYPING=weak bash build_trt_jetson.sh model.onnx                  # force the FP32 graph + --fp16 build
# If <name>_fp16.onnx sits next to the FP32 graph it is built --stronglyTyped first (see section 3).
#
# No sudo, no network. It never changes the power mode, clocks or fan; set the mode yourself first
# (e.g. `sudo nvpmodel -m <id>`). It writes only into ./build_<model name>/ (or $OUT).
# An engine only runs on the TensorRT version and GPU it was built with: build it on the device that uses it.
# The disparity output stays FP32 (trtexec's default I/O format); do not add --outputIOFormats=fp16.
set -u
ONNX="${1:-}"
if [ -z "$ONNX" ] || [ ! -f "$ONNX" ]; then
  echo "usage: bash build_trt_jetson.sh MODEL_fp32.onnx   (the FP32 graph, for example nrs-edge_w640h400_fp32.onnx)"; exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="$(basename "$ONNX" .onnx)"
NAME="${NAME%_fp32}"
OUT="${OUT:-build_$NAME}"
DURATION="${DURATION:-60}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"
ENGINE="$OUT/${NAME}_fp16.engine"

# ---- 1. the system, printed before anything else (no personal data: no hostname, IP, serial or user name) ----
board=$( { tr -d '\0' < /proc/device-tree/model; } 2>/dev/null || true)
l4t=$(head -n1 /etc/nv_tegra_release 2>/dev/null | cut -c1-100 || true)
jetpack=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)
trt_pkg=$(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null || dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null || true)
trt_py=$("$PY" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || true)
cuda=$(cat /usr/local/cuda/version.json 2>/dev/null | grep -m1 -oE '"version" *: *"[0-9.]+"' | grep -oE '[0-9.]+' || true)
pmode=$(nvpmodel -q 2>/dev/null | tr -s '\n' ' ' || true)
{
  echo "board:      ${board:-unknown}"
  echo "l4t:        ${l4t:-unknown}"
  echo "jetpack:    ${jetpack:-unknown}"
  echo "tensorrt:   ${trt_pkg:-unknown (package)}  python: ${trt_py:-not importable}"
  echo "cuda:       ${cuda:-unknown}"
  echo "power mode: ${pmode:-unknown (nvpmodel -q did not answer without sudo; please note the mode yourself)}"
  echo "kernel:     $(uname -r 2>/dev/null)"
  echo "model:      $(basename "$ONNX")  sha256 $(sha256sum "$ONNX" | cut -c1-16)"
} | tee "$OUT/system.txt"

# ---- 2. trtexec ----
TRTEXEC="${TRTEXEC:-}"
if [ -z "$TRTEXEC" ]; then
  for c in /usr/src/tensorrt/bin/trtexec "$(command -v trtexec 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
  done
fi
if [ -z "$TRTEXEC" ]; then
  echo "trtexec not found. On JetPack it is /usr/src/tensorrt/bin/trtexec; or run TRTEXEC=/path/to/trtexec bash $0 ..."
  exit 1
fi

trt_run() {     # trt_run <log> <args...>: retries without options an older trtexec does not know
  local log=$1; shift
  "$TRTEXEC" "$@" > "$log" 2>&1 && return 0
  if grep -qiE "unknown option|unrecognized|invalid.*percentile" "$log"; then
    local args=()
    for a in "$@"; do
      case "$a" in
        --percentile=*) args+=("--percentile=99") ;;
        --skipInference) args+=("--iterations=1" "--duration=0") ;;
        *) args+=("$a") ;;
      esac
    done
    "$TRTEXEC" "${args[@]}" > "$log" 2>&1 && return 0
  fi
  return 1
}
strip_log() { sed -e 's/^\[[0-9/:.-]*\] \[[A-Z]\] //' -e "s#${HOME:-/nonexistent}#~#g"; }

# ---- 3. FP16 engine ----
# Preferred: the shipped mixed-precision graph <name>_fp16.onnx (FP32 islands at the output tail and InstanceNorm),
# built --stronglyTyped: the same precision graph that our desktop TensorRT test checked. Fallback, or TYPING=weak:
# the FP32 graph with --fp16 (weak typing; TensorRT picks per-layer precision, so it is NOT the graph that test checked).
# The self-test below checks whichever engine was built against the FP32 reference.
ONNX16="${ONNX16:-${ONNX%_fp32.onnx}_fp16.onnx}"
TYPING="${TYPING:-auto}"                          # auto | strong | weak
rm -f "$ENGINE"                                   # never time or test an engine from an earlier run
t0=$(date +%s)
BUILT=""
if [ "$TYPING" != "weak" ] && [ "$ONNX16" != "$ONNX" ] && [ -f "$ONNX16" ]; then
  echo "== building $ENGINE (strongly typed, $(basename "$ONNX16"); several minutes on an Orin Nano) ..."
  if trt_run "$OUT/build.log" --onnx="$ONNX16" --stronglyTyped --saveEngine="$ENGINE" --skipInference; then
    BUILT="strongly typed, $(basename "$ONNX16") sha256 $(sha256sum "$ONNX16" | cut -c1-16)"
  else
    echo "STRONGLY TYPED BUILD FAILED; last errors:"; grep -E "\[E\]|ERROR|error" "$OUT/build.log" | strip_log | tail -n 20
    cp "$OUT/build.log" "$OUT/build_strongly_typed_failed.log"
    [ "$TYPING" = "strong" ] && exit 1
    echo "== falling back to the FP32 graph with --fp16 (weak typing)"
  fi
elif [ "$TYPING" = "strong" ]; then
  echo "TYPING=strong but $ONNX16 not found"; exit 1
fi
if [ -z "$BUILT" ]; then
  echo "== building $ENGINE (FP32 graph + --fp16, weak typing; several minutes on an Orin Nano) ..."
  if ! trt_run "$OUT/build.log" --onnx="$ONNX" --fp16 --saveEngine="$ENGINE" --skipInference; then
    echo "BUILD FAILED; last errors:"; grep -E "\[E\]|ERROR|error" "$OUT/build.log" | strip_log | tail -n 20; exit 1
  fi
  BUILT="weakly typed --fp16 from $(basename "$ONNX") (not the graph that our desktop test checked)"
fi
echo "engine:     $BUILT" | tee -a "$OUT/system.txt"
echo "== built in $(( $(date +%s) - t0 )) s; $(grep -m1 -oE 'TensorRT version: [0-9.]+' "$OUT/build.log" || echo 'TensorRT version: see build.log')"

# ---- 4. timing (60 s each: CUDA graph without copies, then end to end) ----
if [ -z "${SKIP_TIMING:-}" ]; then
  TPID=""
  if command -v tegrastats > /dev/null 2>&1; then
    tegrastats --interval 1000 > "$OUT/tegrastats.log" 2>/dev/null &
    TPID=$!
  fi
  trap '[ -n "$TPID" ] && kill "$TPID" 2>/dev/null' EXIT
  echo "== timing ${DURATION} s, compute only (CUDA graph, no transfers) ..."
  trt_run "$OUT/compute.log" --loadEngine="$ENGINE" --useCudaGraph --noDataTransfers \
    --duration="$DURATION" --warmUp=5000 --percentile=50,90,99 || true
  echo "== timing ${DURATION} s, end to end (with host<->device copies) ..."
  trt_run "$OUT/e2e.log" --loadEngine="$ENGINE" --duration="$DURATION" --warmUp=5000 --percentile=50,90,99 || true
  [ -n "$TPID" ] && kill "$TPID" 2>/dev/null
  TPID=""
  {
    echo "compute (CUDA graph, no transfers):"
    grep -E "Throughput:|GPU Compute Time:|Enqueue Time:" "$OUT/compute.log" 2>/dev/null | strip_log | sed 's/^/  /'
    echo "end to end (with transfers):"
    grep -E "Throughput:|Latency:|GPU Compute Time:" "$OUT/e2e.log" 2>/dev/null | strip_log | sed 's/^/  /'
  } | tee "$OUT/timing.txt"
fi

# ---- 5. device self-test against our FP32 reference ----
if [ -z "${SKIP_SELFTEST:-}" ]; then
  if [ -n "$trt_py" ]; then
    "$PY" "$HERE/selftest.py" --engine "$ENGINE" --json "$OUT/selftest.json"
    echo "self-test exit code: $?"
  else
    echo "self-test skipped: the tensorrt Python package is not importable with $PY."
    echo "Use the system Python of JetPack: PYTHON=/usr/bin/python3 bash $0 $ONNX"
    echo "Later, without a new build: /usr/bin/python3 $HERE/selftest.py --engine $ENGINE"
  fi
fi
echo "done; everything is in $OUT/"
