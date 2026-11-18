#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  samples.sh - capture the sample runs into results/*.txt
#
#  C* simulator runs of the three programs (as shipped, 4 x 4 torus), the
#  full C* performance study, the real-hardware study and the CPU + GPU
#  utilization run.  capture.sh then renders these logs as PNG captures.
#  C* runs are pinned to CPUS (default 0-15, the P-cores).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
CPUS="${CPUS:-0-15}"
export CPUS
mkdir -p results

run() {   # run <outfile> <title> <command...>
  local out=$1 title=$2; shift 2
  { echo "### $title"; echo "### command: $*"; echo; "$@"; } > "results/$out" 2>&1
  echo "  results/$out"
}

hw_samples() {
  echo "\$ ./cpu_torus            # 4 x 4 torus, 2 threads per node = 32 threads"
  ./cpu_torus | grep -v CSV; echo
  echo "\$ ./cpu_torus 1 1        # 1 thread, same N, for the speedup"
  ./cpu_torus 1 1 | grep -v CSV; echo
  echo "\$ ./cuda_torus           # 64 x 64 torus of thread blocks, N = 8192"
  ./cuda_torus | grep -v CSV
}

run_quiet() {   # like run, without a "command:" line (the body prints its own)
  local out=$1 title=$2; shift 2
  { echo "### $title"; echo; "$@"; } > "results/$out" 2>&1
  echo "  results/$out"
}

echo "Capturing sample runs ..."
run run-cstar-sequential.txt "Sample run: MatMulSequential.c (C* simulator, 1 processor)" \
    taskset -c "$CPUS" ./runcstar.sh MatMulSequential.c
run run-cstar-torus.txt "Sample run: TorusMatMul.c (C* simulator, 4 x 4 torus, Cannon's algorithm)" \
    taskset -c "$CPUS" ./runcstar.sh TorusMatMul.c 'UTILIZATION 0:15'
run run-cstar-naive.txt "Sample run: NaiveMatMul.c (C* simulator, 4 x 4 torus, host distributes)" \
    taskset -c "$CPUS" ./runcstar.sh NaiveMatMul.c 'UTILIZATION 0:15'
run results-cstar.txt "C* performance study, P = 1 .. 256" ./benchmark.sh
run_quiet run-hw-samples.txt "Sample runs on real hardware" hw_samples
run results-hw.txt "Real-hardware performance study" ./benchmark-hw.sh
./utilization.sh > /dev/null && echo "  results/utilization.txt"
echo "done."
