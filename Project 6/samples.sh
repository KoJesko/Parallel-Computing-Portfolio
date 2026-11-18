#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  samples.sh - capture the sample runs into results/*.txt
#
#  C* simulator runs of the three programs (as shipped, 4 x 4 mesh), the
#  full C* performance study, and the real-hardware study.  capture.sh then
#  renders these logs as PNG screen captures.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p results

run() {   # run <outfile> <title> <command...>
  local out=$1 title=$2; shift 2
  { echo "### $title"; echo "### command: $*"; echo; "$@"; } > "results/$out" 2>&1
  echo "  results/$out"
}

hw_samples() {
  echo "\$ mpirun --map-by core --bind-to core -np 16 ./mesh_integrate 1000000000"
  mpirun --map-by core --bind-to core -np 16 ./mesh_integrate 1000000000 | grep -v CSV; echo
  echo "\$ ./cpu_mesh 24"; ./cpu_mesh 24 | grep -v CSV; echo
  echo "\$ ./cuda_mesh"; ./cuda_mesh | grep -v CSV
}

run_quiet() {   # like run, without a "command:" line (the body prints its own)
  local out=$1 title=$2; shift 2
  { echo "### $title"; echo; "$@"; } > "results/$out" 2>&1
  echo "  results/$out"
}

echo "Capturing sample runs ..."
run run-cstar-sequential.txt "Sample run: IntegrateSequential.c (C* simulator, 1 processor)" \
    ./runcstar.sh IntegrateSequential.c
run run-cstar-mesh.txt "Sample run: MeshIntegrate.c (C* simulator, 4 x 4 mesh, mesh summation)" \
    ./runcstar.sh MeshIntegrate.c 'UTILIZATION 0:15'
run run-cstar-naive.txt "Sample run: MeshIntegrateNaive.c (C* simulator, 4 x 4 mesh, gather at 0)" \
    ./runcstar.sh MeshIntegrateNaive.c 'UTILIZATION 0:15'
run results-cstar.txt "C* performance study, P = 1 .. 144" ./benchmark.sh
run_quiet run-hw-samples.txt "Sample runs on real hardware (n = 1e9)" hw_samples
run results-hw.txt "Real-hardware performance study" ./benchmark-hw.sh
echo "done."
