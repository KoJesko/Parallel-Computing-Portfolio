#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark-hw.sh - Project 7 performance summary on real hardware
#
#    cpu_torus   Cannon's algorithm, Q x Q torus of nodes, W threads per
#                node, N = 4096, pinned over all 32 logical CPUs
#    cuda_torus  Cannon-ordered SGEMM, G x G torus of thread blocks,
#                with cuBLAS as a reference
#  CPU speedups are against the 1-thread time of the same program.
#  Usage: ./benchmark-hw.sh [N_cpu]
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
N=${1:-4096}
make -s cpu_torus cuda_torus || exit 1

line() { printf '  %s\n' "-------------------------------------------------------------------------------"; }
FLOP=$(awk -v n="$N" 'BEGIN { printf "%.0f", 2 * n * n * n }')

echo "==============================================================================="
echo "  PROJECT 7 - MATRIX MULTIPLICATION ON A TORUS: REAL HARDWARE"
echo "==============================================================================="
echo "  Problem : C = A x B, single precision, exact integer-valued entries"
echo "  CPU     : $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) logical CPUs"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader), $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader) max"
echo "==============================================================================="

cpu_table() {   # cpu_table <flag> <title> <"threads:Q:W" ...>
  local flag=$1 title=$2; shift 2
  echo; echo "  $title"
  printf '  %-8s %-7s %-6s %-10s %-10s %-9s %-11s %s\n' Threads Torus "W" "Time(s)" "GFLOP/s" Speedup Efficiency Checks
  line
  for cfg in "$@"; do
    IFS=: read -r p q w <<< "$cfg"
    ./cpu_torus $flag "$q" "$w" "$N" | awk -F, '/^CSV/ { print $2 "," $3 "," $4 "," $7 "," $8 }'
  done | awk -F, -v f="$FLOP" -v n="$N" -v t1="${T1:-0}" '
    { if (t1 == 0) t1 = $4
      printf "  %-8d %-7s %-6d %-10.4f %-10.1f %-9.2f %-11.3f %s\n",
             $1, $2 "x" $2, $3, $4, f/$4/1e9, t1/$4, t1/$4/$1, (2*n - $5) "/" 2*n }'
}

T1=$(./cpu_torus 1 1 "$N" | awk -F, '/^CSV/ { print $7 }')
export T1
cpu_table "" "CPU - pthreads, dynamic work split (N = $N)" \
  1:1:1 2:1:2 4:2:1 8:2:2 16:4:1 24:2:6 32:4:2 32:2:8
cpu_table -s "CPU - pthreads, static split: each thread works only on its own node" \
  8:2:2 16:4:1 24:2:6 32:4:2

echo; echo "  GPU - CUDA, G x G torus of thread blocks (128 x 128 C tile each)"
printf '  %-6s %-8s %-10s %-10s %-10s %-12s %-8s %s\n' N Grid "K order" "Time(s)" "GFLOP/s" "cuBLAS GF/s" "% cuBLAS" Checks
line
for n in 2048 4096 8192; do
  for flag in "" -n; do
    ./cuda_torus $flag "$n" 10 | awk -F, '/^CSV/ {
      f = 2 * $2 * $2 * $2
      printf "  %-6d %-8s %-10s %-10.6f %-10.1f %-12.1f %-8.0f %s\n", $2, $2/128 "x" $2/128,
             ($3 ? "Cannon" : "unskewed"), $4, f/$4/1e9, f/$5/1e9, 100*$5/$4, (2*$2 - $6) "/" 2*$2 }'
  done
done
echo
echo "==============================================================================="
