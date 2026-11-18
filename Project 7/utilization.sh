#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  utilization.sh - CPU and GPU utilization measured during the same run
#
#  Starts the 32-thread CPU torus program and the CUDA torus program
#  together, then samples whole-machine CPU busy (from /proc/stat), top
#  (per-process %CPU, 100% = one logical CPU) and nvidia-smi at the same
#  moments.  Output goes to results/utilization.txt.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
NVSMI="${NVSMI:-nvidia-smi}"
make -s cpu_torus cuda_torus || exit 1
mkdir -p results
out=results/utilization.txt

cpu_busy() {   # whole-machine CPU busy % over 1 s, from /proc/stat
  read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat; sleep 1
  read -r _ u2 n2 s2 i2 w2 q2 sq2 st2 _ < /proc/stat
  awk -v b=$(( (u2+n2+s2+q2+sq2+st2) - (u1+n1+s1+q1+sq1+st1) )) \
      -v i=$(( (i2+w2) - (i1+w1) )) 'BEGIN { printf "%.1f", 100*b/(b+i) }'
}

./cpu_torus 4 2 10240 > results/util-cpu.out &
cpid=$!
./cuda_torus 8192 150 > results/util-gpu.out &
gpid=$!
sleep 3

{
  echo "### CPU + GPU utilization during one run"
  echo "### command: ./cpu_torus 4 2 10240 &   ./cuda_torus 8192 150 &"
  echo
  echo "### Timeline (whole machine, 1 s samples)"
  printf '  %-6s %-24s %-16s %s\n' "t (s)" "CPU busy (all 32 CPUs)" "GPU utilization" "GPU power"
  for t in 1 2 3 4 5 6 7 8; do
    c=$(cpu_busy)
    g=$("$NVSMI" --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    printf '  %-6s %-24s %-16s %s\n' "$t" "$c %" "${g%%,*} %" "${g##*, } W"
  done
  echo
  echo "### command: top -b -n 2 -d 1 -o %CPU   (100% = one logical CPU)"
  top -b -n 2 -d 1 -o %CPU -w 100 | awk '/^top -/{n++} n==2' | head -14
  echo
  echo "### command: mpstat-style per-CPU busy over 1 s (from /proc/stat)"
  { grep '^cpu[0-9]' /proc/stat; sleep 1; grep '^cpu[0-9]' /proc/stat; } |
    awk '{ b=$2+$3+$4+$7+$8+$9; i=$5+$6
           if ($1 in B) { db=b-B[$1]; di=i-I[$1]; printf "%s %5.1f%%%s", $1, 100*db/(db+di), (++n % 8 ? "   " : "\n") }
           else { B[$1]=b; I[$1]=i } }'
  echo
  echo "### command: nvidia-smi   (sampled at the same moment)"
  "$NVSMI" 2>&1 | head -20
} > "$out"

wait $cpid $gpid
{
  echo
  echo "### Program output"
  grep -v CSV results/util-cpu.out | sed 's/^/  /'
  grep -v CSV results/util-gpu.out | sed 's/^/  /'
} >> "$out"
rm -f results/util-cpu.out results/util-gpu.out
cat "$out"
