#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  utilization.sh - CPU and GPU utilization measured during the same run
#
#  Starts the 32-thread pthreads program and the CUDA program together, then
#  samples top (per-process %CPU, where 100% = one logical CPU) and
#  nvidia-smi (GPU utilization) at the same moments.
#  Output goes to results/utilization.txt.
#
#  NVSMI=/path/to/nvidia-smi overrides the nvidia-smi used (needed only when
#  the installed tools are newer than the loaded driver, before a reboot).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
NVSMI="${NVSMI:-nvidia-smi}"
make -s cpu_mesh cuda_mesh || exit 1
mkdir -p results
out=results/utilization.txt

cpu_busy() {   # whole-machine CPU busy % over 1 s, from /proc/stat
  read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat; sleep 1
  read -r _ u2 n2 s2 i2 w2 q2 sq2 st2 _ < /proc/stat
  awk -v b=$(( (u2+n2+s2+q2+sq2+st2) - (u1+n1+s1+q1+sq1+st1) )) \
      -v i=$(( (i2+w2) - (i1+w1) )) 'BEGIN { printf "%.1f", 100*b/(b+i) }'
}

./cpu_mesh 32 40000000000 > results/util-cpu_mesh.out &
cpid=$!
./cuda_mesh 10000000000 32 200 > results/util-cuda_mesh.out &
gpid=$!
sleep 3

{
  echo "### CPU + GPU utilization during one run"
  echo "### command: ./cpu_mesh 32 40000000000 &   ./cuda_mesh 10000000000 32 200 &"
  echo
  echo "### Timeline (whole machine, 1 s samples)"
  printf '  %-6s %-22s %s\n' "t (s)" "CPU busy (all 32 CPUs)" "GPU utilization"
  for t in 1 2 3 4 5 6 7 8; do
    c=$(cpu_busy)
    g=$("$NVSMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    printf '  %-6s %-22s %s\n' "$t" "$c %" "$g %"
  done
  echo
  echo "### command: top -b -n 2 -d 1 -o %CPU   (100% = one logical CPU)"
  top -b -n 2 -d 1 -o %CPU -w 100 | awk '/^top -/{n++} n==2' | head -14
  echo
  echo "### command: nvidia-smi   (sampled at the same moment)"
  "$NVSMI" 2>&1 | head -20
} > "$out"

wait $cpid $gpid
{
  echo
  echo "### Program output"
  sed 's/^/  /' results/util-cpu_mesh.out | grep -v CSV
  sed 's/^/  /' results/util-cuda_mesh.out | grep -v CSV
} >> "$out"
rm -f results/util-cpu_mesh.out results/util-cuda_mesh.out
cat "$out"
