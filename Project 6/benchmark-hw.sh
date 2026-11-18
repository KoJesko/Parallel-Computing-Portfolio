#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark-hw.sh - Project 6 performance summary on real hardware
#
#  Same integral, n = 1e9 trapezoids, three implementations:
#    mesh_integrate  MPI processes on an MPI_Cart_create 2-D mesh
#    cpu_mesh        pthreads pinned one per logical CPU, mailbox mesh
#    cuda_mesh       GPU, mesh of 32 x 8 thread blocks
#  Speedups are against the 1-process / 1-thread time of the same program.
#  Usage: ./benchmark-hw.sh [n]
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
N=${1:-1000000000}
PLIST="1 2 4 8 16 24 32"
make -s mesh_integrate cpu_mesh cuda_mesh || exit 1

line() { printf '  %s\n' "-------------------------------------------------------------------------"; }
table() {   # table <csv lines: p,mesh,result,time>
  awk -F, 'NR==1 { t1 = $4 }
  { err = $3 - 3.14159265358979; if (err < 0) err = -err
    printf "  %-5d %-7s %-16.12f %-10.2e %-10.6f %-8.2f %.3f\n",
           $1, $2, $3, err, $4, t1/$4, t1/$4/$1 }'
}
hdr() { printf '  %-5s %-7s %-16s %-10s %-10s %-8s %s\n' P Mesh Result Error "Time(s)" Speedup Efficiency; line; }

echo "==========================================================================="
echo "  PROJECT 6 - NUMERICAL INTEGRATION ON A 2-D MESH: REAL HARDWARE"
echo "==========================================================================="
echo "  Problem : integral of sqrt(4 - t*t) from 0 to 2 (= pi), n = $N"
echo "  CPU     : $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) logical CPUs"
echo "  GPU     : $(./cuda_mesh 1000 1 | sed -n 's/^CUDA mesh on //p')"
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
echo "==========================================================================="

echo; echo "  MPI (Open MPI, one process per physical core up to $CORES, MPI_Cart_create mesh)"; hdr
for p in $PLIST; do
  if [ "$p" -le "$CORES" ]; then bind="--map-by core --bind-to core"
  else bind="--use-hwthread-cpus --bind-to none"; fi
  mpirun $bind -np "$p" ./mesh_integrate "$N" 2>/dev/null |
    awk -F, '/^CSV/ { print $2 "," $3 "x" $4 "," $6 "," $8 }'
done | table

for mode in dynamic static; do
  flag=""; [ $mode = static ] && flag="-s"
  echo; echo "  PTHREADS ($mode work distribution, mailbox mesh summation)"; hdr
  for p in $PLIST; do
    ./cpu_mesh $flag "$p" "$N" | awk -F, '/^CSV/ { print $2 "," $3 "x" $4 "," $6 "," $7 }'
  done | table
done

echo; echo "  CUDA (grid = side x side blocks of 32 x 8 threads)"
printf '  %-9s %-10s %-16s %-10s %-10s %s\n' Grid Threads Result Error "Time(s)" "G eval/s"; line
for side in 4 8 16 24 32; do
  ./cuda_mesh "$N" "$side" | awk -F, -v n="$N" '/^CSV/ {
    err = $4 - 3.14159265358979; if (err < 0) err = -err
    printf "  %-9s %-10d %-16.12f %-10.2e %-10.6f %.1f\n", $3 "x" $3, $3*$3*256, $4, err, $5, n/$5/1e9 }'
done
echo
echo "==========================================================================="
