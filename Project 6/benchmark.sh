#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark.sh - Project 6 performance summary (C* simulator)
#
#  Runs the mesh integration programs in the C* simulator on M x M meshes
#  (M = 1 .. 12, i.e. P = 1 .. 144) with the same total N, and
#  prints the speedup / efficiency table.  Both the efficient program (mesh
#  summation toward the centre) and the naive control (gather at processor
#  0) are measured, plus the sequential reference for T(1).
#
#  cstar.exe simulates all P processors on one host core, so the runs are
#  started concurrently to keep the host's cores busy.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1

MLIST="${MLIST:-1 2 3 4 5 6 8 10 12}"
BUILD=build
rm -rf "$BUILD"; mkdir -p "$BUILD"

# ---- one run directory per configuration ---------------------------------
jobs_file="$BUILD/jobs.txt"; : > "$jobs_file"
mkdir -p "$BUILD/seq"; cp IntegrateSequential.c math.h "$BUILD/seq/"
echo "$BUILD/seq/IntegrateSequential.c 1" >> "$jobs_file"
for m in $MLIST; do
  p=$((m * m))
  for prog in MeshIntegrate MeshIntegrateNaive; do
    d="$BUILD/${prog}_m$m"; mkdir -p "$d"; cp math.h "$d/"
    sed -e "s/^ARCHITECTURE MESH2(4);/ARCHITECTURE MESH2($m);/" \
        -e "s/^#define M        4 /#define M        $m /" \
        -e "s/^#define P        16 /#define P        $(printf '%-3s' $p)/" \
        "$prog.c" > "$d/$prog.c"
    echo "$d/$prog.c $p" >> "$jobs_file"
  done
done

echo "Running $(wc -l < "$jobs_file") simulator configurations concurrently ..."
xargs -P "$(nproc)" -L1 bash -c './runcstar.sh "$0" "UTILIZATION 0:$(($1 - 1))" > "$(dirname "$0")/run.out" 2>&1' < "$jobs_file"

# ---- collect --------------------------------------------------------------
tpar()  { grep -m1 'PARALLEL EXECUTION TIME' "$1" | awk '{print $NF}'; }
res()   { grep -m1 'Integral' "$1" | awk '{print $NF}'; }
util()  { awk '/PERCENT UTILIZATION/{f=1;next} f&&$1~/^[0-9]+$/&&$1<'"$2"'{s+=$2;n++} END{if(n) printf "%d", s/n}' "$1"; }

TSEQ=$(tpar "$BUILD/seq/run.out")
RSEQ=$(res "$BUILD/seq/run.out")

line() { printf '  %s\n' "-------------------------------------------------------------------------"; }
echo
echo "==========================================================================="
echo "  PROJECT 6 - NUMERICAL INTEGRATION ON A 2-D MESH: PERFORMANCE SUMMARY"
echo "==========================================================================="
echo "  Problem : integral of sqrt(4 - t*t) from 0 to 2 (= pi), N = 28800"
echo "  Machine : C* simulated M x M mesh multicomputer (ARCHITECTURE MESH2)"
echo "  Baseline: IntegrateSequential.c  T(1) = $TSEQ, result $RSEQ"
echo "==========================================================================="
for prog in MeshIntegrate MeshIntegrateNaive; do
  echo
  if [ $prog = MeshIntegrate ]; then
    echo "  EFFICIENT VERSION - neighbour-only mesh summation toward the centre"
  else
    echo "  NAIVE CONTROL - every process sends its sum to processor 0"
  fi
  printf '  %-6s %-4s %-12s %-9s %-11s %-6s %s\n' Mesh P "T(P)" Speedup Efficiency Util% Result
  line
  for m in $MLIST; do
    p=$((m * m)); f="$BUILD/${prog}_m$m/run.out"; t=$(tpar "$f")
    awk -v m="$m" -v p="$p" -v t="$t" -v s="$TSEQ" -v u="$(util "$f" "$p")" -v r="$(res "$f")" \
      'BEGIN { printf "  %-6s %-4d %-12d %-9.2f %-11.3f %-6s %s\n", m "x" m, p, t, s/t, s/t/p, u, r }'
  done
done
echo
echo "  BENEFIT OF MESH SUMMATION"
printf '  %-6s %-4s %-14s %-14s %-11s %s\n' Mesh P "T(P) mesh" "T(P) naive" Naive/Mesh "Time saved"
line
for m in $MLIST; do
  p=$((m * m))
  te=$(tpar "$BUILD/MeshIntegrate_m$m/run.out"); tn=$(tpar "$BUILD/MeshIntegrateNaive_m$m/run.out")
  awk -v m="$m" -v p="$p" -v a="$te" -v b="$tn" \
    'BEGIN { printf "  %-6s %-4d %-14d %-14d %-11.3f %.1f%%\n", m "x" m, p, a, b, b/a, 100*(b-a)/b }'
done
echo
echo "==========================================================================="
