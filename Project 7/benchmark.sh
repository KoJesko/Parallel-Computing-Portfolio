#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark.sh - Project 7 performance summary (C* simulator)
#
#  Runs the matrix multiplication programs in the C* simulator on Q x Q
#  tori (Q = 1 .. 16, i.e. P = 1 .. 256) with the same N = 96, and prints
#  speedup / efficiency tables for:
#    TorusMatMul.c            Cannon's algorithm on the torus (main program)
#    NaiveMatMul.c            host distributes strips, collects C blocks
#    TorusMatMul.c on MESH2   same program with the wraparound links removed
#    TorusMatMul.c OVERLAP 0  same program, communication not overlapped
#  T(1) comes from MatMulSequential.c.
#
#  Each cstar.exe simulates the whole multicomputer itself; separate runs
#  are only separate configurations.  They are pinned to CPUS (default
#  0-15, the 8 P-cores of the i9-14900HX and their hyperthreads) with at
#  most JOBS (default 16) running at once.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1

QLIST="${QLIST:-1 2 3 4 6 8 12 16}"
CPUS="${CPUS:-0-15}"
JOBS="${JOBS:-16}"
BUILD=build
rm -rf "$BUILD"; mkdir -p "$BUILD"

variant() {   # variant <src> <dir> <arch> <q> <overlap>
  local q=$4 p=$(($4 * $4)) nb=$((96 / $4))
  mkdir -p "$2"; cp math.h "$2/"
  sed -e "s/^ARCHITECTURE TORUS(4);/ARCHITECTURE $3($q);/" \
      -e "s/^#define Q        4 /#define Q        $(printf '%-2s' $q) /" \
      -e "s/^#define P        16 /#define P        $(printf '%-3s' $p)/" \
      -e "s/^#define NB       24 /#define NB       $(printf '%-3s' $nb)/" \
      -e "s/^#define OVERLAP  1 /#define OVERLAP  $5 /" \
      "$1" > "$2/$(basename "$1")"
  echo "$2/$(basename "$1") $p"
}

# ---- one run directory per configuration ---------------------------------
jobs_file="$BUILD/jobs.txt"; : > "$jobs_file"
mkdir -p "$BUILD/seq"; cp MatMulSequential.c math.h "$BUILD/seq/"
echo "$BUILD/seq/MatMulSequential.c 1" >> "$jobs_file"
for q in $QLIST; do
  variant TorusMatMul.c "$BUILD/torus_q$q"   TORUS $q 1 >> "$jobs_file"
  variant NaiveMatMul.c "$BUILD/naive_q$q"   TORUS $q 1 >> "$jobs_file"
  variant TorusMatMul.c "$BUILD/mesh_q$q"    MESH2 $q 1 >> "$jobs_file"
  variant TorusMatMul.c "$BUILD/nolap_q$q"   TORUS $q 0 >> "$jobs_file"
done

echo "Running $(wc -l < "$jobs_file") simulator configurations on CPUs $CPUS ($JOBS at a time) ..."
xargs -P "$JOBS" -L1 taskset -c "$CPUS" bash -c \
  './runcstar.sh "$0" "UTILIZATION 0:$(($1 - 1))" > "$(dirname "$0")/run.out" 2>&1' < "$jobs_file"

# ---- collect --------------------------------------------------------------
tpar()  { grep -m1 'T(P)\|T(1)' "$1" | awk '{print $NF}'; }
ok()    { grep -m1 'checks' "$1" | awk '{print $4 "/" $6}'; }
util()  { awk '/PERCENT UTILIZATION/{f=1;next} f&&$1~/^[0-9]+$/&&$1<'"$2"'{s+=$2;n++} END{if(n) printf "%d", s/n}' "$1"; }

TSEQ=$(tpar "$BUILD/seq/run.out")

line() { printf '  %s\n' "-------------------------------------------------------------------------"; }
table() {   # table <dir prefix> <title>
  echo; echo "  $2"
  printf '  %-7s %-4s %-6s %-12s %-9s %-11s %-6s %s\n' Torus P Block "T(P)" Speedup Efficiency Util% Checks
  line
  for q in $QLIST; do
    p=$((q * q)); f="$BUILD/$1_q$q/run.out"
    awk -v q="$q" -v p="$p" -v t="$(tpar "$f")" -v s="$TSEQ" -v u="$(util "$f" "$p")" -v c="$(ok "$f")" \
      'BEGIN { printf "  %-7s %-4d %-6s %-12d %-9.2f %-11.3f %-6s %s\n", q "x" q, p, 96/q "x" 96/q, t, s/t, s/t/p, u, c }'
  done
}
compare() {   # compare <dir a> <dir b> <title> <label a> <label b>
  echo; echo "  $3"
  printf '  %-7s %-4s %-14s %-14s %s\n' Torus P "$4" "$5" "$5 is slower by"
  line
  for q in $QLIST; do
    [ "$q" -eq 1 ] && continue
    p=$((q * q))
    awk -v q="$q" -v p="$p" -v a="$(tpar "$BUILD/$1_q$q/run.out")" -v b="$(tpar "$BUILD/$2_q$q/run.out")" \
      'BEGIN { printf "  %-7s %-4d %-14d %-14d %.2f%%\n", q "x" q, p, a, b, 100*(b-a)/a }'
  done
}

echo
echo "==========================================================================="
echo "  PROJECT 7 - MATRIX MULTIPLICATION ON A TORUS: PERFORMANCE SUMMARY"
echo "==========================================================================="
echo "  Problem : C = A x B, N = 96 (884,736 multiply-adds)"
echo "  Machine : C* simulated Q x Q torus multicomputer (ARCHITECTURE TORUS)"
echo "  Baseline: MatMulSequential.c  T(1) = $TSEQ  ($(ok "$BUILD/seq/run.out") checks)"
echo "  Host    : cstar.exe under Wine, pinned to CPUs $CPUS"
echo "==========================================================================="
table torus "EFFICIENT VERSION - Cannon's algorithm, 1-hop block rotations (TorusMatMul.c)"
table naive "NAIVE CONTROL - processor 0 sends strips and collects blocks (NaiveMatMul.c)"
compare torus naive "CANNON vs NAIVE" "T(P) Cannon" "T(P) naive"
compare torus mesh  "TORUS vs MESH: same program, wraparound links removed (MESH2)" "T(P) torus" "T(P) mesh"
compare torus nolap "OVERLAP vs NO OVERLAP: rotations sent before vs after the multiply" "T(P) overlap" "T(P) after"
echo
echo "==========================================================================="
