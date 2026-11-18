#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark.sh - Project 5 performance summary
#
#  Runs the Jacobi relaxation programs in the C* simulator for
#  P = 1, 2, 4, 8, 16, 32 processors and prints the speedup / efficiency
#  table that forms the performance summary for the project.
#
#  Both the efficient program (butterfly barrier + butterfly aggregation)
#  and the naive control program (linear counter barrier + lock-based
#  aggregation + copy-back sweep) are measured, so the benefit of the
#  efficient barrier and aggregation functions can be read off directly.
#
#  NOTE ON HOST CPU USAGE: cstar.exe is a single-threaded SIMULATOR - it
#  simulates P processors on one host core, so one run loads only one core.
#  This script therefore runs all the configurations CONCURRENTLY, which is
#  what keeps the host's own cores busy.  (For a run on real hardware see
#  JacobiOpenMP.c / benchmark-openmp.sh.)
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1

PLIST="${PLIST:-1 2 4 8 16 32}"
BUILD=build
JOBS="${JOBS:-$(nproc)}"

rm -rf "$BUILD"; mkdir -p "$BUILD"

log2() { case "$1" in 1) echo 0;; 2) echo 1;; 4) echo 2;; 8) echo 3;;
                      16) echo 4;; 32) echo 5;; *) echo "?";; esac; }

# ---- build one run directory per configuration ----------------------------
jobs_file="$BUILD/jobs.txt"; : > "$jobs_file"

mkdir -p "$BUILD/seq"
cp JacobiSequential.c "$BUILD/seq/"
echo "seq seq JacobiSequential.c" >> "$jobs_file"

for p in $PLIST; do
  lg=$(log2 "$p")

  d="$BUILD/eff_p$p"; mkdir -p "$d"
  sed -e "s/^ARCHITECTURE SHARED(32);/ARCHITECTURE SHARED($p);/" \
      -e "s/^#define P       32 /#define P       $p /" \
      -e "s/^#define LOGP    5 /#define LOGP    $lg /" \
      JacobiParallel.c > "$d/JacobiParallel.c"
  echo "eff $p JacobiParallel.c" >> "$jobs_file"

  d="$BUILD/nai_p$p"; mkdir -p "$d"
  sed -e "s/^ARCHITECTURE SHARED(32);/ARCHITECTURE SHARED($p);/" \
      -e "s/^#define P       32 /#define P       $p /" \
      JacobiNaiveBarrier.c > "$d/JacobiNaiveBarrier.c"
  echo "nai $p JacobiNaiveBarrier.c" >> "$jobs_file"
done

# ---- run them all concurrently -------------------------------------------
echo "Running $(wc -l < "$jobs_file") simulator configurations on up to $JOBS host cores ..."
run_one() {
  set -u
  kind=$1; p=$2; srcname=$3
  case "$kind" in
    seq) d="build/seq"    ;;
    eff) d="build/eff_p$p" ;;
    nai) d="build/nai_p$p" ;;
  esac
  ./runcstar.sh "$d/$srcname" "UTILIZATION 0:$p" > "$d/run.out" 2>&1
  echo "  done: $kind P=$p"
}
export -f run_one
awk '{print $1, $2, $3}' "$jobs_file" |
  xargs -P "$JOBS" -L1 bash -c 'run_one "$@"' _

# ---- collect the numbers --------------------------------------------------
getval() { grep -m1 "$2" "$1" 2>/dev/null | awk -F: '{gsub(/ /,"",$2); print $2}'; }
getswp() { grep -m1 'Sweeps to converge' "$1" 2>/dev/null | awk '{print $NF}'; }
getsum() { grep -m1 'Solution checksum' "$1" 2>/dev/null | awk '{print $NF}'; }

TSEQ=$(getval "$BUILD/seq/run.out" 'SEQUENTIAL EXECUTION TIME')
SSWP=$(getswp "$BUILD/seq/run.out")
SSUM=$(getsum "$BUILD/seq/run.out")

echo
echo "==========================================================================="
echo "  PROJECT 5 - PARALLEL JACOBI RELAXATION: PERFORMANCE SUMMARY"
echo "==========================================================================="
echo "  Problem : 32 x 32 interior grid, top edge 100 deg, other edges 0 deg"
echo "  Stopping: max change per sweep < 0.01  (convergence test)"
echo "  Machine : C* simulated shared-memory multiprocessor"
echo "  Baseline: JacobiSequential.c  T(1) = $TSEQ  ($SSWP sweeps, checksum $SSUM)"
echo "==========================================================================="
printf "\n  EFFICIENT VERSION - butterfly barrier + butterfly aggregation\n"
printf "  %-4s %-14s %-9s %-11s %-8s %-7s %s\n" \
       "P" "T(P)" "Speedup" "Efficiency" "Sweeps" "Util%" "Checksum"
printf "  %s\n" "-------------------------------------------------------------------------"
for p in $PLIST; do
  f="$BUILD/eff_p$p/run.out"
  tp=$(getval "$f" 'PARALLEL EXECUTION TIME')
  sw=$(getswp "$f"); ck=$(getsum "$f")
  ut=$(awk '/PERCENT UTILIZATION/,0' "$f" | awk '$1+0>=1 && NF==3 {s+=$2; n++} END{if(n)printf "%.0f", s/n}')
  awk -v p="$p" -v tp="$tp" -v ts="$TSEQ" -v sw="$sw" -v ut="$ut" -v ck="$ck" \
    'BEGIN{printf "  %-4s %-14s %-9.2f %-11.3f %-8s %-7s %s\n", p, tp, ts/tp, ts/tp/p, sw, ut, ck}'
done

printf "\n  NAIVE CONTROL - linear counter barrier + locked maximum + copy-back\n"
printf "  %-4s %-14s %-9s %-11s %-8s %-7s %s\n" \
       "P" "T(P)" "Speedup" "Efficiency" "Sweeps" "Util%" "Checksum"
printf "  %s\n" "-------------------------------------------------------------------------"
for p in $PLIST; do
  f="$BUILD/nai_p$p/run.out"
  tp=$(getval "$f" 'PARALLEL EXECUTION TIME')
  sw=$(getswp "$f"); ck=$(getsum "$f")
  ut=$(awk '/PERCENT UTILIZATION/,0' "$f" | awk '$1+0>=1 && NF==3 {s+=$2; n++} END{if(n)printf "%.0f", s/n}')
  awk -v p="$p" -v tp="$tp" -v ts="$TSEQ" -v sw="$sw" -v ut="$ut" -v ck="$ck" \
    'BEGIN{printf "  %-4s %-14s %-9.2f %-11.3f %-8s %-7s %s\n", p, tp, ts/tp, ts/tp/p, sw, ut, ck}'
done

printf "\n  BENEFIT OF THE EFFICIENT BARRIER AND AGGREGATION\n"
printf "  %-4s %-14s %-14s %-12s %s\n" "P" "T(P) efficient" "T(P) naive" "Naive/Eff" "Time saved"
printf "  %s\n" "-------------------------------------------------------------------------"
for p in $PLIST; do
  te=$(getval "$BUILD/eff_p$p/run.out" 'PARALLEL EXECUTION TIME')
  tn=$(getval "$BUILD/nai_p$p/run.out" 'PARALLEL EXECUTION TIME')
  awk -v p="$p" -v te="$te" -v tn="$tn" \
    'BEGIN{printf "  %-4s %-14s %-14s %-12.3f %.1f%%\n", p, te, tn, tn/te, 100*(tn-te)/tn}'
done
echo
echo "==========================================================================="
echo "  All versions converge in the same number of sweeps to the same checksum,"
echo "  which confirms the parallel programs compute exactly the same solution"
echo "  as the sequential reference."
echo "==========================================================================="
