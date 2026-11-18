#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  benchmark-openmp.sh - Project 5 performance summary on REAL hardware
#
#  Four studies:
#    1. correctness      - every thread count and every synchronisation mode
#                          must return the identical solution
#    2. barrier cost     - the efficient barrier/aggregation measured on its
#                          own against the naive one and against OpenMP's
#    3. strong scaling   - speedup and efficiency of the full solver
#    4. large grids      - throughput once the problem leaves cache
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1

[ -x ./JacobiOpenMP ] || make -s JacobiOpenMP || exit 1

PLIST="${PLIST:-1 2 4 8 16 24 32}"
NCONV="${NCONV:-1024}"      # grid for the scaling study (fits in L3)
NBIG="${NBIG:-8192}"        # grid for the out-of-cache study
SWEEPS="${SWEEPS:-400}"     # fixed sweeps for the scaling study
REPS="${REPS:-3}"
MAXP=$(nproc)

export OMP_PROC_BIND=close
export OMP_PLACES="$(./corelist.sh)"


# ---------------------------------------------------------------------------
#  best3 <mode> <iters> <P>  - lowest of three barrier timings, in ns
# ---------------------------------------------------------------------------
best3() {
  local mode=$1 iters=$2 p=$3 i v best=""
  for i in 1 2 3; do
    v=$(./JacobiOpenMP -b -p "$p" -s "$mode" -I "$iters" -q | awk '{print $3}')
    best=$(awk -v a="$best" -v b="$v" 'BEGIN{ if (a == "" || b + 0 < a + 0) print b; else print a }')
  done
  printf '%s\n' "$best"
}

# let the clocks recover, so one phase cannot throttle the next
cooldown() { sleep "${COOLDOWN:-3}"; }

hr() { printf '  %s\n' "-----------------------------------------------------------------------"; }

echo "==========================================================================="
echo "  PROJECT 5 - PARALLEL JACOBI RELAXATION: REAL HARDWARE SUMMARY"
echo "==========================================================================="
printf "  CPU        : %s\n" "$(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ *//')"
printf "  Cores      : %s hardware threads\n" "$MAXP"
if command -v lscpu >/dev/null 2>&1; then
  lscpu -e=CORE,MAXMHZ 2>/dev/null | tail -n +2 | sort -u -k1,1n |
    awk '{c[$2]++} END{printf "  Core types : "; for (f in c) printf "%d x %.0f MHz  ", c[f], f; print ""}'
fi
printf "  Compiler   : %s\n" "$(${CC:-gcc} --version | head -1)"
printf "  Placement  : one thread per physical core, fastest first\n"
# Power state matters: on battery the package power limit is much lower, so
# the same program reports lower throughput and more run-to-run spread.
ac="unknown"
for f in /sys/class/power_supply/A*/online /sys/class/power_supply/AC*/online; do
  [ -r "$f" ] && { [ "$(cat "$f")" = "1" ] && ac="on AC" || ac="on battery"; break; }
done
printf "  Power      : %s" "$ac"
[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] &&
  printf "   governor=%s" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
[ -r /sys/firmware/acpi/platform_profile ] &&
  printf "   profile=%s" "$(cat /sys/firmware/acpi/platform_profile)"
printf "\n"
echo "==========================================================================="

# ---------------------------------------------------------------------------
#  1. CORRECTNESS
# ---------------------------------------------------------------------------
echo
echo "  1. CORRECTNESS - the solution must not depend on how it is computed"
hr
printf "  %-6s %-10s %-6s %-9s %-18s %s\n" "P" "sync" "bal" "sweeps" "checksum" "centre"
hr
REF=""
for mode in tree linear omp; do
  for p in 1 8 32; do
    for bal in "" "-L"; do
      [ "$p" -gt "$MAXP" ] && continue
      line=$(./JacobiOpenMP -n 256 -p "$p" -s "$mode" -t 0.01 $bal)
      sw=$(echo "$line" | awk '/Sweeps /{print $NF}')
      ck=$(echo "$line" | awk '/checksum/{print $NF}')
      ce=$(echo "$line" | awk '/Centre value/{print $4}')
      [ -z "$REF" ] && REF="$ck"
      mark=""; [ "$ck" != "$REF" ] && mark="  <-- MISMATCH"
      printf "  %-6s %-10s %-6s %-9s %-18s %s%s\n" \
             "$p" "$mode" "${bal:--}" "$sw" "$ck" "$ce" "$mark"
    done
  done
done
hr
echo "  Reference checksum: $REF   (identical everywhere = the parallel"
echo "  program computes exactly what the sequential one computes)"

# ---------------------------------------------------------------------------
#  2. STRONG SCALING
# ---------------------------------------------------------------------------
echo
echo "  2. STRONG SCALING - ${NCONV}x${NCONV} grid, $SWEEPS sweeps, convergence tested every sweep"
hr
printf "  %-6s %-10s %-9s %-7s %-10s %-9s %-7s %s\n" \
       "P" "static s" "Gpt/s" "spd" "adaptive s" "Gpt/s" "spd" "gain"
hr
base_s=""; base_a=""
for p in $PLIST; do
  [ "$p" -gt "$MAXP" ] && continue
  sa=$(./JacobiOpenMP -n "$NCONV" -p "$p" -i "$SWEEPS" -r "$REPS" -q    | awk '{print $4, $8/8}')
  sb=$(./JacobiOpenMP -n "$NCONV" -p "$p" -i "$SWEEPS" -r "$REPS" -q -L | awk '{print $4, $8/8}')
  ts=$(echo "$sa" | cut -d' ' -f1); gs=$(echo "$sa" | cut -d' ' -f2)
  ta=$(echo "$sb" | cut -d' ' -f1); ga=$(echo "$sb" | cut -d' ' -f2)
  [ -z "$base_s" ] && base_s="$ts"
  [ -z "$base_a" ] && base_a="$ts"
  awk -v p="$p" -v ts="$ts" -v gs="$gs" -v ta="$ta" -v ga="$ga" \
      -v bs="$base_s" -v ba="$base_a" \
    'BEGIN{printf "  %-6s %-10.4f %-9.2f %-7.2f %-10.4f %-9.2f %-7.2f %+.0f%%\n",
                  p, ts, gs, bs/ts, ta, ga, ba/ta, 100*(ta>0 ? (ts-ta)/ts : 0)}'
done
hr
echo "  spd = speedup against the 1-thread run.  'gain' is what the adaptive"
echo "  partition adds; it is large exactly where the slow E-cores would"
echo "  otherwise hold back every sweep."


cooldown
# ---------------------------------------------------------------------------
#  3. LARGE GRIDS
# ---------------------------------------------------------------------------
echo
echo "  3. LARGE GRIDS - adaptive, all cores, 50 sweeps"
hr
printf "  %-8s %-12s %-10s %-12s %-12s %s\n" \
       "N" "points" "time s" "Gpt/s" "GFLOP/s" "GB/s touched"
hr
for n in 2048 4096 "$NBIG"; do
  r=$(./JacobiOpenMP -n "$n" -p "$MAXP" -i 50 -r 1 -q -L)
  awk -v n="$n" -v r="$r" 'BEGIN{split(r,f," ");
    printf "  %-8s %-12s %-10.4f %-12.2f %-12.2f %.2f\n",
           n, sprintf("%.1fM", n*n/1e6), f[4], f[8]/8, f[8], f[9]}'
done
hr

cooldown
# ---------------------------------------------------------------------------
#  4. BARRIER + AGGREGATION COST  (run last: it heats the CPU)
# ---------------------------------------------------------------------------
echo
echo "  4. COST OF ONE BARRIER + GLOBAL MAXIMUM  (nanoseconds per operation)"
hr
printf "  %-6s %-14s %-14s %-14s %-10s %s\n" \
       "P" "efficient" "naive" "OpenMP" "naive/eff" "omp/eff"
printf "  %-6s %-14s %-14s %-14s %-10s %s\n" \
       "" "(log P, free)" "(O(P), locked)" "(omp barrier)" "" ""
hr
for p in $PLIST; do
  [ "$p" -gt "$MAXP" ] && continue
  [ "$p" -eq 1 ] && continue
  # a contended lock has huge run-to-run spread, so take the best of 3,
  # and give the expensive modes far fewer iterations
  t=$(best3 tree   100000 "$p");  l=$(best3 linear 3000 "$p");  o=$(best3 omp 5000 "$p")
  awk -v p="$p" -v t="$t" -v l="$l" -v o="$o" \
    'BEGIN{printf "  %-6s %-14.0f %-14.0f %-14.0f %-10s %s\n", p, t, l, o,
                  sprintf("%.1fx", l/t), sprintf("%.1fx", o/t)}'
done
hr

echo
echo "==========================================================================="
echo "  Peak throughput and the best thread count are reported above.  On this"
echo "  hybrid CPU the fastest configuration is NOT always all 32 threads: the"
echo "  last 8 are SMT siblings of cores that are already busy."
echo "==========================================================================="
