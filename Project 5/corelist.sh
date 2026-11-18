#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  corelist.sh - print an OMP_PLACES list holding one hardware thread per
#  physical core, fastest cores first, with the SMT siblings appended last.
#
#  Why this matters: this machine is a hybrid CPU (fast P-cores plus slower
#  E-cores) and the P-cores are SMT.  With the default consecutive numbering
#  two threads can land on the two SMT siblings of ONE core while other
#  cores sit idle, which makes a 2-thread run slower than a 1-thread run.
#  Ordering the places one-per-core, fastest first, gives a scaling curve
#  that means what it appears to mean.
# ---------------------------------------------------------------------------
set -u

if ! command -v lscpu >/dev/null 2>&1; then
  n=$( nproc 2>/dev/null || echo 1 )
  for ((c = 0; c < n; c++)); do printf '{%d},' "$c"; done | sed 's/,$//'
  echo; exit 0
fi

lscpu -e=CPU,CORE,MAXMHZ 2>/dev/null | tail -n +2 |
awk '{ print $1, $2, $3 + 0 }' |        # cpu, core, MHz
sort -k3,3nr -k1,1n |                   # fastest cores first
awk '
  { cpu = $1; core = $2
    if (core in seen) { sibs[++s] = cpu; next }   # an SMT sibling
    seen[core] = 1; prim[++p] = cpu }             # first thread of a core
  END {
    out = ""
    for (i = 1; i <= p; i++) out = out "{" prim[i] "},"
    for (i = 1; i <= s; i++) out = out "{" sibs[i] "},"
    sub(/,$/, "", out)
    print out }'
