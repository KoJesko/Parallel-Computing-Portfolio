#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  runcstar.sh <source.c> [extra C* commands ...]
#
#  Drives Bruce Lester's C* compiler / parallel computer simulator
#  (cstar.exe) non-interactively, so that runs can be scripted and captured.
#
#  The C* system is an interactive program: it prints a '*' prompt and reads
#  commands such as OPEN, RUN, UTILIZATION and EXIT.  This script feeds those
#  commands in on standard input and strips the banner from the output.
#
#  cstar.exe is a Win32 program, so it is started through Wine on Linux/macOS.
#  Set CSTAR=/path/to/cstar.exe to override the search, or use the native
#  binary (cstar / cstar-mac / cstar-m1) by pointing CSTAR at it - a native
#  binary is run directly, without Wine.
#
#  Simulator settings that must be given before RUN (e.g. CONGESTION ON,
#  DELAY 50) go in CSTAR_PRE, separated by ';'.  Commands given as extra
#  arguments are issued after the program has run.
#
#  Examples:
#     ./runcstar.sh TorusMatMul.c
#     ./runcstar.sh TorusMatMul.c 'UTILIZATION 0:15' TIME
#     CSTAR_PRE='CONGESTION ON;DELAY 50' ./runcstar.sh TorusMatMul.c
# ---------------------------------------------------------------------------
set -u

here="$(cd "$(dirname "$0")" && pwd)"

# ---- locate the simulator -------------------------------------------------
if [ -z "${CSTAR:-}" ]; then
  for cand in "$here/cstar.exe" "$here/../cstar.exe" "$here/../parallel/cstar.exe"; do
    [ -f "$cand" ] && CSTAR="$cand" && break
  done
fi
if [ -z "${CSTAR:-}" ] || [ ! -f "$CSTAR" ]; then
  echo "runcstar.sh: cannot find cstar.exe (set CSTAR=/path/to/cstar.exe)" >&2
  exit 1
fi

# ---- a .exe needs Wine, a native binary does not --------------------------
case "$CSTAR" in
  *.exe) runner=(env WINEDEBUG=-all wine "$CSTAR") ;;
  *)     runner=("$CSTAR") ;;
esac

if [ $# -lt 1 ]; then
  echo "usage: runcstar.sh <source.c> [extra C* commands ...]" >&2
  exit 2
fi

src="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
cd "$(dirname "$src")" || exit 1

# ---- feed the command script to the simulator -----------------------------
{
  printf 'OPEN %s\n' "$(basename "$src")"
  if [ -n "${CSTAR_PRE:-}" ]; then printf '%s\n' "$CSTAR_PRE" | tr ';' '\n'; fi
  printf 'RUN\n'
  for cmd in "$@"; do printf '%s\n' "$cmd"; done
  printf 'EXIT\n'
} | timeout 3600 "${runner[@]}" 2>/dev/null |
tr -d '\r' |
sed -n '/^  Basic Commands:/,$p' | tail -n +9 | sed '/^\*$/d'
