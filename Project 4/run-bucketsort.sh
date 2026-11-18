#!/usr/bin/env bash
# Run ParallelBucketSort.c in the C* simulator under Wine.
# usage: ./run-bucketsort.sh [buckets] [skewed]
#   buckets  number of buckets/processes (must divide both 1000 and 2000; default 10)
#   skewed   1 = put ~80% of values in bucket 0 (default 0)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
nb=${1:-10}
skew=${2:-0}
if (( 1000 % nb != 0 || 2000 % nb != 0 )); then
  echo "buckets must divide both 1000 and 2000 (try 2, 4, 5, 8, 10, 20, 25, 40, 50)" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$here"/*.h "$work"/
# C* #define accepts only literal constants, so the variant is generated textually
sed -e "s|^#define buckets 10 |#define buckets $nb |" \
    -e "s|^#define procs 10 |#define procs $nb |" \
    -e "s|^#define width 100 |#define width $((1000 / nb)) |" \
    -e "s|^#define skewed 0 |#define skewed $skew |" \
    "$here/ParallelBucketSort.c" > "$work/ParallelBucketSort.c"

echo "=== Parallel Bucket Sort: n=2000, range [0,999], buckets=$nb, processes=$nb, skewed=$skew ==="
cd "$work"
printf 'OPEN ParallelBucketSort.c\nRUN\nEXIT\n' | wine "$here/cstar.exe" 2>/dev/null | tr -d '\r' \
  | sed -n '/Successfully Compiled/,/TERMINATE/p' | grep -v -e 'LISTFILE' -e 'TERMINATE'
