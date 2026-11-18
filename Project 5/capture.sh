#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  capture.sh - render the captured run logs in results/ as terminal-style
#  PNG screen captures, so the submission carries images of the sample runs.
#
#  Lines beginning "### command:" are redrawn as a shell prompt.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
command -v magick >/dev/null 2>&1 || { echo "capture.sh: ImageMagick not found" >&2; exit 1; }

FONT="${FONT:-DejaVu-Sans-Mono}"
BG="#11131a"; FG="#d6dae4"; PT="${PT:-15}"

render() {                       # render <infile> <outfile> <title>
  local in=$1 out=$2 title=$3 tmp
  tmp=$(mktemp)
  { printf '  %s\n\n' "$title"
    sed -e 's/^### command: /  $ /' -e 's/^### /  # /' "$in"
    printf '\n'
  } | expand > "$tmp"
  magick -background "$BG" -fill "$FG" -font "$FONT" -pointsize "$PT" \
         -interline-spacing 2 label:@"$tmp" \
         -bordercolor "$BG" -border 18 \
         -bordercolor "#2a2f3d" -border 1 "$out"
  rm -f "$tmp"
  printf '  %-44s %s\n' "$(basename "$out")" "$(magick identify -format '%wx%h' "$out")"
}

echo "Rendering screen captures into results/ ..."
render results/run-cstar-sequential.txt results/capture-1-cstar-sequential.png \
       "Project 5 - C* simulator: sequential Jacobi reference run"
render results/run-cstar-parallel.txt   results/capture-2-cstar-parallel.png \
       "Project 5 - C* simulator: 32-processor Jacobi, butterfly barrier"
render results/run-cstar-naive.txt      results/capture-3-cstar-naive.png \
       "Project 5 - C* simulator: 32-processor Jacobi, naive linear barrier"
render results/results-cstar.txt        results/capture-4-cstar-summary.png \
       "Project 5 - C* simulator: performance summary, P = 1 .. 32"
render results/run-openmp-samples.txt   results/capture-5-openmp-runs.png \
       "Project 5 - real hardware: sample runs on 32 cores"
render results/results-openmp.txt       results/capture-6-openmp-summary.png \
       "Project 5 - real hardware: performance summary"
echo "done."
