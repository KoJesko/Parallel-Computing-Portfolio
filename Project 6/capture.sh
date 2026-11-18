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
  [ -f "$in" ] || { echo "  (skipped, no $in)"; return; }
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
       "Project 6 - C* simulator: sequential integration reference run"
render results/run-cstar-mesh.txt       results/capture-2-cstar-mesh.png \
       "Project 6 - C* simulator: 4 x 4 mesh, neighbour-only mesh summation"
render results/run-cstar-naive.txt      results/capture-3-cstar-naive.png \
       "Project 6 - C* simulator: 4 x 4 mesh, naive gather at processor 0"
render results/results-cstar.txt        results/capture-4-cstar-summary.png \
       "Project 6 - C* simulator: performance summary, P = 1 .. 144"
render results/run-hw-samples.txt       results/capture-5-hw-runs.png \
       "Project 6 - real hardware: MPI, pthreads and CUDA sample runs"
render results/results-hw.txt           results/capture-6-hw-summary.png \
       "Project 6 - real hardware: performance summary"
render results/utilization.txt          results/capture-7-utilization.png \
       "Project 6 - CPU and GPU utilization during the same run"
echo "done."
