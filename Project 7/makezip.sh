#!/usr/bin/env bash
# makezip.sh - package source, scripts, README and results for submission
set -u
cd "$(dirname "$0")" || exit 1
name="Project 7 - Ari Kotler"
rm -f "$name.zip"
( cd .. && zip -q -r "$name/$name.zip" "$name" \
    -x "$name/build/*" "$name/$name.zip" "$name/cpu_torus" "$name/cuda_torus" "$name/LISTFILE.TXT" )
echo "$(pwd)/$name.zip"
