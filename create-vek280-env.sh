#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo 'conda is not on PATH' >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | rg -x 'hls4ml-vek280' >/dev/null 2>&1; then
  conda env update -n hls4ml-vek280 -f environment-vek280.yml --prune
else
  conda env create -f environment-vek280.yml
fi
