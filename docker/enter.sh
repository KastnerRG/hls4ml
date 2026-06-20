#!/usr/bin/env bash
set -eo pipefail

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${HLS4ML_CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${HLS4ML_VITIS_ROOT}/Vitis/settings64.sh" ]]; then
    echo "Vitis settings script not found: ${HLS4ML_VITIS_ROOT}/Vitis/settings64.sh" >&2
    exit 1
fi
source "${HLS4ML_VITIS_ROOT}/Vitis/settings64.sh"

for license in /licenses/Xilinx-lic/*.lic; do
    [[ -f "${license}" ]] || continue
    export XILINXD_LICENSE_FILE="${XILINXD_LICENSE_FILE:+${XILINXD_LICENSE_FILE}:}${license}"
    export LM_LICENSE_FILE="${LM_LICENSE_FILE:+${LM_LICENSE_FILE}:}${license}"
done
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-${XILINXD_LICENSE_FILE}}"

if (($#)); then
    exec "$@"
fi

exec bash --noprofile --rcfile /etc/hls4ml.bashrc -i
