#!/bin/bash
# Build the UniVTAC isaac51 conda env (Isaac Sim 5.1, TacEx/libuipc, cuRobo) inside the container.
# Submit from repo root: sbatch -A <account> slurm/install_univtac.sbatch
#SBATCH -J univtac-install
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o ../logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$UNIVTAC_ROOT"
git apply --check "$SLURM_SUBMIT_DIR/patches/univtac-isaac51.patch" 2>/dev/null \
  && git apply "$SLURM_SUBMIT_DIR/patches/univtac-isaac51.patch"  # local fixes, see patches/
nvidia-smi -L
in_univtac bash -c '
  set -euo pipefail
  source "$UNIVTAC_CONDA/etc/profile.d/conda.sh"
  export UNIVTAC_CUDA_ARCH=80 UNIVTAC_BUILD_JOBS=12   # A30/A100 = sm_80
  export VCPKG_OVERLAY_PORTS="$SLURM_SUBMIT_DIR/patches/vcpkg-overlay"
  bash scripts/install.sh
  bash scripts/install.sh --check
  "$CONDA_ENVS_PATH/UniVTAC/bin/python" -m pip install pyzmq   # sim_server.py
  "$CONDA_ENVS_PATH/UniVTAC/bin/python" scripts/smoke_isaac51.py --backend taxim --headless
'
