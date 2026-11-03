#!/bin/bash
# Verify the UniVTAC env and the LeRobot <-> UniVTAC bridge.
# Submit from repo root: sbatch -A <account> slurm/smoke_univtac.sbatch
#SBATCH -J univtac-smoke
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ../logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$UNIVTAC_ROOT"
nvidia-smi -L
in_univtac bash -c '
  set -euo pipefail
  source "$UNIVTAC_CONDA/etc/profile.d/conda.sh"
  UNIVTAC_CUDA_ARCH=80 bash scripts/install.sh --check
  "$CONDA_ENVS_PATH/UniVTAC/bin/python" -m pip install -q pyzmq   # sim_server.py
  "$CONDA_ENVS_PATH/UniVTAC/bin/python" scripts/smoke_isaac51.py --backend taxim --headless
'
echo "== bridge: LeRobot env -> UniVTAC sim (reset + 20 hold-position steps) =="
cd "$SLURM_SUBMIT_DIR"
lerobot_py - <<'PY'
import time, cv2, numpy as np
from lerobot_env_univtac.env import UniVTACEnv
t = time.time(); env = UniVTACEnv("lift_can"); print(f"sim up in {time.time()-t:.0f}s, step_lim={env._max_episode_steps}")
obs, _ = env.reset(seed=0)
print({k: v.shape for k, v in obs["pixels"].items()}, "agent_pos", np.round(obs["agent_pos"], 3))
t = time.time()
for _ in range(20):
    obs, r, term, trunc, info = env.step(obs["agent_pos"])
print(f"20 steps in {time.time()-t:.1f}s, success={info['is_success']}")
cv2.imwrite("../logs/bridge_check.jpg", np.hstack([env.render()[:, :, :]]))
cv2.imwrite("../logs/bridge_check_obs.jpg", np.hstack(list(obs["pixels"].values())))
env.close(); print("BRIDGE OK")
PY
