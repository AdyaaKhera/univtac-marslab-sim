# Shared environment for all jobs. Source this file; it is not a job script.
# WORK = workspace on scratch holding UniVTAC/, this repo, conda envs, containers
export WORK="${WORK:-$RCAC_SCRATCH/univtac}"
export UNIVTAC_ROOT="$WORK/UniVTAC"
export CONDA_ENVS_PATH="$WORK/conda/envs"   # module default is $HOME (25 GB quota)
export CONDA_PKGS_DIRS="$WORK/conda/pkgs"
export PIP_CACHE_DIR="$WORK/cache/pip"
export HF_HOME="$WORK/cache/hf"
export OMNI_KIT_ACCEPT_EULA=YES

# UniVTAC env runs inside an Ubuntu 22.04 container (Isaac Sim 5.1 needs glibc 2.35),
# using a Miniforge install on scratch. See container/univtac.def.
export UNIVTAC_SIF="${UNIVTAC_SIF:-$WORK/containers/univtac.sif}"
export UNIVTAC_CONDA="$WORK/miniforge"
export UNIVTAC_PYTHON="$WORK/univtac-marslab-sim/slurm/univtac_python"  # used by lerobot_env_univtac
in_univtac() { apptainer exec --nv -B "$WORK" "$UNIVTAC_SIF" "$@"; }

# Python 3.12 env with LeRobot 0.6.1 (host). Preload its libstdc++ so torchcodec
# (video decoding) does not pick up the older system one.
export LEROBOT_ENV="$CONDA_ENVS_PATH/lerobot"
lerobot_py() { LD_PRELOAD="$LEROBOT_ENV/lib/libstdc++.so.6" "$LEROBOT_ENV/bin/python" "$@"; }
