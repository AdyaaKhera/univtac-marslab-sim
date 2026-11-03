#!/bin/bash
# Evaluate a trained policy in UniVTAC through lerobot-eval (--env.type=univtac).
# Submit from repo root: sbatch -A <account> slurm/eval_dp.sbatch <training output dir> [task] [episodes]
#SBATCH -J dp-eval
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o ../logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
RUN_DIR=$1; CKPT=$RUN_DIR/checkpoints/last/pretrained_model; TASK=${2:-lift_can}; N=${3:-50}
nvidia-smi -L
lerobot_py -m lerobot.scripts.lerobot_eval \
  --policy.path=$CKPT --env.type=univtac --env.task=$TASK \
  --eval.n_episodes=$N --eval.batch_size=1 --eval.use_async_envs=false \
  --output_dir=$WORK/eval/$(basename $RUN_DIR)_$(date +%m%d_%H%M)
