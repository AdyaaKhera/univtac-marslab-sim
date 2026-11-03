#!/bin/bash
# Train LeRobot Diffusion Policy on a converted UniVTAC dataset.
# Submit from repo root: sbatch -A <account> slurm/train_dp.sbatch [task]
#SBATCH -J dp-train
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o ../logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
TASK=${1:-lift_can}
RUN=dp_${TASK}_$(date +%m%d_%H%M)
nvidia-smi -L
lerobot_py -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=univtac/$TASK --dataset.root=$WORK/datasets/$TASK \
  --policy.type=diffusion --policy.push_to_hub=false \
  --policy.crop_shape="[116,116]" \
  --batch_size=64 --steps=100000 --num_workers=12 \
  --save_freq=20000 --log_freq=200 \
  --output_dir=$WORK/outputs/$RUN --job_name=$RUN --wandb.enable=false
