#!/bin/bash

#SBATCH -vv
#SBATCH --requeue

#SBATCH --job-name="FNO-UNet"
#SBATCH --partition=gpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --time=0-23:59:59
#SBATCH --begin=now
#SBATCH --signal=30@300

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

#SBATCH --mail-type=END,FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=alexander.auras@uni-siegen.de

#SBATCH --chdir=/home/aa609734/Projects/FNO-UNet

module load miniconda3
source /cm/shared/omni/apps/miniconda3/bin/activate FNO-UNet

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aa609734/.conda/envs/FNO-UNet/lib/python3.11/site-packages/nvidia/cufft/lib /home/aa609734/.conda/envs/FNO-UNet/bin/python /home/aa609734/Projects/FNO-UNet/fun/train.py --dataset=ellipses-mixed --no-compile --model=interp --batch-size=32 --max-epochs=10 --device=cuda --num-workers=0 $@