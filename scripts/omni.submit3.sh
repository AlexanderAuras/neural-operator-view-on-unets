#!/bin/bash

#SBATCH -vv
#SBATCH --requeue

#SBATCH --job-name="FNO-UNet"
#SBATCH --partition=gpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=3
#SBATCH --gres=gpu:3
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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aa609734/.conda/envs/FNO-UNet/lib/python3.11/site-packages/nvidia/cufft/lib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/aa609734/.conda/envs/FNO-UNet/bin/python /home/aa609734/Projects/FNO-UNet/fun/__main__.py\
    --no-compile\
    --max-epochs=10\
    --devices cuda:0 cuda:1\
    --num-workers=0\
    $@
    # --batch-size=32\
    # --dataset=ellipses-mixed\
    # --model=fno\
    # --test-only\
    # --weights=runs/mixed/weights/final.pt\