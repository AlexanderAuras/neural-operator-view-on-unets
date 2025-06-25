#!/bin/bash

sbatch -J 'unet-64x64'     $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-unet-64x64'    --dataset=ellipses-64x64    --model=unet
sbatch -J 'unet-128x128'   $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-unet-128x128'  --dataset=ellipses-128x128  --model=unet
sbatch -J 'unet-256x256'   $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-unet-256x256'  --dataset=ellipses-256x256  --model=unet
sbatch -J 'unet-mixed'     $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-unet-mixed'    --dataset=ellipses-mixed    --model=unet

sbatch -J 'fno-64x64'      $(dirname "$0")/omni.submit3.sh --forced-run-name='smooth-fno-64x64'    --dataset=ellipses-64x64    --model=heat        --batch-size=1 --use-checkpointing
sbatch -J 'fno-128x128'    $(dirname "$0")/omni.submit3.sh --forced-run-name='smooth-fno-128x128'  --dataset=ellipses-128x128  --model=heat        --batch-size=1 --use-checkpointing
sbatch -J 'fno-256x256'    $(dirname "$0")/omni.submit3.sh --forced-run-name='smooth-fno-256x256'  --dataset=ellipses-256x256  --model=heat        --batch-size=1 --use-checkpointing
sbatch -J 'fno-mixed'      $(dirname "$0")/omni.submit3.sh --forced-run-name='smooth-fno-mixed'    --dataset=ellipses-mixed    --model=heat        --batch-size=1 --use-checkpointing

sbatch -J 'dncnn-256x256'  $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-dncnn-256x256' --dataset=ellipses-256x256  --model=dncnn                      --use-checkpointing
sbatch -J 'dncnn-mixed'    $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-dncnn-mixed'   --dataset=ellipses-mixed    --model=dncnn                      --use-checkpointing

sbatch -J 'interp-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-interp-64x64'   --dataset=ellipses-64x64   --model=unet-interp --batch-size=6
sbatch -J 'interp-128x128' $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-interp-128x128' --dataset=ellipses-128x128 --model=unet-interp --batch-size=6
sbatch -J 'interp-256x256' $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-interp-256x256' --dataset=ellipses-256x256 --model=unet-interp --batch-size=6
sbatch -J 'interp-mixed'   $(dirname "$0")/omni.submit.sh --forced-run-name='smooth-interp-mixed'   --dataset=ellipses-mixed   --model=unet-interp --batch-size=6