#!/bin/bash

sbatch -J 'unet-32x32'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-32x32'    --dataset=ellipses-32x32    --model=unet --unet-convs=0
sbatch -J 'unet-64x64'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-64x64'    --dataset=ellipses-64x64    --model=unet --unet-convs=0
sbatch -J 'unet-128x128'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-128x128'  --dataset=ellipses-128x128  --model=unet --unet-convs=0
sbatch -J 'unet-256x256'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-256x256'  --dataset=ellipses-256x256  --model=unet --unet-convs=0
sbatch -J 'unet-mixed'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-mixed'    --dataset=ellipses-mixed    --model=unet --unet-convs=0

sbatch -J 'unet-32x32'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-32x32'    --dataset=ellipses-32x32    --model=unet-custom --pooling-base-size=64
sbatch -J 'unet-64x64'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-64x64'    --dataset=ellipses-64x64    --model=unet-custom --pooling-base-size=64
sbatch -J 'unet-128x128'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-128x128'  --dataset=ellipses-128x128  --model=unet-custom --pooling-base-size=64
sbatch -J 'unet-256x256'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-256x256'  --dataset=ellipses-256x256  --model=unet-custom --pooling-base-size=64
sbatch -J 'unet-mixed'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-mixed'    --dataset=ellipses-mixed    --model=unet-custom --pooling-base-size=64

sbatch -J 'interp-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='interp-64x64'   --dataset=ellipses-64x64   --model=unet-interp --interp-mode --batch-size=6
sbatch -J 'interp-128x128' $(dirname "$0")/omni.submit.sh --forced-run-name='interp-128x128' --dataset=ellipses-128x128 --model=unet-interp --interp-mode --batch-size=6
sbatch -J 'interp-256x256' $(dirname "$0")/omni.submit.sh --forced-run-name='interp-256x256' --dataset=ellipses-256x256 --model=unet-interp --interp-mode --batch-size=6
sbatch -J 'interp-mixed'   $(dirname "$0")/omni.submit.sh --forced-run-name='interp-mixed'   --dataset=ellipses-mixed   --model=unet-interp --interp-mode --batch-size=6

# sbatch -J 'interp-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='interp-64x64'   --dataset=ellipses-sweep --model=unet-interp --interp-mode --batch-size=6