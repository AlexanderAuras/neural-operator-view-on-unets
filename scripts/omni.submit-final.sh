#!/bin/bash

NUM='1'

#### TRAINING #####

#sbatch -J "UNet-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM"   --model=unet-custom --dataset=ellipses-64x64   --batch-size=32
#sbatch -J "UNet-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM" --model=unet-custom --dataset=ellipses-128x128 --batch-size=32
#sbatch -J "UNet-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM" --model=unet-custom --dataset=ellipses-256x256 --batch-size=32
#sbatch -J "UNet-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM"   --model=unet-custom --dataset=ellipses-mixed   --batch-size=32

sbatch -J "FNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM"    --model=specRes     --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32
sbatch -J "FNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM"  --model=specRes     --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32
sbatch -J "FNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM"  --model=specRes     --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32
sbatch -J "FNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM"    --model=specRes     --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32

#sbatch -J "UFNO-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"   --model=specResU    --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32
sbatch -J "UFNO-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM" --model=specResU    --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32
sbatch -J "UFNO-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM" --model=specResU    --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32
sbatch -J "UFNO-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"   --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32



#### SWEEPS #####

# sbatch -J "Sweep-UNet-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UNet-64x64-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UNet-64x64-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UNet-128x128-$NUM" $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UNet-128x128-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UNet-128x128-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UNet-256x256-$NUM" $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UNet-256x256-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UNet-256x256-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UNet-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UNet-mixed-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UNet-mixed-$NUM/weights/best-all.pt"

# sbatch -J "Sweep-FNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-FNO-64x64-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/FNO-64x64-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-FNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-FNO-128x128-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/FNO-128x128-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-FNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-FNO-256x256-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/FNO-256x256-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-FNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-FNO-mixed-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/FNO-mixed-$NUM/weights/best-all.pt"

# sbatch -J "Sweep-UFNO-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UFNO-64x64-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UFNO-64x64-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UFNO-128x128-$NUM" $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UFNO-128x128-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UFNO-128x128-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UFNO-256x256-$NUM" $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UFNO-256x256-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UFNO-256x256-$NUM/weights/best-all.pt"
# sbatch -J "Sweep-UFNO-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --forced-run-name="Sweep-UFNO-mixed-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --model-save-freq=9999 --test-only --weights="runs/UFNO-mixed-$NUM/weights/best-all.pt"



