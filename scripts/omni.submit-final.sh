#!/bin/bash

NUM='14'

#### TRAINING #####

# sbatch -J "UNet-64x64-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM"     --model=unet-custom --dataset=ellipses-64x64   --batch-size=32
# sbatch -J "UNet-128x128-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM"   --model=unet-custom --dataset=ellipses-128x128 --batch-size=32
# sbatch -J "UNet-256x256-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM"   --model=unet-custom --dataset=ellipses-256x256 --batch-size=32
# sbatch -J "UNet-mixed-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM"     --model=unet-custom --dataset=ellipses-mixed   --batch-size=32
# # sbatch -J "UNet-rmixed64-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-rmixed64-$NUM"  --model=unet-custom --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=64
# # sbatch -J "UNet-rmixed128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-rmixed128-$NUM" --model=unet-custom --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=128
# # sbatch -J "UNet-rmixed256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-rmixed256-$NUM" --model=unet-custom --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=256
# 
# sbatch -J "CNO-64x64-$NUM"      $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-64x64-$NUM"      --model=cno         --dataset=ellipses-64x64   --batch-size=32
# sbatch -J "CNO-128x128-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-128x128-$NUM"    --model=cno         --dataset=ellipses-128x128 --batch-size=32
# sbatch -J "CNO-256x256-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-256x256-$NUM"    --model=cno         --dataset=ellipses-256x256 --batch-size=32
# # sbatch -J "CNO-rmixed64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-rmixed64-$NUM"   --model=cno         --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=64
# # sbatch -J "CNO-rmixed128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-rmixed128-$NUM"  --model=cno         --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=128
# # sbatch -J "CNO-rmixed256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-rmixed256-$NUM"  --model=cno         --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=256
# 
# sbatch -J "Diff-64x64-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-64x64-$NUM"     --model=diff        --dataset=ellipses-64x64   --batch-size=32
# sbatch -J "Diff-128x128-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-128x128-$NUM"   --model=diff        --dataset=ellipses-128x128 --batch-size=32
# sbatch -J "Diff-256x256-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-256x256-$NUM"   --model=diff        --dataset=ellipses-256x256 --batch-size=32
# sbatch -J "Diff-mixed-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-mixed-$NUM"     --model=diff        --dataset=ellipses-mixed   --batch-size=32
# # sbatch -J "Diff-rmixed64-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-rmixed64-$NUM"  --model=diff        --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=64
# # sbatch -J "Diff-rmixed128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-rmixed128-$NUM" --model=diff        --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=128
# # sbatch -J "Diff-rmixed256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-rmixed256-$NUM" --model=diff        --dataset=ellipses-mixed   --batch-size=32 --resize-input-size=256
# 
# sbatch -J "FNO-64x64-$NUM"      $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM"      --model=specRes     --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "FNO-128x128-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM"    --model=specRes     --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "FNO-256x256-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM"    --model=specRes     --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "FNO-mixed-$NUM"      $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM"      --model=specRes     --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# # sbatch -J "FNO-rmixed64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-rmixed64-$NUM"   --model=specRes     --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=64
# # sbatch -J "FNO-rmixed128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-rmixed128-$NUM"  --model=specRes     --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=128
# # sbatch -J "FNO-rmixed256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-rmixed256-$NUM"  --model=specRes     --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=256
# 
# sbatch -J "UFNO-64x64-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"     --model=specResU    --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UFNO-128x128-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM"   --model=specResU    --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UFNO-256x256-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM"   --model=specResU    --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UFNO-mixed-$NUM"     $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"     --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# # sbatch -J "UFNO-rmixed64-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-rmixed64-$NUM"  --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=64
# # sbatch -J "UFNO-rmixed128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-rmixed128-$NUM" --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=128
# # sbatch -J "UFNO-rmixed256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-rmixed256-$NUM" --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=256
# 
# sbatch -J "UNO-64x64-$NUM"      $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-64x64-$NUM"      --model=uno         --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UNO-128x128-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-128x128-$NUM"    --model=uno         --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UNO-256x256-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-256x256-$NUM"    --model=uno         --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# sbatch -J "UNO-mixed-$NUM"      $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-mixed-$NUM"      --model=uno         --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
# # sbatch -J "UNO-rmixed64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-rmixed64-$NUM"   --model=uno         --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=64
# # sbatch -J "UNO-rmixed128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-rmixed128-$NUM"  --model=uno         --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=128
# # sbatch -J "UNO-rmixed256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-rmixed256-$NUM"  --model=uno         --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state --resize-input-size=256




#### SWEEPS 1 #####

sbatch -J "Sweep-UNet-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNet-64x64-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNet-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNet-128x128-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNet-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNet-256x256-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNet-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNet-mixed-$NUM/weights/best-all.pt

sbatch -J "Sweep-Diff-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-64x64-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/Diff-64x64-$NUM/weights/best-all.pt
sbatch -J "Sweep-Diff-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-128x128-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/Diff-128x128-$NUM/weights/best-all.pt
sbatch -J "Sweep-Diff-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-256x256-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/Diff-256x256-$NUM/weights/best-all.pt
sbatch -J "Sweep-Diff-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-mixed-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/Diff-mixed-$NUM/weights/best-all.pt

sbatch -J "Sweep-FNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/FNO-64x64-$NUM/weights/best-all.pt
sbatch -J "Sweep-FNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/FNO-128x128-$NUM/weights/best-all.pt
sbatch -J "Sweep-FNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/FNO-256x256-$NUM/weights/best-all.pt
sbatch -J "Sweep-FNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/FNO-mixed-$NUM/weights/best-all.pt

sbatch -J "Sweep-UFNO-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UFNO-64x64-$NUM/weights/best-all.pt
sbatch -J "Sweep-UFNO-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UFNO-128x128-$NUM/weights/best-all.pt
sbatch -J "Sweep-UFNO-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UFNO-256x256-$NUM/weights/best-all.pt
sbatch -J "Sweep-UFNO-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UFNO-mixed-$NUM/weights/best-all.pt

sbatch -J "Sweep-UNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-64x64-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNO-64x64-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-128x128-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNO-128x128-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-256x256-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNO-256x256-$NUM/weights/best-all.pt
sbatch -J "Sweep-UNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-mixed-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --test-only --weights=$(dirname "$0")/../runs/UNO-mixed-$NUM/weights/best-all.pt



exit 0
#### SWEEPS 2 #####

sbatch -J "RSweep-UNet-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UNet-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNet-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UNet-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNet-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNet-256x256-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNet-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNet-mixed-$NUM/weights/best-all.pt

sbatch -J "RSweep-CNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-64x64-$NUM"    --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/CNO-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-CNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-128x128-$NUM"  --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/CNO-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-CNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="CNO-256x256-$NUM"  --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/CNO-256x256-$NUM/weights/best-all.pt

sbatch -J "RSweep-Diff-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-64x64-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/Diff-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-Diff-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-128x128-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/Diff-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-Diff-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-256x256-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/Diff-256x256-$NUM/weights/best-all.pt
sbatch -J "RSweep-Diff-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="Diff-mixed-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/Diff-mixed-$NUM/weights/best-all.pt

sbatch -J "RSweep-FNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/FNO-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-FNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/FNO-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-FNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/FNO-256x256-$NUM/weights/best-all.pt
sbatch -J "RSweep-FNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/FNO-mixed-$NUM/weights/best-all.pt

sbatch -J "RSweep-UFNO-64x64-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UFNO-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-UFNO-128x128-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UFNO-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-UFNO-256x256-$NUM" $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UFNO-256x256-$NUM/weights/best-all.pt
sbatch -J "RSweep-UFNO-mixed-$NUM"   $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UFNO-mixed-$NUM/weights/best-all.pt

sbatch -J "RSweep-UNO-64x64-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-64x64-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UNO-64x64-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNO-128x128-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-128x128-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UNO-128x128-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNO-256x256-$NUM"  $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-256x256-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNO-256x256-$NUM/weights/best-all.pt
sbatch -J "RSweep-UNO-mixed-$NUM"    $(dirname "$0")/omni.submit.sh --model-save-freq=9999 --forced-run-name="UNO-mixed-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNO-mixed-$NUM/weights/best-all.pt
