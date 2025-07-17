#!/bin/bash
#SBATCH --partition=psgpu
#SBATCH --time=24:00:00                           # Maximum time requested
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --job-name=sweeps_resize
#SBATCH --mail-type=END,FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=samira.kabri@desy.de         # Email to which notifications will be sent.

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

NUM='1'

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM-sweep"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNet-64x64-$NUM/weights/best-all.pt 
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM-sweep" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNet-128x128-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM-sweep" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNet-256x256-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM-sweep"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNet-mixed-$NUM/weights/best-all.pt

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-64x64-$NUM-sweep"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/Diff-64x64-$NUM/weights/best-all.pt 
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-128x128-$NUM-sweep" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/Diff-128x128-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-256x256-$NUM-sweep" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/Diff-256x256-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-mixed-$NUM-sweep"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/Diff-mixed-$NUM/weights/best-all.pt 

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM-sweep"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/FNO-64x64-$NUM/weights/best-all.pt
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM-sweep"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/FNO-128x128-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM-sweep"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/FNO-256x256-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM-sweep"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/FNO-mixed-$NUM/weights/best-all.pt

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM-sweep"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UFNO-64x64-$NUM/weights/best-all.pt 
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM-sweep" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UFNO-128x128-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM-sweep" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UFNO-256x256-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM-sweep"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UFNO-mixed-$NUM/weights/best-all.pt 

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-64x64-$NUM-sweep"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNO-64x64-$NUM/weights/best-all.pt
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-128x128-$NUM-sweep"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNO-128x128-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-256x256-$NUM-sweep"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNO-256x256-$NUM/weights/best-all.pt  
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-mixed-$NUM-sweep"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=/home/kabrisam/Code/FNO-UNet/runs/UNO-mixed-$NUM/weights/best-all.pt
