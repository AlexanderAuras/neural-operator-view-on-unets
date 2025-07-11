#!/bin/bash
#SBATCH --partition=psgpu
#SBATCH --time=24:00:00                           # Maximum time requested
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --job-name=sweeps_resize
#SBATCH --mail-type=END,FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=samira.kabri@desy.de         # Email to which notifications will be sent.
#SBATCH --constraint=A100

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

NUM='1'

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-64x64-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UNet-64x64-$NUM/weights/best-all.pt 
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-128x128-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UNet-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-256x256-$NUM" --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNet-256x256-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNet-mixed-$NUM"   --model=unet-custom --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNet-mixed-$NUM/weights/best-all.pt

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="CNO-64x64-$NUM"    --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/CNO-64x64-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="CNO-128x128-$NUM"  --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/CNO-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="CNO-256x256-$NUM"  --model=cno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/CNO-256x256-$NUM/weights/best-all.pt 

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-64x64-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/Diff-64x64-$NUM/weights/best-all.pt 
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-128x128-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/Diff-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-256x256-$NUM" --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/Diff-256x256-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="Diff-mixed-$NUM"   --model=diff        --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/Diff-mixed-$NUM/weights/best-all.pt 

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-64x64-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/FNO-64x64-$NUM/weights/best-all.pt
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-128x128-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/FNO-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-256x256-$NUM"  --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/FNO-256x256-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="FNO-mixed-$NUM"    --model=specRes     --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/FNO-mixed-$NUM/weights/best-all.pt

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UFNO-64x64-$NUM/weights/best-all.pt 
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UFNO-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM" --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UFNO-256x256-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"   --model=specResU    --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UFNO-mixed-$NUM/weights/best-all.pt 

python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-64x64-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=64  --test-only --weights=$(dirname "$0")/../runs/UNO-64x64-$NUM/weights/best-all.pt
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-128x128-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=128 --test-only --weights=$(dirname "$0")/../runs/UNO-128x128-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-256x256-$NUM"  --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNO-256x256-$NUM/weights/best-all.pt  
python ../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UNO-mixed-$NUM"    --model=uno         --dataset=ellipses-sweep --batch-size=32 --resize-input-size=256 --test-only --weights=$(dirname "$0")/../runs/UNO-mixed-$NUM/weights/best-all.pt
