#!/bin/bash
#SBATCH --partition=psgpu
#SBATCH --time=24:00:00                           # Maximum time requested
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --job-name=fno_train
#SBATCH --mail-type=END,FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=samira.kabri@desy.de         # Email to which notifications will be sent.

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

NUM='2'

python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-64x64-$NUM"     --model=specResU    --dataset=ellipses-64x64   --batch-size=1  --accumulation-steps=32 --cpu-opt-state 
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-128x128-$NUM"   --model=specResU    --dataset=ellipses-128x128 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-256x256-$NUM"   --model=specResU    --dataset=ellipses-256x256 --batch-size=1  --accumulation-steps=32 --cpu-opt-state
python ../../fun/__main__.py --no-compile --max-epochs=10 --devices cuda:0 --num-workers=0 --model-save-freq=9999 --forced-run-name="UFNO-mixed-$NUM"     --model=specResU    --dataset=ellipses-mixed   --batch-size=1  --accumulation-steps=32 --cpu-opt-state
