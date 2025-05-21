#!/bin/bash

#LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/aa609734/.conda/envs/FNO-UNet/lib/python3.11/site-packages/nvidia/cufft/lib" /home/aa609734/.conda/envs/FNO-UNet/bin/python /home/aa609734/Projects/FNO-UNet/fun/train.py\
#    --dataset=ellipses-256x256\
#    --model=classic\
#    --batch-size=32\
#    --max-epochs=10\
#    --device=cuda\
#    --num-workers=0\
#    $@

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/aa609734/.conda/envs/FNO-UNet/lib/python3.11/site-packages/nvidia/cufft/lib" /home/aa609734/.conda/envs/FNO-UNet/bin/python /home/aa609734/Projects/FNO-UNet/fun/train.py\
    --dataset=ellipses-mixed\
    --model=classic\
    --batch-size=32\
    --test-only\
    --weights=runs/64x64/weights/final.pt\
    --device=cuda\
    --num-workers=0\
    $@ 