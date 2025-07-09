#!/bin/bash

#### FINAL #####

# sbatch -J 'unet-32x32'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-32x32'   --dataset=ellipses-32x32    --model=unet --unet-convs=0
# sbatch -J 'unet-64x64'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-64x64'   --dataset=ellipses-64x64    --model=unet --unet-convs=0
# sbatch -J 'unet-128x128'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-128x128' --dataset=ellipses-128x128  --model=unet --unet-convs=0
# sbatch -J 'unet-256x256'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-256x256' --dataset=ellipses-256x256  --model=unet --unet-convs=0
# sbatch -J 'unet-mixed'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-mixed'   --dataset=ellipses-mixed    --model=unet --unet-convs=0

# sbatch -J 'unet-pool-32x32'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-32x32-pool'   --dataset=ellipses-32x32   --model=unet-custom --pooling-base-size=64
# sbatch -J 'unet-pool-64x64'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-64x64-pool'   --dataset=ellipses-64x64   --model=unet-custom --pooling-base-size=64
# sbatch -J 'unet-pool-128x128'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-128x128-pool' --dataset=ellipses-128x128 --model=unet-custom --pooling-base-size=64
# sbatch -J 'unet-pool-256x256'   $(dirname "$0")/omni.submit.sh --forced-run-name='unet-256x256-pool' --dataset=ellipses-256x256 --model=unet-custom --pooling-base-size=64
# sbatch -J 'unet-pool-mixed'     $(dirname "$0")/omni.submit.sh --forced-run-name='unet-mixed-pool'   --dataset=ellipses-mixed   --model=unet-custom --pooling-base-size=64

# sbatch -J 'interp-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='interp-64x64'   --dataset=ellipses-64x64   --model=unet-interp --interp-mode --batch-size=6
# sbatch -J 'interp-128x128' $(dirname "$0")/omni.submit.sh --forced-run-name='interp-128x128' --dataset=ellipses-128x128 --model=unet-interp --interp-mode --batch-size=6
# sbatch -J 'interp-256x256' $(dirname "$0")/omni.submit.sh --forced-run-name='interp-256x256' --dataset=ellipses-256x256 --model=unet-interp --interp-mode --batch-size=6
# sbatch -J 'interp-mixed'   $(dirname "$0")/omni.submit.sh --forced-run-name='interp-mixed'   --dataset=ellipses-mixed   --model=unet-interp --interp-mode --batch-size=6

#### SWEEP1 #####

# sbatch -J 'SpatResU64'        $(dirname "$0")/omni.submit.sh --forced-run-name='SpatResU64'        --dataset=ellipses-sweep --model=spatResU  --test-only --weights='runs/CNN-Interp/SpatResU64/final64.pt'
# sbatch -J 'SpatResU128'       $(dirname "$0")/omni.submit.sh --forced-run-name='SpatResU128'       --dataset=ellipses-sweep --model=spatResU  --test-only --weights='runs/CNN-Interp/SpatResU128/final128.pt'
# sbatch -J 'SpatResU256'       $(dirname "$0")/omni.submit.sh --forced-run-name='SpatResU256'       --dataset=ellipses-sweep --model=spatResU  --test-only --weights='runs/CNN-Interp/SpatResU256/final256.pt'

# sbatch -J 'SmallResUNet64'    $(dirname "$0")/omni.submit.sh --forced-run-name='SmallResUNet64'    --dataset=ellipses-sweep --model=smallResU --test-only --weights='runs/CNN-Transposed/smallResUNet64/final64.pt'
# sbatch -J 'SmallResUNet128'   $(dirname "$0")/omni.submit.sh --forced-run-name='SmallResUNet128'   --dataset=ellipses-sweep --model=smallResU --test-only --weights='runs/CNN-Transposed/smallResUNet128/final128.pt'
# sbatch -J 'SmallResUNet256'   $(dirname "$0")/omni.submit.sh --forced-run-name='SmallResUNet256'   --dataset=ellipses-sweep --model=smallResU --test-only --weights='runs/CNN-Transposed/smallResUNet256/final256.pt'
# sbatch -J 'SmallResUNetMixed' $(dirname "$0")/omni.submit.sh --forced-run-name='SmallResUNetMixed' --dataset=ellipses-sweep --model=smallResU --test-only --weights='runs/CNN-Transposed/smallResUNetMixed/finalMixed.pt'

# sbatch -J 'SpecRes64'    $(dirname "$0")/omni.submit.sh --forced-run-name='SpecRes64'    --dataset=ellipses-sweep --model=specRes --test-only --weights='runs/FNO-ResNets/SpecRes64/final64.pt'
# sbatch -J 'SpecRes128'   $(dirname "$0")/omni.submit.sh --forced-run-name='SpecRes128'   --dataset=ellipses-sweep --model=specRes --test-only --weights='runs/FNO-ResNets/SpecRes128/final128.pt'
# sbatch -J 'SpecRes256'   $(dirname "$0")/omni.submit.sh --forced-run-name='SpecRes256'   --dataset=ellipses-sweep --model=specRes --test-only --weights='runs/FNO-ResNets/SpecRes256/final256.pt'
# sbatch -J 'SpecResMixed' $(dirname "$0")/omni.submit.sh --forced-run-name='SpecResMixed' --dataset=ellipses-sweep --model=specRes --test-only --weights='runs/FNO-ResNets/SpecResMixed/finalMixed.pt'

# sbatch -J 'SpecResU64'    $(dirname "$0")/omni.submit.sh --forced-run-name='SpecResU64'    --dataset=ellipses-sweep --model=specResU --test-only --weights='runs/FNO-ResUNets/SpecResU64/final64.pt'
# sbatch -J 'SpecResU128'   $(dirname "$0")/omni.submit.sh --forced-run-name='SpecResU128'   --dataset=ellipses-sweep --model=specResU --test-only --weights='runs/FNO-ResUNets/SpecResU128/final128.pt'
# sbatch -J 'SpecResU256'   $(dirname "$0")/omni.submit.sh --forced-run-name='SpecResU256'   --dataset=ellipses-sweep --model=specResU --test-only --weights='runs/FNO-ResUNets/SpecResU256/final256.pt'
# sbatch -J 'SpecResUMixed' $(dirname "$0")/omni.submit.sh --forced-run-name='SpecResUMixed' --dataset=ellipses-sweep --model=specResU --test-only --weights='runs/FNO-ResUNets/SpecResUMixed/finalMixed.pt'

#### SWEEP2 #####

# sbatch -J 'fewconv-unet-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-fewconv-unet-64x64'   --dataset=ellipses-sweep --model=unet --unet-convs=0 --test-only --weights='runs/__final/fewconv-unet-64x64/weights/final.pt'
# sbatch -J 'fewconv-unet-128x128' $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-fewconv-unet-128x128' --dataset=ellipses-sweep --model=unet --unet-convs=0 --test-only --weights='runs/__final/fewconv-unet-128x128/weights/final.pt'
# sbatch -J 'fewconv-unet-256x256' $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-fewconv-unet-256x256' --dataset=ellipses-sweep --model=unet --unet-convs=0 --test-only --weights='runs/__final/fewconv-unet-256x256/weights/final.pt'
# sbatch -J 'fewconv-unet-mixed'   $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-fewconv-unet-mixed'   --dataset=ellipses-sweep --model=unet --unet-convs=0 --test-only --weights='runs/__final/fewconv-unet-mixed/weights/final.pt'

# sbatch -J 'unet-pool-64x64'   $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-unet-64x64-pool'   --dataset=ellipses-sweep --model=unet-custom --pooling-base-size=64 --test-only --weights='runs/unet-64x64-pool/weights/final.pt'
# sbatch -J 'unet-pool-128x128' $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-unet-128x128-pool' --dataset=ellipses-sweep --model=unet-custom --pooling-base-size=64 --test-only --weights='runs/unet-128x128-pool/weights/final.pt'
# sbatch -J 'unet-pool-256x256' $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-unet-256x256-pool' --dataset=ellipses-sweep --model=unet-custom --pooling-base-size=64 --test-only --weights='runs/unet-256x256-pool/weights/final.pt'
sbatch -J 'unet-pool-mixed'   $(dirname "$0")/omni.submit.sh --forced-run-name='sweep-unet-mixed-pool'   --dataset=ellipses-sweep --model=unet-custom --pooling-base-size=64 --test-only --weights='runs/unet-mixed-pool/weights/final.pt'