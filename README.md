# FNO-UNets: UNets based on fourier neural operators

## Setup

```bash
conda create -n FNO-UNet python=3.11
conda activate FNO-UNet
git clone --recurse-submodules https://github.com/AlexanderAuras/FNO-UNet/
cd FNO-UNet
conda install ninja
conda install -c nvidia/label/cuda-12.1.0 nvidia/label/cuda-12.1.0::cuda-toolkit
pip install poetry
poetry install -v --with dev
```
