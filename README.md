# FNO-UNets: UNets based on fourier neural operators

## Setup

```bash
conda create -n FNO-UNet python=3.11
conda activate FNO-UNet
git clone https://github.com/AlexanderAuras/FNO-UNet/
cd FNO-UNet
git submodule init
git submodule update
pip install poetry
poetry install -v --with dev
```
