#!/bin/bash

_CONDA_ENV_NAME=FNO-UNet
_PROJECT_PATH='/home/aa609734/Projects/FNO-UNet'

if [[ $- == *i* ]]; then
    SCRIPT_FILE_NAME=$(uuid).sh
    echo -e "#!/bin/bash\n\
             rm -f $SCRIPT_FILE_NAME\n\
             source ~/.bashrc\n\
             module load miniconda3\n\
             source /cm/shared/omni/apps/miniconda3/bin/activate $_CONDA_ENV_NAME\n\
             alias python=$HOME/.conda/envs/$_CONDA_ENV_NAME/bin/python\n\
             alias python3=$HOME/.conda/envs/$_CONDA_ENV_NAME/bin/python3\n\
             alias pip=$HOME/.conda/envs/$_CONDA_ENV_NAME/bin/pip\n\
             alias pip3=$HOME/.conda/envs/$_CONDA_ENV_NAME/bin/pip3"\n\
        > $HOME/$SCRIPT_FILE_NAME
    srun --job-name='dev'\
     --partition=gpu\
     --ntasks=1\
     --cpus-per-task=8\
     --mem=32G\
     --gpus-per-task=1\
     --gres=gpu:1\
     --time=0-23:59:59\
     --chdir=$_PROJECT_PATH\
     --pty /bin/bash --init-file $HOME/$SCRIPT_FILE_NAME
else
    echo 'Please execute in an interactive shell, e.g. via `source`'
fi