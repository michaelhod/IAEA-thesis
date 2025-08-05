#!/bin/bash
#SBATCH --gres=gpu:1                  # one GPU
#SBATCH --partition=AMD7-A100-T       # omit for “any GPU”
#SBATCH --time=2-00:00:00             # 2 days max; adjust if needed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mjh24@ic.ac.uk
#SBATCH --output=job_%j.out           # %j = job-ID

# 1--Set up environment
export PATH=/vol/bitbucket/mjh24/.venv/bin:$PATH
source /vol/bitbucket/mjh24/.venv/bin/activate
. /vol/cuda/12.0.0/setup.sh           # matches current PyTorch wheels

# 2--Run your notebook (now a .py)
python /vol/bitbucket/mjh24/IAEA-thesis/GAT.py