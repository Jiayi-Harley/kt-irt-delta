#!/bin/bash -l
# SGE job script for the multi-task Δθ run (方向①) on Myriad.
# Usage:  qsub run_mt.sh [SEEDS] [EMB] [EPOCHS] [CORR_W]
#   e.g.  qsub run_mt.sh 0,1,2 256 15 1.0     (multi-task smoke, 3 seeds)
#         qsub run_mt.sh 0,1,2,3,4 256 15 0.0 (control: correctness off)
#$ -l gpu=1
#$ -l h_rt=3:00:00
#$ -l mem=16G
#$ -N kt_mt
#$ -cwd
module purge
module load pytorch/2.1.0/gpu
export PYTHONPATH=$PWD/pykt-toolkit:$PYTHONPATH

SEEDS=${1:-0,1,2,3,4,5,6,7,8,9}
EMB=${2:-256}
EP=${3:-15}
CORR=${4:-1.0}
echo "SEEDS=$SEEDS EMB=$EMB EPOCHS=$EP CORR_W=$CORR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u models/qikt_delta_mt.py --emb_size $EMB --epochs $EP --seeds $SEEDS --corr_w $CORR