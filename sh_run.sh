#!/bin/bash
#SBATCH --job-name=DL
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -e %j.err
#SBATCH -o %j.log
#SBATCH -p GPUFEE05
#SBATCH --constraint=Python

source /gpfs/home/E22201116/anaconda3/bin/activate
conda activate sapt
srun python run_seq2seq01.py