#!/bin/bash

#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --account=kwf@v100
#SBATCH --time=20:00:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --error=slurm-err-%j.out
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g



srun python main.py 0 $1 maxcut $SLURM_ARRAY_TASK_ID 999 10 20 48 2 64 10 ring 0.5 1 NoTest False -1 XT
