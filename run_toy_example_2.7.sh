#!/bin/bash
#SBATCH --job-name=toy2.7      # create a short name for your job
#SBATCH --output=toy2.7.log    # job output file
#SBATCH --partition=pvc9       # cluster partition to be used
#SBATCH --account=support-gpu  # slurm project account
#SBATCH --nodes=2              # number of nodes
#SBATCH --gres=gpu:4           # number of allocated gpus per node
#SBATCH --time=01:00:00        # total run time limit (HH:MM:SS)
#SBATCH --exclude=pvc-s-216,pvc-s-217

source lightning-setup-2.7.sh
source run_toy_example.sh
