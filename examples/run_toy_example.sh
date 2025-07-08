#!/bin/bash
#SBATCH --job-name=toy2.3      # create a short name for your job
#SBATCH --output=%x.log        # job output file
#SBATCH --partition=pvc9       # cluster partition to be used
#SBATCH --account=support-gpu  # slurm project account
#SBATCH --nodes=2              # number of nodes
#SBATCH --gres=gpu:4           # number of allocated gpus per node
#SBATCH --time=01:00:00        # total run time limit (HH:MM:SS)
 
# Script for running lightning example.
#
# It's assumed that the environment for running lightning applications
# can be set up with:
# source ../install/lightning-setup-${PYTORCH_VERSION}.sh
# where ${PYTORCH_VERSION} identifies the underlying PyTorch version.
#
# If PYTORCH_VERSION isn't defined explicitly, an attempt will be made
# to extract it from SLURM_JOB_NAME.
#
# This script can be run interactively:
#     ./run_toy_example.sh
#     export SLURM_JOB_NAME=toy2.7; ./run_toy_example.sh # use PyTorch 2.7
# or can be submitted to a Slurm batch system.
#     sbatch run_toy_example.sh
#     sbatch --job-name=toy2.7 run_toy_example.sh # use PyTorch 2.7

# Exit at first failure.
set -e

# Ensure that Slurm environment variables are set,
# also if running outside of Slurm environment.
if [[ -z "${SLURM_JOB_NAME}" ]]; then
    SLURM_JOB_NAME="toy2.3"
fi

if [[ -z "${SLURM_NNODES}" ]]; then
    SLURM_NNODES=1
fi

if [[ -z "${SLURM_GPUS_ON_NODE}" ]]; then
    SLURM_NTASKS_PER_NODE=1
else
    SLURM_NTASKS_PER_NODE=$((2*${SLURM_GPUS_ON_NODE}))
fi

# If PYTORCH_VERSION not defined explicitly,
# try extracting from the Slurm job name.
if [[ -z "${PYTORCH_VERSION}" ]]; then
    PYTORCH_VERSION=$(echo "${SLURM_JOB_NAME}" | grep -Eo "[0-9]+(\.[0-9]+)?")
fi
source ../install/lightning-setup-${PYTORCH_VERSION}.sh

# Define command to run depending on availability of srun.
APP="toy_example.py"
if command -v srun 1>/dev/null 2>&1
then
    echo "Nodes being used:"
    srun hostname
    CMD="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} python ${APP}"
else
    echo "Hostname: $(hostname)"
    CMD="python ${APP}"
fi
echo ""

# Run and time application.
T0=${SECONDS}
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T0})) seconds"
