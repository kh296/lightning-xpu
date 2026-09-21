#!/bin/bash
#SBATCH --job-name=toy2.8      # create a short name for the job
#SBATCH --output=%x.log        # job output file
#SBATCH --partition=pvc9       # cluster partition to be used
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
# to extract it from SLURM_JOB_NAME.  If this fails, the value
# of ${DEFAULT_PYTORCH_VERSION}, defined in this script is used.
#
# This script can be run interactively:
#     ./run_toy_example.sh # use default PyTorch version
#     export SLURM_JOB_NAME=toy2.3; ./run_toy_example.sh # use PyTorch 2.3
# or can be submitted to a Slurm batch system.
#     sbatch run_toy_example.sh # use default PyTorch version
#     sbatch --job-name=toy2.3 run_toy_example.sh # use PyTorch 2.3
T1=${SECONDS}
echo "Job start on $(hostname): $(date)"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"

# Exit at first failure.
set -e

# Define default PyTorch version.
DEFAULT_PYTORCH_VERSION=2.8

# Ensure that Slurm environment variables are set correctly.
# The variables specifying number of tasks per node (SLURM_NTASKS_PER_NODE),
# number of tasks (SLURM_NTASKS), and number of CPU cores per task
# (SLURM_CPUS_PER_TASK) are used by Lightning in device allocation.
# These numbers depend on the number of nodes allocated, on the
# number of GPUs per node, and on whether GPUs are used in "FLAT" mode
# or in "COMPOSITE" mode.
#
# On Dawn, if a single node is allocated, then it may be allocated with
# 1, 2, 3, or 4 GPUs (but not with 0 GPUs).  If more than one node is
# allocated, all must # be allocated with all (4) GPUs.  If GPUs are
# used in "FLAT" mode, the two stacks of each GPU are treated as two
# root devices.  If GPUs are used in "COMPOSITE" mode, the two stacks
# of each GPU are treated as a single root device.  For more information
# about modes for Intel GPUs, see:
# https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/exposing-device-hierarchy.html
if [[ -z "${SLURM_JOB_NAME}" || "_interactive" == "${SLURM_JOB_NAME}" ]]; then
    SLURM_JOB_NAME="toy${DEFAULT_PYTORCH_VERSION}"
fi

# Default to 1 node allocated if running outside of Slurm.
if [[ -z "${SLURM_NNODES}" ]]; then
    SLURM_NNODES=1
fi

# Determine number of root devices per GPU on Dawn.
if [[ "FLAT" == ${ZE_FLAT_DEVICE_HIERARCHY} ]]; then
    DEVICES_PER_GPU=2
else
    DEVICES_PER_GPU=1
fi

# Determine number of tasks per node, with one task per GPU root device,
# or defaulting to 1 if there are no GPUs.
if [[ -z "${SLURM_GPUS_ON_NODE}" ]]; then
    SLURM_NTASKS_PER_NODE=1
else
    SLURM_NTASKS_PER_NODE=$((${SLURM_GPUS_ON_NODE}*${DEVICES_PER_GPU}))
fi

# Determine total number of tasks.
SLURM_NTASKS=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

# Determine number of CPU cores per task.
if [[ -z "${SLURM_CPUS_ON_NODE}" ]]; then
    SLURM_CPUS_ON_NODE=1
fi
#export SLURM_CPUS_PER_TASK=$((${SLURM_CPUS_ON_NODE}/${SLURM_NTASKS_PER_NODE}))

# Unset and set Slurm variables for compatibility with srun.
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_NODE
SLURM_EXPORT_ENV=ALL

# Perform environment setup.
PROJECT_HOME=$(cd $(dirname "$0")/..; pwd)
if [[ ${PROJECT_HOME} == /var/spool/* ]]; then
    PROJECT_HOME=$(dirname $(pwd))
fi
SETUP_SCRIPT="${PROJECT_HOME}/envs/lightning-setup.sh"
if [[ ! -f "${SETUP_SCRIPT}" ]]; then
    # Set PyTorch version.
    SETUP_SCRIPT_PREFIX="${PROJECT_HOME}/install/lightning-setup-"
    SETUP_SCRIPT_SUFFIX=".sh"
    if [[ -z "${PYTORCH_VERSION}" ]]; then
        PYTORCH_VERSION=${DEFAULT_PYTORCH_VERSION}
        PYTORCH_VERSION_SOURCE="\${DEFAULT_PYTORCH_VERSION}"
        if [ 0 -ne  $(echo "${SLURM_JOB_NAME}" | grep -Eoc "[0-9]+(\.[0-9]+)?") ]; then
            VERSION=$(echo "${SLURM_JOB_NAME}" | grep -Eo "[0-9]+(\.[0-9]+)?")
            if [ -f "${SETUP_SCRIPT_PREFIX}${VERSION}${SETUP_SCRIPT_SUFFIX}" ]; then
                PYTORCH_VERSION=${VERSION}
                PYTORCH_VERSION_SOURCE="\${SLURM_JOB_NAME}"
	    fi
        fi
    else
        PYTORCH_VERSION_SOURCE="\${PYTORCH_VERSION}"
    fi
    echo ""
    echo "From ${PYTORCH_VERSION_SOURCE}, using PyTorch version: ${PYTORCH_VERSION}"

    # Perform environment setup.
    SETUP_SCRIPT="${SETUP_SCRIPT_PREFIX}${PYTORCH_VERSION}${SETUP_SCRIPT_SUFFIX}"
    if ! [ -f ${SETUP_SCRIPT} ]; then
        echo "Setup script for Python version ${PYTHON_VERSION} not found:"
        echo "    ${SETUP_SCRIPT}"
        exit
    fi
fi
SETUP="source ${SETUP_SCRIPT}"
echo ${SETUP}
${SETUP}
echo ""

# Define command to run depending on availability of srun.
APP="toy_example.py"
if command -v srun 1>/dev/null 2>&1
then
    # List nodes allocated.
    SRUN_ONE_PER_NODE="srun --nodes=${SLURM_NNODES} --ntasks-per-node=1"
    echo "Nodes used:"
    eval "${SRUN_ONE_PER_NODE} hostname"
    echo ""
    # Initial package import can be slow.  Perform before running
    # application, so that the initial time isn't included in
    # the application timing.
    echo "Performing initial import of lightning on each node"
    T2=${SECONDS}
    eval "${SRUN_ONE_PER_NODE} python -c 'import lightning'"
    echo "Import time 1: $((${SECONDS}-${T2})) seconds"
    echo "Performing second import of lightning on each node"
    T2=${SECONDS}
    eval "${SRUN_ONE_PER_NODE} python -c 'import lightning'"
    echo "Import time 2: $((${SECONDS}-${T2})) seconds"
    # Define command to run application.
    CMD="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} python ${APP}"
else
    # List node allocated.
    echo "Hostname: $(hostname)"
    echo ""
    # Initial package import can be slow.  Perform before running
    # application, so that the initial time isn't included in
    # the application timing.
    echo "Performing initial import of lightning"
    T2=${SECONDS}
    echo "Import time: $((${SECONDS}-${T2})) seconds"
    python -c "import lightning"
    # Define command to run application.
    CMD="python ${APP}"
fi

# Ensure that data needed are downloaded before running application.
echo ""
echo "Downloading/checking dataset"
T3=${SECONDS}
python -c "import torchvision as tv; tv.datasets.MNIST('.', download=True)"
echo "Time downloading/checking dataset: $((${SECONDS}-${T3})) seconds"

# Run and time application.
T4=${SECONDS}
echo ""
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo ""
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T4})) seconds"
echo ""
echo "Job time: $((${SECONDS}-${T1})) seconds"
