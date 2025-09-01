#!/bin/bash
#SBATCH --job-name=toy2.8      # job output file
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

# Ensure that Slurm environment variables are set,
# also if running outside of Slurm environment.
if [[ -z "${SLURM_JOB_NAME}" || "_interactive" == "${SLURM_JOB_NAME}" ]]; then
    SLURM_JOB_NAME="toy${DEFAULT_PYTORCH_VERSION}"
fi

if [[ -z "${SLURM_NNODES}" ]]; then
    SLURM_NNODES=1
fi

if [[ -z "${SLURM_GPUS_ON_NODE}" ]]; then
    SLURM_NTASKS_PER_NODE=1
else
    SLURM_NTASKS_PER_NODE=$((2*${SLURM_GPUS_ON_NODE}))
fi
SLURM_NTASKS=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

# Unset and set Slurm variables for compatibility with srun.
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_NODE
SLURM_EXPORT_ENV=ALL

# Set PyTorch version.
SETUP_SCRIPT_PREFIX="../install/lightning-setup-"
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
SETUP="source ${SETUP_SCRIPT}"
echo ${SETUP}
${SETUP}
echo ""

# Define command to run depending on availability of srun.
APP="toy_example.py"
if command -v srun 1>/dev/null 2>&1
then
    echo "Nodes used:"
    srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 hostname
    echo ""
    echo "Performing initial import of lightning_xpu on each node"
    T2=${SECONDS}
    srun python -c "import lightning_xpu"
    echo "Import time 1: $((${SECONDS}-${T2})) seconds"
    echo "Performing second import of lightning_xpu on each node"
    T2=${SECONDS}
    srun python -c "import lightning_xpu"
    echo "Import time 2: $((${SECONDS}-${T2})) seconds"
    CMD="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} python ${APP}"
else
    echo "Hostname: $(hostname)"
    echo ""
    echo "Performing initial import of lightning_xpu"
    T2=${SECONDS}
    echo "Import time: $((${SECONDS}-${T2})) seconds"
    python -c "import lightning_xpu"
    CMD="python ${APP}"
fi

echo ""
echo "Checking/downloading dataset"
T3=${SECONDS}
python -c "import torchvision as tv; tv.datasets.MNIST('.', download=True)"
echo "Time checking/downloading dataset: $((${SECONDS}-${T3})) seconds"

# Run and time application.
T4=${SECONDS}
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo ""
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T4})) seconds"
echo ""
echo "Job time: $((${SECONDS}-${T1})) seconds"
