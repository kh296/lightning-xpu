#!/bin/bash
#SBATCH --job-name=light2.7     # create a short name for your job
#SBATCH --output=install2.7.log # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of allocated gpus per node
#SBATCH --time=01:00:00         # total run time limit (HH:MM:SS)

# Script for installing lightning on Dawn supercomputer,
# including user installation of pytorch (version 2.7).
#
# This installation relies on the user having a miniforge installation
# at ~/miniforge3/bin/activate.  For instruction for installing miniforge, see:
# https://conda-forge.org/download/
#
# After installation, the environment for running lightning applications
# can be activated by sourcing the file lightning-setup-2.7.sh, created
# in this directory.

T0=${SECONDS}
echo "Lightning installation started: $(date)"

# Create script for environment setup.
VERSION="2.7"
cat <<EOF >lightning-setup-${VERSION}.sh
# Setup script for enabling lightning on Dawn supercomputer
# Generated: $(date)

module purge
module load rhel9/default-dawn
module load intel-oneapi-ccl/2021.15.0

# Initialise conda.
source ~/miniforge3/bin/activate

# Activate environment.
EOF

# Define installation environment.
source lightning-setup-${VERSION}.sh

# Create and activate conda environment.
ENV_NAME="lightning-${VERSION}"
cat <<EOF >${ENV_NAME}.yml
name: ${ENV_NAME}
channels:
  - https://software.repos.intel.com/python/conda
  - conda-forge
  - nodefaults
dependencies:
  - intelpython3_full
  - python=3.12
  - pip
  - pip:
    - --index-url https://download.pytorch.org/whl/xpu
    - --extra-index-url https://pypi.org/simple
    - --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    - deepspeed
    - lightning[extra]
    - litmodels
    - mpi4py
    - py-cpuinfo
    - torch==2.7.0
    - torchaudio==2.7.0
    - torchvision==0.22.0
    - intel-extension-for-pytorch==2.7.10+xpu
    - oneccl_bind_pt==2.7.0+xpu
    - setuptools==80.8.0
    - -e ..
EOF

conda env remove -n ${ENV_NAME} -y
conda env create -f ${ENV_NAME}.yml
CMD="conda activate ${ENV_NAME}"
echo ${CMD} >> lightning-setup-${VERSION}.sh

echo "Lightning installation completed: $(date)"
echo "Installation time: $((${SECONDS}-${T0})) seconds"
