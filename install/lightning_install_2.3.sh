#!/bin/bash
#SBATCH --job-name=light2.3     # create a short name for your job
#SBATCH --output=install2.3.log # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of allocated gpus per node
#SBATCH --time=00:30:00         # total run time limit (HH:MM:SS)

# Script for installing lightning on Dawn supercomputer,
# making use of system installation of pytorch (version 2.3.1).
#
# After installation, the environment for running lightning applications
# can be activated by sourcing the file lightning-setup-2.3.sh,
# created in this directory.

T0=${SECONDS}
echo "Lightning installation started: $(date)"

# Create script for environment setup.
VERSION="2.3"
cat <<EOF >lightning-setup-${VERSION}.sh
# Setup script for enabling lightning on Dawn supercomputer
# Generated: $(date)

module purge
module load rhel9/default-dawn
module load intel-oneapi-ccl/2021.15.0
EOF

# Define installation environment.
source lightning-setup-${VERSION}.sh
module load intelpython-conda/2025.0
conda activate pytorch-gpu-2.3.1

# Create virtual environment, add its activation to setup script,
# and install packages needed in addition to the system PyTorch.
rm -rf lightning-${VERSION}-venv
python -m venv --system-site-packages lightning-${VERSION}-venv
CMD="source $(pwd)/lightning-${VERSION}-venv/bin/activate"
echo ${CMD} >> lightning-setup-${VERSION}.sh
${CMD}
pip install --upgrade pip
pip install py-cpuinfo
pip install deepspeed lightning[extra] litmodels mpi4py
pip install -e ..

echo "Lightning installation completed: $(date)"
echo "Installation time: $((${SECONDS}-${T0})) seconds"
