#!/bin/bash
#SBATCH --job-name=light2.3    # create a short name for your job
#SBATCH --output=install2.3.log   # job output file
#SBATCH --partition=pvc9       # cluster partition to be used
#SBATCH --account=support-gpu  # slurm project account
#SBATCH --nodes=1              # number of nodes
#SBATCH --gres=gpu:1           # number of allocated gpus per node
#SBATCH --time=00:30:00        # total run time limit (HH:MM:SS)
T0=${SECONDS}
echo "Lightning installation started: $(date)"

cat <<EOF >lightning-setup-2.3.sh
# Setup script for enabling lightning on Dawn supercomputer
# Generated: $(date)

module purge
module load rhel9/default-dawn
module load intel-oneapi-ccl/2021.15.0
EOF

source lightning-setup-2.3.sh
module load intelpython-conda/2025.0
conda activate pytorch-gpu-2.3.1

rm -rf lightning-venv
python -m venv --system-site-packages lightning-venv
CMD="source $(pwd)/lightning-venv/bin/activate"
echo ${CMD} >> lightning-setup-2.3.sh
${CMD}
pip install --upgrade pip
pip install py-cpuinfo
pip install deepspeed lightning[extra] litmodels mpi4py

echo "Lightning installation completed: $(date)"
echo "Installation time: $((${SECONDS}-${T0})) seconds"
