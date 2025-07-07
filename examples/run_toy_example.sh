#!/bin/bash
rm -rf lightning_logs

if [[ -z "${SLURM_NNODES}" ]]; then
    SLURM_NNODES=1
fi

if [[ -z "${SLURM_GPUS_ON_NODE}" ]]; then
    SLURM_NTASKS_PER_NODE=1
else
    SLURM_NTASKS_PER_NODE=$((2*${SLURM_GPUS_ON_NODE}))
fi

T1=${SECONDS}
echo "Lightning initial imports started: $(date)"
srun hostname
srun python -c "import lightning_xpu"
echo "Lightning initial imports completed: $(date)"
echo "Initial imports: $((${SECONDS}-${T1})) seconds"
echo ""

APP="toy_example.py"
CMD="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} python ${APP}"

T2=${SECONDS}
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T2})) seconds"
