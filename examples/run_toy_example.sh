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

APP="toy_example.py"
CMD="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} python ${APP}"

T0=${SECONDS}
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T0})) seconds"
