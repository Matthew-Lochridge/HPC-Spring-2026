#!/bin/bash

PAR=$1
N_BASE=${2:-10000000}
METHODS=("async" "dask" "joblib" "mp" "mpi" "mpire" "ppe" "thread")
SCALING_MODES=("strong" "weak")

for SCALING in "${SCALING_MODES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        sbatch "run_p${PAR}.slurm" $METHOD $SCALING $N_BASE
    done
done
squeue -u mdl220000