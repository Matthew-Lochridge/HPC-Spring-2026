#!/bin/bash

MAX_PARALLELISM=${1:-64}
N_BASE=${2:-10000000}
METHODS=("async" "dask" "joblib" "mp" "mpi" "mpire" "ppe" "thread")
SCALING_MODES=("strong" "weak")

for SCALING in "${SCALING_MODES[@]}"; do
    for ((p=1; p<=$MAX_PARALLELISM; p*=2)); do
        for METHOD in "${METHODS[@]}"; do
            sbatch "run_p${p}.slurm" $METHOD $SCALING $N_BASE
        done
    done
done
squeue -u mdl220000