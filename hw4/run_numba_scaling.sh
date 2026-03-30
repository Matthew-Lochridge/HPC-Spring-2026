#!/bin/bash

MAX_PARALLELISM=${1:-128}
N_BASE=${2:-10000000}
N=$N_BASE
SCALING_MODES=("strong" "weak")

for SCALING in "${SCALING_MODES[@]}"; do
    for ((p=1; p<=$MAX_PARALLELISM; p*=2)); do
        if [[ $p -gt $MAX_PARALLELISM ]]; then
            break
        fi
        if [[ "$SCALING" == "weak" ]]; then
            N=$(( N_BASE * p ))
        else
            N=$N_BASE
        fi
        python3 "src/numba_lorentz.py" $SCALING $N 1 $p
    done
done