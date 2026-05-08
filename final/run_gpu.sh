#!/bin/bash

for (( n=4 ; n <= 10; n+=1 )); do
    sbatch ./run_gpu.slurm "($n,0)-CNT"
    sbatch ./run_gpu.slurm "($n,$n)-CNT"
done
