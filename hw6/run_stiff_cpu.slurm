#!/bin/bash
#SBATCH --job-name=stiff-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

python3 src/trbdf2.py 0