#!/bin/bash
#SBATCH --job-name=rk-gpu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

python3 src/runge_kutta.py