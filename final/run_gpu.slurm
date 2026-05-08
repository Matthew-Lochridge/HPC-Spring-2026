#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=04:00:00
#SBATCH --exclusive

allotrope=${1:-"Graphene"}
include_H=${2:-0}

python3 main.py $allotrope $include_H 1 0