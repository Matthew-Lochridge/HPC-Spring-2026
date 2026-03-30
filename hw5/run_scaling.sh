#!/bin/bash

n=${1:-16384}
maxpar=${2:-16}

./run_serial.sh $n

python3 plot_serial.py

./run_omp.sh $maxpar $n

./run_mpi.sh $maxpar $n

./run_mpi_shared.sh $maxpar $n

python3 plot_parallel.py $n