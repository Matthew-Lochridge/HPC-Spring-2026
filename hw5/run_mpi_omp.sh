#!/bin/bash

function compile() {
    compiler="mpicxx -fopenmp -O3 -std=c++17"
    echo "Compiling source code..."
    $compiler -c $1.cc  # Compile the source file
    echo "Linking objects..."
    $compiler -o $1 $1.o -lm # Link the object file
    rm $1.o
}

function run() {
    compile $1
    TIMEFORMAT='%3R'
    echo "Running $1 with $4 bodies, $2 ranks, and $3 threads..."
    export OMP_NUM_THREADS=$3
    real_time=$( { time mpiexec -n $2 ./$1 $4; } 2>&1 )
    echo "Finished in $real_time seconds."
    rm $1
}

n_ranks=${1:-2}
n_threads=${2:-4}
n=${3:-512}
run nbody_mpi_omp $n_ranks $n_threads $n
