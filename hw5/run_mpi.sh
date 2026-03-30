#!/bin/bash

function compile() {
    compiler="mpicxx -O3 -std=c++17"
    echo "Compiling source code..."
    $compiler -c $1.cc  # Compile the source file
    echo "Linking objects..."
    $compiler -o $1 $1.o -lm # Link the object file
    rm $1.o
}

function run() {
    compile $1
    TIMEFORMAT='%3R'
    echo "Running $1 with $3 bodies and $2 ranks..."
    real_time=$( { time mpiexec -n $n_ranks ./$1 $3; } 2>&1 )
    echo "Finished in $real_time seconds."
    rm $1
}

function run_all() {
    compile $1
    TIMEFORMAT='%3R'
    file="local_scaling/mpi_scaling_N=$3.txt"
    echo "# Ranks    Runtime (s)" > $file
    for (( n_ranks=1 ; n_ranks <= $2 ; n_ranks*=2 )); do
        echo "Running $1 with $3 bodies and $n_ranks ranks..."
        real_time=$( { time mpiexec -n $n_ranks ./$1 $3; } 2>&1 )
        echo "Finished in ${real_time} seconds."
        echo "${n_ranks}    ${real_time}" >> $file
    done
    echo "Completed all runs."
    rm $1
}

n_ranks=${1:-4}
N=${2:-512}
run_all nbody_mpi $n_ranks $N
