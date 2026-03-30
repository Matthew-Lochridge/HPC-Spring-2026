#!/bin/bash

function compile() {
    compiler="g++ -fopenmp -O3 -std=c++17"
    echo "Compiling source code..."
    $compiler -c $1.cc  # Compile the source file
    echo "Linking objects..."
    $compiler -o $1 $1.o -lm # Link the object file
    rm $1.o
}

function run() {
    compile $1
    TIMEFORMAT='%3R'
    echo "Running $1 with $3 bodies and $2 threads..."
    export OMP_NUM_THREADS=$2
    real_time=$( { time ./$1 $3; } 2>&1 )
    echo "Finished in $real_time seconds."
    rm $1
}

function run_all() {
    compile $1
    TIMEFORMAT='%3R'
    file="local_scaling/omp_scaling_N=$3.txt"
    echo "# Threads    Runtime (s)" > $file
    for (( n_threads=1 ; n_threads <= $2 ; n_threads*=2 )); do
        echo "Running $1 with $3 bodies and $n_threads threads..."
        export OMP_NUM_THREADS=$n_threads
        real_time=$( { time ./$1 $3; } 2>&1 )
        echo "Finished in ${real_time} seconds."
        echo "${n_threads}    ${real_time}" >> $file
    done
    echo "Completed all runs."
    rm $1
}

n_threads=${1:-4}
N=${2:-512}
run_all nbody_omp $n_threads $N
