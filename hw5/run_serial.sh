#!/bin/bash

function compile() {
    compiler="g++ -O3 -std=c++17"
    echo "Compiling source code..."
    $compiler -c $1.cc  # Compile the source file
    echo "Linking objects..."
    $compiler -o $1 $1.o -lm  # Link the object file
    rm $1.o
}

function run() {
    compile $1
    TIMEFORMAT='%3R'
    echo "Running $1 with $2 bodies..."
    real_time=$( { time ./$1 $3; } 2>&1 )
    echo "Finished in $real_time seconds."
    rm $1
}

function run_all() {
    compile $1
    TIMEFORMAT='%3R'
    file="local_scaling/serial_scaling.txt"
    echo "# N    Runtime (s)" > $file
    for (( N=128 ; N <= $2 ; N*=2 )); do
        echo "Running $1 with $N bodies..."
        real_time=$( { time ./$1 $N; } 2>&1 )
        echo "Finished in ${real_time} seconds."
        echo "${N}    ${real_time}" >> $file
    done
    echo "Completed all runs."
    rm $1
}

N=${1:-512}
run_all nbody $N
