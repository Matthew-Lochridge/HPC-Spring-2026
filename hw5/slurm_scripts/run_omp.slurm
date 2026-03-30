#!/bin/bash
#SBATCH --job-name=omp
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=1 # total ranks
#SBATCH --cpus-per-task=128 # CPUs per rank
#SBATCH --time=00:15:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

set -euo pipefail

max_threads=${1:-128}
N=${2:-4096}

program="nbody_omp"
compiler="g++ -fopenmp -O3 -std=c++17"
echo "Compiling source code..."
$compiler -c $program.cc  # Compile the source file
echo "Linking objects..."
$compiler -o $program $program.o -lm # Link the object file
rm $program.o

TIMEFORMAT='%3R'
file="scaling_data/omp_scaling_N=$N.txt"
echo "# Threads    Runtime (s)" > $file
for (( n_threads=1 ; n_threads <= $max_threads ; n_threads*=2 )); do
    echo "Running $program with $N bodies and $n_threads threads..."
    export OMP_NUM_THREADS=$n_threads
    real_time=$( { time ./$program $N; } 2>&1 )
    echo "Finished in ${real_time} seconds."
    echo "${n_threads}    ${real_time}" >> $file
done
echo "Completed all runs."
rm $program
