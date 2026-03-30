#!/bin/bash
#SBATCH --job-name=serial
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=1 # total ranks
#SBATCH --cpus-per-task=1 # CPUs per rank
#SBATCH --time=00:15:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

set -euo pipefail

N_max=${1:-4096}

program="nbody"
compiler="g++ -O3 -std=c++17"
echo "Compiling source code..."
$compiler -c $program.cc  # Compile the source file
echo "Linking objects..."
$compiler -o $program $program.o -lm  # Link the object file
rm $program.o

TIMEFORMAT='%3R'
file="scaling_data/serial_scaling.txt"
echo "# N    Runtime (s)" > $file
for (( N=128 ; N <= $N_max ; N*=2 )); do
    echo "Running $program with $N bodies..."
    real_time=$( { time ./$program $N; } 2>&1 )
    echo "Finished in ${real_time} seconds."
    echo "${N}    ${real_time}" >> $file
done
echo "Completed all runs."
rm $program
