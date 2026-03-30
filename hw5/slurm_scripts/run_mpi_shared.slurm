#!/bin/bash
#SBATCH --job-name=mpi-shared
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=128 # total ranks
#SBATCH --cpus-per-task=1 # CPUs per rank
#SBATCH --time=00:15:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

set -euo pipefail

max_ranks=${1:-128}
N=${2:-4096}

program="nbody_mpi_shared"
compiler="mpicxx -O3 -std=c++17"
echo "Compiling source code..."
$compiler -c $program.cc  # Compile the source file
echo "Linking objects..."
$compiler -o $program $program.o -lm # Link the object file
rm $program.o

TIMEFORMAT='%3R'
file="scaling_data/mpi_shared_scaling_N=$N.txt"
echo "# Ranks    Runtime (s)" > $file
for (( n_ranks=1 ; n_ranks <= $max_ranks ; n_ranks*=2 )); do
    echo "Running $program with $N bodies and $n_ranks ranks..."
    real_time=$( { time mpiexec -n $n_ranks ./$program $N; } 2>&1 )
    echo "Finished in ${real_time} seconds."
    echo "${n_ranks}    ${real_time}" >> $file
done
echo "Completed all runs."
rm $program
