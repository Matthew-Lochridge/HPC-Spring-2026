#!/bin/bash
#SBATCH --job-name=mpi-omp
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=2 # total ranks
#SBATCH --cpus-per-task=4 # CPUs per rank
#SBATCH --time=00:15:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdl220000@utdallas.edu

set -euo pipefail

n_ranks=${1:-2}
n_threads=${2:-4}
N=${3:-4096}

export OMP_NUM_THREADS=$n_threads

program="nbody_mpi_omp"
compiler="mpicxx -fopenmp -O3 -std=c++17"
echo "Compiling source code..."
$compiler -c $program.cc  # Compile the source file
echo "Linking objects..."
$compiler -o $program $program.o -lm # Link the object file
rm $program.o

TIMEFORMAT='%3R'
echo "Running $program with $N bodies, $n_ranks ranks, and $n_threads threads..."
real_time=$( { time mpiexec -n $n_ranks ./$program $N; } 2>&1 )
echo "Finished in $real_time seconds."
rm $program
