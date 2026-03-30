#!/bin/bash
#SBATCH --job-name=hw4
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=128 # total ranks
#SBATCH --cpus-per-task=1 # CPUs per rank
#SBATCH --time=00:10:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --mail-type=END
#SBATCH --mail-user=mdl220000@utdallas.edu

set -euo pipefail

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

METHOD=$1
SCALING=$2
N=${3:-10000000}
P=128

if [[ "$SCALING" == "weak" ]]; then
    N=$((N * P))
fi

if [[ "$METHOD" == "mpi" ]]; then
    mpiexec -n $P python3 "src/${METHOD}_lorentz.py" $SCALING $N
else
    prun python3 "src/${METHOD}_lorentz.py" $SCALING $N $P
fi
