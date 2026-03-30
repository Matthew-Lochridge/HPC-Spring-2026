#!/bin/bash
#SBATCH -J hw4-lorentz
#SBATCH -p compute
#SBATCH -N 2 # nodes
#SBATCH -n 4 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:10:00 # walltime
#SBATCH --exclusive # avoid interference

set -euo pipefail

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running each method with one processor."

echo "Running threading method..."
python3 src/thread_lorentz.py strong 1e7 1

echo "Running multiprocessing method..."
python3 src/mp_lorentz.py strong 1e7 1

echo "Running ProcessPoolExecutor method..."
python3 src/ppe_lorentz.py strong 1e7 1

echo "Running AsyncIO method..."
python3 src/async_lorentz.py strong 1e7 1

echo "Running Dask method..."
python3 src/dask_lorentz.py strong 1e7 1

echo "Running Numba method..."
python3 src/numba_lorentz.py strong 1e7 1

echo "Running Joblib method..."
python3 src/joblib_lorentz.py strong 1e7 1

echo "Running mpire method..."
python3 src/mpire_lorentz.py strong 1e7 1

echo "Running MPI method..."
mpiexec -n 1 python3 src/mpi_lorentz.py