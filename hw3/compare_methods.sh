#!/bin/bash

set -euo pipefail

n=${1:-1e6}
echo "Executing each method with $n samples:"
# echo "Pure Python: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
# python pi_python.py $n
echo "NumPy: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
python pi_numpy.py $n
echo "Serial Numba: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
python pi_numba.py $n 'serial'
echo "Parallel Numba: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
python pi_numba.py $n 'parallel'
echo "Cython: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
# python setup_cython.py build_ext --inplace
python test_cython.py $n
echo "C++: Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
# g++ -O3 main.cpp -o mc_pi
./mc_pi $n