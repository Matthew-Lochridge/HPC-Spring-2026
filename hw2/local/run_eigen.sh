#!/bin/bash

for N in 10 20 40; do
    for potential in 'well' 'harmonic' 'sinusoidal'; do
        echo "Running eigen.py with N=$N and potential=$potential"
        python src/eigen.py --N $N --potential $potential --n_eigs 6 --bc 'dirichlet' --save_gs 1
    done
done
