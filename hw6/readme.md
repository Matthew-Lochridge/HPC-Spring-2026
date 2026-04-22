# HW6 - CuPy-based ODE IVP Solvers

This repository contains the work of Matthew Lochridge for the titular assignment in PHYS 5v48.002 for Spring 2026 at UTD.

Python scripts for several ODE IVP solvers as well as visualization are included in the src folder. The results folder is intended to store .txt files that are generated from the solver scripts, and the figures folder is intended to store plots and tables generated from these data. Also included are Slurm batch scripts for running larger problems on Ganymede2 using a CPU or GPU. 

To run an ODE IVP solver from the command line, enter: 
    python src/<script_name> <prefer_gpu>
where <prefer_gpu> is an optional Boolean input for runge_kutta.py, logistic.py, stiff.py, and trbdf2.py. GPU is preferred by default. 

To run any of the Slurm scripts on Ganymede2, enter:
    sbatch <script_name.slurm>

To visualize all output files in the results folder, enter:
    python src/plot_results.py
