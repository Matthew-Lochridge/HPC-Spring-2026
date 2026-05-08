
# 5v48.002 Spring 2026 Final Project: Electronic band structure of carbon nanotubes using NumPy vs. CuPy

## Required Python packages

* NumPy
* CuPy

## Configuration

* All configuration is done in config.py and via command line arguments.
* In config.py:
  * Physical constants and Kurokawa pseudopotential parameters should not be altered.
  * Nanostructure configuration parameters (in the case of carbon nanotubes):
    * N_a refers to the number of repeated helical motifs in the unit cell. For an infinitely long nanotube, this should be set to 1.
    * N_x refers to the spacing between nanotubes in the supercell scheme along the axial direction. For an infinitely long nanotube, this should be set to 0.
    * N_y and N_z are the supercell spacing in y and z directions. They are not used in this case, in favor of automatically generating a square cross-section of the length of two nanotube diameters.
  * Other parameters:
    * E_cut refers to the cutoff energy in Rydbergs beyond which reciprocal lattice vectors are discarded. According to Fischetti and Vandenberghe, 15 is low enough for convergence.
    * max_G refers to the maximum Manhattan distance of reciprocal lattice vectors generated as multiples of primitive reciprocal lattice vectors. In my testing, 100 was high enough to generate all reciprocal lattice vectors within E_cut for several different structures without costing much computation time (less than 4 seconds).
    * n_x refers to the number of real-space points in each dimension along which to evaluate wavefunctions (currently not implemented due to insufficient memory).
    * n_k refers to the number of reciprocal-space points in the axial direction along which to evaluate band structure. In my testing, 101 is plenty to achieve smooth curves in the band structure.

## Instructions

* Local:
  * In the Final-Project directory, enter the terminal command: python main.py allotrope includeH preferGPU showSupercell
    * allotrope = a string describing the structure to be used, e.g. "(5,0)-CNT".
      * For more examples, see how these strings are parsed in construct_allotrope.py.
    * includeH = a Boolean integer, where 1 will terminate any dangling bonds with hydrogen atoms. For infinitely long nanotubes, this should be 0.
    * preferGPU = a Boolean integer, where 1 will attempt to use the CuPy backend to run on a GPU.
    * showSupercell = a Boolean integer, where 1 will pause after constructing the allotrope to plot the supercell. This is used for verifying the construction of new allotrope classes.
      * See default settings for command line arguments at the start of main.py.
  * Output files will be stored in the data folder:
    * allotrope_bands_backend.txt contains the energy bands.
    * scaling_backend.txt contains timing data for different runs with either the NumPy or CuPy backend.
  * Band structures will be plotted and saved automatically in figures/allotrope_bands_backend.pdf.
  * To plot the scaling behavior, ensure that both scaling .txt files contain data for the same allotropes run on both backends, and enter the terminal command: python plot_scaling.py. Runtime plots will be saved to the figures folder.
* Ganymede2 batch jobs:
  * Edit shell files to submit desired batch jobs on CPU or GPU. These will call the corresponding Slurm file, which submits a job to run main.py with the specified command line inputs.
  * In the Final-Project directory, enter the terminal command: ./run_cpu.sh or ./run_gpu.sh.

## Reproduction

* All presented results used the default settings for config.py, includeH, and showSupercell.
* Band structures were presented with (5,5)-CNT and (10,0)-CNT for validation with the results of Fischetti and Vandenberghe.
* Scaling behavior was presented for (n,0)-CNT, with n in (4,5,6,7,8).
