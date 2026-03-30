# mp_lorentz.py
import multiprocessing
import sys
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

def initializer():
    return np.random.default_rng(int(time()))

def run_multiproc(n, n_cores=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using processes.
    """
    # Split n samples among processes
    chunks = (n // n_cores) * np.ones(n_cores, dtype=int)
    chunks[:n % n_cores] += 1 # Distribute remainder
    # Use partial function to reset default arguments (bins, xmin, xmax)
    from functools import partial
    lorentzian_hist_func = partial(lorentzian_histogram, bins=bins, xmin=xmin, xmax=xmax)
    # Use Pool to distribute chunks to processes
    with multiprocessing.Pool(n_cores, initializer) as pool:
        results = pool.map(lorentzian_hist_func, chunks)
    return np.sum(results, axis=0)

if __name__ == "__main__":

    method = 'multiprocessing'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    n_cores = 4 # Default number of cores
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python mp_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        n_cores = int(float(sys.argv[3])) # Overwrite default n_cores
    if len(sys.argv) >= 5:
        bins = int(float(sys.argv[4])) # Overwrite default bins
    if len(sys.argv) >= 6:
        xmin = int(float(sys.argv[5])) # Overwrite default xmin
    if len(sys.argv) == 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    if len(sys.argv) >= 8:
        print("Usage: python mp_lorentz.py <weak|strong> <n> <n_cores> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")

    start_time = perf_counter()
    counts = run_multiproc(n, n_cores, bins, xmin, xmax)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {n_cores}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{n_cores}\t{end_time-start_time}\n")