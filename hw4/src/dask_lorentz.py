# dask_lorentz.py
import dask
from dask import delayed
import sys
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

@delayed
def delayed_lorentzian_histogram(seed, n, bins=100, xmin=-10, xmax=10):
    """
    Delayed function for lorentzian_histogram.
    """
    np.random.default_rng(seed)
    return lorentzian_histogram(n, bins, xmin, xmax)

def run_dask(seeds, n, n_tasks=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using Dask.
    """
    dask.config.set(scheduler='processes') # Override the default single-threaded scheduler
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    tasks = [delayed_lorentzian_histogram(seeds[i], chunks[i], bins, xmin, xmax) for i in range(n_tasks)]
    results = dask.compute(*tasks) # Compute all tasks
    return np.sum(results, axis=0) # Aggregate results

if __name__ == "__main__":

    method = 'Dask'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    n_tasks = 4 # Default number of tasks
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python dask_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        n_tasks = int(float(sys.argv[3])) # Overwrite default n_tasks
    if len(sys.argv) >= 5:
        bins = int(float(sys.argv[4])) # Overwrite default bins
    if len(sys.argv) >= 6:
        xmin = int(float(sys.argv[5])) # Overwrite default xmin
    if len(sys.argv) == 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    elif len(sys.argv) >= 8:
        print("Usage: python dask_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")
        
    seed_seq = np.random.SeedSequence(int(time()))
    seeds = seed_seq.spawn(n_tasks)

    start_time = perf_counter()
    counts = run_dask(seeds, n, n_tasks, bins, xmin, xmax)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {n_tasks}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{n_tasks}\t{end_time-start_time}\n")