# ppe_lorentz.py
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

def initializer():
    return np.random.default_rng(int(time()))

def run_ppe(n, max_workers=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using ProcessPoolExecutor.
    """
    chunks = (n // max_workers) * np.ones(max_workers, dtype=int) # Split n samples among workers
    chunks[:n % max_workers] += 1 # Distribute remainder
    with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer) as executor:
        futures = [executor.submit(lorentzian_histogram, chunk, bins, xmin, xmax) for chunk in chunks]
        results = [f.result() for f in futures] # Collect results
    return np.sum(results, axis=0)

if __name__ == "__main__":

    method = 'ProcessPoolExecutor'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    max_workers = 4 # Default number of workers
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python ppe_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        max_workers = int(float(sys.argv[3])) # Overwrite default max_workers
    if len(sys.argv) >= 5:
        bins = int(float(sys.argv[4])) # Overwrite default bins
    if len(sys.argv) >= 6:
        xmin = int(float(sys.argv[5])) # Overwrite default xmin
    if len(sys.argv) == 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    if len(sys.argv) >= 8:
        print("Usage: python pps_lorentz.py <weak|strong> <n> <max_workers> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")

    start_time = perf_counter()
    counts = run_ppe(n, max_workers, bins, xmin, xmax)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {max_workers}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{max_workers}\t{end_time-start_time}\n")