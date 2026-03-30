# mpire_lorentz.py
import sys
from mpire import WorkerPool
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

def init_func(worker_state):
    # Initialize rng and counts for each worker
    worker_state['seed'] = int(time())

def mpire_lorentzian_histogram(worker_state, n, bins=100, xmin=-10, xmax=10):
    np.random.default_rng(worker_state['seed'])
    worker_state['counts'] = lorentzian_histogram(n, bins, xmin, xmax)

def exit_func(worker_state):
    # Return counts
    return worker_state['counts']

def run_mpire(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using mpire.
    """
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    with WorkerPool(n_jobs=n_jobs, use_worker_state=True) as pool:
        # See mpire docs for argument passing; alternatively use starmap
        # pool.set_initializer(initializer)
        pool.map(mpire_lorentzian_histogram, chunks, worker_init=init_func, worker_exit=exit_func)
        counts = pool.get_exit_results()
    if n_jobs > 1:
        counts = np.sum(counts, axis=0)
    return counts

if __name__ == "__main__":

    method = 'mpire'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    n_jobs = 4 # Default number of jobs
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python mpire_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        n_jobs = int(float(sys.argv[3])) # Overwrite default n_jobs
    if len(sys.argv) >= 5:
        bins = int(float(sys.argv[4])) # Overwrite default bins
    if len(sys.argv) >= 6:
        xmin = int(float(sys.argv[5])) # Overwrite default xmin
    if len(sys.argv) == 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    elif len(sys.argv) >= 8:
        print("Usage: python mpire_lorentz.py <weak|strong> <n> <n_jobs> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")

    start_time = perf_counter()
    counts = run_mpire(n, n_jobs, bins, xmin, xmax)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {n_jobs}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{n_jobs}\t{end_time-start_time}\n")