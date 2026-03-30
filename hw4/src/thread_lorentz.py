# thread_lorentz.py
import threading
import sys
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

def add_chunk(seed, n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    np.random.default_rng(seed)
    local_counts = lorentzian_histogram(n, bins, xmin, xmax)
    # Acquire lock to merge partial counts into global
    with lock:
        counts += local_counts

def run_threaded(seeds, n, n_threads=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using threads.
    """
    # Split n samples among processes
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int)
    chunks[:n % n_threads] += 1 # Distribute remainder
    threads = [None] * n_threads # Thread list
    counts = np.zeros(bins) # Global counts
    lock = threading.Lock() # Lock for global data
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(seeds[i], chunks[i], counts, lock, bins, xmin, xmax))
        t.start() # Start thread
        threads[i] = t
    for t in threads:
        t.join() # Wait for all threads to finish
    return counts

if __name__ == "__main__":

    method = 'threading'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    n_threads = 4 # Default number of threads
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python thread_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        n_threads = int(float(sys.argv[3])) # Overwrite default n_threads
    if len(sys.argv) >= 5:
        bins = int(float(sys.argv[4])) # Overwrite default bins
    if len(sys.argv) >= 6:
        xmin = int(float(sys.argv[5])) # Overwrite default xmin
    if len(sys.argv) == 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    if len(sys.argv) >= 8:
        print("Usage: python thread_lorentz.py <weak|strong> <n> <n_threads> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")
        
    seed_seq = np.random.SeedSequence(int(time()))
    seeds = seed_seq.spawn(n_threads)

    start_time = perf_counter()
    counts = run_threaded(seeds, n, n_threads, bins, xmin, xmax)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {n_threads}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{n_threads}\t{end_time-start_time}\n")