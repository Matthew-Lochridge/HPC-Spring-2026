# numba_lorentz.py
import sys
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
from time import time, perf_counter
from base_lorentz import plot_samples

@cuda.jit
def lorentzian_histogram_numba(rng_states, n, bins, xmin, xmax, counts):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling with CUDA's random number generator.
    Make a histogram with the specified bin count and range. Modifies counts in-place.
    """
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)  # Total number of threads
    idx = thread_id
    
    # Each thread processes multiple items with stride
    while idx < n:
        xfac = bins / (xmax - xmin)
        # Generate random number using this thread's RNG state
        local_u = xoroshiro128p_uniform_float32(rng_states, thread_id)
        local_x = 1. / np.tan(np.pi * local_u)
        local_cdf = np.arctan(local_x) / np.pi + 0.5
        ix = int((local_x - xmin) * xfac)
        if 0 <= ix < bins:
            cuda.atomic.add(counts, ix, 1)
        
        idx += stride  # Move to next item assigned to this thread

if __name__ == "__main__":

    method = 'Numba'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    blocks_per_grid = 1 # Default number of blocks
    threads_per_block = 4 # Default number of threads
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python numba_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Error: Scaling not specified.")
        sys.exit(1)
    if len(sys.argv) >= 3:
        n = int(float(sys.argv[2])) # Overwrite default n
    if len(sys.argv) >= 4:
        blocks_per_grid = int(float(sys.argv[3])) # Overwrite default blocks_per_grid
    if len(sys.argv) >= 5:
        threads_per_block = int(float(sys.argv[4])) # Overwrite default threads_per_block
    if len(sys.argv) >= 6:
        bins = int(float(sys.argv[5])) # Overwrite default bins
    if len(sys.argv) >= 7:
        xmin = int(float(sys.argv[6])) # Overwrite default xmin
    if len(sys.argv) == 8:
        xmax = int(float(sys.argv[7])) # Overwrite default xmax
    if len(sys.argv) >= 9:
        print("Usage: python numba_lorentz.py <weak|strong> <n> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")
        
    start_time = perf_counter()
    
    # Create per-thread random number states for the GPU
    num_threads = blocks_per_grid * threads_per_block
    rng_states = create_xoroshiro128p_states(num_threads, seed=int(time()))
    
    # Allocate output arrays
    counts = np.zeros(bins, dtype=np.int32)
    
    # Launch kernel with RNG states as first argument
    lorentzian_histogram_numba[blocks_per_grid, threads_per_block](rng_states, n, bins, xmin, xmax, counts)
    
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {num_threads}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{num_threads}\t{end_time-start_time}\n")