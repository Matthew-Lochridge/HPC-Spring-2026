# async_lorentz.py
import asyncio
import sys
import numpy as np
from time import time, perf_counter
from base_lorentz import lorentzian_histogram, plot_samples

async def async_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Async wrapper for lorentzian_histogram. Since lorentzian_histogram
    is CPU-bound and synchronous, we call it directly.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

async def add_chunk(seed, n, counts, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Generate n samples in subchunks and add to global counts.
    """
    np.random.default_rng(seed)
    # Split n samples among sub-chunks
    sub_chunks = (n // n_subchunks) * np.ones(n_subchunks, dtype=int)
    sub_chunks[:n % n_subchunks] += 1 # Distribute remainder
    # Gather results from subchunks
    local_counts = await asyncio.gather(*[
        async_lorentzian_histogram(chunk, bins, xmin, xmax)
        for chunk in sub_chunks
    ])
    counts += np.sum(local_counts, axis=0) 

async def get_counts(seeds, n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Async function to run the Lorentzian sampling in parallel using asyncio.
    """
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    counts = np.zeros(bins)
    tasks = [
        asyncio.create_task(add_chunk(seeds[i], chunks[i], counts, bins, xmin, xmax, n_subchunks))
        for i in range(n_tasks)
    ]
    await asyncio.gather(*tasks) # Wait for all tasks to finish
    return counts

def run_async(seeds, n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Run the Lorentzian sampling in parallel using asyncio.
    """
    return asyncio.run(get_counts(seeds, n, n_tasks, bins, xmin, xmax, n_subchunks))

if __name__ == "__main__":

    method = 'AsyncIO'
    scaling = sys.argv[1] # 'weak' or 'strong'
    n = int(1e7) # Default number of samples
    n_tasks = 4 # Default number of tasks
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax
    n_subchunks = 10 # Default number of sub-chunks

    # Parse command line arguments, if provided
    if len(sys.argv) < 2:
        print("Usage: python async_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
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
    if len(sys.argv) >= 7:
        xmax = int(float(sys.argv[6])) # Overwrite default xmax
    if len((sys.argv)) == 8:
        n_subchunks = int(float(sys.argv[7])) # Overwrite default n_subchunks
    if len(sys.argv) >= 9:
        print("Usage: python async_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
        print("Additional arguments will be ignored.")
        
    seed_seq = np.random.SeedSequence(int(time()))
    seeds = seed_seq.spawn(n_tasks)

    start_time = perf_counter()
    counts = run_async(seeds, n, n_tasks, bins, xmin, xmax, n_subchunks)
    end_time = perf_counter()
    print(f"{method}")
    print(f"Concurrency: {n_tasks}")
    print(f"Samples: {n}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    with open(f"results/{scaling}_scaling_{method}.txt", "a", encoding="utf-8") as f:
        f.write(f"{n_tasks}\t{end_time-start_time}\n")