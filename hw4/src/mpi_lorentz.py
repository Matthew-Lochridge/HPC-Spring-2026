# mpi_lorentz.py
import sys
import numpy as np
from mpi4py import MPI
from time import time, perf_counter
from base_lorentz import plot_samples

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    x = 1. / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)

scaling = sys.argv[1] # 'weak' or 'strong'
n_total = 10_000_000
bins = 100
xmin = -10
xmax = 10

# Parse command line arguments, if provided
if len(sys.argv) < 2:
    print("Usage: python mpi_lorentz.py <weak|strong> <n> <n_tasks> <bins> <xmin> <xmax> <n_subchunks>")
    print("Error: Scaling not specified.")
    sys.exit(1)
if len(sys.argv) >= 3:
    n_total = int(float(sys.argv[2])) # Overwrite default n
if len(sys.argv) >= 4:
    bins = int(float(sys.argv[3])) # Overwrite default bins
if len(sys.argv) >= 5:
    xmin = int(float(sys.argv[4])) # Overwrite default xmin
if len(sys.argv) == 6:
    xmax = int(float(sys.argv[5])) # Overwrite default xmax
if len(sys.argv) >= 7:
    print("Usage: python mpi_lorentz.py <weak|strong> <n> <n_cores> <bins> <xmin> <xmax>")
    print("Additional arguments will be ignored.")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start timing
if rank == 0:
    start_time = perf_counter()

# Independent RNG stream per rank
seed = int(time())
ss = np.random.SeedSequence(seed)
child = ss.spawn(size)[rank]
rng = np.random.default_rng(child)

chunks = np.full(size, n_total // size, dtype=int)
chunks[: n_total % size] += 1
local = lorentzian_histogram(int(chunks[rank]), bins=bins, xmin=xmin, xmax=xmax, rng=rng)
global_counts = np.empty_like(local)
comm.Allreduce(local, global_counts, op=MPI.SUM)

if rank == 0:
    end_time = perf_counter()
    print("MPI")
    print(f"Concurrency: {size}")
    print(f"Samples: {n_total}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")

    # Save results
    bin_edges = np.linspace(-10, 10, 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    np.savetxt("lorentzian_histogram.txt",
    np.column_stack([bin_centers, global_counts]),
    fmt="%.6f %d")
    with open(f"results/{scaling}_scaling_MPI.txt", "a", encoding="utf-8") as f:
        f.write(f"{size}\t{end_time-start_time}\n")

