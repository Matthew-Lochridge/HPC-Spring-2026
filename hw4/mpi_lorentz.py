# mpi_lorentz.py
from mpi4py import MPI
import numpy as np
from time import time, perf_counter
from base_lorentz import plot_samples

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    x = 1. / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)

scaling = 'strong'
n_total = 10_000_000
bins = 100
xmin = -10
xmax = 10 

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

counts = lorentzian_histogram(int(chunks[rank]), bins=bins, xmin=xmin, xmax=xmax, rng=rng)

# Gather results
all_counts = np.array(comm.reduce(counts, op=MPI.SUM, root=0))

if rank == 0:
    # Stop timer
    end_time = perf_counter()
    print(f"Total samples: {n_total}")
    print(f"Total ranks: {size}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    print(f"Samples per second: {n_total / (end_time - start_time):.0f}")

    # Save results
    bin_edges = np.linspace(-10, 10, 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    np.savetxt("lorentzian_histogram.txt",
                np.column_stack([bin_centers, all_counts]),
                fmt="%.6f %d")
    print("Results saved to lorentzian_histogram.txt")

    with open(f"results/{scaling}_scaling_MPI.txt", "a", encoding="utf-8") as f:
        f.write(f"{size}\t{end_time-start_time}\n")

    if size == 1 and scaling == 'strong':
        plot_samples('MPI', all_counts, n_total, bins, xmin, xmax)