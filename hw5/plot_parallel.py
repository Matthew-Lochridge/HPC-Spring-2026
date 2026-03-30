import sys
import matplotlib.pyplot as plt
from numpy import loadtxt

def fetch_data(method, n):
    data = loadtxt(f"local_scaling/{method}_scaling_N={n}.txt")
    n_processors = data[:, 0]
    runtime = data[:, 1]
    speedup = runtime[0]/runtime
    efficiency = speedup/n_processors
    return n_processors, runtime, speedup, efficiency

if __name__ == "__main__":

    n = 16384 # Default number of masses
    if len(sys.argv) > 1:
        n = int(sys.argv[1]) # Overwrite default n

    np_omp, t_omp, S_omp, E_omp = fetch_data('omp', n)
    np_mpi, t_mpi, S_mpi, E_mpi = fetch_data('mpi', n)
    np_mpi_shared, t_mpi_shared, S_mpi_shared, E_mpi_shared = fetch_data('mpi_shared', n)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12,8))

    ax1.plot(np_omp, t_omp/(n**2), "-.o", label="Shared-memory OpenMP")
    ax1.plot(np_mpi, t_mpi/(n**2), "-o", label="Distributed-memory MPI")
    ax1.plot(np_mpi_shared, t_mpi_shared/(n**2), "--o", label="Shared-memory MPI")
    ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime / $N^2$ (s)")
    ax1.legend()

    ax2.plot(np_omp, S_omp, "-.o", label="Shared-memory OpenMP")
    ax2.plot(np_mpi, S_mpi, "-o", label="Distributed-memory MPI")
    ax2.plot(np_mpi_shared, S_mpi_shared, "--o", label="Shared-memory MPI")
    ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Speedup $S_1/S_p$")

    ax3.plot(np_omp, E_omp, "-.o", label="Shared-memory OpenMP")
    ax3.plot(np_mpi, E_mpi, "-o", label="Distributed-memory MPI")
    ax3.plot(np_mpi_shared, E_mpi_shared, "--o", label="Shared-memory MPI")
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log", base=10)
    ax3.set_xlabel("Number of parallel processors ($p$)")
    ax3.set_ylabel("Efficiency $S_p/p$")

    fig.suptitle(f"Strong Scaling with $N=${n}")
    fig.savefig(f"local_scaling/parallel_scaling_N={n}.pdf")