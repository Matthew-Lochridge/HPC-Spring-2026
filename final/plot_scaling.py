import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load and parse cpu data
    data = np.loadtxt("data/scaling_numpy.txt", delimiter=' ', skiprows=1, usecols=(1,2,3,4,5,6,7))
    sort_natoms = np.argsort(data[:, 0])
    sort_nG = np.argsort(data[:, 1])
    n_atoms = data[sort_natoms, 0]
    n_G = data[sort_nG, 1]
    cell_time_cpu = data[sort_natoms, 2]
    G_time_cpu = data[sort_nG, 3]
    U_time_cpu = data[sort_nG, 4]
    E_time_cpu = data[sort_nG, 5]
    total_time_cpu = data[sort_nG, 6]

    # Load and parse gpu data
    data = np.loadtxt("data/scaling_cupy.txt", delimiter=' ', skiprows=1, usecols=(1,2,3,4,5,6,7))
    sort_natoms = np.argsort(data[:, 0])
    sort_nG = np.argsort(data[:, 1])
    n_atoms = data[sort_natoms, 0]
    n_G = data[sort_nG, 1]
    cell_time_gpu = data[sort_natoms, 2]
    G_time_gpu = data[sort_nG, 3]
    U_time_gpu = data[sort_nG, 4]
    E_time_gpu = data[sort_nG, 5]
    total_time_gpu = data[sort_nG, 6]

    # Cell runtime vs n_G
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(n_atoms, cell_time_cpu, label="NumPy")
    # ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime (s)")
    ax1.legend()
    ax2.plot(n_atoms, cell_time_gpu, label="CuPy")
    # ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Runtime (s)")
    ax2.legend()
    ax3.plot(n_atoms, cell_time_cpu / cell_time_gpu)
    # ax3.set_xscale("log", base=10)
    # ax3.set_yscale("log", base=10)
    ax3.set_xlabel("Num. atoms")
    ax3.set_ylabel("Speedup")
    fig.suptitle("Supercell construction")
    fig.savefig(f"figures/cell_runtime.pdf", bbox_inches="tight")

    # G runtime vs n_G
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(n_G, G_time_cpu, label="NumPy")
    # ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime (s)")
    ax1.legend()
    ax2.plot(n_G, G_time_gpu, label="CuPy")
    # ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Runtime (s)")
    ax2.legend()
    ax3.plot(n_G, G_time_cpu / G_time_gpu)
    # ax3.set_xscale("log", base=10)
    # ax3.set_yscale("log", base=10)
    ax3.set_xlabel(r"Num. $G$ vectors")
    ax3.set_ylabel("Speedup")
    fig.suptitle(r"$G$ vector generation")
    fig.savefig(f"figures/G_runtime.pdf", bbox_inches="tight")

    # H runtime vs n_G
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(n_G, U_time_cpu, label="NumPy")
    ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime (s)")
    ax1.legend()
    ax2.plot(n_G, U_time_gpu, label="CuPy")
    ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Runtime (s)")
    ax2.legend()
    ax3.plot(n_G, U_time_cpu / U_time_gpu)
    # ax3.set_xscale("log", base=10)
    # ax3.set_yscale("log", base=10)
    ax3.set_xlabel(r"Num. $G$ vectors")
    ax3.set_ylabel("Speedup")
    fig.suptitle("Pseudopotential calculation")
    fig.savefig(f"figures/U_runtime.pdf", bbox_inches="tight")

    # E runtime vs n_G
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(n_G, E_time_cpu, label="NumPy")
    ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime (s)")
    ax1.legend()
    ax2.plot(n_G, E_time_gpu, label="CuPy")
    ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Runtime (s)")
    ax2.legend()
    ax3.plot(n_G, E_time_cpu / E_time_gpu)
    # ax3.set_xscale("log", base=10)
    # ax3.set_yscale("log", base=10)
    ax3.set_xlabel(r"Num. $G$ vectors")
    ax3.set_ylabel("Speedup")
    fig.suptitle(r"$H$ diagonalization")
    fig.savefig(f"figures/E_runtime.pdf", bbox_inches="tight")

    # Total runtime vs n_G
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(n_G, total_time_cpu, label="NumPy")
    ax1.set_yscale("log", base=10)
    ax1.set_ylabel("Runtime (s)")
    ax1.legend()
    ax2.plot(n_G, total_time_gpu, label="CuPy")
    ax2.set_yscale("log", base=10)
    ax2.set_ylabel("Runtime (s)")
    ax2.legend()
    ax3.plot(n_G, total_time_cpu / total_time_gpu)
    # ax3.set_xscale("log", base=10)
    # ax3.set_yscale("log", base=10)
    ax3.set_xlabel(r"Num. $G$ vectors")
    ax3.set_ylabel("Speedup")
    fig.suptitle("Total runtime")
    fig.savefig(f"figures/total_runtime.pdf", bbox_inches="tight")
