import sys
import numpy as np
from config import parameters
from backend import get_backend, sync, to_cpu, Timer
from construct_allotrope import construct_allotrope
from kurokawa import V_C, V_H
from plot_funcs.plot_supercell import plot_supercell
from plot_funcs.plot_bands import plot_bands
from plot_funcs.plot_probs import plot_probs

if __name__ == "__main__":
    '''
    Main function in serial
    Ibackend.xputs:
        Allotrope name (string)
        Include H (bool)
        Show supercell (bool)
    '''

    # Parse command line arguments
    allotrope_name = ""
    include_H = 0
    prefer_gpu = 0
    show_supercell = 0
    match len(sys.argv):
        case 2:
            allotrope_name = sys.argv[1]
        case 3:
            allotrope_name = sys.argv[1]
            include_H = int(sys.argv[2])
        case 4:
            allotrope_name = sys.argv[1]
            include_H = int(sys.argv[2])
            prefer_gpu = int(sys.argv[3])
        case 5:
            allotrope_name = sys.argv[1]
            include_H = int(sys.argv[2])
            prefer_gpu = int(sys.argv[3])
            show_supercell = int(sys.argv[4])
        case _:
            print("Error: Allotrope not specified.")
            exit(-1)

    # Initialize backend (GPU if available and preferred, else CPU)
    backend = get_backend(prefer_gpu)  # or get_backend(prefer_gpu=False) for CPU-only

    # Fetch configuration parameters
    config = parameters()
    
    # Construct allotrope
    print("Constructing allotrope...")
    with Timer(backend) as timer:
        allotrope = construct_allotrope(allotrope_name, config, include_H, backend)
        sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
    cell_time = timer.dt
    print(f"Finished in {cell_time} seconds.")
    if allotrope is None:
        print(f"Error: {allotrope_name} not recognized.")
        exit(-1)

    # Optionally show supercell for validation
    if show_supercell:
        plot_supercell(allotrope, config)

    print("Generating reciprocal lattice vectors...")
    with Timer(backend) as timer:
        # Generators of reciprocal lattice vectors
        G_gen = 2*np.pi/allotrope.V_cell * backend.xp.array([backend.xp.cross(allotrope.R_gen[1, :], allotrope.R_gen[2, :]),
                                            backend.xp.cross(allotrope.R_gen[2, :], allotrope.R_gen[0, :]),
                                            backend.xp.cross(allotrope.R_gen[0, :], allotrope.R_gen[1, :])])
        # Generate all reciprical lattice vectors within max_G Manhattan distance
        combo_gen = backend.xp.arange(-config.max_G, config.max_G+1, 1)
        combo = backend.xp.array([backend.xp.kron(backend.xp.kron(combo_gen, backend.xp.ones_like(combo_gen)), backend.xp.ones_like(combo_gen)), 
                backend.xp.kron(backend.xp.kron(backend.xp.ones_like(combo_gen), combo_gen), backend.xp.ones_like(combo_gen)), 
                backend.xp.kron(backend.xp.kron(backend.xp.ones_like(combo_gen), backend.xp.ones_like(combo_gen)), combo_gen)]).T
        G = combo @ G_gen
        # Retain vectors within the cutoff energy
        G2 = backend.xp.linalg.norm(G, axis=1)**2
        keep_idx = backend.xp.where(G2 < config.E_cut)[0]
        G = G[keep_idx, :]
        n_G = G.shape[0]
        GmG =  backend.xp.kron(backend.xp.ones((1, n_G)), G) - backend.xp.kron(backend.xp.ones((n_G, 1)), backend.xp.reshape(G, (1, 3*n_G))) # Compute differences between reciprocal lattice vectors
        sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
    G_time = timer.dt
    print(f"Finished in {G_time} seconds.")
    print(f"Using {n_G} reciprocal lattice vectors.")

    print("Computing pseudopotentials...")
    with Timer(backend) as timer:
        U = backend.xp.zeros((n_G, n_G))
        for i_atom, r_atom in enumerate(allotrope.atom_rad):
            match r_atom:
                case 1:
                    U = U + 1/allotrope.V_cell * backend.xp.exp(1j * GmG @ backend.xp.kron(backend.xp.identity(n_G), allotrope.atom_pos[i_atom, :]).T) * V_H(backend.xp.sqrt(GmG**2 @ backend.xp.kron(backend.xp.identity(n_G), backend.xp.ones(3)).T), config)
                case config.r_C:
                    U = U + 1/allotrope.V_cell * backend.xp.exp(1j * GmG @ backend.xp.kron(backend.xp.identity(n_G), allotrope.atom_pos[i_atom, :]).T) * V_C(backend.xp.sqrt(GmG**2 @ backend.xp.kron(backend.xp.identity(n_G), backend.xp.ones(3)).T), config)
        sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
    U_time = timer.dt
    print(f"Finished in {U_time} seconds.")

    print("Diagonalizing Hamiltonian...")
    with Timer(backend) as timer:
        E = backend.xp.zeros((config.n_k, n_G))
        # u = np.cdouble(1) * backend.xp.ones((n_G, 6)) # Store eigenvectors with k = 0 for the highest 3 valence bands and lowest 3 conduction bands
        for i, k_i in enumerate(allotrope.k):
            T = backend.xp.diag(backend.xp.sum(backend.xp.absolute(G + backend.xp.array([backend.xp.ones(n_G), backend.xp.ones(n_G), backend.xp.ones(n_G)]).T * k_i)**2, axis=1))
            E_vec, u_mat = backend.xp.linalg.eigh(T + U)
            sort_idx = backend.xp.argsort(E_vec.real)
            E[i, :] = E_vec[sort_idx].real
            '''
            if backend.xp.linalg.norm(k_i) == 0.:
                u_mat = u_mat[:, sort_idx]
                u[:, :] = u_mat[:, allotrope.n_valence-3:allotrope.n_valence+3]
            '''
        E = E - backend.xp.max(E[:, allotrope.n_valence-1]) * backend.xp.ones_like(E) # Set valence band maximum to zero
        sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
    E_time = timer.dt
    print(f"Finished in {E_time} seconds.")

    print(f"Saving band structure...")
    cpu_E = to_cpu(E)
    with open(f"data/{allotrope.name}_bands_{backend.name}.txt", "w", encoding="utf-8") as file:
        for i, _ in enumerate(allotrope.k):
            for j in range(n_G):
                file.write(f"{cpu_E[i, j]} ")
            file.write("\n")
    print(f"Saved data/{allotrope.name}_bands_{backend.name}.txt")
    plot_bands(cpu_E, allotrope, config, backend)

    '''
    print("Computing ground state wavefunctions...")
    psi2 = backend.xp.zeros((n_G, n_G, 6))
    with Timer(backend) as timer:
        combo_gen = backend.xp.linspace(-0.5, 0.5, n_G)
        combo = backend.xp.array([backend.xp.kron(backend.xp.kron(combo_gen, backend.xp.ones_like(combo_gen)), backend.xp.ones_like(combo_gen)), 
                backend.xp.kron(backend.xp.kron(backend.xp.ones_like(combo_gen), combo_gen), backend.xp.ones_like(combo_gen)), 
                backend.xp.kron(backend.xp.kron(backend.xp.ones_like(combo_gen), backend.xp.ones_like(combo_gen)), combo_gen)]).T
        r = combo @ allotrope.R_gen
        for n in range(6):
            psi2[:, :, n] = backend.xp.average(backend.xp.absolute(backend.xp.sum(u[:, n] * backend.xp.exp(1j * backend.xp.sum(G @ r.T, axis=1)))) ** 2) / allotrope.V_cell
        sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
    psi2_time = timer.dt
    print(f"Finished in {psi2_time} seconds.")

    print(f"Saving probability densities...")
    cpu_psi2 = to_cpu(psi2)
    for n in range(6):
        n_band = allotrope.n_valence - 3 + n
        with open(f"data/{allotrope.name}_probs_band{n_band+1}_{backend.name}.txt", "w", encoding="utf-8") as file:
            for i in range(config.n_x):
                for j in range(config.n_x):
                    file.write(f"{cpu_psi2[i, j, n_band]} ")
                file.write("\n")
        print(f"Saved data/{allotrope.name}_probs_band{n_band+1}_{backend.name}.txt")
    plot_probs(cpu_psi2, allotrope, config, backend)
    '''

    total_time = cell_time + G_time + U_time + E_time # + psi2_time
    # Save scaling info
    with open(f"data/scaling_{backend.name}.txt", "a+", encoding="utf-8") as file:
        if file.tell() == 0: # If file is empty, write header
            file.write("# allotrope, n_atoms, n_G, cell_time (s), G_time (s), U_time (s), E_time (s), total_time (s)\n")
        file.write(f"{allotrope.name} {int(allotrope.n_atoms)} {n_G} {cell_time} {G_time} {U_time} {E_time} {total_time}\n")
    print(f"Saved data/scaling_{backend.name}.txt")
