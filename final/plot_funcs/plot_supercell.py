import numpy as np
from math import isclose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_supercell(allotrope, config):
    max_R_norm = np.max(np.linalg.norm(allotrope.R_gen, axis=1))
    nm = 1e9 * config.r_H
    # Plot atom positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(allotrope.atom_pos[0:allotrope.n_C, 0]*nm, allotrope.atom_pos[0:allotrope.n_C, 1]*nm, allotrope.atom_pos[0:allotrope.n_C, 2]*nm, marker='.', facecolors='black', edgecolors='black')
    labels = ["C"]
    if allotrope.n_H > 0:
        ax.scatter3D(allotrope.atom_pos[allotrope.n_C:allotrope.n_atoms, 0]*nm, allotrope.atom_pos[allotrope.n_C:allotrope.n_atoms, 1]*nm, allotrope.atom_pos[allotrope.n_C:allotrope.n_atoms, 2]*nm, marker='o', facecolors='none', edgecolors='black')
        labels.append("H")
    # Plot interatomic bonds
    for i in range(len(allotrope.atom_rad)-1):
        for j in range(i+1, len(allotrope.atom_rad)):
            dist = np.sqrt( (allotrope.atom_pos[i, 0] - allotrope.atom_pos[j, 0])**2 + (allotrope.atom_pos[i, 1] - allotrope.atom_pos[j, 1])**2 + (allotrope.atom_pos[i, 2] - allotrope.atom_pos[j, 2])**2)
            if np.isclose(dist, config.a_CC, rtol=1e-1):
                ax.plot([allotrope.atom_pos[i, 0]*nm, allotrope.atom_pos[j, 0]*nm], [allotrope.atom_pos[i, 1]*nm, allotrope.atom_pos[j, 1]*nm], [allotrope.atom_pos[i, 2]*nm, allotrope.atom_pos[j, 2]*nm], '-k')
            if allotrope.n_H > 0 and np.isclose(dist, config.a_CH):
                ax.plot([allotrope.atom_pos[i, 0]*nm, allotrope.atom_pos[j, 0]*nm], [allotrope.atom_pos[i, 1]*nm, allotrope.atom_pos[j, 1]*nm], [allotrope.atom_pos[i, 2]*nm, allotrope.atom_pos[j, 2]*nm], '--k')
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
    ax.set_zlabel(r"$z$ (nm)")
    ax.set_xlim((-1.5 * max_R_norm/2 * nm, 1.5 * max_R_norm/2 * nm))
    ax.set_ylim((-1.5 * max_R_norm/2 * nm, 1.5 * max_R_norm/2 * nm))
    ax.set_zlim((-1.5 * max_R_norm/2 * nm, 1.5 * max_R_norm/2 * nm))
    ax.legend(labels)
    plt.title(f"{allotrope.name} Supercell")
    plt.show()
            