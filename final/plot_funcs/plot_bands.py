import numpy as np
import matplotlib.pyplot as plt

def plot_bands(E, allotrope, config, backend):
    print("Plotting band structure...")
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(E*config.Ry)
    ax.set_xlim((np.min(allotrope.k[:, 0]), np.max(allotrope.k[:, 0])))
    ax.set_ylim((-5, 5))
    ax.set_xlabel(allotrope.k_label)
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_xticks(allotrope.k_ticks)
    ax.set_xticklabels(allotrope.k_ticklabels)
    ax.set_title(allotrope.name)
    fig.savefig(f"figures/{allotrope.name}_bands_{backend.name}.pdf", bbox_inches="tight")
    print(f"Saved figures/{allotrope.name}_bands_{backend.name}.pdf.")
