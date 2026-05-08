import matplotlib.pyplot as plt

def plot_probs(psi2, allotrope, config, backend):
    print("Plotting band structure...")
    fig, ax = plt.subplots(2, 3)
    for n in range(6):
        ax[n].imshow(psi2[:, :, n] / (1e9*config.r_H)**3, origin='lower', extent=(-allotrope.R_gen[2,2]/2*config.r_H*1e9, allotrope.R_gen[2,2]/2*config.r_H*1e9, -allotrope.R_gen[3,3]/2*config.r_H*1e9, allotrope.R_gen[3,3]/2*config.r_H*1e9))
        ax[n].colorbar()
        ax[n].set_xlabel(r"$y$ (nm)")
        ax[n].set_ylabel(r"$z$ (nm)")
        ax[n].set_title(f"Band {allotrope.n_valence - 2 + n}")
    fig.savefig(f"figures/{allotrope.name}_probs_{backend.name}.pdf", bbox_inches="tight")
    print(f"Saved figures/{allotrope.name}_probs_{backend.name}.pdf.")
