import numpy as np
import matplotlib.pyplot as plt


def load_table(fname):
    data = np.loadtxt(fname, comments="#")
    z = data[:, 0]
    vals = data[:, 1]
    # ensure increasing z for interpolation
    order = np.argsort(z)
    return z[order], vals[order]


def main():
    z_hi, xHI = load_table("data/reion_history_Thesan1.dat")
    z_sfrd, sfrd = load_table("data/sfrd_Thesan1.dat")

    # reion history: xHI vs z
    plt.figure()
    plt.plot(z_hi, xHI, lw=1)
    plt.xlabel("z (redshift)")
    plt.ylabel("x_HI (volume-weighted neutral hydrogen fraction)")
    plt.title("Global cosmic reionization history")
    plt.gca().invert_xaxis()
    plt.savefig("plots/reion.pdf", bbox_inches="tight")
    plt.close()

	# split y-axis: linear [0.1,1] on top, log [1e-4,0.1] on bottom
    fig, (ax_top, ax_bot) = plt.subplots(
		2,
		1,
		gridspec_kw={"height_ratios": [1, 1]},
		figsize=(6, 6),
	)
    ax_top.plot(z_hi, xHI, lw=1)
    ax_top.set_xlim(0.9*min(z_hi), 1.1*max(z_hi))
    ax_top.set_ylim(0.1, 1.1)
    ax_top.set_yscale("linear")
    ax_top.spines["bottom"].set_visible(False)
    ax_top.xaxis.set_ticks([])

    ax_bot.plot(z_hi, xHI, lw=1)
    ax_bot.set_xlim(0.9*min(z_hi), 1.1*max(z_hi))
    ax_bot.set_ylim(1e-4, 0.1)
    ax_bot.set_yscale("log")
    ax_bot.spines["top"].set_visible(False)

    # diagonal break markers
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    kwargs = dict(transform=ax_bot.transAxes, color="k", clip_on=False)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_bot.set_xlabel("z (redshift)")
    ax_bot.set_ylabel("x_HI")
    ax_top.invert_xaxis()
    ax_bot.invert_xaxis()
    fig.suptitle("Global cosmic reionization history")
    plt.savefig("plots/reion_split.pdf", bbox_inches="tight")
    plt.close()

    # SFRD vs z (use log scale for y)
    plt.figure()
    plt.plot(z_sfrd, sfrd, lw=1)
    plt.xlabel("z (redshift)")
    plt.ylabel("SFRD (Msun/yr/Mpc^3)")
    plt.yscale("log")
    plt.title("Global star formation rate density")
    plt.gca().invert_xaxis()
    plt.savefig("plots/sfrd.pdf", bbox_inches="tight")
    plt.close()

    # Map SFRD onto xHI by interpolating SFRD to the redshifts of xHI
    sfrd_on_hi = np.interp(z_hi, z_sfrd, sfrd)
    plt.figure()
    plt.plot(xHI, sfrd_on_hi, lw=1)
    plt.xlabel("x_HI (volume-weighted neutral hydrogen fraction)")
    plt.ylabel("SFRD (Msun/yr/Mpc^3)")
    plt.yscale("log")
    plt.title("Global star formation rate density")
    plt.gca().invert_xaxis()
    plt.savefig("plots/sfrd_HI.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main() 
