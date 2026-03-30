import matplotlib.pyplot as plt
from numpy import loadtxt

if __name__ == "__main__":
    data = loadtxt("local_scaling/serial_scaling.txt")
    n = data[:, 0]
    t = data[:, 1]
    plt.figure()
    plt.plot(n, t, "-o")
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    plt.xlabel("N")
    plt.ylabel("Runtime (s)")
    plt.title("Serial Scaling")
    plt.savefig("local_scaling/serial_scaling.pdf")