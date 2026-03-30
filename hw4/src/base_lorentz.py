# base_lorentz.py
import sys
import numpy as np
from time import time, perf_counter

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax)) # Generate histogram
    return counts

def plot_samples(method, counts, n, bins, xmin, xmax):
    import matplotlib.pyplot as plt

    x_bins = np.linspace(xmin, xmax, bins) # Bin centers
    dx = (xmax - xmin) / bins # Bin width
    pdf = 1. / (np.pi * (1. + x_bins**2)) # Probability density function
    counts_norm = counts / (n * dx) # Normalized counts

    # Plot histogram
    plt.figure(figsize=(12, 8))
    plt.bar(x_bins, counts_norm, label='Samples')
    plt.plot(x_bins, pdf, label='PDF', color='r')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f'{method} method: Lorentzian Distribution with {n:0.0e} Samples and {bins} Bins')
    plt.legend()
    plt.savefig(f'plots/hist_{method}_{n:0.0e}samples_{bins}bins.pdf', bbox_inches="tight")

def lorentzian_histogram_KS(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax)) # Generate histogram
    
    # Large (~n) calculations for correctness checks
    cdf = np.arctan(x) / np.pi + 0.5 # Analytical cumulative distribution function

    return counts, u, x, cdf

def plot_samples_KS(method, counts, u, x_samples, cdf, n, bins, xmin, xmax):
    from scipy.stats.mstats import mquantiles, kstest
    import matplotlib.pyplot as plt
    
    # Sort u, permute x and CDF to match
    order = np.argsort(u)
    u = np.array(u)
    u = u[order]
    x_samples = np.array(x_samples)
    x_samples = x_samples[order]
    cdf = np.array(cdf)
    cdf = cdf[order]
    cdf = cdf[::-1] # Reverse order to account for missing - sign in inverse-transform sampling

    x_bins = np.linspace(xmin, xmax, bins) # Bin centers
    dx = (xmax - xmin) / bins # Bin width

    # Analytical values
    pdf = 1. / (np.pi * (1. + x_bins**2)) # Probability density function
    xq_analytical = np.linspace(-1, 1, 3) # 4-quantiles
    p_q = 1. / (np.pi * (1. + xq_analytical**2)) # PDF evaluated at 4-quantiles

    # Empirical values
    counts_norm = counts / (n * dx) # Normalized counts
    xq_empirical = mquantiles(x_samples) # 4-quantiles
    q_idx = np.zeros(3)
    counts_q = np.zeros(3)
    for i in range(3):
        q_idx[i] = np.abs(x_bins-xq_empirical[i]).argmin() # Indices of bins containing 4-quantiles
        counts_q[i] = counts_norm[q_idx[i].astype(int)] # Counts in bins containing 4-quantiles

    # Kolmogorov-Smirnov test
    ks_test = kstest(u, cdf)
    ks_idx = np.abs(u - ks_test.statistic_location).argmin() # Index of value of u corresponding to K-S statistic

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Visualize u and CDF with K-S statistic
    ax1.semilogx(np.abs(x_samples), u, label='Uniform(0,1)')
    ax1.semilogx(np.abs(x_samples), cdf, color='orange', linestyle='dashed', label='CDF')
    ax1.annotate(text="", 
                     xy=(np.abs(x_samples[ks_idx]), ks_test.statistic_location),
                     xytext=(np.abs(x_samples[ks_idx]), ks_test.statistic_location + ks_test.statistic_sign*ks_test.statistic),
                     arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='k'))
    ax1.set_title(f'{method} method: K-S Test')
    ax1.legend(handles=[plt.Line2D([], [], label='Uniform(0,1)'),
                        plt.Line2D([], [], color='orange', linestyle='dashed', label='CDF'),
                        plt.Line2D([0], [0], color="black", marker="|", linestyle="None", label="KS Statistic")],
               labels=['Uniform(0,1)', 'CDF', 'K-S Statistic'],
               loc='upper left')
    # Visualize residuals of cdf - u
    ax2.semilogx(x_samples, cdf - u) # Reverse order of CDF to account for missing - sign in inverse-transform sampling
    ax2.annotate(text="", 
                     xy=(np.abs(x_samples[ks_idx]), ks_test.statistic_location - u[ks_idx]),
                     xytext=(np.abs(x_samples[ks_idx]), ks_test.statistic_location - u[ks_idx] + ks_test.statistic_sign*ks_test.statistic),
                     arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='k'))
    ax2.set_title(f'{method} method: K-S Test')
    ax2.legend(handles=[plt.Line2D([], [], label='CDF - Uniform(0,1)'),
                        plt.Line2D([0], [0], color="black", marker="|", linestyle="None", label="KS Statistic")],
               labels=['CDF - Uniform(0,1)', 'K-S Statistic'])
    ax2.set_xlabel('|x|')
    ax2.set_title('Residuals of Transformed Samples')
    fig.savefig(f'plots/CDF_KS_{method}_{n:0.0e}samples.pdf', bbox_inches="tight")

    # Plot histogram
    plt.figure(figsize=(12, 8))
    plt.bar(x_bins, counts_norm, label='Samples')
    plt.plot(x_bins, pdf, label='PDF', color='r')
    plt.plot(xq_analytical, p_q, 'o', label='Analytical 4-quantiles', color='r')
    plt.plot(xq_empirical, counts_q, 'o', label='Empirical 4-quantiles', color='k')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f'{method} method: Lorentzian Distribution with {n:0.0e} Samples and {bins} Bins')
    plt.legend()
    plt.savefig(f'plots/hist_{method}_{n:0.0e}samples_{bins}bins.pdf', bbox_inches="tight")

if __name__ == "__main__":

    method = 'base'
    n = int(1e6) # Default number of samples
    bins = 100 # Default number of bins
    xmin = -10 # Default xmin
    xmax = 10 # Default xmax

    # Parse command line arguments, if provided
    if len(sys.argv) >= 2:
        n = int(float(sys.argv[1])) # Overwrite default n
    if len(sys.argv) >= 3:
        bins = int(float(sys.argv[2])) # Overwrite default bins
    if len(sys.argv) >= 4:
        xmin = int(float(sys.argv[3])) # Overwrite default xmin
    if len(sys.argv) == 5:
        xmax = int(float(sys.argv[4])) # Overwrite default xmax
    if len(sys.argv) >= 6:
        print("Usage: python base_lorentz.py <n> <bins> <xmin> <xmax>")
        print("Additional arguments will be ignored.")

    np.random.default_rng(int(time())) # Use current time to set seed

    start_time = perf_counter()
    counts, u, x, cdf = lorentzian_histogram_KS(n, bins, xmin, xmax)
    end_time = perf_counter()
    print(f'Method: {method}')
    print(f'Runtime: {end_time-start_time:0.3f}s')
    plot_samples_KS(method, counts, u, x, cdf, n, bins, xmin, xmax)