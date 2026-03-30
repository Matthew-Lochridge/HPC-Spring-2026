import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pandas import DataFrame

def amdahl_func(n_proc, p_frac=1):
    return 1./(1-p_frac+p_frac/n_proc)

def gustafson_func(n_proc, p_frac=1):
    return 1 + (n_proc-1)*p_frac

def read_single(method, scaling, maxpar=32):
    num_conc = int(np.log2(maxpar)+1)
    parallelism_unique = np.zeros(num_conc)
    runtime_unique = np.zeros(num_conc)

    data = np.loadtxt(f"results/{scaling}_scaling_{method}.txt")
    parallelism = np.array(data[:,0])
    order = np.argsort(parallelism)
    parallelism = np.sort(parallelism)
    parallelism = parallelism.tolist()
    runtime = data[:,1]
    runtime = runtime[order]

    for j in range(num_conc):
        loc_unique = parallelism.index(2**j)
        parallelism_unique[j] = parallelism[loc_unique]
        runtime_unique[j] = runtime[loc_unique]

    return parallelism_unique, runtime_unique

def average_trials(method, scaling, maxpar=32, trials=5):
    num_conc = int(np.log2(maxpar)+1)
    parallelism_unique = np.zeros(num_conc)
    all_runtimes = np.zeros((num_conc, trials))
    for i in range:
        data = np.loadtxt(f"results/trial{i}/{scaling}_scaling_{method}.txt")
        parallelism = np.array(data[:,0])
        order = np.argsort(parallelism)
        parallelism = np.sort(parallelism)
        parallelism = parallelism.tolist()
        runtime = np.array(data[:,1])
        runtime = runtime.tolist()

        for j in range(num_conc):
            loc_unique = parallelism.index(2**j)
            parallelism_unique[j] = parallelism[loc_unique]
            all_runtimes[j, i] = runtime[loc_unique]

    std_dev = np.std(all_runtimes, axis=1)
    avg_runtimes = np.mean(all_runtimes, axis=1)

    return parallelism_unique, avg_runtimes, std_dev/np.sqrt(trials)

def plot_single(method, maxpar=32, n_base=int(1e7)):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Weak scaling
    parallelism, wruntime = read_single(method, 'weak', maxpar)
    ax1.plot(parallelism, wruntime, '-o', label='weak scaling')
    ax1.set_xlabel('parallelism (p)')
    ax1.set_ylabel('runtime (s)')
    ax1.legend()

    # Strong scaling
    parallelism, sruntime = read_single(method, 'strong', maxpar)
    sspeedup = sruntime[0]/sruntime
    sefficiency = sspeedup/parallelism

    ax2.plot(parallelism, sruntime, '-o', label='strong scaling')
    ax2.set_xlabel('parallelism (p)')
    ax2.set_ylabel('runtime (s)')
    ax2.legend()
    
    ax4.plot(parallelism, sefficiency, '-o', label='empirical efficiency')
    ax4.set_xlabel('parallelism (p)')
    ax4.set_ylabel('efficiency ($E_p=S_p/p$)')
    ax4.legend()

    opt, cov = curve_fit(amdahl_func, parallelism, sspeedup, 1)
    p_frac_amdahl = opt[0]
    p_frac_amdahl_err = np.sqrt(cov[0][0])
    fit_amdahl = amdahl_func(parallelism, p_frac_amdahl)

    opt, cov = curve_fit(gustafson_func, parallelism, sspeedup, 1)
    p_frac_gustafson = opt[0]
    p_frac_gustafson_err = np.sqrt(cov[0][0])
    fit_gustafson = gustafson_func(parallelism, p_frac_gustafson)

    ax3.plot(parallelism, sspeedup, '-o', label='empirical speedup')
    ax3.plot(parallelism, fit_amdahl, linestyle='--', label=f"Amdahl's law with serial fraction = ({1-p_frac_amdahl:.3f} ± {p_frac_amdahl_err:.3f})")
    ax3.plot(parallelism, fit_gustafson, linestyle='--', label=f"Gustafson's law with serial fraction = ({1-p_frac_gustafson:.3f} ± {p_frac_gustafson_err:.3f})")
    ax3.set_xlabel('parallelism (p)')
    ax3.set_ylabel('speedup ($S_p=T_1/T_p$)')
    ax3.legend()

    fig.suptitle(f"{method} method: scaling with {n_base:0.0e} base samples")
    fig.savefig(f'plots/final_scaling_{method}_single_maxpar={maxpar}_nbase={n_base:0.0e}.pdf', bbox_inches="tight")

    df = DataFrame({
        "Concurrency": parallelism,
        "Speedup": sspeedup
    })

    # Render the table and save it as an image
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(f"data/{method}_final_scaling_table.png")

def plot_average(method, maxpar=32, trials=5, n_base=int(1e7)):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Weak scaling
    parallelism, wruntime, wstd_err = average_trials(method, 'weak', trials)
    ax1.errorbar(parallelism, wruntime, yerr=wstd_err, fmt='-o', linewidth=2, capsize=6, label='weak scaling')
    ax1.set_xlabel('parallelism (p)')
    ax1.set_ylabel('runtime (s)')
    ax1.legend()

    # Strong scaling
    parallelism, sruntime, sstd_err = average_trials(method, 'strong', trials)
    sspeedup = sruntime[0]/sruntime
    sspeedup_err = np.sqrt((sstd_err[0]/sruntime)**2 + (-sruntime[0]*sstd_err/(sruntime**2))**2)
    sefficiency = sspeedup/parallelism
    sefficiency_err = sspeedup_err/parallelism

    ax2.errorbar(parallelism, sruntime, yerr=sstd_err, fmt='-o', linewidth=2, capsize=6, label='strong scaling')
    ax2.set_xlabel('parallelism (p)')
    ax2.set_ylabel('runtime (s)')
    ax2.legend()
    
    ax4.errorbar(parallelism, sefficiency, yerr=sefficiency_err, fmt='-o', linewidth=2, capsize=6, label='empirical efficiency')
    ax4.set_xlabel('parallelism (p)')
    ax4.set_ylabel('efficiency ($E_p=S_p/p$)')
    ax4.legend()

    opt, cov = curve_fit(amdahl_func, parallelism, sspeedup, 1)
    p_frac_amdahl = opt[0]
    p_frac_amdahl_err = np.sqrt(cov[0][0])
    fit_amdahl = amdahl_func(parallelism, p_frac_amdahl)

    opt, cov = curve_fit(gustafson_func, parallelism, sspeedup, 1)
    p_frac_gustafson = opt[0]
    p_frac_gustafson_err = np.sqrt(cov[0][0])
    fit_gustafson = gustafson_func(parallelism, p_frac_gustafson)

    ax3.errorbar(parallelism, sspeedup, yerr=sspeedup_err, fmt='-o', linewidth=2, capsize=6, label='empirical speedup')
    ax3.plot(parallelism, fit_amdahl, linestyle='--', label=f"Amdahl's law with serial fraction = ({1-p_frac_amdahl:.3f} ± {p_frac_amdahl_err:.3f})")
    ax3.plot(parallelism, fit_gustafson, linestyle='--', label=f"Gustafson's law with serial fraction = ({1-p_frac_gustafson:.3f} ± {p_frac_gustafson_err:.3f})")
    ax3.set_xlabel('parallelism (p)')
    ax3.set_ylabel('speedup ($S_p=T_1/T_p$)')
    ax3.legend()

    fig.suptitle(f"{method} method: scaling with {n_base:0.0e} base samples and {trials} trials")
    fig.savefig(f'plots/scaling_{method}_maxpar={maxpar}_trials={trials}_nbase={n_base:0.0e}.pdf', bbox_inches="tight")

    df = DataFrame({
        "Concurrency": parallelism,
        "Speedup": sspeedup
    })

    # Render the table and save it as an image
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(f"data/{method}_scaling_table.png")

if __name__ == "__main__":
    
    maxpar = 128
    trials = 5
    n_base = int(1e9)
    if len(sys.argv) >= 2:
        trials = int(sys.argv[1])
    if len(sys.argv) == 3:
        n = int(sys.argv[2])

    methods = ['threading', 'multiprocessing', 'ProcessPoolExecutor', 'AsyncIO', 'Dask', 'Numba', 'Joblib', 'mpire', 'MPI']

    for method in methods:
        if method == 'MPI':
            plot_single(method, maxpar, n_base)