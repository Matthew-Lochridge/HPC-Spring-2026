import random, sys
from numba import jit, njit, prange
from time import time
import numpy as np

@jit(nopython=True, nogil=True, parallel=True)
def calc_pi_parallel(n):
    h = 0
    for _ in prange(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1:
            h += 1
    return 4. * h / n

if __name__ == "__main__":
    if len(sys.argv) == 1:
        n_ests = int(10) # Default number of pi estimates to compute
    elif len(sys.argv) == 2:
        n_ests = int(float(sys.argv[1])) # Number of pi estimates to compute
    else:
        print("Usage: python pi_comprehensive.py <n_estimates>") # argv = ['pi_comprehensive.py', 'n_estimates']
        sys.exit(1)

    log_n_samps = range(3,10) # log10(n_samples) ranges from 3 to 9

    # Pre-allocate lists for results
    n_samps = [0] * len(log_n_samps) # Number of samples for each log10(log_n_samps) value
    avg_pi_ests = [0] * len(log_n_samps) # Average pi estimates for each number of samples
    err_pi_ests = [0] * len(log_n_samps) # Absolute error of pi estimates for each number of samples
    std_pi_ests = [0] * len(log_n_samps) # Standard deviation of pi estimates for each number of samples
    avg_exec_times = [0] * len(log_n_samps) # Average execution times for each number of samples
    avg_samp_times = [0] * len(log_n_samps) # Average time per sample for each number of samples

    for i in range(len(log_n_samps)): # Iterating over different numbers of samples
        n_samps[i] = int(10**log_n_samps[i])

        # Pre-allocate lists for repeated estimates and times for current number of samples
        pi_ests = [0.0] * n_ests
        exec_times = [0.0] * n_ests
        samp_times = [0.0] * n_ests

        for j in range(n_ests): # Iterating over repeated estimates for current number of samples
            exec_start = time()
            pi_est = calc_pi_parallel(n_samps[i])
            exec_end = time()
            pi_ests[j] = pi_est
            exec_times[j] = exec_end - exec_start
            samp_times[j] = exec_times[j] / n_samps[i]

        avg_pi_ests[i] = np.mean(pi_ests)
        err_pi_ests[i] = abs(avg_pi_ests[i] - np.pi)
        std_pi_ests[i] = np.std(pi_ests)
        avg_exec_times[i] = np.mean(exec_times)
        avg_samp_times[i] = np.mean(samp_times)
    avg_samp_rates = [1./t for t in avg_samp_times] # Average sample rates (samples per second) for each number of samples
    
    # Write results in tabular format
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write("n\tpi estimate\tabsolute error\tstandard deviation\truntime\ttime per sample\tsamples/sec\n")
        for i in range(len(log_n_samps)):
            file.write(f"{n_samps[i]}\t{avg_pi_ests[i]}\t{err_pi_ests[i]}\t{std_pi_ests[i]}\t{avg_exec_times[i]}\t{avg_samp_times[i]}\t{avg_samp_rates[i]}\n")
    # Print results to console
    with open("output.txt", "r", encoding="utf-8") as f:
        print(f.read())