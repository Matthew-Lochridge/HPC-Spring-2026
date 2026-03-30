import random, sys
from numba import jit, njit, prange
from time import time

@njit
def calc_pi_numba(n):
    h = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y < 1.:
            h += 1
    return 4. * h / n

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
        n = int(1e6) # Default number of samples (default format is decimal)
        mode = 'serial' # Default mode for execution
    elif len(sys.argv) == 2:
        n = int(float(sys.argv[1])) # Number of samples (default format is decimal)
        mode = 'serial' # Default mode for execution
    elif len(sys.argv) == 3:
        n = int(float(sys.argv[1])) # Number of samples (default format is decimal)
        mode = sys.argv[2] # Mode for execution (serial or parallel)
        if mode not in ['serial', 'parallel']:
            print("Usage: python pi_numba.py <n> [serial|parallel]") # argv = ['pi_numba.py', 'n', 'serial|parallel']
            sys.exit(1)
    else:
        print("Usage: python pi_numba.py <n> [serial|parallel]") # argv = ['pi_numba.py', 'n', 'serial|parallel']
        sys.exit(1)
    
    if mode == 'serial':
        exec_start = time()
        pi_est = calc_pi_numba(n)
        exec_end = time()
        total_time = exec_end - exec_start
        sample_time = total_time / n
        print(f"{pi_est:0.6f}\t{n:0.0e}\t{total_time:0.6e}\t{round(1./sample_time):0.6e}\t{sample_time:0.6e}")
    elif mode == 'parallel':
        exec_start = time()
        pi_est = calc_pi_parallel(n)
        exec_end = time()
        total_time = exec_end - exec_start
        sample_time = total_time / n
        print(f"{pi_est:0.6f}\t{n:0.0e}\t{total_time:0.6e}\t{round(1./sample_time):0.6e}\t{sample_time:0.6e}")