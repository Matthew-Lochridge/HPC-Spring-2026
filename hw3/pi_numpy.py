from numpy import sum
from numpy.random import rand
import sys
from time import time

def calc_pi_numpy(n):
    sample_start = time()
    h = sum(rand(n)**2 + rand(n)**2 < 1.)
    sample_end = time()
    return 4. * float(h) / float(n), (sample_end-sample_start)/n # Estimate pi

if __name__ == "__main__":
    if len(sys.argv) == 1:
        n = int(1e6) # Default number of samples (default format is decimal)
    elif len(sys.argv) == 2:
        n = int(float(sys.argv[1])) # Number of samples (default format is decimal)
    else:
        print("Usage: python pi_numpy.py <n>") # argv = ['pi_numpy.py', 'n']
        sys.exit(1)
    
    exec_start = time()
    pi_est, sample_time = calc_pi_numpy(n)
    exec_end = time()
    print(f"{pi_est:0.6f}\t{n:0.0e}\t{exec_end-exec_start:0.6e}\t{round(1./sample_time):0.6e}\t{sample_time:0.6e}")