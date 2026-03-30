from calc_pi import calc_pi_cython
import sys
from time import time

if len(sys.argv) == 1:
    n = int(1e6) # Default number of samples (default format is decimal)
elif len(sys.argv) == 2:
    n = int(float(sys.argv[1])) # Number of samples (default format is decimal)
else:
    print("Usage: python pi_cython.py <n>") # argv = ['pi_cython.py', 'n']
    sys.exit(1)
    
exec_start = time() # Start timer
pi_est, sample_time = calc_pi_cython(n) # Calculate pi
exec_end = time() # End timer
print(f"{pi_est:0.6f}\t{n:0.0e}\t{exec_end-exec_start:0.6e}\t{round(1./sample_time):0.6e}\t{sample_time:0.6e}")