# cython: language_level=3
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from time import time

@cython.boundscheck(False) # Disable bounds checking for performance
@cython.wraparound(False) # Disable negative indexing for performance
def calc_pi_cython(long long int n):
    cdef:
        long long int i, h = 0
        double x, y, sample_start, sample_end
    sample_start = time()
    for i in range(n):
        x = rand() / RAND_MAX # Generate a random x coordinate
        y = rand() / RAND_MAX # Generate a random y coordinate
        # Check if the point (x, y) is inside the unit circle
        if x*x + y*y < 1.:
            h += 1
    sample_end = time()
    # Estimate Pi using the ratio of points inside the circle to the total points
    return 4. * h / n, (sample_end-sample_start)/n