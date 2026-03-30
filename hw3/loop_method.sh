#!/bin/bash

set -euo pipefail

logn_max=${2:-6}    
echo "Iterating $1 up to 10^$logn_max samples:"
echo "Pi estimate, Samples, Total time (s), Sample rate (1/s), Sample time (s)"
logns=$(seq 1 $logn_max)
for logn in $logns; do
    n=$(echo "10^$logn" | bc -l)
    $1 $n
done