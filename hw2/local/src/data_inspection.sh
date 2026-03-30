#!/bin/bash

set -euo pipefail

get_headers () {
  datafile=$1
  headers=$(head -n 1 "$datafile" | cut -d'#' -f2 | cut -d'[' -f1 | xargs)
  printf "Column headers: %s\n" "$headers"
}

get_num_rows () {
  datafile=$1
  num_rows=$(tail -n +2 "$datafile" | wc -l)
  printf "Number of data rows (excluding header/comments): %d\n" $num_rows
}

get_midpoint_redshift () {
  datafile=$1
  target="0.5"
  awk -v target="$target" '
  function abs(x){return x<0?-x:x}
  function isnum(s){return s ~ /^[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$/}
  BEGIN{min=1e99; best_z=""; best_xhi=""}
  
  !/^[[:space:]]*#/ {
    z=$1; xhi=$2;
    if(!isnum(z) || !isnum(xhi)) next;
    diff = abs(xhi - target);
    if(diff < min){ min = diff; best_z = z; best_xhi = xhi }
  }
  
  END{
    if(best_z != ""){
      printf("Midpoint of reionization occurs at redshift: %.4f\n", best_z)
    } else {
      print "No valid data rows found in" ARGV[1] > "/dev/stderr";
      exit 1
    }
  }' "$datafile"
}

reion_datafile="data/reion_history_Thesan1.dat"
printf "File: %s\n" $reion_datafile
get_headers "$reion_datafile"
get_num_rows "$reion_datafile"
get_midpoint_redshift "$reion_datafile"

sfrd_datafile="data/sfrd_Thesan1.dat"
printf "File: %s\n" $sfrd_datafile
get_headers "$sfrd_datafile"
get_num_rows "$sfrd_datafile"
