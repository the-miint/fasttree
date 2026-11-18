#!/bin/sh
# check_tree.sh LABEL REFERENCE COMMAND
#
# Runs COMMAND, one shell command line that may redirect its input, and
# passes when it exits 0 and its standard output equals REFERENCE, a saved
# Newick tree. On a failure it says which check failed and prints the end
# of the command's standard error. The output is kept in test-out/.
set -u
label=$1
ref=$2
cmd=$3
mkdir -p test-out
base=test-out/$(printf '%s' "$label" | tr -c 'A-Za-z0-9.-' '_')
sh -c "$cmd" > "$base.nwk" 2> "$base.err"
status=$?
if [ "$status" -ne 0 ]; then
  echo "FAIL: $label (exit status $status)"
elif ! cmp -s "$base.nwk" "$ref"; then
  echo "FAIL: $label (tree differs from $ref)"
else
  echo "PASS: $label"
  exit 0
fi
tail -n 5 "$base.err" | sed 's/^/  /'
exit 1
