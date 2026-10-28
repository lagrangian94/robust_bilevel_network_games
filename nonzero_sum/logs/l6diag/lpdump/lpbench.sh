G=/c/gurobi1301/win64/bin/gurobi_cl
for f in "$1"/ok_*.mps; do
  echo "## $(basename $f)"
  for cfg in "Method=1 Presolve=0" "Method=1 Presolve=-1" "Method=0 Presolve=-1" "Method=2 Crossover=0 Threads=1" "Method=2 Crossover=0 Threads=4" "Method=2 Threads=1" "Method=2 Threads=4" "Method=2 Threads=12"; do
    out=$($G Threads=1 TimeLimit=300 FeasibilityTol=1e-7 OptimalityTol=1e-7 $cfg $f 2>&1)
    t=$(echo "$out" | grep -E '^(Solved|Stopped|Barrier performed|Infeasible)' | grep -oE '[0-9.]+ seconds' | tail -1)
    st=$(echo "$out" | grep -E '^(Optimal objective|Sub-optimal|Infeasible model|Time limit|Numerical)' | tail -1)
    printf "  %-30s %12s | %s\n" "$cfg" "$t" "$st"
  done
done
