#!/bin/bash
# Run plot_phaseAB_3way.jl for all eps/beta combinations (excluding computational)
# Parses x solutions from logs automatically

LOGBASE="logs/factor"
SCRIPT="plot_phaseAB_3way.jl"
cd "$(dirname "$0")"

for dir in $LOGBASE/eps_*/; do
    # skip computational
    [[ "$dir" == *computational* ]] && continue

    dirname=$(basename "$dir")
    # parse eps and beta from folder name: eps_Xp_beta_Yp
    # e.g. eps_0p2_beta_0p4 → eps=0.2, beta=0.4
    eps_raw=$(echo "$dirname" | sed 's/eps_\([^_]*\)_beta_.*/\1/' | sed 's/p/./')
    beta_raw=$(echo "$dirname" | sed 's/eps_[^_]*_beta_//' | sed 's/p/./')

    # skip already done (need both png and jls for all 5 networks)
    plotdir="plots/$dirname"
    if [ -d "$plotdir" ] && [ "$(ls $plotdir/*.jls 2>/dev/null | wc -l)" -ge 5 ]; then
        echo "[skip] $dirname — all jls files exist"
        continue
    fi

    echo ""
    echo "====== $dirname (eps=$eps_raw, beta=$beta_raw) ======"

    for net in grid5x5 polska abilene nobel_us sioux_falls; do
        # Parse x arcs from logs
        nom_log="$dir/${net}_nominal.log"
        sl_log="$dir/${net}_single_l.log"
        dbl_log="$dir/${net}_double.log"

        if [ ! -f "$nom_log" ] || [ ! -f "$sl_log" ] || [ ! -f "$dbl_log" ]; then
            echo "  [skip] $net — missing log(s)"
            continue
        fi

        # Extract arc indices: "x arcs = [28, 37]" → "28,37"
        x_nom=$(grep 'x arcs' "$nom_log" | tail -1 | sed 's/.*\[//;s/\].*//;s/, */,/g')
        x_sl=$(grep 'x arcs' "$sl_log" | tail -1 | sed 's/.*\[//;s/\].*//;s/, */,/g')
        x_dbl=$(grep 'x arcs' "$dbl_log" | tail -1 | sed 's/.*\[//;s/\].*//;s/, */,/g')

        if [ -z "$x_nom" ] || [ -z "$x_sl" ] || [ -z "$x_dbl" ]; then
            echo "  [skip] $net — could not parse x arcs"
            continue
        fi

        echo "  $net: nom=[$x_nom] sl=[$x_sl] dbl=[$x_dbl]"

        # Always pass beta_risk explicitly (0 = expectation)
        beta_arg="beta_risk=$beta_raw"

        julia "$SCRIPT" "$net" factor "$x_nom" "$x_sl" "$x_dbl" \
            eps="$eps_raw" $beta_arg \
            label1=nominal label2=single_l label3=double
    done
done

echo ""
echo "All phaseAB plots done!"
