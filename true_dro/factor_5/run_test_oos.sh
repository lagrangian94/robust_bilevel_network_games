#!/bin/bash
# Run OOS test (seed=43) for beta=0.7 and beta=0.05
# Each network uses its validation-optimal epsilon
cd "$(dirname "$0")"
SCRIPT="plot_phaseAB_3way.jl"

# ── beta=0.7: out → test_seed43/beta_0p7 ──
echo "===== beta=0.7 ====="

julia "$SCRIPT" grid5x5 factor "27,28" "27,28" "27,28" \
    beta_risk=0.7 eps=0.1 oos_seed=43 out_folder=test_seed43/beta_0p7 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" polska factor "13,34" "13,34" "13,34" \
    beta_risk=0.7 eps=0.1 oos_seed=43 out_folder=test_seed43/beta_0p7 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" abilene factor "26,27" "26,27" "5,27" \
    beta_risk=0.7 eps=0.2 oos_seed=43 out_folder=test_seed43/beta_0p7 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" nobel_us factor "6,9" "6,9" "9,20" \
    beta_risk=0.7 eps=0.2 oos_seed=43 out_folder=test_seed43/beta_0p7 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" sioux_falls factor "54,72" "35,72" "35,72" \
    beta_risk=0.7 eps=0.1 oos_seed=43 out_folder=test_seed43/beta_0p7 \
    label1=nominal label2=single_l label3=double

# ── beta=0.05: out → test_seed43/beta_0p05 ──
echo "===== beta=0.05 ====="

julia "$SCRIPT" grid5x5 factor "28,37" "28,37" "28,37" \
    beta_risk=0.05 eps=0.1 oos_seed=43 out_folder=test_seed43/beta_0p05 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" polska factor "3,33" "3,33" "13,34" \
    beta_risk=0.05 eps=0.2 oos_seed=43 out_folder=test_seed43/beta_0p05 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" abilene factor "5,26" "26,27" "5,11" \
    beta_risk=0.05 eps=0.2 oos_seed=43 out_folder=test_seed43/beta_0p05 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" nobel_us factor "6,9" "6,9" "6,9" \
    beta_risk=0.05 eps=0.1 oos_seed=43 out_folder=test_seed43/beta_0p05 \
    label1=nominal label2=single_l label3=double

julia "$SCRIPT" sioux_falls factor "54,72" "54,72" "54,72" \
    beta_risk=0.05 eps=0.2 oos_seed=43 out_folder=test_seed43/beta_0p05 \
    label1=nominal label2=single_l label3=double

echo ""
echo "All test OOS done!"
