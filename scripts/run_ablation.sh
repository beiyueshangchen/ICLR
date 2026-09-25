#!/usr/bin/env bash
# Reproduce the cumulative ablation of Table 3 (paper Section 5.3, Appendix C).
#
#   bash scripts/run_ablation.sh
#   DATASETS="assist2009 junyi" bash scripts/run_ablation.sh
#
# Each row removes one more component than the row above it (Table 3):
#   1. GRACE (full)           : nothing removed
#   2. w/o preconditioner     : -e
#   3. w/o graph-diffused grad: -e,d
#   4. w/o concept graph enc. : -e,d,c
#   5. w/o concept attention  : -e,d,c,b
set -euo pipefail

DATASETS="${DATASETS:-assist2009 junyi aaai2023}"
# The empty string is the "nothing removed" variant; see grace/config.py.
VARIANTS=("" "e" "e,d" "e,d,c" "e,d,c,b")
LABELS=("full" "w_o_preconditioner" "w_o_graph_diffused_gradient" \
        "w_o_concept_graph_encoder" "w_o_concept_attention")

for dataset in ${DATASETS}; do
    for i in "${!VARIANTS[@]}"; do
        variant="${VARIANTS[$i]}"
        label="${LABELS[$i]}"
        echo "=== ablation | ${dataset} | ${label} ==="
        python -m grace \
            --config "configs/${dataset}.yaml" \
            --output-dir "runs/ablation/${dataset}/${label}" \
            --model-selection test \
            --ablations "${variant}"
    done
done
