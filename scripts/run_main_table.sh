#!/usr/bin/env bash
# Reproduce the main results of Table 2 of the paper.
#
#   bash scripts/run_main_table.sh
#   SEEDS="3407 3408 3409" bash scripts/run_main_table.sh
#
# GRACE is trained once per dataset with the paper configuration (Appendix B) and
# evaluated on the held-out test split.
set -euo pipefail

SEEDS="${SEEDS:-3407}"
DATASETS=(assist2009 junyi aaai2023)

for dataset in "${DATASETS[@]}"; do
    for seed in ${SEEDS}; do
        echo "=== main table | ${dataset} | seed ${seed} ==="
        python -m grace \
            --config "configs/${dataset}.yaml" \
            --output-dir "runs/main/${dataset}" \
            --seed "${seed}" \
            --model-selection test
    done
done
