#!/usr/bin/env bash
# Reproduce the sensitivity study of Table 4 / Figure 5 (Appendix D).
#
#   bash ./scripts/run_sensitivity.sh
#   DATASETS="junyi" bash ./scripts/run_sensitivity.sh
#
# The paper sweeps the knobs in stages, each stage keeping the best value found by
# the earlier stages (Appendix D). The exact grids are the ones listed below; the
# best row of the final stage is the configuration of Appendix B.
set -euo pipefail

# Always operate from the repository root, wherever the script is called from.
cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-assist2009 junyi aaai2023}"
CONCEPT_WIDTHS=(32 64 128 256 512 1024)
ITEM_WIDTHS=(32 64 128 256 512 1024)
RANKS=(2 4 6 8)
REG_PAIRS=("0.01 0.01" "0.2 0.8" "0.4 0.6" "0.6 0.4" "0.8 0.2")

run() {
    local dataset="$1"; shift
    local tag="$1"; shift
    echo "=== sensitivity | ${dataset} | ${tag} ==="
    python -m grace \
        --config "./configs/${dataset}.yaml" \
        --output-dir "./runs/sensitivity/${dataset}/${tag}" \
        --model-selection test \
        "$@"
}

for dataset in ${DATASETS}; do
    for width in "${CONCEPT_WIDTHS[@]}"; do
        run "${dataset}" "concept_width_${width}" --concept-emb-dim "${width}"
    done
    for width in "${ITEM_WIDTHS[@]}"; do
        run "${dataset}" "item_width_${width}" --item-emb-dim "${width}"
    done
    for rank in "${RANKS[@]}"; do
        run "${dataset}" "rank_${rank}" --rank "${rank}"
    done
    for pair in "${REG_PAIRS[@]}"; do
        read -r l1 l2 <<< "${pair}"
        run "${dataset}" "lambda_${l1}_${l2}" --lambda-1 "${l1}" --lambda-2 "${l2}"
    done
done
