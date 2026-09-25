#!/usr/bin/env bash
# Train GRACE on a single dataset.
#
#   bash scripts/train.sh assist2009
#   CONFIG=configs/junyi.yaml OUTPUT_DIR=runs/junyi bash scripts/train.sh junyi
#
# ASSISTments 2009 ships with the repository and needs no path configuration;
# Junyi and AAAI2023 must be reachable either through --dataset-root / --data-root
# in the config or through the GRACE_*_ROOT / GRACE_DATA_ROOT environment variables.
set -euo pipefail

DATASET="${1:-assist2009}"
CONFIG="${CONFIG:-configs/${DATASET}.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== GRACE | dataset=${DATASET} | config=${CONFIG} | output_dir=${OUTPUT_DIR} ==="
python -m grace \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}
