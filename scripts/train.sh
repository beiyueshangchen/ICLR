#!/usr/bin/env bash
# Train GRACE on a single dataset.
#
#   bash ./scripts/train.sh                            # Junyi (shipped, no setup)
#   CONFIG=./configs/junyi.yaml bash ./scripts/train.sh junyi
#
# Junyi ships with the repository and needs no path configuration. ASSISTments
# 2009 and AAAI2023 are not redistributed, so they must be reachable either
# through --dataset-root / --data-root in the config or through the
# GRACE_*_ROOT / GRACE_DATA_ROOT environment variables.
set -euo pipefail

# Always operate from the repository root, wherever the script is called from.
cd "$(dirname "$0")/.."

DATASET="${1:-junyi}"
CONFIG="${CONFIG:-./configs/${DATASET}.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== GRACE | dataset=${DATASET} | config=${CONFIG} | output_dir=${OUTPUT_DIR} ==="
python -m grace \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}
