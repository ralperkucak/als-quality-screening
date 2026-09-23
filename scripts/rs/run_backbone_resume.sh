#!/usr/bin/env bash
# Resume the second-backbone downstream pass after the source calibration step.
#
# Step 03 already wrote source_D*_point_transformer.npz for every fold, so this
# picks up at the calibrator fit and runs through to the comparison table.
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
B=point_transformer
echo "=== $(date '+%H:%M:%S') 04 fit calibrators ==="
$PY -u scripts/pivot/04_fit_calibrators.py --backbone $B
echo "=== $(date '+%H:%M:%S') 05 cache unit statistics ==="
$PY -u scripts/pivot/05_cache_unit_statistics.py --backbone $B --families source target
echo "=== $(date '+%H:%M:%S') 51 backbone comparison ==="
$PY -u scripts/rs/51_backbone_compare.py
echo "=== $(date '+%H:%M:%S') done ==="
