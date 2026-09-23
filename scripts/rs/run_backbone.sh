#!/usr/bin/env bash
# Finish the second backbone once training has produced all five checkpoints.
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
B=point_transformer
$PY scripts/pivot/03_fit_source_calibration.py --backbone $B
$PY scripts/pivot/04_fit_calibrators.py --backbone $B
$PY scripts/pivot/05_cache_unit_statistics.py --backbone $B --families source target
$PY scripts/rs/51_backbone_compare.py
