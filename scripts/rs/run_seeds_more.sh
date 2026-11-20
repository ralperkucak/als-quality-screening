#!/usr/bin/env bash
# Two further training initializations for D1, D3 and D4, so that every region
# has three runs. Each run repeats the whole downstream chain, so its screening
# numbers come from exactly the code that produced the primary ones.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
FOLDS="D1 D3 D4"

for V in randlanet_s2 randlanet_s3; do
  echo "=== $(date '+%H:%M:%S') train $V on $FOLDS ==="
  $PY -u scripts/07_train_source_lodo.py --backbone "$V" --folds $FOLDS
  echo "=== $(date '+%H:%M:%S') calibration $V ==="
  $PY -u scripts/pivot/03_fit_source_calibration.py --backbone "$V" --folds $FOLDS
  $PY -u scripts/pivot/04_fit_calibrators.py --backbone "$V" --folds $FOLDS
  echo "=== $(date '+%H:%M:%S') statistics $V ==="
  $PY -u scripts/pivot/05_cache_unit_statistics.py --backbone "$V" \
      --folds $FOLDS --families source target
done
echo "=== $(date '+%H:%M:%S') done ==="
