#!/usr/bin/env bash
# Seed sensitivity on two representative regions.
#
# The regions are the ones where the screening is weakest and strongest, D2 and
# D5, so the spread over initialisations is measured at both ends of the result
# rather than in the middle. Each variant repeats the whole downstream chain,
# so its screening numbers are produced by exactly the code that produced the
# primary ones.
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
FOLDS="D2 D5"

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
