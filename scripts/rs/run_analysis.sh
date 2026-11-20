#!/usr/bin/env bash
# Rebuild every numeric result of the study from the cached per-unit
# statistics: the exponent surface, the retrieval and calibration tables, the
# robustness controls, the second-encoder comparison and the seed-sensitivity
# check. No GPU and no dataset download are needed for this script -- it reads
# the parquet and npz files written by scripts/pivot/02-05 (see the top-level
# README) and writes CSV, JSON and parquet summaries to outputs/rs/.
#
# Figures and LaTeX tables are written by scripts/rs/1x, 20, 21, 52 and 64;
# see the README.
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"

echo "== exponent surface, controls, retrieval, calibration, regional, per-class =="
$PY scripts/rs/01_exponent_surface.py
$PY scripts/rs/02_controls.py
$PY scripts/rs/03_retrieval.py
$PY scripts/rs/04_calibration.py
$PY scripts/rs/05_regional.py
$PY scripts/rs/06_per_class.py
$PY scripts/rs/07_operational.py
$PY scripts/rs/08_robustness.py
$PY scripts/rs/09_derived.py

echo "== class-universe control, extra baselines, target-side calibration, support sensitivity =="
$PY scripts/rs/53_conventions.py
$PY scripts/rs/54_extra_baselines.py
$PY scripts/rs/55_target_calibration.py
$PY scripts/rs/56_support_sensitivity.py
$PY scripts/rs/67_criterion_support.py

echo "== point thinning, patch subsets, supervised learning curves, operational checks =="
$PY scripts/rs/60_support_perturbation.py
$PY scripts/rs/61_baseline_learning_curve.py
$PY scripts/rs/62_operational_checks.py
$PY scripts/rs/63_baseline_features.py
$PY scripts/rs/65_baseline_table_rows.py

echo "== second encoder and seed sensitivity (skipped if their inputs are not cached) =="
$PY scripts/rs/51_backbone_compare.py || echo "   second encoder not cached, skipping"
$PY scripts/rs/57_seed_compare.py || echo "   extra seeds not cached, skipping"
$PY scripts/rs/66_seed_ensemble.py || echo "   extra seeds not cached, skipping"
$PY scripts/rs/68_baseline_seeds.py || echo "   extra seeds not cached, skipping"

echo "== done: see outputs/rs/*.csv, *.json, *.parquet =="
