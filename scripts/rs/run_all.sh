#!/usr/bin/env bash
# Rebuild every analysis output, figure, table and the manuscript itself from
# the cached per-patch statistics. No GPU, no dataset download, no retraining.
#
# This is the exact script used to build the submitted manuscript, kept here
# for transparency: it shows how each analysis output became a table cell or a
# figure panel. Its final two stages expect a `manuscript_rs/` directory (the
# paper's LaTeX source, with `sections/*.tex` already written) alongside this
# repository, which is not distributed here. To reproduce only the numeric
# results, without that dependency, run `run_analysis.sh` instead.
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"

echo "== analysis =="
$PY scripts/rs/01_exponent_surface.py
$PY scripts/rs/02_controls.py
$PY scripts/rs/03_retrieval.py
$PY scripts/rs/04_calibration.py
$PY scripts/rs/05_regional.py
$PY scripts/rs/06_per_class.py
$PY scripts/rs/07_operational.py
$PY scripts/rs/08_robustness.py
$PY scripts/rs/09_derived.py
# The architecture check needs the second encoder cached; skip it quietly when
# only the primary backbone is present so a fresh clone still runs end to end.
$PY scripts/rs/51_backbone_compare.py || echo "   second encoder not cached, skipping"
$PY scripts/rs/57_seed_compare.py || echo "   extra seeds not cached, skipping"

echo "== figures =="
$PY scripts/rs/10_figure_exponent.py
$PY scripts/rs/11_figures.py
$PY scripts/rs/12_figures_spatial.py
$PY scripts/rs/13_figure_gallery.py
$PY scripts/rs/52_figure_backbone.py || echo "   second encoder not cached, skipping"

echo "== tables and layout =="
$PY scripts/rs/20_tables.py
$PY scripts/rs/21_configuration.py
$PY scripts/rs/29_clean_floats.py
$PY scripts/rs/30_layout.py

echo "== manuscript =="
cd manuscript_rs
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
cd ..

echo "== audit =="
$PY scripts/rs/40_audit.py
