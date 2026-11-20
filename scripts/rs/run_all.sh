#!/usr/bin/env bash
# Rebuild every analysis output, figure and LaTeX table from the cached
# per-patch statistics. No GPU, no dataset download, no retraining.
# Figures and tables are written to manuscript_rs/figures and
# manuscript_rs/tables (override the figure folder with RS_FIGDIR).
set -eu
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
mkdir -p manuscript_rs/figures manuscript_rs/tables

bash scripts/rs/run_analysis.sh

echo "== figures =="
$PY scripts/rs/10_figure_exponent.py
$PY scripts/rs/11_figures.py
$PY scripts/rs/12_figures_spatial.py
$PY scripts/rs/13_figure_gallery.py
$PY scripts/rs/15_figure_architecture.py
$PY scripts/rs/52_figure_backbone.py || echo "   second encoder not cached, skipping"
$PY scripts/rs/64_figure_support.py

echo "== tables =="
$PY scripts/rs/20_tables.py
$PY scripts/rs/21_configuration.py
