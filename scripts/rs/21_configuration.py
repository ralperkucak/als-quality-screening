"""The configuration table of the appendix, read from the experiment record."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript_rs" / "tables"
CONFIG = ROOT / "configs" / "experiment" / "main.yaml"

ROWS = [
    ("Preprocessing", "voxel size", "0.25~m"),
    ("", "retained patches", "94{{,}}000"),
    ("", "coordinate reference system", "EPSG:2154"),
    ("Sampling", "points presented per patch", "{sampling[num_points]}"),
    ("", "nearest neighbours", "{sampling[k_neighbours]}"),
    ("", "hierarchy levels", "{sampling[num_layers]}"),
    ("", "subsampling ratio between levels", "{sampling[sub_ratio]}"),
    ("", "coordinate scale", "{sampling[coord_scale_m]}~m"),
    ("Model", "architecture", "RandLA-Net"),
    ("", "encoder widths", "16, 64, 128, 256"),
    ("", "trainable parameters", "1{{,}}214{{,}}175"),
    ("", "second architecture", "Point Transformer"),
    ("", "attention widths", "16, 48, 96, 200"),
    ("", "trainable parameters, second", "1{{,}}204{{,}}327"),
    ("", "dropout before the classifier", "{model[dropout]}"),
    ("", "input features", "{model[in_features]}"),
    ("Training", "patches per source region", "{splits[n_source_train_per_domain]}"),
    ("", "epochs", "{source_training[epochs]}"),
    ("", "batch size", "{source_training[batch_size]}"),
    ("", "optimiser", "Adam"),
    ("", "initial learning rate", "$4\\times10^{{-3}}$"),
    ("", "learning rate floor", "$10^{{-5}}$"),
    ("", "gradient norm clipping", "{source_training[grad_clip]}"),
    ("", "class weighting of the loss", "inverse square root of frequency"),
    ("Recalibration", "source points sampled per patch", "200"),
    ("", "source points per fold", "240{{,}}000"),
    ("", "class-wise shrinkage reaches full weight at", "1000 points"),
    ("", "cross-fitting folds", "5"),
    ("", "calibration error bins", "20, equal width"),
    ("Tiles", "primary spacing", "2~km"),
    ("", "sensitivity spacings", "1~km, 4~km"),
    ("", "minimum patches per tile", "8"),
    ("", "tiles in total", "1048"),
    ("Screening", "review budgets", "0.10, 0.20, 0.30"),
    ("", "score exponents examined", "$-1.0$ to $2.0$ in steps of 0.05"),
    ("", "criterion exponents examined", "$-0.5$ to $1.5$ in steps of 0.05"),
    ("", "null distribution", "hypergeometric, exact"),
    ("", "bootstrap draws", "2000 for the ridge, 4000 for contrasts"),
    ("", "spatial block sizes", "6, 10, 16~km"),
    ("", "criterion class sets examined",
     "union of prediction and reference, reference only, all seven"),
    ("", "score class sets examined", "predicted classes, all seven"),
    ("", "minimum predicted support examined", "0, 10, 50, 100, 500, 1000"),
    ("", "confidence quantiles examined", "0.05, 0.10, 0.25"),
    ("", "supervised baselines",
     "ridge with generalised cross-validation, gradient boosting"),
    ("", "calibration cross-fitting unit", "patch"),
    ("", "target calibration binning", "predicted class"),
    ("", "analysis seed", "20260920"),
    ("", "training seed", "{seed_base}"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conf = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lines = []
    for stage, name, template in ROWS:
        value = template.format(**conf)
        lines.append(f"{stage} & {name} & {value} \\\\")
    body = "\n".join(lines)
    table = (
        "\\begin{table}[H]\n"
        "\\caption{Configuration of every stage of the study."
        "\\label{tab:configuration}}\n"
        "\\small\n"
        "\\begin{tabularx}{\\textwidth}{llX}\n"
        "\\toprule\n"
        "Stage & Parameter & Value\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n")
    (OUT / "table_configuration.tex").write_text(table, encoding="utf-8")
    print("wrote table_configuration with", len(ROWS), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
