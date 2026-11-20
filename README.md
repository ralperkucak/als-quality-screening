# Label-Free Quality Screening of Cross-Region Airborne Lidar Segmentation

Code for the paper *"Label-Free Quality Screening of Cross-Region Airborne
Lidar Segmentation by Class-Weighted Confidence"* (submitted to *Remote
Sensing*). Given a segmentation model applied to a newly surveyed region with
no reference labels, this study orders the region's tiles by how likely they
are to be poorly segmented, using only the model's own point-level confidence.
The central finding is that the class weighting used to reduce point
confidences to one score per tile has to match the class weighting of the
accuracy criterion a delivery is judged by; get that wrong and the ordering
performs no better than chance.

This repository holds the full code path from the raw FRACTAL archives to the
tables and figures reported in the paper: ingestion and region assignment,
leave-one-region-out training of two segmentation backbones, the source-side
confidence calibration pipeline, and the analysis layer that turns cached
per-tile statistics into every number in the manuscript.

## Data

[FRACTAL](https://huggingface.co/datasets/IGNF/FRACTAL) (Institut national de
l'information géographique et forestière, Etalab Open Licence 2.0): 100,000
airborne lidar patches of 50×50 m, seven semantic classes, EPSG:2154. It is
not redistributed here; `scripts/01_download_fractal.py` downloads and
preprocesses it.

## Environment

Python 3.11.

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` pins the exact versions (CUDA 12.8 build of PyTorch)
used to produce the results in the paper, for readers who want to match the
environment exactly rather than the loosened bounds in `requirements.txt`.

Bulk data is kept outside the repository. By default it is written to
`~/geoquery_store`; set `GEOQUERY_DATA_ROOT` to change that:

```bash
export GEOQUERY_DATA_ROOT=/path/to/data
```

## Layout

```
configs/experiment/main.yaml   every number that controls training and sampling
src/geoquery_als/
  config.py        repository paths and FRACTAL constants
  data/            preprocessing, region assignment, split construction, the patch store
  models/          RandLA-Net and the Point Transformer variant
  reliability/      prototypes, confidence calibration, per-patch sufficient statistics
  adaptation/       training and inference loop shared by both backbones
  rs/              the self-contained analysis layer (Section "Reproducing the analysis")
scripts/
  00-07            ingestion, region assignment, split construction, training
  pivot/02-05      source-side calibration and per-unit statistics caching
  rs/              analysis, and the manuscript's table/figure generation
```

## Reproducing the results

Each stage reads the previous stage's output; none of the intermediate
artefacts (preprocessed point store, trained checkpoints, cached statistics)
are distributed here, since together they run past what a repository should
hold. Expect this to take from a few hours (analysis only, from cached
statistics) to about a day per encoder (full training on a single GPU).

**1. Ingestion and splits.**

```bash
python scripts/00_environment_check.py     # optional: records hardware and package versions
python scripts/01_download_fractal.py      # downloads and preprocesses FRACTAL
python scripts/03_audit_dataset.py         # clusters patches into five regions
python scripts/04_build_domain_splits.py   # builds the leave-one-region-out roles
```

**2. Training.** One call per fold and per backbone; `--backbone` selects
`randlanet`, `point_transformer`, or a seed variant such as `randlanet_s2`
(same architecture, different initialisation, used for the seed-sensitivity
check; `scripts/rs/run_seeds_more.sh` trains the further initializations and
recomputes their statistics).

```bash
python scripts/07_train_source_lodo.py --backbone randlanet
```

**3. Source-side calibration and per-unit statistics**, once per backbone:

```bash
python scripts/pivot/02_build_aggregation_units.py
python scripts/pivot/03_fit_source_calibration.py --backbone randlanet
python scripts/pivot/04_fit_calibrators.py --backbone randlanet
python scripts/pivot/05_cache_unit_statistics.py --backbone randlanet --families source target
```

**4. The analysis layer.** With the statistics above cached for at least the
primary backbone, this reproduces every numeric result in the paper (the
exponent surface, the retrieval and calibration tables, the robustness
controls, and, once their inputs are cached, the second-encoder and
seed-sensitivity checks):

```bash
bash scripts/rs/run_analysis.sh
```

Results are written to `outputs/rs/*.csv`, `*.json` and `*.parquet`.

**5. Figures and tables.** `scripts/rs/10-13`, `15`, `20-21`, `52` and `64`
write the figures and LaTeX tables of the paper from the outputs above, and
`scripts/rs/run_all.sh` runs the analysis followed by these scripts. Output goes
to `manuscript_rs/figures` and `manuscript_rs/tables`.

## What this repository does not include

- The FRACTAL archives themselves (public, linked above).
- Trained model checkpoints and the cached per-patch and per-unit statistics
  (regenerable from the steps above; too large for a code repository).
- The active-learning acquisition code from an earlier, separate study that
  shares this codebase's data and model layer but contributes nothing to the
  results reported in this paper.
- The paper's LaTeX source.

## Citation

Ramazan Alper Kuçak, Department of Geomatics Engineering, Faculty of Civil
Engineering, Istanbul Technical University. A citation entry will be added
once the paper is assigned a DOI.

## License

MIT, see `LICENSE`.
