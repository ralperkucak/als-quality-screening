r"""Place every float next to the paragraph that first refers to it.

Captions live here rather than in the prose so that a figure and the sentence
describing it are edited together. The inserter is idempotent: a float that is
already present is left alone, so the script can be rerun after the text changes.
A final pass refuses to write any file containing a control character, which is
the failure mode that a shell escaping accident produces and that LaTeX absorbs
without a warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "manuscript_rs" / "sections"
TABLES_DIR = ROOT / "manuscript_rs" / "tables"
FIG_DIR = ROOT / "manuscript_rs" / "figures"

FIGURES = {
    "fig:study-area": dict(
        file="fig_study_area", wide=True,
        caption=(r"The five geographic regions. (\textbf{a}) Patch centroids "
                 r"over a political basemap. (\textbf{b}) Share of reference "
                 r"points held by each class. (\textbf{c}) The 2~km tiles over "
                 r"shaded relief. Basemap credits: Esri, HERE, Garmin, "
                 r"\copyright{} OpenStreetMap contributors and the GIS user "
                 r"community in (\textbf{a}); Esri and USGS in (\textbf{c}).")),
    "fig:workflow": dict(
        file="fig_workflow", wide=True,
        caption=(r"The screening procedure. The recalibration map is fitted "
                 r"once on labelled source data, and no reference label from "
                 r"the new region enters any score.")),
    "fig:distribution": dict(
        file="fig_tile_quality", wide=True,
        caption=(r"Tile mean intersection over union in each held-out region. "
                 r"Each dot is one 2~km tile, the solid rule the mean over "
                 r"tiles and the dashed rule the pooled value.")),
    "fig:decomposition": dict(
        file="fig_decomposition", wide=False, width=r"0.74\textwidth",
        caption=(r"Per-class intersection over union in D2, pooled over the "
                 r"region and averaged over the tiles in which the class "
                 r"appears. Connector width is proportional to the "
                 r"contribution to the covariance term of "
                 r"Equation~\eqref{eq:decomposition}, and the right-hand "
                 r"column gives the number of tiles holding the class.")),
    "fig:coverage": dict(
        file="fig_review_budget", wide=True,
        caption=(r"Screening lift against the review budget in each held-out "
                 r"region, for the same calibrated confidence aggregated per "
                 r"class and per point. The normalised area under the risk "
                 r"coverage curve is printed in each panel, where lower is "
                 r"better.")),
    "fig:surface": dict(
        file="fig_exponent_surface", wide=True,
        caption=(r"Screening over the two exponents. (\textbf{a}) Lift at a 20 "
                 r"per cent budget, averaged over the five regions; the solid "
                 r"line is the best score exponent, the band the exponents "
                 r"within one standard error of it, the dashed line equal "
                 r"exponents and the rings the two published conventions. "
                 r"(\textbf{b}) The best exponent of each region. "
                 r"(\textbf{c}) What each control leaves of the lift, with 95 "
                 r"per cent intervals.")),
    "fig:perclass": dict(
        file="fig_per_class", wide=False, width=r"0.86\textwidth",
        caption=(r"Rank correlation between a tile's per-class intersection "
                 r"over union and that class's confidence, over tiles where "
                 r"the model predicts at least 500 points of the class. The "
                 r"bracketed number is the correlation of the class-averaged "
                 r"score on the same tiles, and a dash marks too few eligible "
                 r"tiles.")),
    "fig:gallery": dict(
        file="fig_gallery", wide=True,
        caption=(r"One tile of D4 for each screening outcome at a 20 per cent "
                 r"budget, shown through the patch holding the most points. "
                 r"(\textbf{a}) Reference classes over a shaded plan view. "
                 r"(\textbf{b}) The cells where the prediction disagrees with "
                 r"the reference.")),
    "fig:maps": dict(
        file="fig_quality_maps", wide=True,
        caption=(r"Outcome of the screening at a 20 per cent review budget. "
                 r"Each square is one 2~km tile at its position. Blue marks a "
                 r"flagged tile genuinely in the worst fifth, gold a flagged "
                 r"tile that is not, red a tile in the worst fifth that was "
                 r"missed, and grey the remainder.")),
    "fig:backbone": dict(
        file="fig_backbone", wide=True,
        caption=(r"The screening on both encoders. (\textbf{a}) Best score "
                 r"exponent at each criterion exponent, with the interval that "
                 r"ties with it shaded. (\textbf{b}) Screening lift with the "
                 r"score weighting matched to the criterion, filled, and "
                 r"reversed, open.")),
}
# Where each float goes: the text it follows, and the section file.
PLACEMENT = [
    ("methods.tex", "to D5 in order of increasing easting (Figure~\\ref{fig:study-area}).",
     "fig:study-area"),
    ("methods.tex", "the figure to read for how the map behaves in use.", "fig:workflow"),
    ("results.tex", "already correct.", "fig:distribution"),
    ("results.tex", "segmented poorly, so the pooled value for water is set by the lake\n(Figure~\\ref{fig:decomposition}).", "fig:decomposition"),
    ("results.tex", "(Figure~\\ref{fig:coverage}).", "fig:coverage"),
    ("results.tex", "ordering of the two reverses, to 0.66 and $-0.02$.", "fig:surface"),
    ("results.tex", "and the peak sits at a negative\nexponent in all five regions.", "fig:perclass"),
    ("results.tex", "scattered across a region\n(Figure~\\ref{fig:maps}).", "fig:maps"),
    ("results.tex", "Figure~\\ref{fig:gallery} shows one tile of each outcome as the reviewer would open it.", "fig:gallery"),
    ("results.tex", "the other criterion (Figure~\\ref{fig:backbone}b).", "fig:backbone"),
]

TABLES = [
    # Each table is placed after the paragraph that first refers to it, so the
    # reader meets the sentence before the numbers it points at.
    ("results.tex", "effort evenly over a delivery therefore spends most of it on ground that is\nalready correct.", "table_regions"),
    ("results.tex", "point-weighted average assigns almost no weight.", "table_transfer"),
    ("results.tex", "budget it recovers 0.427 of the worst tenth and at three tenths it recovers\n0.607.", "table_retrieval"),
    ("results.tex", "pooled over points inherits the behaviour of the pooled mean.", "table_baselines"),
    ("results.tex", "(Table~\\ref{tab:exponents}, Figure~\\ref{fig:surface}b).", "table_exponents"),
    ("results.tex", "class-averaged criterion from 0.494 to\n0.291 and raises it against the point-weighted criterion from 0.656 to 0.712.", "table_controls"),
    ("results.tex", "which is why the control does not arise elsewhere in the paper.", "table_universe"),
    ("results.tex", "relaxed to 200 predicted points and in all 29 when it is tightened to 1000\n(Figure~\\ref{fig:perclass}).", "table_per_class"),
    ("results.tex", "neither changes the direction of the result (Table~\\ref{tab:robustness}).", "table_robustness"),
    ("results.tex", "and the two families with the lowest calibration error give the two\nlowest recalls.", "table_calibration"),
    ("results.tex", "the mistakes it\nmakes.", "table_backbone"),
    ("results.tex", "recalibration procedure and the budget unchanged, so that only the trained\nmodel differs (Table~\\ref{tab:seeds}).", "table_seeds"),
]


def float_block(label: str) -> str:
    spec = FIGURES[label]
    width = spec.get("width", "\\fulllength" if spec["wide"] else "\\textwidth")
    lines = ["", "\\begin{figure}[H]"]
    if spec["wide"]:
        lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    lines += ["\\centering",
              "\\includegraphics[width=%s]{figures/%s}" % (width, spec["file"]),
              "\\caption{%s\\label{%s}}" % (spec["caption"], label)]
    if spec["wide"]:
        lines.append("\\end{adjustwidth}")
    lines += ["\\end{figure}", ""]
    return "\n".join(lines) + "\n"


def insert_after(text: str, anchor: str, block: str, marker: str) -> str:
    if marker in text:
        return text
    at = text.find(anchor)
    if at < 0:
        raise SystemExit("anchor not found: %r" % anchor[:60])
    at += len(anchor)
    return text[:at] + "\n" + block + text[at:]


def main() -> int:
    edited = {}
    for name, anchor, label in PLACEMENT:
        path = SECTIONS / name
        # As for tables, a figure whose analysis has not been run is skipped.
        if not (FIG_DIR / f"{FIGURES[label]['file']}.pdf").exists():
            print("no", FIGURES[label]["file"], "yet, not placed")
            continue
        text = edited.get(name, path.read_text(encoding="utf-8"))
        edited[name] = insert_after(text, anchor, float_block(label),
                                    "\\label{%s}}" % label)
    for name, anchor, table in TABLES:
        path = SECTIONS / name
        # A table whose analysis has not been run is skipped rather than
        # inserted as an input that would not resolve.
        if not (TABLES_DIR / f"{table}.tex").exists():
            print("no", table, "yet, not placed")
            continue
        text = edited.get(name, path.read_text(encoding="utf-8"))
        edited[name] = insert_after(text, anchor,
                                    "\n\\input{tables/%s}\n" % table,
                                    "tables/%s}" % table)

    for name, text in edited.items():
        raw = text.encode("utf-8")
        bad = sorted({c for c in raw if c < 0x20 and c not in (10, 13)})
        if bad:
            raise SystemExit("control characters in %s: %s" % (name, bad))
        (SECTIONS / name).write_text(text, encoding="utf-8")
        print("laid out", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
