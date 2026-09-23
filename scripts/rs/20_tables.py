"""Every table of the manuscript, written from the analysis outputs.

Captions live here so that a number and the sentence describing it cannot drift
apart. Wide tables are set across the margin with the width the journal class
provides for that purpose.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "outputs" / "rs"
OUT = ROOT / "manuscript_rs" / "tables"
CLASS_LABEL = {"other": "Other", "ground": "Ground", "vegetation": "Vegetation",
               "building": "Building", "water": "Water", "bridge": "Bridge",
               "permanent_structure": "Permanent structure"}


# A numeric cell holding a negative value should carry a minus sign rather than
# the hyphen a bare "-" produces in text mode, whether it stands alone in the
# cell or inside a bracketed interval.
MINUS = re.compile(r"(?<=[&\[ ])(-\d+\.\d+)(?=[\s&,\]\\])")


def typeset_minus(table: str) -> str:
    return MINUS.sub(lambda m: f"${m.group(1)}$", table)


def interval(lo: float, hi: float, digits: int = 3) -> str:
    """A confidence interval that a column cannot break in half."""
    return f"\\mbox{{[{lo:.{digits}f}, {hi:.{digits}f}]}}"


def fmt_p(p: float) -> str:
    """A p value in the form a journal prints it."""
    if p <= 0:
        return r"$<10^{-300}$"
    if p >= 1e-3:
        return f"{p:.3f}"
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / 10.0 ** exponent
    return r"$%.1f \times 10^{%d}$" % (mantissa, exponent)


def wide(body: str, caption: str, label: str, spec: str, header: str,
         note: str = "") -> str:
    footer = ("\n\\noindent{\\small " + note + "}") if note else ""
    return (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        "\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\footnotesize\n"
        f"\\begin{{tabularx}}{{\\fulllength}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header}\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        f"{footer}\n"
        "\\end{adjustwidth}\n"
        "\\end{table}\n")


def narrow(body: str, caption: str, label: str, spec: str, header: str,
           note: str = "") -> str:
    footer = ("\n\\noindent{\\small " + note + "}") if note else ""
    return (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header}\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}"
        f"{footer}\n"
        "\\end{table}\n")


def table_regions() -> str:
    d = pd.read_csv(DATA / "regions.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            r["region"], f"{int(r['tiles'])}",
            f"{r['points_million']:.1f}", f"{r['pooled_oa']:.3f}",
            f"{r['pooled_miou']:.3f}", f"{r['tile_mean_miou']:.3f}",
            f"{r['covariance_term']:+.3f}", f"{r['class_presence_term']:+.3f}",
            f"{r['tile_miou_p10']:.3f}", f"{r['tile_miou_median']:.3f}",
            f"{r['tile_miou_p90']:.3f}"]) + r" \\")
    return wide(
        "\n".join(rows),
        "The five held-out regions. Pooled values come from the confusion matrix "
        "summed over the region, tile values from its 2\\,km tiles. The last "
        "two terms are the decomposition of "
        "Equation~\\eqref{eq:decomposition}.",
        "tab:regions",
        "l" + "C" * 10,
        "Region & Tiles & Points (10$^6$) & OA & Pooled mIoU & "
        "Tile mIoU & Covariance & Presence & $P_{10}$ & Median & $P_{90}$")


def table_transfer() -> str:
    d = pd.read_csv(DATA / "transfer_summary.csv", header=[0, 1], index_col=0)
    d.columns = [a if b.startswith("Unnamed") else f"{a}_{b}"
                 for a, b in d.columns]
    rows = []
    for name in CLASS_LABEL:
        if name not in d.index:
            continue
        r = d.loc[name]
        rows.append(" & ".join([
            CLASS_LABEL[name],
            f"{100 * r['share_source']:.2f}", f"{100 * r['share_target']:.2f}",
            f"{r['iou_source']:.3f}", f"{r['iou_target']:.3f}",
            f"{r['loss']:.3f}"]) + r" \\")
    return narrow(
        "\n".join(rows),
        "Per-class accuracy on the labelled source validation data and on the "
        "held-out region, averaged over the five folds and computed from the "
        "confusion matrix summed over each partition.",
        "tab:transfer",
        "l" + "C" * 5,
        "Class & Share, source (\\%) & Share, target (\\%) & IoU, source & "
        "IoU, target & Loss")


def table_backbone() -> str:
    """Only emitted once the second encoder has been trained and cached."""
    path = DATA / "backbone_summary.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    # The primary encoder leads, as the surrounding text reads it.
    d = d.set_index("backbone").reindex(
        ["RandLA-Net", "Point Transformer"]).reset_index()
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            r["backbone"],
            f"{r['pooled_miou']:.3f}", f"{r['tile_mean_miou']:.3f}",
            f"{r['recall_class_averaged']:.3f}",
            f"{r['recall_point_weighted']:.3f}",
            f"{r['naurc_class_averaged']:.3f}",
            f"{r['naurc_point_weighted']:.3f}",
            f"{r['monotone_rho']:.3f}",
            f"{r['alpha_class_averaged']:.2f}",
            f"{r['alpha_point_weighted']:.2f}"]) + r" \\")
    return wide(
        "\n".join(rows),
        "Screening on two encoders that differ only in how a point aggregates "
        "its neighbours, averaged over the five held-out regions. Each "
        "encoder is measured against its own errors.",
        "tab:backbone",
        "l" + "C" * 9,
        "Encoder & Pooled mIoU & Tile mIoU & R@20 ($\\alpha=0$) & "
        "R@20 ($\\alpha=1$) & NAURC ($\\alpha=0$) & NAURC ($\\alpha=1$) & "
        "Ridge $\\rho$ & $\\alpha^{\\ast}$ at $\\beta=0$ & "
        "$\\alpha^{\\ast}$ at $\\beta=1$")


def table_baselines() -> str:
    """Orderings that are harder to beat than a pooled mean confidence."""
    path = DATA / "extra_baselines.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path).sort_values("recall", ascending=False)
    label = {
        "class-averaged calibrated confidence": "Class-averaged calibrated confidence",
        "point-weighted calibrated confidence": "Point-weighted calibrated confidence",
        "weakest predicted class": "Weakest predicted class",
        "fifth percentile of point confidence": "Fifth percentile of point confidence",
        "tenth percentile of point confidence": "Tenth percentile of point confidence",
        "quarter percentile of point confidence": "Twenty-fifth percentile of point confidence",
        "class-averaged tenth percentile": "Class-averaged tenth percentile",
        "supervised meta-regressor, ridge": "Supervised regressor, ridge",
        "supervised meta-regressor, gradient boosting": "Supervised regressor, gradient boosting",
    }
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            label.get(r["method"], r["method"]),
            f"{r['recall']:.3f}", f"{r['lift']:.3f}",
            f"{r['naurc']:.3f}", f"{r['spearman']:.3f}",
            fmt_p(r["p_exact"])]) + r" \\")
    return wide(
        "\n".join(rows),
        "Orderings compared on the same tiles, criterion and budget as "
        "Table~\\ref{tab:retrieval}. The supervised regressors are fitted on "
        "labelled source tiles and read no label from the held-out region.",
        "tab:baselines",
        "l" + "C" * 5,
        "Ordering & R@20 & Lift & NAURC & $\\rho$ & Largest $p$")


def table_universe() -> str:
    """Which classes the criterion averages over, and which the score does."""
    conventions = DATA / "criterion_conventions.csv"
    variants = DATA / "missing_class_variant.csv"
    if not conventions.exists() or not variants.exists():
        return ""
    v = pd.read_csv(variants)
    order = ["union", "fixed", "reference"]
    name = {"union": "Prediction and reference",
            "fixed": "All seven classes",
            "reference": "Reference only"}
    rows = []
    for rule in order:
        sub = v[v["rule"] == rule].set_index("score")
        cells = [name[rule]]
        for which in ("predicted class set", "fixed class universe",
                      "predicted class count"):
            r = sub.loc[which]
            cells += [f"{r['recall']:.3f}", f"{r['lift']:.3f}"]
        rows.append(" & ".join(cells) + r" \\")
    return wide(
        "\n".join(rows),
        "Screening when the class set the criterion averages over and the class "
        "set the score averages over are varied independently, at the "
        "class-averaged operating point. The last pair of columns orders the "
        "tiles on the predicted class count alone.",
        "tab:universe",
        "l" + "C" * 6,
        "Criterion averages over & R@20 & Lift & R@20 & Lift & R@20 & Lift",
        note="Column pairs use, in order, the score over the predicted class "
             "set, the score over the fixed class universe, and the number of "
             "predicted classes on its own.")


def table_seeds() -> str:
    """Only emitted once at least one extra initialisation has been cached."""
    path = DATA / "seed_by_region.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path).sort_values(["region", "seed"])
    rows = []
    for region, block in d.groupby("region"):
        for i, (_, r) in enumerate(block.iterrows()):
            rows.append(" & ".join([
                region if i == 0 else "",
                f"{int(r['seed'])}",
                f"{r['pooled_miou']:.3f}", f"{r['tile_mean_miou']:.3f}",
                f"{r['recall']:.3f}", f"{r['lift']:.3f}",
                f"{r['naurc']:.3f}", f"{r['spearman']:.3f}",
                f"{r['alpha_at_beta0']:.2f}", f"{r['alpha_at_beta1']:.2f}",
                fmt_p(r["p_exact"])]) + r" \\")
    return wide(
        "\n".join(rows),
        "The protocol repeated from three initialisations on the two regions that "
        "returned the lowest and the highest recall in the main run. Only the "
        "trained model changes across rows.",
        "tab:seeds",
        "l l" + "C" * 9,
        "Region & Seed & Pooled mIoU & Tile mIoU & R@20 & Lift & NAURC & "
        "$\\rho$ & $\\alpha^{\\ast}$ at $\\beta=0$ & "
        "$\\alpha^{\\ast}$ at $\\beta=1$ & Exact $p$")


def table_retrieval() -> str:
    d = pd.read_csv(DATA / "retrieval_summary.csv")
    order = ["class-specific ATC", "class-specific DoC",
             "class-averaged raw confidence",
             "class-averaged calibrated confidence",
             "class-averaged negative entropy",
             "average thresholded confidence", "mean raw confidence",
             "mean calibrated confidence", "mean negative entropy",
             "feature support reliability"]
    family = {a: ("Class averaged" if a == 0.0 else "Point weighted")
              for a in (0.0, 1.0)}
    d = d.set_index("method").reindex(order).reset_index()
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            r["method"], family[r["alpha"]],
            f"{r['recall_10']:.3f}", f"{r['recall_20']:.3f}",
            f"{r['recall_30']:.3f}", f"{r['captured_20']:.3f}",
            f"{r['aurc_normalised']:.3f}", f"{r['spearman']:.3f}",
            fmt_p(r["max_p_20"])]) + r" \\")
    return wide(
        "\n".join(rows),
        "Screening performance averaged over the five held-out regions. Recall "
        "at a budget is the share of the genuinely worst tiles among the same "
        "number of lowest ranked tiles, against 0.20 for a random order. "
        "NAURC is zero for a perfect ordering and one for a random one.",
        "tab:retrieval",
        "l l" + "C" * 7,
        "Score & Aggregation & R@10 & R@20 & R@30 & Captured deficit & NAURC & "
        "$\\rho$ & max $p$",
        "R@$q$: recall at a review budget of $q$. NAURC: normalised area under "
        "the risk coverage curve. $\\rho$: Spearman correlation between the "
        "score and the tile intersection over union.")


def table_exponents() -> str:
    d = pd.read_csv(DATA / "exponent_summary.csv")
    label = {"calibrated": "Calibrated confidence",
             "confidence": "Maximum softmax confidence",
             "atc_classwise": "Thresholded confidence"}
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            label[r["statistic"]],
            f"{r['monotone_rho']:.3f}",
            f"[{r['monotone_lo']:.2f}, {r['monotone_hi']:.2f}]",
            f"{r['alpha_class_averaged']:.2f}",
            f"[{r['alpha_class_averaged_lo']:.2f}, {r['alpha_class_averaged_hi']:.2f}]",
            f"{r['lift_class_averaged']:.3f}",
            f"{r['lift_class_averaged_mismatched']:.3f}",
            f"{r['alpha_point_weighted']:.2f}",
            f"[{r['alpha_point_weighted_lo']:.2f}, {r['alpha_point_weighted_hi']:.2f}]",
            f"{r['lift_point_weighted']:.3f}",
            f"{r['lift_point_weighted_mismatched']:.3f}"]) + r" \\")
    return wide(
        "\n".join(rows),
        "How the best score exponent follows the criterion. For each convention "
        "the table gives the best exponent, the exponents tying with it inside "
        "one standard error over regions, and the lift attained at that "
        "exponent and at the opposite one. The best exponent is a maximum over "
        "the same grid, so the lift beside it is optimistic.",
        "tab:exponents",
        "l" + "C" * 10,
        "Point statistic & $\\rho(\\beta,\\alpha^{\\ast})$ & 95\\% interval & "
        "$\\alpha^{\\ast}$ at $\\beta=0$ & Band & Lift & Lift if $\\alpha=1$ & "
        "$\\alpha^{\\ast}$ at $\\beta=1$ & Band & Lift & Lift if $\\alpha=0$")


def table_controls() -> str:
    d = pd.read_csv(DATA / "control_ladder.csv")
    label = {"matched exponents": "Equal exponents ($\\alpha=\\beta$)",
             "mismatched exponents": "Opposite exponents",
             "reference class weights": "Reference class weights",
             "composition only": "Class composition only",
             "class count only": "Class count only"}
    order = list(label)
    rows = []
    for control in order:
        cells = [label[control]]
        for criterion in ("class averaged", "point weighted"):
            r = d[(d["control"] == control) & (d["criterion"] == criterion)].iloc[0]
            cells += [f"{r['lift']:.3f}", interval(r["lo"], r["hi"])]
        rows.append(" & ".join(cells) + r" \\")
    return wide(
        "\n".join(rows),
        "What each control leaves of the screening lift, with one ingredient of "
        "the score removed per row. Intervals resample the regions and then "
        "the tiles, except for the class count row, where they come from "
        "repeating the tie breaking.",
        "tab:controls",
        "l" + "C" * 4,
        "Control & Lift ($\\beta=0$) & 95\\% interval & Lift ($\\beta=1$) & "
        "95\\% interval")


def table_per_class() -> str:
    d = pd.read_csv(DATA / "per_class_summary.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            CLASS_LABEL[r["class"]], f"{int(r['regions'])}",
            f"{int(r['eligible_tiles'])}", f"{r['mean_iou']:.3f}",
            f"{r['spearman_own_class']:.3f}",
            f"{r['spearman_class_balanced']:+.3f}",
            f"{r['recall_own_class']:.3f}",
            f"{r['recall_class_balanced']:.3f}", fmt_p(r["max_p"])]) + r" \\")
    return wide(
        "\n".join(rows),
        "Screening for a single class, over the tiles in which the model predicts "
        "at least 500 points of it. Only the confidence the ordering reads "
        "varies; the tiles, the budget and the criterion are fixed.",
        "tab:perclass",
        "l" + "C" * 8,
        "Class & Regions & Eligible tiles & Mean IoU & $\\rho$ own class & "
        "$\\rho$ class averaged & R@20 own class & R@20 class averaged & max $p$")


def table_calibration() -> str:
    d = pd.read_csv(DATA / "calibration_summary.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            r["family"].capitalize() if r["family"][0].islower() else r["family"],
            f"{r['ece_in_sample']:.4f}", f"{r['ece_out_of_sample']:.4f}",
            f"{r['brier_out_of_sample']:.4f}",
            f"{r['recall_20_class_averaged']:.3f}",
            f"{r['recall_20_point_weighted']:.3f}"]) + r" \\")
    return narrow(
        "\n".join(rows),
        "Recalibration families fitted on the labelled source validation points "
        "of each fold and carried unchanged to the held-out region. Cross "
        "fitting over five folds is grouped by patch.",
        "tab:calibration",
        "l" + "C" * 5,
        "Recalibration & ECE in sample & ECE out of sample & Brier & "
        "R@20 ($\\alpha=0$) & R@20 ($\\alpha=1$)")


def table_robustness() -> str:
    d = pd.read_csv(DATA / "robustness_summary.csv")
    d = d.sort_values(["variation", "setting"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            r["variation"].capitalize(), r["setting"], f"{int(r['tiles'])}",
            f"{r['recall']:.3f}", f"{r['naurc']:.3f}", f"{r['spearman']:.3f}",
            f"{int(r['regions_above_random'])} of 5", fmt_p(r["max_p"])])
            + r" \\")
    return wide(
        "\n".join(rows),
        "Sensitivity of the screening to the way the tiles were built. The "
        "score, the criterion and the review budget are held fixed throughout "
        "and only the tiling changes.",
        "tab:robustness",
        "l l" + "C" * 6,
        "Variation & Setting & Tiles & R@20 & NAURC & $\\rho$ & Regions above "
        "random & max $p$")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_regions": table_regions(),
        "table_transfer": table_transfer(),
        "table_retrieval": table_retrieval(),
        "table_exponents": table_exponents(),
        "table_controls": table_controls(),
        "table_per_class": table_per_class(),
        "table_baselines": table_baselines(),
        "table_universe": table_universe(),
        "table_seeds": table_seeds(),
        "table_calibration": table_calibration(),
        "table_backbone": table_backbone(),
        "table_robustness": table_robustness(),
    }
    for name, body in tables.items():
        if not body:
            print("skipped", name, "(no data yet)")
            continue
        (OUT / f"{name}.tex").write_text(typeset_minus(body), encoding="utf-8")
        print("wrote", name, len(body.splitlines()), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
