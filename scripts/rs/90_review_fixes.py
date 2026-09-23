r"""Text corrections from the review round, applied once and checked.

Each entry names the file, the exact text it replaces and the replacement, so a
rerun is a no-op and a failed match is reported rather than silently skipped.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript_rs"

EDITS = [
    # Rounding corrected against derived_values.json.
    ("sections/results.tex",
     "it spans 0.042 across the five regions while the pooled\nmIoU spans 0.157.",
     "it spans 0.041 across the five regions while the pooled\nmIoU spans 0.156."),
    ("sections/results.tex",
     "understates their error by a factor of three to five, at 0.0003 in sample",
     "understates their error by a factor of four to five, at 0.0003 in sample"),
    ("sections/results.tex",
     "Such tiles are\ntwo to four times more frequent in the worst fifth than elsewhere,",
     "Such tiles are\n1.7 to 4.0 times more frequent in the worst fifth than elsewhere,"),

    # The normalised risk coverage range is across scores, not across regions.
    ("sections/results.tex",
     "The normalised area under the risk coverage curve, which is zero for an\n"
     "ordering that cannot be improved and one for an arbitrary order, is 0.250 to\n"
     "0.281 for the class-averaged scores and 0.794 to 0.849 for the point-weighted\n"
     "ones, with no overlap between the two groups (Figure~\\ref{fig:coverage}).",
     "The normalised area under the risk coverage curve, which is zero for an\n"
     "ordering that cannot be improved and one for an arbitrary order, averages\n"
     "0.250 to 0.281 over the regions for the class-averaged scores and 0.794 to\n"
     "0.849 for the point-weighted ones. Per region the class-averaged value runs\n"
     "from 0.16 in D4 to 0.39 in D1 and the point-weighted value from 0.74 in D5 to\n"
     "0.90 in D2, so the two families do not overlap in any region, although in D1\n"
     "the two curves meet above a budget of about a third of the tiles\n"
     "(Figure~\\ref{fig:coverage})."),

    # Why the calibrated score is the one carried through the figures.
    ("sections/results.tex",
     "The five class-averaged scores\nagree closely, and the calibrated confidence used in the figures and the\n"
     "controls below averages 0.596,",
     "The five class-averaged scores\nagree closely. The calibrated confidence is carried through the figures and the\n"
     "controls below because a recalibration step is what a production office would\n"
     "deploy and because it is the weakest of the five, so the mechanism results\n"
     "rest on the least favourable member of the family. It averages 0.596,"),

    # Bridge is the one class whose retrieval is not separated from random everywhere.
    ("sections/results.tex",
     "The class confidence leads it in all 27 class and\nregion comparisons that meet the eligibility rule,",
     "Bridge is the one class for which the retrieval is not separated from random\n"
     "review in every region, with a largest exact $p$ value of 0.404 across the\n"
     "three regions in which it is eligible. The class confidence leads the\n"
     "class-averaged score in all 27 class and region comparisons that meet the\n"
     "eligibility rule,"),

    # The patch accounting between the protocol and the tiles.
    ("sections/methods.tex",
     "contiguous block of 3000 patches and a further 800 patches that share no ground\n"
     "area with that block, and the screening analysis uses both, since no reference\n"
     "label from the held-out region enters any score.",
     "contiguous block of 3000 patches and a further 800 patches that share no ground\n"
     "area with that block, and the screening analysis uses both, since no reference\n"
     "label from the held-out region enters any score. Between 3628 and 3717 of\n"
     "those 3800 patches fall in cells that meet the inclusion rule below and reach\n"
     "the analysis."),
    ("sections/methods.tex",
     "six that were not read all belong to the released validation split, so the\n"
     "training and test splits are complete.",
     "six that were not read all belong to the released validation split, so the\n"
     "training and test splits are complete and the patches used for model\n"
     "selection and recalibration are drawn from the archives that were read."),

    # A paper does not annotate which of its equations lack a citation.
    ("sections/methods.tex",
     "Cosine similarity is used in place of a Mahalanobis distance because a\n"
     "32-dimensional class covariance estimated from a few hundred points of a rare\n"
     "class cannot be inverted stably. This statistic was constructed for the present\n"
     "study and carries no citation; it is included so that the comparison covers a\n"
     "score that reads more than the predictive distribution alone.",
     "Cosine similarity is used in place of a Mahalanobis distance because a\n"
     "32-dimensional class covariance estimated from a few hundred points of a rare\n"
     "class cannot be inverted stably. Including this statistic keeps a score in the\n"
     "comparison that reads the feature space as well as the posterior."),

    # Miller et al. is an accuracy-on-the-line study, not a disagreement estimator.
    ("sections/introduction.tex",
     "Agreement-based estimators use the disagreement of independently trained\n"
     "networks \\citep{jiang2022assessing, miller2021accuracy, baek2022agreement}, and\n"
     "others use auxiliary quantities such as a projection norm",
     "Agreement-based estimators use the disagreement of independently trained\n"
     "networks \\citep{jiang2022assessing, baek2022agreement}, a related line relates\n"
     "in-distribution to out-of-distribution accuracy directly\n"
     "\\citep{miller2021accuracy}, and others use auxiliary quantities such as a\n"
     "projection norm"),

    # Section heading register.
    ("sections/results.tex",
     "\\subsection{Sensitivity to the Tiling, Spatial Layout and Structural Blind Spots}",
     "\\subsection{Sensitivity to the Tiling and the Layout of the Review Set}"),
    ("sections/results.tex",
     "\\subsection{What Recalibration Contributes}",
     "\\subsection{Contribution of the Recalibration Step}"),
]


def main() -> int:
    missing = 0
    for name, old, new in EDITS:
        path = MS / name
        text = path.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            print("  MISS", name, repr(old[:60]))
            missing += 1
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print("applied", name, repr(new[:48]))
    print(f"\n{len(EDITS) - missing} of {len(EDITS)} edits in place")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
