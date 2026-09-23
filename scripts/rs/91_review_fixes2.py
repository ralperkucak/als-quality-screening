r"""Second round of text corrections from the review.

Every entry names the file, the text it replaces and the replacement, so a rerun
is a no-op and a failed match is reported rather than silently skipped.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript_rs"

EDITS = [
    # The control that replaced the permutation, which could not fail at alpha = 0.
    ("sections/results.tex",
     "Permuting which class each weight belongs to, while keeping the weights\n"
     "themselves, leaves 0.014 with an interval from $-0.024$ to 0.058.",
     "Ordering tiles by the number of classes the model predicts in them, which is\n"
     "what a class-averaged mean reads when the values it averages carry nothing,\n"
     "falls below random review at $-0.124$ with an interval from $-0.144$ to\n"
     "$-0.103$."),
    ("sections/discussion.tex",
     "Permuting which class each weight belongs to removes the effect entirely, so the\n"
     "weights have to be attached to the classes they describe.",
     "Ordering by the number of predicted classes alone falls below random review, so\n"
     "the result is not an arithmetic consequence of averaging over a variable number\n"
     "of classes."),

    # The mismatch is not at the level of random review at both ends.
    ("sections/conclusions.tex",
     "score with the wrong convention costs almost all of the signal, from 0.50 to\n"
     "0.12 of the available lift at one end and from 0.68 to zero at the other.",
     "score with the wrong convention costs most of the signal, from 0.50 to 0.12 of\n"
     "the available lift at one end and from 0.68 to none at the other."),

    # Held-out selection, refreshed and with the spread over regions.
    ("sections/results.tex",
     "Selecting the exponent on four regions and applying it to the fifth shows that\n"
     "this does not depend on the region it was measured on. Across the five\n"
     "leave-one-region-out splits the exponent chosen for the class-averaged criterion\n"
     "averages 0.03 and delivers a lift of 0.481 on the held-out region, while the\n"
     "exponent chosen for the point-weighted criterion averages 1.05 and delivers\n"
     "0.680.",
     "The exponent was also selected on four regions and applied to the fifth. Across\n"
     "the five splits the exponent chosen for the class-averaged criterion averages\n"
     "0.03 and delivers a lift of 0.481 on the held-out region, with a range of 0.27\n"
     "to 0.62, while the exponent chosen for the point-weighted criterion is 1.05 in\n"
     "every split and delivers 0.680, with a range of 0.37 to 0.87."),

    # The point-weighted end is the loosely identified one, and the band reaches the grid.
    ("sections/results.tex",
     "1.05 and the exponents that tie with it span 0.75 to 2.00, so the optimum is\n"
     "broad and contains the point-weighted convention without being pinned to it.",
     "1.05 and the exponents that tie with it span 0.75 to the largest value examined,\n"
     "so the optimum is broad, contains the point-weighted convention, and is not\n"
     "bounded from above by the grid searched."),

    # Precision corrections against the outputs.
    ("sections/results.tex",
     "feature-support reliability recover between 0.265 and 0.297 of the worst tiles,",
     "feature-support reliability recover between 0.264 and 0.282 of the worst tiles,"),
    ("sections/results.tex",
     "0.250 to 0.281 over the regions for the class-averaged scores",
     "0.250 to 0.280 over the regions for the class-averaged scores"),
    ("sections/results.tex",
     "over union runs from 0.636 for bridge to 0.903 for water, and no single region\n"
     "and class pair falls below 0.54.",
     "over union runs from 0.636 for bridge to 0.902 for water, and no single region\n"
     "and class pair falls below 0.53."),
    ("sections/results.tex",
     "understates their error by a factor of four to five, at 0.0003 in sample\n"
     "against 0.0011 and 0.0014 out of sample,",
     "understates their error by a factor of four to five, at 0.0003 in sample\n"
     "against 0.0014 and 0.0011 out of sample,"),
    ("sections/results.tex",
     "0.0011 for equal-mass histogram binning, a sevenfold\n"
     "improvement, and the weighted Brier score improves from 0.0515 to 0.0497",
     "0.0011 for equal-mass histogram binning, a sevenfold\n"
     "improvement, and the weighted Brier score improves from 0.0515 for raw\n"
     "confidence to 0.0497 for isotonic regression"),
    ("sections/results.tex",
     "Translating the 2~km lattice by 500, 1000 and 1500~m in each axis gives recall at\n"
     "a fifth of the budget between 0.482 and 0.573 against 0.596 at the original\n"
     "alignment, with all five regions above random review in every case and a largest\n"
     "exact $p$ value of $2\\times10^{-4}$.",
     "Shifting the 2~km lattice to four other origins, by 500~m in both axes, by\n"
     "1000~m in both, by 500~m east and 1500~m north, and by 1500~m east and 500~m\n"
     "north, gives recall at a fifth of the budget between 0.482 and 0.573 against\n"
     "0.596 at the original alignment, with all five regions above random review in\n"
     "every case and a largest exact $p$ value of $1.9\\times10^{-4}$."),

    # Constructions whose work is denial rather than statement.
    ("sections/results.tex",
     "The advantage does not come from flagging large tiles. The selected fifth holds",
     "The selected fifth holds"),
    ("sections/results.tex",
     "The optimum is interior in both directions. Weighting rare classes more heavily\n"
     "than an equal average, which places the score exponent below zero, degrades\n"
     "screening against the class-averaged criterion, and weighting them less heavily\n"
     "degrades it in the other direction.",
     "Screening degrades on both sides of the optimum. A score exponent below zero\n"
     "concentrates the weight on classes holding few points and loses performance\n"
     "against the class-averaged criterion, and an exponent above it concentrates the\n"
     "weight on the abundant classes and loses performance in the same way."),
    ("sections/results.tex",
     "The tiles the screening flags are scattered across a region rather than\n"
     "concentrated in one part of it (Figure~\\ref{fig:maps}).",
     "The tiles the screening flags are scattered across a region\n"
     "(Figure~\\ref{fig:maps})."),
    ("sections/discussion.tex",
     "how much of the result it could explain, and it is a structural limit rather than\n"
     "a tuning problem: no confidence statistic exists for a class the model never\n"
     "proposed.",
     "how much of the result it could explain. No confidence statistic exists for a\n"
     "class the model never proposed, so the limit is structural."),
    ("sections/methods.tex",
     "Equation~\\eqref{eq:score-family} are the escort distribution of the predicted\n"
     "class shares, so the family is scale free and its two published members are\n"
     "recovered exactly rather than approximately.",
     "Equation~\\eqref{eq:score-family} are normalised powers of the predicted class\n"
     "shares, so the family is scale free and both published members are recovered\n"
     "exactly. In the vocabulary of multi-class evaluation, $\\alpha=1$ is\n"
     "micro-averaging and $\\alpha=0$ is macro-averaging."),
    ("sections/methods.tex",
     "sampling meets the inclusion rule. Basemaps are context only "
     "and no part of the analysis reads them. Basemap credits: "
     "Esri, Maxar, Earthstar Geographics and the GIS User "
     "Community.",
     "sampling meets the inclusion rule. Basemap credits: panel (\\textbf{a}) Esri, "
     "HERE, Garmin, \\copyright{} OpenStreetMap contributors and the GIS user "
     "community; panel (\\textbf{c}) Esri and USGS."),

    # Multiplicity, stated once.
    ("sections/methods.tex",
     "Every stage from the cached per-patch statistics to the final tables is a\n"
     "deterministic script with a fixed seed,",
     "Taking the largest $p$ value across the five regions tests the hypothesis that\n"
     "the ordering beats random review in every one of them, which is conservative\n"
     "and needs no further adjustment for the number of regions. The per-class and\n"
     "tiling families are reported as descriptive, and no conclusion rests on a\n"
     "single $p$ value near the conventional threshold.\n"
     "\n"
     "Every stage from the cached per-patch statistics to the final tables is a\n"
     "deterministic script with a fixed seed,"),

    # The Brier weighting.
    ("sections/methods.tex",
     "equal-width bins and the weighted Brier score as a secondary summary.",
     "equal-width bins and, as a secondary summary, the Brier score of the\n"
     "recalibrated confidence against correctness, each point weighted by the number\n"
     "of stored points it represents."),

    # The promised alternative conventions, now reported rather than promised.
    ("sections/methods.tex",
     "Restricting $\\mathcal{C}_B$ to classes present in the reference, and\n"
     "extending it to all seven classes with absent classes scored zero, are reported\n"
     "as alternative conventions.",
     "Restricting $\\mathcal{C}_B$ to the classes present in the reference, and\n"
     "extending it to all seven classes with absent classes scored zero, are reported\n"
     "in Section~\\ref{sec:results-robustness} as alternative conventions. The\n"
     "identity in Equation~\\eqref{eq:oa-identity} holds under the union rule and not\n"
     "under the other two."),
]


def main() -> int:
    missing = 0
    for name, old, new in EDITS:
        path = MS / name
        text = path.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            print("  MISS", name, repr(old[:62]))
            missing += 1
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print("applied", name, repr(new[:46]))
    print(f"\n{len(EDITS) - missing} of {len(EDITS)} edits in place")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
