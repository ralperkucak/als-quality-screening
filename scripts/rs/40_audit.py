r"""Checks the manuscript against the rules it is written to.

Three families of check. Whether the source is intact, which catches the
escaping accidents that a shell can introduce and that LaTeX absorbs silently.
Whether the prose obeys the style the paper is written in. And whether every
number in the prose exists in an analysis output, which catches a figure quoted
from memory rather than from a file.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript_rs"
SECTIONS = MS / "sections"
DATA = ROOT / "outputs" / "rs"

BODY = ("introduction", "methods", "results", "discussion", "conclusions")

BANNED = {
    "em dash": "\u2014",
    "en dash between words": "\u2013 ",
    "double hyphen": "--",
    "inline enumeration": r"\(i\)\s|\(ii\)\s|\(iii\)\s",
    "in conclusion": r"(?i)\bin conclusion\b",
    "overall comma": r"(?i)^overall,",
    "to summarise": r"(?i)\bto summari[sz]e\b",
    "it is worth noting": r"(?i)\bit is worth noting\b",
    "delve": r"(?i)\bdelve[sd]?\b",
    "leverage as a verb": r"(?i)\bleverag(e|es|ed|ing)\b",
    "crucial": r"(?i)\bcrucial\b",
    "pivotal": r"(?i)\bpivotal\b",
    "robustly": r"(?i)\brobustly\b",
    "internal identifier": r"atc_classwise|unit_table|randlanet\.py|alpha=|_parquet",
}

# Style rule: a sentence that denies one thing to assert another.
DEFENSIVE = re.compile(
    r"\b(?:is|are|was|were|does|do|did)\s+not\s+[^.,;]{0,60},\s+but\s", re.I)


def strip_latex(text: str) -> str:
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", text, flags=re.S)
    text = re.sub(r"\\(?:cite[tp]?|ref|eqref|label|input)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = re.sub(r"[{}$~\\]", " ", text)
    return re.sub(r"\s+", " ", text)


def known_numbers() -> set:
    """Every number any analysis output holds, at the precision it is quoted."""
    values = set()

    def add(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return
        if v != v:
            return
        for digits in (0, 1, 2, 3, 4):
            values.add(f"{abs(v):.{digits}f}".rstrip("0").rstrip("."))
            values.add(f"{abs(v):.{digits}f}")
        if v != 0:
            exponent = int(np.floor(np.log10(abs(v))))
            mantissa = abs(v) / 10.0 ** exponent
            for digits in (1, 2):
                values.add(f"{mantissa:.{digits}f}")

    for path in sorted(DATA.glob("*.csv")):
        frame = pd.read_csv(path)
        for column in frame.columns:
            for x in frame[column].tolist():
                add(x)
    for path in sorted(DATA.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            else:
                add(item)
    return values


OPENER_LIMIT = 9


def short_openers(text: str) -> list:
    """First sentences of paragraphs that are too short to carry a paragraph."""
    out = []
    for block in text.split("\n\n"):
        if not block.strip() or block.lstrip().startswith("\\"):
            continue
        clean = strip_latex(block).strip()
        if not clean:
            continue
        first = re.split(r"(?<=[.!?])\s", clean)[0].strip()
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", first)
        if 0 < len(words) <= OPENER_LIMIT:
            out.append((len(words), first))
    return out


def main() -> int:
    failures = 0

    print("== source integrity ==")
    for path in sorted(SECTIONS.glob("*.tex")) + [MS / "main.tex"]:
        raw = path.read_bytes()
        bad = sorted({c for c in raw if c < 0x20 and c not in (10, 13)})
        if bad:
            print(f"  FAIL {path.name}: control bytes {[hex(b) for b in bad]}")
            failures += 1
    stray = []
    for path in sorted(SECTIONS.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        stray += [(path.name, m) for m in
                  re.findall(r"(?<![\\\w])(?:textbf|begin|end|ref|cite[tp]?|"
                             r"label|includegraphics|section|subsection)\{",
                             text)]
    if stray:
        print(f"  FAIL macro name without a backslash: {stray[:5]}")
        failures += 1
    if not failures:
        print("  no control characters, no stripped macro names")

    print("\n== length ==")
    total = 0
    for name in BODY:
        words = len(strip_latex(
            (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")).split())
        total += words
        print(f"  {name:<14} {words:5d}")
    print(f"  {'body total':<14} {total:5d}")
    main = (MS / "main.tex").read_text(encoding="utf-8")
    start = main.index("\\abstract{") + len("\\abstract{")
    end = main.index("\\keyword{")
    print(f"  {'abstract':<14} {len(strip_latex(main[start:end]).split()):5d}")

    print("\n== style ==")
    hits = 0
    for name in BODY:
        text = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")
        prose = strip_latex(text)
        for label, pattern in BANNED.items():
            found = (text.count(pattern) if len(pattern) < 4
                     and not any(ch in pattern for ch in "\\[(")
                     else len(re.findall(pattern, prose)))
            if found:
                print(f"  {name}: {label} x{found}")
                hits += found
        for match in DEFENSIVE.finditer(prose):
            print(f"  {name}: defensive construction: "
                  f"...{match.group(0).strip()}...")
            hits += 1
    if not hits:
        print("  clean")

    openers = []
    for name in BODY:
        text = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")
        openers += [(name, n, s) for n, s in short_openers(text)]
    if openers:
        print(f"  {len(openers)} paragraphs open with a sentence of "
              f"{OPENER_LIMIT} words or fewer:")
        for name, n, s in openers:
            print(f"    {name} ({n}): {s[:70]}")
        hits += len(openers)

    print("\n== numbers ==")
    values = known_numbers()
    pattern = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")
    unmatched = []
    for name in BODY:
        prose = strip_latex(
            (SECTIONS / f"{name}.tex").read_text(encoding="utf-8"))
        for token in pattern.findall(prose):
            if token not in values and token.lstrip("0") not in values:
                unmatched.append((name, token))
    print(f"  decimal figures in prose: "
          f"{sum(len(pattern.findall(strip_latex((SECTIONS / f'{n}.tex').read_text(encoding='utf-8')))) for n in BODY)}")
    if unmatched:
        print(f"  not found in any analysis output ({len(unmatched)}):")
        for name, token in unmatched:
            print(f"    {name}: {token}")
    else:
        print("  every decimal figure traces to an analysis output")

    print("\n== compile ==")
    log = MS / "main.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        for label, pattern in (("errors", r"^! "),
                               ("undefined citations", r"Citation .*undefined"),
                               ("undefined references", r"Reference .*undefined"),
                               ("overfull boxes", r"Overfull \\hbox")):
            print(f"  {label:<22} {len(re.findall(pattern, text, re.M))}")
        pages = re.search(r"Output written on .*?\((\d+) pages", text)
        if pages:
            print(f"  {'pages':<22} {pages.group(1)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
