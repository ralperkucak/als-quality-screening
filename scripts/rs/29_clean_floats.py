r"""Remove every float that the layout step inserts, so it can insert them again.

Renaming a table or a figure leaves the previous inclusion behind, because the
layout step only adds. Running this first makes the layout idempotent under
renames as well as under reruns.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "manuscript_rs" / "sections"
TABLES = ROOT / "manuscript_rs" / "tables"

KEEP = {
    "table_regions", "table_transfer", "table_retrieval", "table_exponents",
    "table_controls", "table_per_class", "table_calibration",
    "table_robustness", "table_backbone", "table_configuration",
    "table_baselines", "table_universe", "table_seeds",
}

INPUT = re.compile(r"\n*\\input\{tables/[A-Za-z0-9_]+\}\n*")
FLOAT = re.compile(r"\n*\\begin\{figure\}.*?\\end\{figure\}\n*", re.S)
BLANK = re.compile(r"\n{3,}")


def main() -> int:
    for name in ("results.tex", "methods.tex", "discussion.tex"):
        path = SECTIONS / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        cleaned = BLANK.sub("\n\n", FLOAT.sub("\n\n", INPUT.sub("\n\n", text)))
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")
            print("cleared floats from", name)

    for path in sorted(TABLES.glob("*.tex")):
        if path.stem not in KEEP:
            path.unlink()
            print("removed stale table", path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
