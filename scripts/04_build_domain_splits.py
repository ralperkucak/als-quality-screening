"""Phase 3: build and freeze the leave-one-domain-out splits, then audit leakage.

Outputs
-------
data/manifests/lodo_splits.parquet   frozen patch -> (fold, role) assignment
data/manifests/split_config.json     the configuration that produced them
docs/leakage_audit.md                human-readable audit
outputs/audit/leakage_audit.json     machine-readable audit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoquery_als.config import DOC_DIR, MANIFEST_DIR, OUTPUT_DIR, PATHS  # noqa: E402
from geoquery_als.data.splits import (SplitConfig, build_folds, check_leakage,  # noqa: E402
                                      config_as_dict, folds_to_frame)
from geoquery_als.data.store import PatchStore  # noqa: E402


def main() -> int:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "main.yaml"
    conf = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = SplitConfig(**conf["splits"])

    index = pd.read_parquet(PATHS.patch_index)
    store = PatchStore()
    available = store.available_patches()
    print(f"patches in index {len(index):,}; downloaded and preprocessed {len(available):,}")

    folds = build_folds(index, cfg, available=available)
    frame = folds_to_frame(folds)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(MANIFEST_DIR / "lodo_splits.parquet", index=False)
    (MANIFEST_DIR / "split_config.json").write_text(
        json.dumps({"split_config": config_as_dict(cfg),
                    "n_available_patches": len(available),
                    "fold_diagnostics": {f.target_domain: f.diagnostics for f in folds}},
                   indent=2), encoding="utf-8")

    report = check_leakage(folds, index)
    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit" / "leakage_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, folds)

    for f in folds:
        d = f.diagnostics
        print(f"{f.target_domain}: src_train {len(f.source_train):5d} "
              f"src_val {len(f.source_val):4d} pool {len(f.target_pool):5d} "
              f"test {len(f.target_test):4d} | pool NN {d['pool_median_nn_m']:6.0f} m "
              f"block {d['pool_extent_x_km']:.0f}x{d['pool_extent_y_km']:.0f} km "
              f"| pool->test min {d['min_pool_to_test_m']:.0f} m")
    print("leakage audit passed:", report["passed"])
    return 0 if report["passed"] else 1


def _write_markdown(report: dict, folds) -> None:
    lines = ["# Leakage audit", "",
             f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**", "",
             "Every fold is checked for identifier overlap between roles, for the",
             "presence of any target-domain patch on the source side, and for the",
             "geometric separation of the candidate pool and the target test block.",
             ""]
    for dom, r in report["folds"].items():
        lines += [f"## Target domain {dom}", "",
                  "| check | result |", "|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {'pass' if v else 'FAIL'} |"
                  for k, v in r["checks"].items()]
        lines += ["",
                  f"- sizes: {r['sizes']}",
                  f"- minimum candidate-pool to target-test distance: "
                  f"{r['min_pool_to_test_distance_m']:.1f} m",
                  f"- minimum source to target-test distance: "
                  f"{r['min_source_to_target_test_distance_m']/1000:.1f} km", ""]
    for f in folds:
        lines += [f"### {f.target_domain} candidate block geometry", "",
                  "```json", json.dumps(f.diagnostics, indent=2), "```", ""]
    (DOC_DIR / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
