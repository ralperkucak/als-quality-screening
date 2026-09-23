"""Record the hardware and software environment the results were produced on."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoquery_als.config import DATA_ROOT, DOC_DIR, REPO_ROOT  # noqa: E402


def main() -> int:
    import numpy
    import scipy
    import torch

    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "total_memory_GiB": round(p.total_memory / 2**30, 2),
               "capability": f"{p.major}.{p.minor}",
               "multi_processor_count": p.multi_processor_count,
               "cuda_runtime": torch.version.cuda, "n_devices": torch.cuda.device_count()}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception:                                                # noqa: BLE001
        smi = ""

    du = shutil.disk_usage(DATA_ROOT)
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": __import__("os").cpu_count(),
        "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
        "gpu": gpu, "nvidia_driver": smi,
        "data_root": str(DATA_ROOT),
        "data_root_free_GiB": round(du.free / 2**30, 1),
        "data_root_used_GiB": round(du.used / 2**30, 1),
    }
    (DOC_DIR / "compute_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    lines = ["# Compute environment", "",
             "Every number in this study was produced on the machine described here.", "",
             "| item | value |", "|---|---|"]
    for k, v in env.items():
        lines.append(f"| {k} | {json.dumps(v) if isinstance(v, dict) else v} |")
    lines += ["", "The GPU is a single laptop-class device with 8 GiB of memory, which is",
              "below the lowest compute tier the original protocol anticipated. The",
              "experiment was rescaled accordingly; see `plan_review.md` section 7 and",
              "`decision_log.md` entries D06 and D10. No leakage control, baseline or",
              "statistical procedure was weakened to fit the hardware."]
    (DOC_DIR / "compute_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
