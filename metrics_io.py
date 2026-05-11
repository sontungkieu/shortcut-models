import csv
import os
from pathlib import Path
from typing import Mapping

import numpy as np


def metrics_csv_path(path: str | None) -> str | None:
    if not path:
        return None
    metrics_path = Path(path)
    if metrics_path.suffix in (".jsonl", ".json"):
        return str(metrics_path.with_suffix(".csv"))
    return f"{path}.csv"


def append_metrics_csv(path: str | None, payload: Mapping[str, object]) -> None:
    csv_path = metrics_csv_path(path)
    if not csv_path:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    phase = payload.get("phase", "")
    step = payload.get("step", "")
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "step", "metric", "value"])
        if write_header:
            writer.writeheader()
        for name, value in sorted(payload.items()):
            if name in ("phase", "step"):
                continue
            arr = np.asarray(value)
            if arr.shape != ():
                continue
            try:
                scalar = float(arr)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(scalar):
                continue
            writer.writerow(
                {
                    "phase": phase,
                    "step": step,
                    "metric": name,
                    "value": scalar,
                }
            )


def clear_metrics_csv(path: str | None) -> None:
    csv_path = metrics_csv_path(path)
    if csv_path and os.path.exists(csv_path):
        os.remove(csv_path)
