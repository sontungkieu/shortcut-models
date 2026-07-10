from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping


def parse_eval_fid_seeds(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        raw_values = [token.strip() for token in value.split(",")]
        seeds = [int(token) for token in raw_values if token]
    else:
        seeds = [int(seed) for seed in value]
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        raise ValueError("At least one eval FID seed is required.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Eval FID seeds must be non-negative.")
    return seeds


def sample_mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(max(variance, 0.0))


def summarize_fid_repeat_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for record in records:
        metric_name = str(record["metric_name"])
        grouped[metric_name].append(
            {
                "eval_seed": int(record["eval_seed"]),
                "value": float(record["value"]),
            }
        )

    metrics = {}
    for metric_name, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["eval_seed"]))
        values = [float(row["value"]) for row in rows]
        mean, sample_std = sample_mean_std(values)
        metrics[metric_name] = {
            "n": len(values),
            "mean": mean,
            "sample_std": sample_std,
            "standard_error": sample_std / math.sqrt(len(values)),
            "min": min(values),
            "max": max(values),
            "by_seed": rows,
        }
    return {"metrics": metrics}
