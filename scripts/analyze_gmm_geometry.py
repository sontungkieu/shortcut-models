#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _fmt(value, digits=4):
    value = _finite_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _pairwise_cosine(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.maximum(norms, 1e-12)
    cos = unit @ unit.T
    mask = ~np.eye(cos.shape[0], dtype=bool)
    values = cos[mask]
    if values.size == 0:
        values = np.array([1.0], dtype=np.float32)
    return values


def _pairwise_dist(x):
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    mask = ~np.eye(dist.shape[0], dtype=bool)
    values = dist[mask]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    return values


def _nearest_neighbor_cosines(x, k_neighbors: int):
    if x.shape[0] <= 1:
        return np.array([1.0], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.maximum(norms, 1e-12)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    k = min(k_neighbors, x.shape[0] - 1)
    ids = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    row = np.arange(x.shape[0])[:, None]
    return np.sum(unit[row, :] * unit[ids], axis=-1).reshape(-1)


def analyze_npz(path: Path):
    data = np.load(path)
    required = {"mu", "var", "pi"}
    if not required.issubset(set(data.files)):
        return None
    mu = np.asarray(data["mu"], dtype=np.float64)
    var = np.asarray(data["var"], dtype=np.float64)
    pi = np.asarray(data["pi"], dtype=np.float64)
    data_var = np.asarray(data["data_var"], dtype=np.float64) if "data_var" in data.files else None
    data_mean = np.asarray(data["data_mean"], dtype=np.float64) if "data_mean" in data.files else None

    center_norm = np.linalg.norm(mu, axis=1)
    component_noise = np.sqrt(np.sum(np.maximum(var, 0.0), axis=1))
    snr = center_norm / np.maximum(component_noise, 1e-12)
    pair_cos = _pairwise_cosine(mu)
    pair_dist = _pairwise_dist(mu)
    nn2_cos = _nearest_neighbor_cosines(mu, 2)
    nn4_cos = _nearest_neighbor_cosines(mu, 4)
    data_trace = float(np.sum(data_var)) if data_var is not None else None
    data_mean_norm = float(np.linalg.norm(data_mean)) if data_mean is not None else None
    data_rms = math.sqrt(max(data_trace or 0.0, 0.0))
    if data_mean_norm is not None:
        data_rms = math.sqrt(data_rms * data_rms + data_mean_norm * data_mean_norm)

    pi_safe = np.maximum(pi, 1e-12)
    pi_entropy = -float(np.sum(pi_safe * np.log(pi_safe)))
    pi_entropy_norm = pi_entropy / math.log(max(len(pi), 2))

    return {
        "path": str(path),
        "run_name": path.parent.parent.name if path.parent.name == "diagnostics" else path.parent.name,
        "num_modes": int(mu.shape[0]),
        "dim": int(mu.shape[1]),
        "center_norm_mean": float(np.mean(center_norm)),
        "center_norm_min": float(np.min(center_norm)),
        "center_norm_max": float(np.max(center_norm)),
        "component_noise_mean": float(np.mean(component_noise)),
        "component_noise_min": float(np.min(component_noise)),
        "component_noise_max": float(np.max(component_noise)),
        "center_noise_snr_mean": float(np.mean(snr)),
        "center_noise_snr_min": float(np.min(snr)),
        "center_noise_snr_max": float(np.max(snr)),
        "data_trace": data_trace,
        "data_rms": data_rms,
        "center_norm_to_data_rms": float(np.mean(center_norm) / max(data_rms, 1e-12)),
        "pair_cos_mean": float(np.mean(pair_cos)),
        "pair_cos_p05": float(np.quantile(pair_cos, 0.05)),
        "pair_cos_min": float(np.min(pair_cos)),
        "pair_dist_mean": float(np.mean(pair_dist)),
        "pair_dist_min": float(np.min(pair_dist)),
        "nearest2_cos_mean": float(np.mean(nn2_cos)),
        "nearest2_cos_p05": float(np.quantile(nn2_cos, 0.05)),
        "nearest4_cos_mean": float(np.mean(nn4_cos)),
        "nearest4_cos_p05": float(np.quantile(nn4_cos, 0.05)),
        "pi_entropy_normalized": pi_entropy_norm,
    }


def analyze_gmm_metrics(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    center_mean = _finite_float(data.get("center_distance_mean"))
    center_min = _finite_float(data.get("center_distance_min"))
    trace_mean = _finite_float(data.get("component_variance_trace_mean"))
    trace_min = _finite_float(data.get("component_variance_trace_min"))
    if center_mean is None or trace_mean is None:
        return None
    noise_mean = math.sqrt(max(trace_mean, 0.0))
    noise_min = math.sqrt(max(trace_min or 0.0, 0.0))
    return {
        "path": str(path),
        "run_name": path.parent.parent.name if path.parent.name == "diagnostics" else path.parent.name,
        "num_modes": data.get("num_modes"),
        "center_distance_mean": center_mean,
        "center_distance_min": center_min,
        "component_noise_mean": noise_mean,
        "component_noise_min": noise_min,
        "center_dist_noise_ratio": center_mean / max(noise_mean, 1e-12),
        "center_min_noise_ratio": (center_min / max(noise_mean, 1e-12)) if center_min is not None else None,
        "component_variance_mean": _finite_float(data.get("component_variance_mean")),
        "data_variance_mean": _finite_float(data.get("data_variance_mean")),
        "pi_entropy_normalized": _finite_float(data.get("pi_entropy_normalized")),
        "valid_count_ratio": _finite_float(data.get("valid_count_ratio")),
        "valid_dead_components": data.get("valid_dead_components"),
        "overlap_proxy_max": _finite_float(data.get("overlap_proxy_max")),
        "var_floor_hit_rate": _finite_float(data.get("var_floor_hit_rate")),
    }


def summarize_logged_geometry(train_csv_paths):
    rows = []
    wanted = (
        "geometry/",
        "tide/topk_mu_",
        "tide/x0_tide_base/",
        "tide/mu_tide_base_mu/",
    )
    for path in train_csv_paths:
        last = {}
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric = row.get("metric", "")
                    if metric.startswith(wanted):
                        last[metric] = row.get("value")
        except Exception:
            continue
        if last:
            rows.append({
                "path": str(path),
                "run_name": path.parent.parent.name if path.parent.name == "diagnostics" else path.parent.name,
                "metrics": {k: _finite_float(v) for k, v in sorted(last.items())},
            })
    return rows


def write_report(output_md: Path, npz_rows, metrics_rows, logged_rows):
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GMM/TIDE Geometry Analysis",
        "",
        "Mục tiêu: kiểm tra giả thuyết source GMM/TIDE có thể giữ khoảng cách nhưng phá hình học góc của latent.",
        "",
        "## Kết Luận Ngắn",
        "",
    ]
    low_ratio = [r for r in metrics_rows if (r.get("center_dist_noise_ratio") or 0.0) < 1.0]
    if metrics_rows:
        ratios = [r["center_dist_noise_ratio"] for r in metrics_rows if r.get("center_dist_noise_ratio") is not None]
        lines += [
            f"- Có {len(metrics_rows)} `gmm_metrics.json` có đủ `center_distance` và `component_variance_trace`.",
            f"- {len(low_ratio)}/{len(metrics_rows)} run có `center_distance_mean / sqrt(component_variance_trace_mean) < 1`. Khi tỷ lệ này nhỏ hơn 1, khoảng cách tâm component nhỏ hơn độ nhiễu RMS trong component; source sample dễ bị nhiễu hướng và cosine/angle trở thành metric bắt buộc.",
            f"- Tỷ lệ này có min/median/max = `{_fmt(min(ratios))}` / `{_fmt(float(np.median(ratios)))}` / `{_fmt(max(ratios))}`.",
        ]
    if npz_rows:
        snrs = [r["center_noise_snr_mean"] for r in npz_rows]
        nn2 = [r["nearest2_cos_mean"] for r in npz_rows]
        lines += [
            f"- Có {len(npz_rows)} `gmm_stats.npz` đủ `mu/var`; mean center/noise SNR = `{_fmt(float(np.mean(snrs)))}`, nearest-2 center cosine mean = `{_fmt(float(np.mean(nn2)))}`.",
            "- SNR tâm/noise thấp nghĩa là mỗi Gaussian component không định nghĩa một hướng latent sắc; weighted top-k càng dễ sinh `x0_tide` nằm giữa nhiều hướng.",
        ]
    if not logged_rows:
        lines += [
            "- Không tìm thấy geometry cosine metrics trong các `train_metrics.csv` cũ. Điều này xác nhận các run cũ chưa trực tiếp đo góc `x0`-`x1` hoặc angular dispersion top-k.",
        ]
    else:
        lines += [f"- Tìm thấy {len(logged_rows)} run đã có geometry metrics trong `train_metrics.csv`."]

    lines += [
        "",
        "## GMM Center/Noise Proxy Tệ Nhất",
        "",
        "| run | K | center/noise | min center/noise | center mean | noise mean | pi entropy | dead valid | count ratio | floor hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metrics_rows, key=lambda r: r.get("center_dist_noise_ratio") or 999.0)[:30]:
        lines.append(
            f"| `{row['run_name']}` | {row.get('num_modes', '')} | {_fmt(row.get('center_dist_noise_ratio'))} | "
            f"{_fmt(row.get('center_min_noise_ratio'))} | {_fmt(row.get('center_distance_mean'))} | "
            f"{_fmt(row.get('component_noise_mean'))} | {_fmt(row.get('pi_entropy_normalized'))} | "
            f"{row.get('valid_dead_components', '')} | {_fmt(row.get('valid_count_ratio'))} | {_fmt(row.get('var_floor_hit_rate'))} |"
        )

    lines += [
        "",
        "## GMM Stats Angular Metrics",
        "",
        "| run | K | center/noise SNR | center/data RMS | pair cos mean | pair cos p05 | nearest2 cos | nearest4 cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(npz_rows, key=lambda r: r.get("center_noise_snr_mean") or 999.0):
        lines.append(
            f"| `{row['run_name']}` | {row['num_modes']} | {_fmt(row['center_noise_snr_mean'])} | "
            f"{_fmt(row['center_norm_to_data_rms'])} | {_fmt(row['pair_cos_mean'])} | "
            f"{_fmt(row['pair_cos_p05'])} | {_fmt(row['nearest2_cos_mean'])} | {_fmt(row['nearest4_cos_mean'])} |"
        )

    lines += [
        "",
        "## Cách Đọc",
        "",
        "- `center/noise`: khoảng cách trung bình giữa tâm GMM chia cho RMS noise trong một component. Nhỏ hơn 1 là cảnh báo hình học: các component tách theo Euclidean nhưng sample trong component có độ nhiễu đủ lớn để làm hướng bị mờ.",
        "- `center/noise SNR`: norm tâm component chia cho RMS noise component. Thấp nghĩa là hướng từ gốc tới component không sắc.",
        "- `nearest-k cos`: cosine giữa tâm component và các tâm gần nhất theo Euclidean. Nếu thấp hoặc âm, weighted top-k có thể trộn các hướng khác nhau và kéo source vào vùng giữa mode.",
        "",
        "## Khắc Phục Đề Xuất",
        "",
        "1. Log trực tiếp cosine/angle trong training: `geometry/x0_x1/*`, `geometry/v_x1/*`, `tide/topk_mu_angular_dispersion`, `tide/x0_tide_base/*`.",
        "2. Khi `topk_mu_angular_dispersion` cao, ưu tiên source sparse hơn: `topk=1`, `topk=2` với temperature thấp, hoặc hard-sample một component thay vì weighted mean nhiều component.",
        "3. Với joint router, có thể bật regularizer `gmm_router_geometry_weight > 0` để phạt `tide/topk_mu_angular_dispersion` mà vẫn giữ GMM cố định.",
        "4. Không rank source bằng NLL/khoảng cách đơn lẻ. Rank theo FID + flow curvature + geometry cosine + collapse metrics.",
        "5. Nếu muốn sửa GMM fit, cân nhắc spherical/cosine-aware preprocessing hoặc angular penalty cho centers; nhưng thay đổi này nên sau khi đã có logs cosine trên CelebA.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm-stats-glob", action="append", default=["outputs/**/*.npz"])
    parser.add_argument("--gmm-metrics-glob", action="append", default=["outputs/**/gmm_metrics.json"])
    parser.add_argument("--train-metrics-glob", action="append", default=["outputs/**/train_metrics.csv"])
    parser.add_argument("--output-json", default="reports/gmm_tide_geometry_analysis.json")
    parser.add_argument("--output-md", default="reports/gmm_tide_geometry_analysis.md")
    args = parser.parse_args()

    npz_rows = []
    for pattern in args.gmm_stats_glob:
        for path in glob.glob(pattern, recursive=True):
            row = analyze_npz(Path(path))
            if row is not None:
                npz_rows.append(row)

    metrics_rows = []
    for pattern in args.gmm_metrics_glob:
        for path in glob.glob(pattern, recursive=True):
            row = analyze_gmm_metrics(Path(path))
            if row is not None:
                metrics_rows.append(row)

    train_paths = []
    for pattern in args.train_metrics_glob:
        train_paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    logged_rows = summarize_logged_geometry(train_paths)

    payload = {
        "npz_rows": npz_rows,
        "gmm_metrics_rows": metrics_rows,
        "logged_geometry_rows": logged_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(Path(args.output_md), npz_rows, metrics_rows, logged_rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
