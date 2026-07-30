from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from helper_eval import compute_trajectory_metrics, parse_trajectory_save_steps
from scripts.plot_denoising_trajectories import (
    comparability_audit,
    fit_randomized_pca,
    load_trajectory,
    plot_individual_paths,
    plot_mean_paths,
    plot_metric_boxes,
    project_trajectories,
)
from scripts.submit_gmm_tide_fm_jobs import make_notebook


def _write_fixture(path: Path, *, seed: int, offset: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(8, 1, 2, 2, 2)).astype(np.float32)
    directions = rng.normal(size=(8, 1, 2, 2, 2)).astype(np.float32)
    times = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    states = starts + times[None, :, None, None, None] * directions + offset
    per_sample, _ = compute_trajectory_metrics(states)
    metadata = {
        "checkpoint_step": 400000,
        "trajectory_seed": 42,
        "trajectory_num_samples": 8,
        "trajectory_timesteps": 128,
        "trajectory_save_steps": [0, 32, 64, 96, 128],
    }
    np.savez_compressed(
        path,
        states=states,
        times=times,
        save_steps=np.asarray(metadata["trajectory_save_steps"], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)),
        **per_sample,
    )


def test_parse_trajectory_save_steps_adds_endpoints_and_validates_range():
    assert parse_trajectory_save_steps("32,64,32", 128) == [0, 32, 64, 128]
    default = parse_trajectory_save_steps("", 128)
    assert default[0] == 0
    assert default[-1] == 128
    assert len(default) == 17
    with pytest.raises(ValueError, match="must lie"):
        parse_trajectory_save_steps("129", 128)


def test_compute_trajectory_metrics_is_exact_for_straight_paths():
    starts = np.zeros((3, 1, 1, 1, 2), dtype=np.float32)
    direction = np.asarray([[[[[3.0, 4.0]]]]], dtype=np.float32)
    times = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    states = starts + times[None, :, None, None, None] * direction
    per_sample, summary = compute_trajectory_metrics(states)
    assert per_sample["path_length"] == pytest.approx(np.full(3, 5.0))
    assert per_sample["endpoint_displacement"] == pytest.approx(np.full(3, 5.0))
    assert per_sample["straightness_ratio"] == pytest.approx(np.ones(3))
    assert per_sample["curvature_proxy"] == pytest.approx(np.zeros(3), abs=1e-6)
    assert summary["straightness_ratio_mean"] == pytest.approx(1.0)


def test_randomized_pca_returns_orthonormal_components():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(64, 12)).astype(np.float32)
    _, components, explained = fit_randomized_pca(matrix)
    assert components.shape == (2, 12)
    assert components @ components.T == pytest.approx(np.eye(2), abs=1e-5)
    assert np.all(explained > 0)


def test_multi_model_plotting_uses_one_basis_and_writes_all_outputs(tmp_path):
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _write_fixture(first_path, seed=1)
    _write_fixture(second_path, seed=2, offset=0.2)
    records = [
        load_trajectory("top-4 c=0.75", first_path),
        load_trajectory("top-4 c=1.25", second_path),
    ]
    projected, pca_info = project_trajectories(records)
    assert pca_info["fit_samples"] == 80
    assert [coords.shape for coords in projected] == [(8, 5, 2), (8, 5, 2)]
    assert comparability_audit(records)["status"] == "PASS"

    outputs = []
    outputs.extend(plot_individual_paths(records, projected, tmp_path, max_paths=4))
    outputs.extend(plot_mean_paths(records, projected, tmp_path))
    outputs.extend(plot_metric_boxes(records, tmp_path))
    assert len(outputs) == 6
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs)


def test_kaggle_notebook_renders_eval_only_trajectory_mode():
    notebook = make_notebook(
        {
            "run_name": "trajectory-smoke",
            "execution_mode": "trajectory_eval",
            "trajectory_seed": 42,
            "trajectory_num_samples": 64,
            "trajectory_timesteps": 128,
            "trajectory_decode_samples": 8,
        }
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert '"trajectory_eval"' in source
    assert '"--mode", "eval-trajectory"' in source
    assert '"--trajectory_output_path"' in source
    for index, cell in enumerate(notebook["cells"]):
        cell_source = "".join(cell.get("source", []))
        if cell_source.strip():
            compile(cell_source, f"cell_{index}.py", "exec")
