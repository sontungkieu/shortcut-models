from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gmm_utils import (
    centered_component_params_from_ids,
    component_params_from_ids,
    infer_component_params,
    mixture_mean_from_stats,
    sample_components,
)
from baselines.targets_naive import get_targets
from scripts.submit_gmm_tide_fm_jobs import load_grid, make_notebook


LATENT_SHAPE = (1, 1, 2)


def raw_gmm_state():
    return {
        "pi": jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        "mu": jnp.asarray([[2.0, -1.0], [6.0, 3.0]], dtype=jnp.float32),
        "var": jnp.asarray([[1.0, 4.0], [9.0, 16.0]], dtype=jnp.float32),
        "transform_type": "raw",
    }


def channel_whiten_gmm_state():
    return {
        "pi": jnp.asarray([0.4, 0.6], dtype=jnp.float32),
        "mu": jnp.asarray([[-1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32),
        "var": jnp.asarray([[0.5, 1.0], [1.5, 0.75]], dtype=jnp.float32),
        "transform_type": "channel_whiten",
        "latent_shape": jnp.asarray(LATENT_SHAPE, dtype=jnp.int32),
        "channel_mean": jnp.asarray([1.0, -2.0], dtype=jnp.float32),
        "channel_unwhiten": jnp.asarray([[2.0, 0.5], [0.0, 1.5]], dtype=jnp.float32),
    }


def standardized_gmm_state():
    return {
        "pi": jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        "mu": jnp.asarray([[-1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32),
        "var": jnp.asarray([[0.5, 1.0], [1.5, 0.75]], dtype=jnp.float32),
        "transform_type": "standardize",
        "mean": jnp.asarray([10.0, -2.0], dtype=jnp.float32),
        "std": jnp.asarray([2.0, 3.0], dtype=jnp.float32),
    }


@pytest.mark.parametrize("center_scale", [0.0, 0.5, 0.75, 1.0, 1.3, 1.75])
def test_weighted_centered_component_mean_is_zero(center_scale):
    state = raw_gmm_state()
    component_ids = jnp.arange(2)
    source_mu, _ = centered_component_params_from_ids(
        state,
        component_ids,
        LATENT_SHAPE,
        center_scale=center_scale,
    )
    weighted_mean = jnp.sum(state["pi"][:, None, None, None] * source_mu, axis=0)
    assert weighted_mean == pytest.approx(jnp.zeros(LATENT_SHAPE), abs=1e-6)


@pytest.mark.parametrize("center_scale", [0.0, 0.5, 0.75, 1.0, 1.3, 1.75])
def test_pairwise_component_distance_scales_by_c(center_scale):
    state = raw_gmm_state()
    original_mu, _ = component_params_from_ids(state, jnp.arange(2), LATENT_SHAPE)
    source_mu, _ = centered_component_params_from_ids(
        state,
        jnp.arange(2),
        LATENT_SHAPE,
        center_scale=center_scale,
    )
    original_distance = jnp.linalg.norm(original_mu[1] - original_mu[0])
    source_distance = jnp.linalg.norm(source_mu[1] - source_mu[0])
    assert float(source_distance) == pytest.approx(center_scale * float(original_distance), abs=1e-6)


@pytest.mark.parametrize("state", [raw_gmm_state(), standardized_gmm_state(), channel_whiten_gmm_state()])
def test_center_scaling_preserves_exact_within_component_residual_and_sigma(state):
    key = jax.random.PRNGKey(19)
    component_ids = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    x_original, mu_original, sigma_original = sample_components(
        key,
        state,
        component_ids,
        LATENT_SHAPE,
    )
    x_centered, mu_centered, sigma_centered = sample_components(
        key,
        state,
        component_ids,
        LATENT_SHAPE,
        center_scale=1.3,
    )
    assert x_centered - mu_centered == pytest.approx(x_original - mu_original, abs=1e-6)
    assert sigma_centered == pytest.approx(sigma_original, abs=1e-7)


@pytest.mark.parametrize("state", [standardized_gmm_state(), channel_whiten_gmm_state()])
def test_centering_happens_after_inverse_transform_to_latent_space(state):
    source_mu, _ = centered_component_params_from_ids(
        state,
        jnp.arange(2),
        LATENT_SHAPE,
        center_scale=1.3,
    )
    weighted_mean = jnp.sum(state["pi"][:, None, None, None] * source_mu, axis=0)
    assert weighted_mean == pytest.approx(jnp.zeros(LATENT_SHAPE), abs=1e-6)


def test_c_zero_keeps_component_specific_covariance_conditioning():
    state = raw_gmm_state()
    component_ids = jnp.asarray([0, 1], dtype=jnp.int32)
    source_mu, source_sigma = centered_component_params_from_ids(
        state,
        component_ids,
        LATENT_SHAPE,
        center_scale=0.0,
    )
    assert source_mu == pytest.approx(jnp.zeros_like(source_mu), abs=1e-7)
    assert not jnp.allclose(source_sigma[0], source_sigma[1])


def test_posterior_pairing_is_unchanged_by_source_centering():
    state = raw_gmm_state()
    x_1 = jnp.asarray([[[[2.1, -0.9]]], [[[5.9, 3.2]]]], dtype=jnp.float32)
    k_original, q_original, log_p_original, mu_original, sigma_original = infer_component_params(state, x_1)
    k_centered, q_centered, log_p_centered, mu_centered, sigma_centered = infer_component_params(
        state,
        x_1,
        center_scale=0.75,
    )
    assert k_centered == pytest.approx(k_original)
    assert q_centered == pytest.approx(q_original, abs=1e-7)
    assert log_p_centered == pytest.approx(log_p_original, abs=1e-7)
    expected_mu = 0.75 * (mu_original - mixture_mean_from_stats(state, LATENT_SHAPE))
    assert mu_centered == pytest.approx(expected_mu, abs=1e-6)
    assert sigma_centered == pytest.approx(sigma_original, abs=1e-7)


def test_centered_target_builder_uses_hard_one_component_source():
    flags = SimpleNamespace(
        dataset_name="dummy_latent",
        model={
            "train_type": "gmm-centered",
            "gmm_source_center_scale": 0.5,
            "class_dropout_prob": 0.0,
            "num_classes": 1,
            "denoise_timesteps": 128,
        },
    )
    x_1 = jnp.asarray(
        [
            [[[0.0, 0.0, 2.1, -0.9]]],
            [[[0.0, 0.0, 5.9, 3.2]]],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.zeros((2,), dtype=jnp.int32)
    outputs = get_targets(flags, jax.random.PRNGKey(7), None, x_1, labels, gmm_state=raw_gmm_state())
    x_t, v_t, _, _, _, info, source_mu, source_sigma = outputs
    assert x_t.shape == v_t.shape == source_mu.shape == source_sigma.shape == (2,) + LATENT_SHAPE
    assert float(info["gmm/source_is_centered"]) == 1.0
    assert float(info["gmm/source_center_scale"]) == pytest.approx(0.5)
    assert not jnp.allclose(source_mu[0], source_mu[1])


def test_grid_contains_requested_scales_and_renders_centered_flag():
    repo_root = Path(__file__).resolve().parents[1]
    grid_path = repo_root / "configs/gmm_centered_source_c_grid.json"
    jobs = load_grid(grid_path)
    assert [job["gmm_source_center_scale"] for job in jobs] == [0.0, 0.5, 0.75, 1.0, 1.3, 1.75]
    assert {job["model_train_type"] for job in jobs} == {"gmm-centered"}

    notebook = make_notebook(jobs[0])
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert '"--model.train_type", str(CONFIG.get("model_train_type", "gmm-tide"))' in source
    assert '"--model.gmm_source_center_scale"' in source
    assert "Skipping router training for model_train_type=" in source

    protocol = json.loads((repo_root / "configs/gmm_centered_source_c_protocol.json").read_text())
    assert protocol["primary_factor"]["values"] == [0.0, 0.5, 0.75, 1.0, 1.3, 1.75]


def test_gmm_tide_shift_scale_grid_renders_all_requested_scales():
    repo_root = Path(__file__).resolve().parents[1]
    grid_path = repo_root / "configs/gmm_tide_moe2_shift_scale_raw_200k_grid.json"
    jobs = load_grid(grid_path)

    assert [job["gmm_source_center_scale"] for job in jobs] == [0.75, 0.875, 1.125, 1.25]
    assert {job["gmm_source_shift_mean"] for job in jobs} == {1}
    assert {job["model_train_type"] for job in jobs} == {"gmm-tide"}
    assert {job["gmm_transform"] for job in jobs} == {"raw"}
    assert {job["train_max_steps"] for job in jobs} == {200000}

    for job in jobs:
        notebook = make_notebook(job)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert '"--model.gmm_source_shift_mean"' in source
        assert '"--model.gmm_source_center_scale"' in source


def test_negative_center_scale_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        centered_component_params_from_ids(
            raw_gmm_state(),
            jnp.asarray([0]),
            LATENT_SHAPE,
            center_scale=-0.1,
        )
