from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_population import (
    PosteriorMoments,
    compare_gmm_to_population,
    gmm_component_moments_in_latent_space,
    posterior_population_summary,
)
from utils.datasets import get_dataset_for_statistics
from utils.stable_vae import StableVAE


def test_streaming_aggregated_posterior_matches_closed_form():
    posterior_mean = np.asarray(
        [[1.0, 2.0], [3.0, 0.0], [-1.0, 4.0]], dtype=np.float64
    )
    posterior_var = np.asarray(
        [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]], dtype=np.float64
    )
    state = PosteriorMoments.empty(2)
    state.update(posterior_mean[:2], posterior_var[:2])
    state.update(posterior_mean[2:], posterior_var[2:])
    moments = state.finalize()

    expected_mean = np.mean(posterior_mean, axis=0)
    centered = posterior_mean - expected_mean
    expected_between = centered.T @ centered / posterior_mean.shape[0]
    expected_noise = np.mean(posterior_var, axis=0)
    expected_total = expected_between + np.diag(expected_noise)

    assert moments["mean"] == pytest.approx(expected_mean)
    assert moments["between_covariance"] == pytest.approx(expected_between)
    assert moments["posterior_noise_variance"] == pytest.approx(expected_noise)
    assert moments["aggregated_covariance"] == pytest.approx(expected_total)


def test_population_summary_reports_trace_decomposition_and_effective_dimension():
    state = PosteriorMoments.empty(2)
    state.update(
        np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
        np.asarray([[0.0, 0.25], [0.0, 0.25]]),
    )
    moments = state.finalize()
    metrics, _ = posterior_population_summary(
        moments,
        np.asarray([1.0, 1.0]),
        mean_epsilon=1e-3,
        dead_variance_threshold=1e-8,
    )
    assert metrics["trace_decomposition_abs_error"] == pytest.approx(0.0)
    assert metrics["posterior_noise_trace_fraction"] == pytest.approx(0.2)
    assert metrics["components_99pct"] == 2


def test_posterior_mean_mode_excludes_posterior_noise_from_covariance():
    state = PosteriorMoments.empty(2)
    state.update(
        np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
        np.asarray([[0.0, 4.0], [0.0, 4.0]]),
    )
    moments = state.finalize()
    metrics, eigenvalues = posterior_population_summary(
        moments,
        np.asarray([1.0, 1.0]),
        mean_epsilon=1e-3,
        dead_variance_threshold=1e-8,
        population_mode="posterior_mean",
    )
    assert metrics["population_mode"] == "posterior_mean"
    assert metrics["selected_covariance_trace"] == pytest.approx(1.0)
    assert metrics["covariance_trace"] == pytest.approx(1.0)
    assert np.sort(eigenvalues) == pytest.approx([0.0, 1.0])
    assert metrics["components_99pct"] == 1


def test_raw_gmm_global_moments_include_within_and_between_covariance():
    state = {
        "pi": np.asarray([0.25, 0.75]),
        "mu": np.asarray([[0.0, 0.0], [2.0, 4.0]]),
        "var": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "transform_type": "raw",
        "latent_shape": np.asarray([1, 1, 2]),
    }
    moments = gmm_component_moments_in_latent_space(state)
    expected_mean = np.asarray([1.5, 3.0])
    centered = state["mu"] - expected_mean
    expected = np.diag(np.sum(state["pi"][:, None] * state["var"], axis=0))
    expected += centered.T @ (centered * state["pi"][:, None])
    assert moments["mixture_mean"] == pytest.approx(expected_mean)
    assert moments["mixture_covariance"] == pytest.approx(expected)


def test_standardized_gmm_is_unscaled_before_moment_comparison():
    state = {
        "pi": np.asarray([1.0]),
        "mu": np.asarray([[1.0, -2.0]]),
        "var": np.asarray([[0.5, 2.0]]),
        "mean": np.asarray([10.0, 20.0]),
        "std": np.asarray([2.0, 3.0]),
        "transform_type": "standardize",
        "latent_shape": np.asarray([1, 1, 2]),
    }
    moments = gmm_component_moments_in_latent_space(state)
    assert moments["mixture_mean"] == pytest.approx([[12.0, 14.0]][0])
    assert np.diag(moments["mixture_covariance"]) == pytest.approx([2.0, 18.0])


def test_channel_whiten_gmm_preserves_exact_cross_channel_covariance():
    unwhiten = np.asarray([[2.0, 0.5], [0.0, 1.5]])
    state = {
        "pi": np.asarray([1.0]),
        "mu": np.asarray([[1.0, -1.0]]),
        "var": np.asarray([[0.5, 2.0]]),
        "channel_mean": np.asarray([3.0, -2.0]),
        "channel_unwhiten": unwhiten,
        "transform_type": "channel_whiten",
        "latent_shape": np.asarray([1, 1, 2]),
    }
    moments = gmm_component_moments_in_latent_space(state)
    expected_mean = state["mu"] @ unwhiten.T + state["channel_mean"]
    expected_covariance = unwhiten @ np.diag(state["var"][0]) @ unwhiten.T
    assert moments["mixture_mean"] == pytest.approx(expected_mean[0])
    assert moments["mixture_covariance"] == pytest.approx(expected_covariance)
    assert moments["component_covariance_blocks"][0, 0] == pytest.approx(
        expected_covariance
    )


def test_gmm_comparison_detects_exact_population_match():
    state = {
        "pi": np.asarray([1.0]),
        "mu": np.asarray([[1.0, -1.0]]),
        "var": np.asarray([[2.0, 3.0]]),
        "transform_type": "raw",
        "latent_shape": np.asarray([1, 1, 2]),
    }
    moments = gmm_component_moments_in_latent_space(state)
    metrics, _ = compare_gmm_to_population(
        moments["mixture_mean"],
        moments["mixture_covariance"],
        moments,
        mean_epsilon=1e-3,
        dead_variance_threshold=1e-8,
    )
    assert metrics["mean_gap_rms"] == pytest.approx(0.0)
    assert metrics["covariance_relative_frobenius_error"] == pytest.approx(0.0)
    assert metrics["covariance_trace_ratio"] == pytest.approx(1.0)


def test_analytics_api_does_not_replace_training_sampler_or_augment_dataset():
    encode_source = inspect.getsource(StableVAE.encode)
    analytics_source = inspect.getsource(StableVAE.encode_posterior_moments)
    dataset_source = inspect.getsource(get_dataset_for_statistics)
    assert ".latent_dist.sample(key)" in encode_source
    assert "latent_dist.mean" in analytics_source
    assert "latent_dist.var" in analytics_source
    assert "random_flip" not in dataset_source
    assert ".shuffle(" not in dataset_source
    assert ".repeat(" not in dataset_source
