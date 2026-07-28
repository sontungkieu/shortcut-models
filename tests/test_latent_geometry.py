from __future__ import annotations

import numpy as np
import pytest

from latent_geometry import (
    binary_auc,
    diagonal_gaussian_nll,
    deterministic_train_test_split,
    fit_whitening,
    knn_precision_recall,
    local_pca_dimensions,
    ppca_nll,
    sample_gmm,
    split_half_covariance_baseline,
    whitening_diagnostics,
)


def test_deterministic_split_is_disjoint_and_reproducible():
    train_a, test_a = deterministic_train_test_split(100, 0.8, 7)
    train_b, test_b = deterministic_train_test_split(100, 0.8, 7)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert len(train_a) == 80
    assert len(test_a) == 20
    assert not set(train_a).intersection(test_a)


def test_split_half_covariance_is_small_for_large_iid_sample():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(4000, 4)).astype(np.float32)
    summary, rows = split_half_covariance_baseline(samples, repeats=3, seed=5)
    assert len(rows) == 3
    assert summary["relative_error_mean"] < 0.15
    assert summary["definition"] == "||Sigma_A-Sigma_B||_F/||Sigma_A||_F"


def test_whitening_and_ppca_use_train_covariance_only():
    rng = np.random.default_rng(11)
    transform = np.asarray(
        [[2.0, 0.3, 0.0], [0.0, 1.2, 0.2], [0.0, 0.0, 0.5]]
    )
    samples = (rng.normal(size=(600, 3)) @ transform.T).astype(np.float32)
    train, test = deterministic_train_test_split(600, 0.8, 13)
    whitening = fit_whitening(samples, train, 1e-8)
    assert whitening["floor_hit_count"] == 0
    train_nll = ppca_nll(
        samples[train],
        whitening["mean"],
        whitening["eigenvectors"],
        whitening["eigenvalues"],
        rank=2,
    )
    test_nll = ppca_nll(
        samples[test],
        whitening["mean"],
        whitening["eigenvectors"],
        whitening["eigenvalues"],
        rank=2,
    )
    assert np.isfinite(train_nll)
    assert np.isfinite(test_nll)
    assert abs(test_nll - train_nll) < 1.0


def test_whitening_reports_finite_sample_gaussian_reference(tmp_path):
    rng = np.random.default_rng(41)
    samples = rng.normal(size=(1200, 8)).astype(np.float32)
    train, test = deterministic_train_test_split(1200, 0.8, 43)
    whitening = fit_whitening(samples, train, 1e-8)
    summary, rows = whitening_diagnostics(
        samples,
        test,
        whitening,
        projection_count=8,
        seed=47,
        output_dir=tmp_path,
    )
    expected_mean = 8 * (960 + 1) / (960 - 8 - 2)
    assert summary["mahalanobis_finite_sample_reference_mean"] == pytest.approx(
        expected_mean
    )
    assert summary["mahalanobis_radius_sq_mean_to_reference_ratio"] == pytest.approx(
        1.0, abs=0.2
    )
    assert len(rows) == 8
    assert (tmp_path / "whitened_mahalanobis_finite_sample_qq.png").is_file()


def test_diagonal_gaussian_nll_matches_standard_normal_expectation():
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(20000, 2))
    nll = diagonal_gaussian_nll(samples, np.zeros(2), np.ones(2))
    expected = np.log(2.0 * np.pi) + 1.0
    assert nll == pytest.approx(expected, abs=0.03)


def test_local_pca_recovers_a_line(tmp_path):
    rng = np.random.default_rng(19)
    coordinate = np.linspace(-2.0, 2.0, 300)
    samples = np.stack(
        [
            coordinate,
            1e-4 * rng.normal(size=coordinate.size),
            1e-4 * rng.normal(size=coordinate.size),
        ],
        axis=1,
    ).astype(np.float32)
    summary, rows = local_pca_dimensions(
        samples,
        pool_size=250,
        query_count=40,
        neighbor_counts=(10, 20),
        variance_fraction=0.9,
        seed=23,
        output_dir=tmp_path,
    )
    assert len(rows) == 80
    assert summary["k10_median"] == pytest.approx(1.0)
    assert summary["k20_median"] == pytest.approx(1.0)
    assert (tmp_path / "local_pca_dimension_histogram.png").is_file()


def test_gmm_sampling_applies_saved_standardization_inverse():
    state = {
        "pi": np.asarray([1.0]),
        "mu": np.asarray([[0.0, 0.0]]),
        "var": np.asarray([[1e-12, 1e-12]]),
        "mean": np.asarray([10.0, -3.0]),
        "std": np.asarray([2.0, 4.0]),
        "transform_type": "standardize",
    }
    samples = sample_gmm(state, sample_count=8, seed=29)
    assert np.mean(samples, axis=0) == pytest.approx([10.0, -3.0], abs=1e-4)


def test_auc_and_knn_precision_recall_sanity():
    labels = np.asarray([0, 0, 1, 1])
    assert binary_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(31)
    samples = rng.normal(size=(128, 4)).astype(np.float32)
    metrics = knn_precision_recall(
        samples,
        samples.copy(),
        subset_size=128,
        k=3,
        seed=37,
    )
    assert metrics["precision"] > 0.8
    assert metrics["recall"] > 0.8
