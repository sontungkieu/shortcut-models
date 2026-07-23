from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from scripts.audit_gmm_tide_router_geometry import (
    channel_covariance_metrics,
    posterior_metrics,
    topk_jaccard,
)


def test_topk_jaccard_matches_sets_not_rank_order():
    p = jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    q = jnp.asarray([[0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    values = topk_jaccard(p, q, topk=2)
    assert values.tolist() == pytest.approx([1.0, 1.0 / 3.0])


def test_posterior_metrics_are_exact_for_identical_targets():
    q = jnp.asarray([[0.75, 0.25], [0.2, 0.8]])
    metrics = posterior_metrics("router/x1", q, q)
    assert float(metrics["router/x1/kl_gmm_to_phi"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["router/x1/brier"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["router/x1/top1_agreement"]) == pytest.approx(1.0)


def test_channel_covariance_metrics_report_effective_rank():
    x = jnp.asarray(
        [
            [[[1.0, 0.0], [-1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, -1.0]]],
        ]
    )
    metrics = channel_covariance_metrics("x", x)
    assert metrics["x/channel_cov_trace"] > 0.0
    assert metrics["x/channel_cov_effective_rank"] == pytest.approx(2.0, rel=1e-5)


def test_embedded_audit_script_adds_execution_cwd_to_import_path():
    source = (Path(__file__).resolve().parents[1] / "scripts/audit_gmm_tide_router_geometry.py").read_text()
    assert 'Path(os.environ.get("GMM_TIDE_REPO_ROOT", Path.cwd()))' in source
    assert "sys.path.insert(0, str(REPO_ROOT))" in source
