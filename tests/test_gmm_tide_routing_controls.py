from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from baselines.targets_gmm_tide import make_tide_source
from gmm_utils import mixture_mean_from_stats


class DummyRouter:
    def apply(self, variables, x, train=False):
        del variables, train
        values = jnp.reshape(x, (x.shape[0], -1))[:, 0]
        return jnp.stack((-values, values), axis=-1)


def gmm_state():
    return {
        "pi": jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        "mu": jnp.asarray([[-2.0], [2.0]], dtype=jnp.float32),
        "var": jnp.asarray([[0.25], [0.25]], dtype=jnp.float32),
        "transform_type": "raw",
    }


def router_state():
    return {"model_def": DummyRouter(), "params": {}}


def make(policy: str):
    return make_tide_source(
        jax.random.PRNGKey(17),
        gmm_state(),
        router_state(),
        batch_size=64,
        latent_shape=(1, 1, 1),
        topk=1,
        source_mode="weighted",
        routing_policy=policy,
    )


def test_gmm_oracle_routes_match_gmm_posterior():
    _, _, _, info = make("gmm_oracle")
    assert float(info["router/routing_policy_is_gmm_oracle"]) == 1.0
    assert float(info["router/route_kl_to_gmm_base"]) == pytest.approx(0.0, abs=1e-7)
    assert float(info["router/route_top1_agreement_to_gmm_base"]) == pytest.approx(1.0)


def test_matched_random_preserves_batch_usage_and_entropy():
    _, _, _, router_info = make("router")
    _, _, _, random_info = make("matched_random")
    assert float(random_info["router/routing_policy_is_matched_random"]) == 1.0
    assert float(random_info["router/assign_entropy"]) == pytest.approx(
        float(router_info["router/assign_entropy"]), abs=1e-7
    )
    assert float(random_info["router/soft_usage_entropy"]) == pytest.approx(
        float(router_info["router/soft_usage_entropy"]), abs=1e-7
    )
    assert float(random_info["router/route_top1_agreement_to_gmm_base"]) < 1.0


def test_non_router_controls_reject_gradient_relaxations():
    with pytest.raises(ValueError, match="require gradient_mode=topk"):
        make_tide_source(
            jax.random.PRNGKey(17),
            gmm_state(),
            router_state(),
            batch_size=8,
            latent_shape=(1, 1, 1),
            topk=1,
            gradient_mode="gumbel_st",
            routing_policy="gmm_oracle",
        )


def test_shift_mean_preserves_router_and_only_translates_final_source():
    state = {
        "pi": jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        "mu": jnp.asarray([[-1.0], [3.0]], dtype=jnp.float32),
        "var": jnp.asarray([[0.25], [0.25]], dtype=jnp.float32),
        "transform_type": "raw",
    }
    kwargs = {
        "key": jax.random.PRNGKey(23),
        "gmm_state": state,
        "router_state": router_state(),
        "batch_size": 64,
        "latent_shape": (1, 1, 1),
        "topk": 2,
        "source_mode": "weighted",
        "routing_policy": "router",
    }
    x_raw, mu_raw, sigma_raw, info_raw = make_tide_source(**kwargs)
    x_shift, mu_shift, sigma_shift, info_shift = make_tide_source(
        **kwargs,
        shift_mixture_mean=True,
    )
    mixture_mean = mixture_mean_from_stats(state, (1, 1, 1))

    assert x_shift == pytest.approx(x_raw - mixture_mean, abs=1e-6)
    assert mu_shift == pytest.approx(mu_raw - mixture_mean, abs=1e-6)
    assert sigma_shift == pytest.approx(sigma_raw, abs=1e-7)
    assert info_shift["router/route_top1_agreement_to_gmm_base"] == pytest.approx(
        info_raw["router/route_top1_agreement_to_gmm_base"],
        abs=1e-7,
    )
    assert float(info_raw["tide/source_shift_mixture_mean"]) == 0.0
    assert float(info_shift["tide/source_shift_mixture_mean"]) == 1.0


@pytest.mark.parametrize("center_scale", [0.75, 0.875, 1.0, 1.125, 1.25])
def test_shifted_center_scale_preserves_residual_sigma_and_router(center_scale):
    state = {
        "pi": jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        "mu": jnp.asarray([[-2.0], [4.0]], dtype=jnp.float32),
        "var": jnp.asarray([[0.25], [1.0]], dtype=jnp.float32),
        "transform_type": "raw",
    }
    kwargs = {
        "key": jax.random.PRNGKey(29),
        "gmm_state": state,
        "router_state": router_state(),
        "batch_size": 64,
        "latent_shape": (1, 1, 1),
        "topk": 2,
        "source_mode": "weighted",
        "routing_policy": "router",
    }
    x_base, mu_base, sigma_base, info_base = make_tide_source(
        **kwargs,
        shift_mixture_mean=True,
        center_scale=1.0,
    )
    x_scaled, mu_scaled, sigma_scaled, info_scaled = make_tide_source(
        **kwargs,
        shift_mixture_mean=True,
        center_scale=center_scale,
    )

    assert mu_scaled == pytest.approx(center_scale * mu_base, abs=1e-6)
    assert x_scaled - mu_scaled == pytest.approx(x_base - mu_base, abs=1e-6)
    assert sigma_scaled == pytest.approx(sigma_base, abs=1e-7)
    for metric in (
        "router/topk_mass",
        "router/route_entropy",
        "router/route_top1_agreement_to_gmm_base",
        "router/route_kl_to_gmm_base",
    ):
        assert info_scaled[metric] == pytest.approx(info_base[metric], abs=1e-7)
    assert float(info_scaled["tide/source_shift_mixture_mean"]) == 1.0
    assert float(info_scaled["tide/source_center_scale"]) == pytest.approx(center_scale)


def test_center_scale_one_is_exact_shift_only_behavior():
    kwargs = {
        "key": jax.random.PRNGKey(31),
        "gmm_state": gmm_state(),
        "router_state": router_state(),
        "batch_size": 16,
        "latent_shape": (1, 1, 1),
        "topk": 2,
        "shift_mixture_mean": True,
    }
    default = make_tide_source(**kwargs)
    explicit = make_tide_source(**kwargs, center_scale=1.0)
    for default_value, explicit_value in zip(default[:3], explicit[:3]):
        assert jnp.array_equal(default_value, explicit_value)


@pytest.mark.parametrize("center_scale", [-0.1, float("nan"), float("inf")])
def test_invalid_tide_center_scale_is_rejected(center_scale):
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_tide_source(
            jax.random.PRNGKey(37),
            gmm_state(),
            router_state(),
            batch_size=4,
            latent_shape=(1, 1, 1),
            topk=1,
            center_scale=center_scale,
        )
