from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from baselines.targets_gmm_tide import make_tide_source


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
