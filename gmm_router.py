import os
import pickle
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class GMMRouter(nn.Module):
    num_modes: int
    hidden_channels: int = 128
    mlp_hidden_size: int = 256
    depth: int = 3
    dropout_rate: float = 0.0
    norm_type: str = "none"
    dtype: object = jnp.bfloat16

    def _maybe_norm(self, h, name: str):
        norm_type = str(self.norm_type).lower().replace("-", "_")
        if norm_type in ("none", "", "off"):
            return h
        if norm_type in ("layernorm", "layer_norm", "ln"):
            return nn.LayerNorm(dtype=self.dtype, name=name)(h)
        if norm_type in ("groupnorm", "group_norm", "gn"):
            return self._group_norm(h, name=name)
        raise ValueError(f"Unsupported router norm_type {self.norm_type!r}")

    def _group_norm(self, h, name: str, eps: float = 1e-6):
        channels = int(h.shape[-1])
        max_groups = min(32, channels)
        groups = max(group for group in range(max_groups, 0, -1) if channels % group == 0)
        x = h.astype(jnp.float32)
        x_grouped = jnp.reshape(x, x.shape[:-1] + (groups, channels // groups))
        reduction_axes = tuple(range(1, x_grouped.ndim - 2)) + (x_grouped.ndim - 1,)
        mean = jnp.mean(x_grouped, axis=reduction_axes, keepdims=True)
        var = jnp.mean(jnp.square(x_grouped - mean), axis=reduction_axes, keepdims=True)
        y = jnp.reshape((x_grouped - mean) * jax.lax.rsqrt(var + eps), x.shape)
        scale = self.param(f"{name}_scale", nn.initializers.ones, (channels,), jnp.float32)
        bias = self.param(f"{name}_bias", nn.initializers.zeros, (channels,), jnp.float32)
        return (y * scale + bias).astype(self.dtype)

    def _maybe_dropout(self, h, train: bool):
        if float(self.dropout_rate) <= 0.0:
            return h
        return nn.Dropout(rate=float(self.dropout_rate))(h, deterministic=not train)

    @nn.compact
    def __call__(self, x, train: bool = False, return_activations: bool = False):
        activations = {}
        h = x.astype(self.dtype)
        channels = int(self.hidden_channels)
        for i in range(int(self.depth)):
            h = nn.Conv(
                features=channels,
                kernel_size=(3, 3),
                strides=(1, 1) if i == 0 else (2, 2),
                padding="SAME",
                dtype=self.dtype,
                name=f"conv_{i}",
            )(h)
            h = self._maybe_norm(h, name=f"norm_conv_{i}")
            h = nn.silu(h)
            h = self._maybe_dropout(h, train=train)
            activations[f"conv_{i}"] = h
            channels = min(channels * 2, int(self.hidden_channels) * 4)

        h = jnp.mean(h, axis=(1, 2))
        activations["pooled"] = h
        h = self._maybe_norm(h, name="norm_pooled")
        h = nn.Dense(int(self.mlp_hidden_size), dtype=self.dtype, name="mlp_0")(h)
        h = self._maybe_norm(h, name="norm_mlp_0")
        h = nn.silu(h)
        h = self._maybe_dropout(h, train=train)
        activations["mlp_hidden"] = h
        logits = nn.Dense(int(self.num_modes), dtype=jnp.float32, name="logits")(h.astype(jnp.float32))
        activations["logits"] = logits
        if return_activations:
            return logits, activations
        return logits


def router_metrics(logits, q_target, eps: float = 1e-8) -> Dict[str, jnp.ndarray]:
    q_target = jax.lax.stop_gradient(q_target)
    q_pred = jax.nn.softmax(logits, axis=-1)
    log_pred = jax.nn.log_softmax(logits, axis=-1)
    q_safe = jnp.maximum(q_target, eps)
    target_entropy = -jnp.sum(q_safe * jnp.log(q_safe), axis=-1)
    cross_entropy = -jnp.sum(q_target * log_pred, axis=-1)
    kl = cross_entropy - target_entropy

    pred_ids = jnp.argmax(q_pred, axis=-1)
    target_ids = jnp.argmax(q_target, axis=-1)
    counts = jnp.bincount(pred_ids, length=q_pred.shape[-1])
    usage = counts / jnp.maximum(logits.shape[0], 1)
    usage_safe = jnp.maximum(usage, eps)
    usage_entropy = -jnp.sum(usage_safe * jnp.log(usage_safe))
    usage_entropy_norm = usage_entropy / jnp.log(jnp.asarray(q_pred.shape[-1], dtype=jnp.float32))

    return {
        "router/kl_to_gmm": jnp.mean(kl),
        "router/cross_entropy": jnp.mean(cross_entropy),
        "router/target_entropy": jnp.mean(target_entropy),
        "router/top1_agreement": jnp.mean(pred_ids == target_ids),
        "router/top1_prob_mean": jnp.mean(jnp.max(q_pred, axis=-1)),
        "router/target_top1_prob_mean": jnp.mean(jnp.max(q_target, axis=-1)),
        "router/usage_entropy": usage_entropy,
        "router/usage_entropy_normalized": usage_entropy_norm,
        "router/assign_max_frac": jnp.max(usage),
        "router/num_unique_clusters": jnp.sum(counts > 0),
    }


def save_router_checkpoint(path: str, params, config: Dict[str, object]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params_np = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), params)
    payload = {
        "params": params_np,
        "config": dict(config),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_router_checkpoint(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["params"] = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["params"])
    return payload


def load_router_state(path: str) -> Dict[str, object]:
    payload = load_router_checkpoint(path)
    config = payload["config"]
    model_def = GMMRouter(
        num_modes=int(config["num_modes"]),
        hidden_channels=int(config.get("hidden_channels", 128)),
        mlp_hidden_size=int(config.get("mlp_hidden_size", 256)),
        depth=int(config.get("depth", 3)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
        norm_type=str(config.get("norm_type", "none")),
    )
    return {
        "model_def": model_def,
        "params": payload["params"],
        "config": config,
    }
