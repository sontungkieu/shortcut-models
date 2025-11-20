from jax._src.nn.initializers import _compute_fans
from jax._src import dtypes
from jax._src import core
from math_utils import get_2d_sincos_pos_embed, modulate
import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# class ConditionalInstanceNorm2dNHWC(nn.Module):
#     num_channels: int
#     special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
#     eps: float = 1e-5
#     use_affine: bool = True

#     # thêm cấu hình nhỏ cho t-embedding
#     time_embed_dim: int = 256       # giống frequency_embedding_size
#     time_max_period: float = 10000.0
#     mlp_hidden_dim: int = 64
#     t_scale: float = 0.5

#     @nn.compact
#     def __call__(self, x, t):
#         # x: [B,H,W,C], t: [B] (giống DiT.__call__)
#         # 1) Instance norm cơ bản

#         # ========= 0) T-EMBED GIỐNG KIỂU TimestepEmbedder =========
#         def timestep_embedding(t_scalar, dim, max_period):
#             # t_scalar: [B]
#             t_scalar = jax.lax.convert_element_type(t_scalar, jnp.float32)
#             half = dim // 2
#             freqs = jnp.exp(
#                 -math.log(max_period)
#                 * jnp.arange(half, dtype=jnp.float32)
#                 / half
#             )  # [half]
#             args = t_scalar[:, None] * freqs[None, :]  # [B, half]
#             emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
#             return emb  # [B, dim]

#         mean = jnp.mean(x, axis=(1, 2), keepdims=True)
#         var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
#         x_norm = (x - mean) / jnp.sqrt(var + self.eps)

#         # 2) Affine phụ thuộc t: γ_b = 1 + α Δγ_b, β_b = α Δβ_b
#         x_tilde = x_norm
#         if self.use_affine:
#             B = x.shape[0]

#             # Embed t → [B, time_embed_dim]
#             t_emb = timestep_embedding(
#                 t, self.time_embed_dim, self.time_max_period)

#             # MLP nhỏ để lấy Δγ, Δβ
#             h = nn.Dense(
#                 features=self.mlp_hidden_dim,
#                 kernel_init=nn.initializers.normal(0.02),
#             )(t_emb)
#             h = nn.silu(h)
#             h = nn.Dense(
#                 features=2 * self.num_channels,
#                 kernel_init=nn.initializers.normal(0.02),
#             )(h)                          # [B, 2C]

#             delta_gamma, delta_beta = jnp.split(h, 2, axis=-1)  # [B,C] mỗi bên

#             # scale nhỏ để ổn định (α)
#             delta_gamma = self.t_scale * delta_gamma
#             delta_beta = self.t_scale * delta_beta

#             # [B,1,1,C]
#             gamma = 1.0 + delta_gamma.reshape(B, 1, 1, self.num_channels)
#             beta = delta_beta.reshape(B, 1, 1, self.num_channels)  # [B,1,1,C]

#             x_tilde = x_norm * gamma + beta
#         # nếu use_affine = False thì x_tilde = x_norm

#         # 3) Gating theo t đặc biệt (giữ y nguyên idea cũ)
#         special_t = jnp.asarray(self.special_t, dtype=t.dtype)       # [K]
#         mask = (t[:, None] == special_t[None, :]).any(axis=-1)       # [B]
#         # [B,1,1,1]
#         mask = mask[:, None, None, None]

#         # 4) Compute masked MSE difference (so với x_tilde – tức norm + affine)
#         sq_err = (x - x_tilde) ** 2                          # [B,H,W,C]
#         mse_per_sample = sq_err.mean(axis=(1, 2, 3))         # [B]

#         mask_f = mask.astype(jnp.float32)                    # [B,1,1,1]
#         denom = jnp.maximum(mask_f.sum(), 1.0)

#         masked_avg_mse = (mse_per_sample * mask_f.reshape(-1)).sum() / denom
#         avg_mse = jnp.mean(mse_per_sample)
#         norm_percentage = jnp.mean(mask_f)

#         # 5) Chỉ norm tại các t thuộc special_t
#         y = jnp.where(mask, x_tilde, x)
#         return y, masked_avg_mse, avg_mse, norm_percentage


class ConditionalInstanceNorm2dNHWC(nn.Module):
    num_channels: int
    special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
    eps: float = 1e-5
    use_affine: bool = True

    @nn.compact
    def __call__(self, x, t):
        # x: [B,H,W,C], t: [B] (giống DiT.__call__)
        # 1) Instance norm cơ bản
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # 2) Affine learnable (gamma, beta)
        if self.use_affine:
            gamma = self.param(
                "gamma",
                nn.initializers.ones,
                (1, 1, 1, self.num_channels),
            )
            beta = self.param(
                "beta",
                nn.initializers.zeros,
                (1, 1, 1, self.num_channels),
            )
            x_norm = x_norm * gamma + beta

        # 3) Gating theo t đặc biệt
        special_t = jnp.asarray(self.special_t, dtype=t.dtype)       # [K]
        # diff = jnp.abs(t[:, None] - special_t[None, :])              # [B,K]
        # mask = (diff < 1e-6).any(axis=-1)                            # [B]
        mask = (t[:, None] == special_t[None, :]).any(axis=-1)
        # [B,1,1,1]
        mask = mask[:, None, None, None]

        # 4) Compute masked MSE difference
        sq_err = (x - x_norm) ** 2                            # [B,H,W,C]
        mse_per_sample = sq_err.mean(axis=(1, 2, 3))            # [B]
        mask_f = mask.astype(jnp.float32)              # [B]
        denom = jnp.maximum(mask_f.sum(), 1.0)         # tránh chia 0

        masked_avg_mse = (mse_per_sample * mask_f).sum() / denom
        avg_mse = jnp.mean(mse_per_sample)
        norm_percentage = jnp.mean(mask_f)

        # Chỉ norm nếu t thuộc special_t, còn lại giữ nguyên x
        return jnp.where(mask, x_norm, x), masked_avg_mse, avg_mse, norm_percentage

class TimeBatchNorm2dNHWC(nn.Module):
    num_channels: int
    special_t: Sequence[float]
    eps: float = 1e-5
    use_affine: bool = True
    momentum: float = 0.99

    @nn.compact
    def __call__(self, x, t, train: bool):
        # x: [B,H,W,C], t: [B]
        B, H, W, C = x.shape

        special_t = jnp.asarray(self.special_t, dtype=t.dtype)   # [K]
        K = special_t.shape[0]

        # 1) sample nào thuộc special_t?
        is_special = (t[:, None] == special_t[None, :])          # [B,K]
        mask_b = is_special.any(axis=-1)                         # [B]
        mask_bhwc = mask_b[:, None, None, None]                  # [B,1,1,1]

        # 2) EMA stats cho từng t_k
        mean_ema = self.variable(
            "batch_stats",
            "mean",
            lambda rng, shape: jnp.zeros(shape, dtype=x.dtype),
            (K, 1, 1, C),
        )
        var_ema = self.variable(
            "batch_stats",
            "var",
            lambda rng, shape: jnp.ones(shape, dtype=x.dtype),
            (K, 1, 1, C),
        )

        # 3) batch stats per special_t (chỉ dùng khi train=True)
        def stats_for_one_k(mask_b_k):
            # mask_b_k: [B]
            m = mask_b_k[:, None, None, None]                    # [B,1,1,1]
            count = jnp.maximum(m.sum((0, 1, 2)), 1.0)           # scalar
            mean_k = (x * m).sum((0, 1, 2), keepdims=True) / count
            var_k = ((x - mean_k) ** 2 * m).sum((0, 1, 2), keepdims=True) / count
            return mean_k, var_k

        mean_batch_list = []
        var_batch_list = []
        has_sample_list = []
        for k in range(K):
            mask_b_k = is_special[:, k]                          # [B]
            mean_k, var_k = stats_for_one_k(mask_b_k)
            mean_batch_list.append(mean_k)                       # [1,1,1,C]
            var_batch_list.append(var_k)
            has_sample_list.append(mask_b_k.any())

        mean_batch = jnp.concatenate(mean_batch_list, axis=0)    # [K,1,1,C]
        var_batch = jnp.concatenate(var_batch_list, axis=0)      # [K,1,1,C]
        has_sample = jnp.stack(has_sample_list, axis=0)          # [K]
        has_sample_bc = has_sample[:, None, None, None]          # [K,1,1,1]

        if train:
            # update EMA chỉ cho những t_k xuất hiện trong batch
            new_mean = jnp.where(
                has_sample_bc,
                self.momentum * mean_ema.value + (1.0 - self.momentum) * mean_batch,
                mean_ema.value,
            )
            new_var = jnp.where(
                has_sample_bc,
                self.momentum * var_ema.value + (1.0 - self.momentum) * var_batch,
                var_ema.value,
            )
            mean_ema.value = new_mean
            var_ema.value = new_var
            mean_used = mean_batch
            var_used = var_batch
        else:
            # eval: dùng EMA
            mean_used = mean_ema.value
            var_used = var_ema.value

        # 4) map stats theo sample (b) dựa trên t
        weights_bk = is_special.astype(x.dtype)                  # [B,K]
        denom_b = jnp.maximum(weights_bk.sum(axis=1, keepdims=True), 1.0)
        weights_bk = weights_bk / denom_b                        # [B,K]

        w_bkhwc = weights_bk[:, :, None, None, None]             # [B,K,1,1,1]
        mean_b = (w_bkhwc * mean_used[None, ...]).sum(axis=1)    # [B,1,1,C]
        var_b = (w_bkhwc * var_used[None, ...]).sum(axis=1)      # [B,1,1,C]

        x_norm = (x - mean_b) / jnp.sqrt(var_b + self.eps)

        # 5) Affine
        if self.use_affine:
            gamma = self.param(
                "gamma",
                nn.initializers.ones,
                (1, 1, 1, self.num_channels),
            )
            beta = self.param(
                "beta",
                nn.initializers.zeros,
                (1, 1, 1, self.num_channels),
            )
            x_norm = x_norm * gamma + beta

        # 6) chỉ norm nếu t thuộc special_t
        x_out = jnp.where(mask_bhwc, x_norm, x)

        # 7) metrics giống class cũ
        sq_err = (x - x_norm) ** 2
        mse_per_sample = sq_err.mean(axis=(1, 2, 3))
        mask_f = mask_b.astype(jnp.float32)
        denom = jnp.maximum(mask_f.sum(), 1.0)

        masked_avg_mse = (mse_per_sample * mask_f).sum() / denom
        avg_mse = mse_per_sample.mean()
        norm_percentage = mask_f.mean()

        return x_out, masked_avg_mse, avg_mse, norm_percentage
