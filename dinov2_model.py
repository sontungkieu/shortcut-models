from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class Mlp(nn.Module):
    hidden_dim: int
    out_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x


class EncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            use_bias=True,
            dtype=self.dtype,
        )(y)
        x = x + y
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = Mlp(
            hidden_dim=int(self.embed_dim * self.mlp_ratio),
            out_dim=self.embed_dim,
            dtype=self.dtype,
        )(y)
        return x + y


class DINOHead(nn.Module):
    out_dim: int
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.bottleneck_dim, dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.out_dim, dtype=jnp.float32)(x)
        return x.astype(jnp.float32)


class DINOv2ViT(nn.Module):
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    out_dim: int = 8192
    patch_out_dim: int = 8192
    dtype: jnp.dtype = jnp.bfloat16

    def _pos_embed(self, patch_grid: Tuple[int, int]):
        base_grid = self.image_size // self.patch_size
        prefix_tokens = 1 + self.num_register_tokens
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, prefix_tokens + base_grid * base_grid, self.embed_dim),
        )
        prefix_pos = pos_embed[:, :prefix_tokens]
        patch_pos = pos_embed[:, prefix_tokens:]
        patch_pos = patch_pos.reshape(1, base_grid, base_grid, self.embed_dim)
        if patch_grid != (base_grid, base_grid):
            patch_pos = jax.image.resize(
                patch_pos,
                (1, patch_grid[0], patch_grid[1], self.embed_dim),
                method="bicubic",
            )
        patch_pos = patch_pos.reshape(1, patch_grid[0] * patch_grid[1], self.embed_dim)
        return jnp.concatenate([prefix_pos, patch_pos], axis=1)

    @nn.compact
    def __call__(
        self,
        images,
        patch_mask: Optional[jnp.ndarray] = None,
        return_patch_logits: bool = True,
    ):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype,
            name="patch_embed",
        )(images)
        batch_size, grid_h, grid_w, _ = x.shape
        x = x.reshape(batch_size, grid_h * grid_w, self.embed_dim)

        if patch_mask is not None:
            mask_token = self.param(
                "mask_token",
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim),
            )
            x = jnp.where(patch_mask[..., None], mask_token.astype(x.dtype), x)

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        cls_token = jnp.broadcast_to(cls_token, (batch_size, 1, self.embed_dim))

        if self.num_register_tokens:
            register_tokens = self.param(
                "register_tokens",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_register_tokens, self.embed_dim),
            )
            register_tokens = jnp.broadcast_to(
                register_tokens,
                (batch_size, self.num_register_tokens, self.embed_dim),
            )
            x = jnp.concatenate([cls_token, register_tokens, x], axis=1)
        else:
            x = jnp.concatenate([cls_token, x], axis=1)

        x = x + self._pos_embed((grid_h, grid_w)).astype(x.dtype)
        for _ in range(self.depth):
            x = EncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dtype=self.dtype,
            )(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        cls = x[:, 0]
        patch_tokens = x[:, 1 + self.num_register_tokens :]
        cls_logits = DINOHead(out_dim=self.out_dim, dtype=self.dtype, name="dino_head")(cls)
        if not return_patch_logits:
            return cls_logits
        patch_logits = DINOHead(
            out_dim=self.patch_out_dim,
            dtype=self.dtype,
            name="ibot_head",
        )(patch_tokens)
        return cls_logits, patch_logits
