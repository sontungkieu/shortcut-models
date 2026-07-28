
from functools import partial, cached_property

import jax
from diffusers import FlaxAutoencoderKL
from einops import rearrange
from flax import struct

from jaxtyping import Array, PyTree, Key, Float, Shaped, Int, UInt8, jaxtyped
from typeguard import typechecked
from functools import partial
typecheck = partial(jaxtyped, typechecker=typechecked)

@struct.dataclass
class StableVAE:
    params: PyTree[Float[Array, "..."]]
    module: FlaxAutoencoderKL = struct.field(pytree_node=False)

    @classmethod
    def create(cls) -> "VAE":
        # module, params = FlaxAutoencoderKL.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae"
        # )
        module, params = FlaxAutoencoderKL.from_pretrained(
            "pcuenq/sd-vae-ft-mse-flax"
        )
        params = jax.device_get(params)
        return cls(
            params=params,
            module=module,
        )

    @partial(jax.jit, static_argnames="scale")
    def encode(
        self, key: Key[Array, ""], images: Float[Array, "b h w 3"], scale: bool = True
    ) -> Float[Array, "b lh lw 4"]:
        images = rearrange(images, "b h w c -> b c h w")
        latents = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist.sample(key)
        if scale:
            latents *= self.module.config.scaling_factor
        return latents

    @partial(jax.jit, static_argnames="scale")
    def encode_posterior_moments(
        self, images: Float[Array, "b h w 3"], scale: bool = True
    ):
        """Return the diagonal VAE posterior moments without changing sampling."""
        images = rearrange(images, "b h w c -> b c h w")
        latent_dist = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist
        mean = latent_dist.mean
        var = latent_dist.var
        if scale:
            scaling_factor = self.module.config.scaling_factor
            mean *= scaling_factor
            var *= scaling_factor**2
        return mean, var

    @partial(jax.jit, static_argnames="scale")
    def decode(
        self, latents: Float[Array, "b lh lw 4"], scale: bool = True
    ) -> Float[Array, "b h w 3"]:
        if scale:
            latents /= self.module.config.scaling_factor
        images = self.module.apply(
            {"params": self.params}, latents, method=self.module.decode
        ).sample
        # convert to channels-last
        images = rearrange(images, "b c h w -> b h w c")
        return images

    @cached_property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.module.block_out_channels) - 1)
