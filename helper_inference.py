import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from functools import partial
from absl import app, flags
from utils.sit_transport import get_transport_dt_base, sit_sample

flags.DEFINE_integer('inference_timesteps', 128,
                     'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096,
                     'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')
flags.DEFINE_enum('inference_transport', 'ode', ['ode', 'sde'],
                  'Transport sampler to use for SiT inference.')
flags.DEFINE_enum('inference_sampling_method', 'heun', ['euler', 'heun'],
                  'Fixed-step solver used by SiT inference.')
flags.DEFINE_enum('inference_diffusion_form', 'sigma',
                  ['constant', 'sbdm', 'sigma', 'linear', 'decreasing', 'increasing-decreasing'],
                  'Diffusion coefficient form used by the SiT SDE sampler.')
flags.DEFINE_float('inference_diffusion_norm', 1.0,
                   'Diffusion coefficient multiplier for the SiT SDE sampler.')
flags.DEFINE_enum('inference_last_step', 'mean', ['none', 'mean', 'tweedie', 'euler'],
                  'Optional last-step correction for the SiT SDE sampler.')
flags.DEFINE_float('inference_last_step_size', 0.04,
                   'Step size for the optional SiT SDE last-step correction.')


def do_inference(
    FLAGS,
    train_state,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    with jax.spmd_mode('allow_all'):
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(
            batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(
            batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])  # Null token
        should_compute_fid = get_fid_activations is not None and truth_fid_stats is not None and fid_from_stats is not None

        def process_img(img):
            # Nhận (N,H,W,C) hoặc (H,W,C). Chỉ cắt batch nếu THỰC SỰ có batch.
            if img.ndim == 4:
                img = img[0]                  # (H,W,C)
            elif img.ndim != 3:
                raise ValueError(
                    f"Unexpected image ndim={img.ndim}, expected 3 or 4")

            # Nếu grayscale với C=1 thì chuyển về (H,W) cho imshow.
            if img.shape[-1] == 1:
                img = img[..., 0]             # (H,W)

            # Nếu dùng Stable-VAE latent -> decode sau khi chuẩn hoá shape
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]  # -> (H,W,3)

            # Chuẩn hoá về [0,1]
            img = jnp.clip(img * 0.5 + 0.5, 0, 1)

            # Trả về NumPy host array cho matplotlib
            return np.array(img)

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False)
            return output

        def sit_model_output(x, t_vector, labels_cond, cfg_scale):
            dt_base = get_transport_dt_base(FLAGS.model['denoise_timesteps'], t_vector.shape[0])
            t_vector, dt_base = shard_data(t_vector, dt_base)
            if cfg_scale == 1:
                return call_model(train_state, x, t_vector, dt_base, labels_cond)
            if cfg_scale == 0:
                return call_model(train_state, x, t_vector, dt_base, labels_uncond)
            v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
            v_cond = call_model(train_state, x, t_vector, dt_base, labels_cond)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        def sample_batch(x, labels, num_steps, cfg_scale, sample_key):
            if FLAGS.model.train_type == 'sit':
                def model_fn(current_x, t_vector):
                    return sit_model_output(current_x, t_vector, labels, cfg_scale)

                return sit_sample(
                    rng=sample_key,
                    x=x,
                    model_fn=model_fn,
                    num_steps=num_steps,
                    path_type=FLAGS.model['transport_path_type'],
                    prediction=FLAGS.model['transport_prediction'],
                    train_eps=FLAGS.model['transport_train_eps'],
                    sample_eps=FLAGS.model['transport_sample_eps'],
                    transport_type=FLAGS.inference_transport,
                    sampling_method=FLAGS.inference_sampling_method,
                    diffusion_form=FLAGS.inference_diffusion_form,
                    diffusion_norm=FLAGS.inference_diffusion_norm,
                    last_step=FLAGS.inference_last_step,
                    last_step_size=FLAGS.inference_last_step_size,
                )

            delta_t = 1.0 / num_steps
            current_x = x
            for ti in range(num_steps):
                t = ti / num_steps  # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((current_x.shape[0],), t)
                if FLAGS.model.train_type == 'naive':
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                    dt_base = jnp.ones(current_x.shape[0], dtype=jnp.int32) * dt_flow
                else:
                    dt_flow = np.log2(num_steps).astype(jnp.int32)
                    dt_base = jnp.ones(current_x.shape[0], dtype=jnp.int32) * dt_flow
                    if FLAGS.model.train_type == 'livereflow' and num_steps < 128:
                        dt_base = jnp.zeros_like(dt_base)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, current_x, t_vector, dt_base, labels)
                elif cfg_scale == 0:
                    v = call_model(train_state, current_x, t_vector, dt_base, labels_uncond)
                else:
                    v_pred_uncond = call_model(train_state, current_x, t_vector, dt_base, labels_uncond)
                    v_pred_label = call_model(train_state, current_x, t_vector, dt_base, labels)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                if FLAGS.model.train_type == 'consistency':
                    eps_step = shard_data(jax.random.normal(jax.random.fold_in(sample_key, ti), current_x.shape))
                    x1pred = current_x + v * (1 - t)
                    current_x = x1pred * (t + delta_t) + eps_step * (1 - t - delta_t)
                else:
                    current_x = current_x + v * delta_t
            return current_x

        if FLAGS.mode == 'interpolate':
            seed = 5
            eps0 = jax.random.normal(
                jax.random.PRNGKey(seed), batch_images[0].shape)
            eps1 = jax.random.normal(
                jax.random.PRNGKey(seed+1), batch_images[0].shape)
            labels = jnp.ones(FLAGS.batch_size,).astype(jnp.int32) * 555
            i = jnp.linspace(0, 1, FLAGS.batch_size)
            i_neg = np.sqrt(1-i**2)
            x = eps0[None] * i_neg[:, None, None, None] + \
                eps1[None] * i[:, None, None, None]
            t_vector = jnp.full((FLAGS.batch_size, ), 0)
            dt_vector = jnp.zeros_like(t_vector)
            cfg_scale = FLAGS.inference_cfg_scale
            v = call_model(train_state, x, t_vector, dt_vector, labels)
            x = x + v * 1.0
            x = vae_decode(x)  # Image is in [-1, 1] space.
            x_render = np.array(
                jax.experimental.multihost_utils.process_allgather(x))
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
            breakpoint()

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        x_render = []
        images_shape = batch_images.shape
        print(
            f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        activations = []
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x = jax.random.normal(eps_key, images_shape)
            labels = jax.random.randint(
                label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            x, labels = shard_data(x, labels)
            x = sample_batch(x, labels, denoise_timesteps, cfg_scale, eps_key)
            if FLAGS.model.use_stable_vae:
                x = vae_decode(x)  # Image is in [-1, 1] space.
                if num_generations < 10000:
                    x_render.append(
                        np.array(jax.experimental.multihost_utils.process_allgather(x)))
            if should_compute_fid:
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                activations.append(np.array(acts))

        if jax.process_index() == 0:
            if should_compute_fid:
                activations = np.concatenate(activations, axis=0)
                activations = activations.reshape((-1, activations.shape[-1]))
                mu1 = np.mean(activations, axis=0)
                sigma1 = np.cov(activations, rowvar=False)
                fid = fid_from_stats(
                    mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                print(f"FID is {fid}")
                print(f"FID is {fid}")
                print(f"FID is {fid}")
            else:
                print("FID skipped because --fid_stats was not provided.")

            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                if x_render:
                    x_render = np.concatenate(x_render, axis=0)
                    np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
