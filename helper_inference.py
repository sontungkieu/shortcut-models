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

from baselines.targets_gmm_tide import make_tide_source
from gmm_utils import sample_prior_components

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')


def _ode_time_edges(FLAGS, denoise_timesteps):
    schedule = str(getattr(FLAGS.model, "eval_ode_schedule", "uniform")).strip().lower()
    power = float(getattr(FLAGS.model, "eval_ode_power", 1.0))
    base = np.linspace(0.0, 1.0, int(denoise_timesteps) + 1, dtype=np.float32)
    if schedule in ("", "none", "uniform"):
        edges = base
    elif schedule in ("end_dense", "end-dense", "power_end"):
        edges = np.power(base, max(power, 1e-6))
    elif schedule in ("start_dense", "start-dense", "power_start"):
        edges = 1.0 - np.power(1.0 - base, max(power, 1e-6))
    else:
        raise ValueError(f"Unknown eval_ode_schedule={schedule!r}")
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges.astype(np.float32)


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
    gmm_state=None,
    router_state=None,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        use_tide = FLAGS.model.train_type == 'gmm-tide' and gmm_state is not None and router_state is not None
        use_gmm = FLAGS.model.train_type in ('naive', 'gmm-tide') and gmm_state is not None
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        if use_tide:
            eps, eps_gmm_mu, eps_gmm_sigma, _ = make_tide_source(
                key,
                gmm_state,
                router_state,
                batch_images.shape[0],
                batch_images.shape[1:],
                topk=FLAGS.model.gmm_router_topk,
                temperature=FLAGS.model.gmm_router_temperature,
                gradient_mode=FLAGS.model.gmm_router_gradient_mode,
                gumbel_tau=FLAGS.model.gmm_router_gumbel_tau,
                source_mode=FLAGS.model.gmm_router_source_mode,
            )
        elif use_gmm:
            eps, eps_gmm_mu, eps_gmm_sigma, _ = sample_prior_components(key, gmm_state, batch_images.shape[0], batch_images.shape[1:])
        else:
            eps = jax.random.normal(key, batch_images.shape)
            eps_gmm_mu = None
            eps_gmm_sigma = None

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img
        
        @partial(jax.jit, static_argnames=("use_ema",))
        def call_model(train_state, images, t, dt, labels, gmm_mu=None, gmm_sigma=None, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False, gmm_mu=gmm_mu, gmm_sigma=gmm_sigma)
            return output
        
        if FLAGS.mode == 'interpolate':
            seed = 5
            eps0 = jax.random.normal(jax.random.PRNGKey(seed), batch_images[0].shape)
            eps1 = jax.random.normal(jax.random.PRNGKey(seed+1), batch_images[0].shape)
            labels = jnp.ones(FLAGS.batch_size,).astype(jnp.int32) * 555
            i = jnp.linspace(0, 1, FLAGS.batch_size)
            i_neg = np.sqrt(1-i**2)
            x = eps0[None] * i_neg[:, None, None, None] + eps1[None] * i[:, None, None, None]
            t_vector = jnp.full((FLAGS.batch_size, ), 0)
            dt_vector = jnp.zeros_like(t_vector)
            cfg_scale = FLAGS.inference_cfg_scale
            v = call_model(train_state, x, t_vector, dt_vector, labels)
            x = x + v * 1.0
            x = vae_decode(x) # Image is in [-1, 1] space.
            x_render = np.array(jax.experimental.multihost_utils.process_allgather(x))
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
            breakpoint()

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        x0 = []
        x1 = []
        lab = []
        x_render = []
        activations = []
        images_shape = batch_images.shape
        t_edges = _ode_time_edges(FLAGS, denoise_timesteps)
        print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            if use_tide:
                x, sample_gmm_mu, sample_gmm_sigma, _ = make_tide_source(
                    eps_key,
                    gmm_state,
                    router_state,
                    images_shape[0],
                    images_shape[1:],
                    topk=FLAGS.model.gmm_router_topk,
                    temperature=FLAGS.model.gmm_router_temperature,
                    gradient_mode=FLAGS.model.gmm_router_gradient_mode,
                    gumbel_tau=FLAGS.model.gmm_router_gumbel_tau,
                    source_mode=FLAGS.model.gmm_router_source_mode,
                )
            elif use_gmm:
                x, sample_gmm_mu, sample_gmm_sigma, _ = sample_prior_components(eps_key, gmm_state, images_shape[0], images_shape[1:])
            else:
                x = jax.random.normal(eps_key, images_shape)
                sample_gmm_mu = None
                sample_gmm_sigma = None
            labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            if use_gmm:
                x, labels, sample_gmm_mu, sample_gmm_sigma = shard_data(x, labels, sample_gmm_mu, sample_gmm_sigma)
            else:
                x, labels = shard_data(x, labels)
            x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            for ti in range(denoise_timesteps):
                t = float(t_edges[ti]) # From x_0 (noise) to x_1 (data)
                delta_t = float(t_edges[ti + 1] - t_edges[ti])
                t_vector = jnp.full((images_shape[0], ), t)
                if FLAGS.model.train_type in ('naive', 'gmm-tide'):
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow # Smallest dt.
                else: # shortcut
                    dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                    # print(dt_base)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, x, t_vector, dt_base, labels, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                elif cfg_scale == 0:
                    v = call_model(train_state, x, t_vector, dt_base, labels_uncond, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                else:
                    v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    v_pred_label = call_model(train_state, x, t_vector, dt_base, labels, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                if FLAGS.model.train_type == 'consistency':
                    eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                    x1pred = x + v * (1-t)
                    x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
                else:
                    x = x + v * delta_t # Euler sampling.
            x1.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            lab.append(np.array(jax.experimental.multihost_utils.process_allgather(labels)))
            if FLAGS.model.use_stable_vae:
                x = vae_decode(x) # Image is in [-1, 1] space.
                if num_generations < 10000:
                    x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
            x = jnp.clip(x, -1, 1)
            acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
            acts = jax.experimental.multihost_utils.process_allgather(acts)
            acts = np.array(acts)
            activations.append(acts)
        
        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            print(f"FID is {fid}")
            print(f"FID is {fid}")
            print(f"FID is {fid}")


            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                x_render = np.concatenate(x_render, axis=0)
                np.save(FLAGS.save_dir + f'/x_render.npy', x_render)

                # x0 = np.concatenate(x0, axis=0)
                # x1 = np.concatenate(x1, axis=0)
                # lab = np.concatenate(lab, axis=0)
                # os.makedirs(FLAGS.save_dir, exist_ok=True)
                # np.save(FLAGS.save_dir + f'/x0.npy', x0)
                # np.save(FLAGS.save_dir + f'/x1.npy', x1)
                # np.save(FLAGS.save_dir + f'/lab.npy', lab)
