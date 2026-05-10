import jax
import jax.experimental
import json
import os
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from functools import partial

from baselines.targets_gmm_tide import make_tide_source
from gmm_utils import infer_component_params, json_default, sample_prior_components


def _append_metrics_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default))
        f.write('\n')


def _parse_eval_fid_timesteps(FLAGS):
    values = []
    for token in str(FLAGS.eval_fid_timesteps).split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        values = [1, 4, 32]
    if FLAGS.model.denoise_timesteps == 128 and 128 not in values:
        values.append(128)
    return list(dict.fromkeys(values))

def eval_model(
    FLAGS,
    train_state,
    train_state_teacher,
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
        eval_metrics = {}
        use_tide = FLAGS.model.train_type == 'gmm-tide' and gmm_state is not None and router_state is not None
        use_gmm = FLAGS.model.train_type in ('naive', 'gmm-tide') and gmm_state is not None
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        if 'latent' in FLAGS.dataset_name:
            eps_valid = valid_images[..., :valid_images.shape[-1]//2]
            batch_images = batch_images[..., batch_images.shape[-1]//2:]
            valid_images = valid_images[..., valid_images.shape[-1]//2:]
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
        
        def condition_for_data(latents):
            if not use_gmm:
                return None, None
            _, _, _, gmm_mu, gmm_sigma = infer_component_params(gmm_state, jnp.asarray(latents))
            return jax.device_get(gmm_mu), jax.device_get(gmm_sigma)

        @partial(jax.jit, static_argnames=("use_ema",))
        def call_model(train_state, images, t, dt, labels, gmm_mu=None, gmm_sigma=None, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False, gmm_mu=gmm_mu, gmm_sigma=gmm_sigma)
            return output

        print("Training Loss per T.")
        if FLAGS.model.denoise_timesteps == 128:
            fig, axs = plt.subplots(5, 8, figsize=(15, 12))
            d_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            fig, axs = plt.subplots(3, 6, figsize=(15, 8))
            d_list = [0, 1, 2, 3, 4, 5]
        for d in d_list:
            infos = None
            for t in np.arange(0, 32):
                t = t * (1.0 / 32)

                batch_images_n, batch_labels_n = next(dataset)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images_n = vae_encode(key, batch_images_n)
                batch_images_sharded, batch_labels_sharded = shard_data(batch_images_n, batch_labels_n)
                _, info = update(train_state, train_state_teacher, batch_images_sharded, batch_labels_sharded, force_t=t, force_dt=d)
                info = jax.experimental.multihost_utils.process_allgather(info)
                if infos is None:
                    infos = jax.tree_map(lambda x: [x], info)
                else:
                    infos = jax.tree_map(lambda x, y: y + [x], info, infos)
            time_axis = np.arange(0, 32) / 32
            axs[0, d].plot(time_axis, infos['loss'])
            axs[0, d].set_title(f"All {d}")
            if FLAGS.model['train_type'] == 'shortcut':
                axs[1, d].plot(time_axis, infos['loss_flow'])
                axs[1, d].set_title(f"Flow {d}")
                axs[2, d].plot(time_axis, infos['loss_bootstrap'])
                axs[2, d].set_title(f"Bootstrap {d}")

            if jax.process_index() == 0:
                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)


        print("One-step Denoising at various t.")
        if 'latent' in FLAGS.dataset_name:
            eps = eps_valid
        for dt_type in ['flow', 'shortcut']:
            if len(jax.local_devices()) == 8:
                if dt_type == 'flow':
                    t = jnp.arange(8) / 8 # between 0 and 0.875
                    t = jnp.tile(t, valid_images.shape[0] // 8) # [batch, etc]
                    dt = 0
                    dt_base = jnp.ones_like(t) * np.log2(FLAGS.model.denoise_timesteps)
                elif dt_type == 'shortcut':
                    dt_base = jnp.array([0,0,0,1,2,3,4,5])
                    if FLAGS.model.denoise_timesteps == 128:
                        dt_base = jnp.array([0,1,2,3,4,5,6,7])
                    dt_base = jnp.tile(dt_base, valid_images.shape[0] // 8) # [batch, etc]
                    dt = 2.0 ** (-dt_base)
                    t = 1 - dt
                eps_tile = jnp.repeat(eps, 8, axis=0)[:valid_images.shape[0]]
                valid_images_tile = jnp.repeat(valid_images, 8, axis=0)[:valid_images.shape[0]]
                t_full = t[..., None, None, None]
                x_t = (1 - (1 - 1e-5) * t_full) * eps_tile + t_full * valid_images_tile
                gmm_mu_tile, gmm_sigma_tile = condition_for_data(valid_images_tile)
                if use_gmm:
                    x_t, t, dt_base, gmm_mu_tile, gmm_sigma_tile = shard_data(x_t, t, dt_base, gmm_mu_tile, gmm_sigma_tile)
                else:
                    x_t, t, dt_base = shard_data(x_t, t, dt_base)
                v_pred = call_model(
                    train_state,
                    x_t,
                    t,
                    dt_base,
                    valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond,
                    gmm_mu=gmm_mu_tile,
                    gmm_sigma=gmm_sigma_tile,
                )
                x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
                x_t = jax.experimental.multihost_utils.process_allgather(x_t) # [devices, batch, H, W, C]
                x_1_pred = jax.experimental.multihost_utils.process_allgather(x_1_pred) # [devices, batch, H, W, C]
                valid_images_gather = jax.experimental.multihost_utils.process_allgather(shard_data(valid_images_tile)) # [devices, batch, H, W, C]
                if jax.process_index() == 0:
                    # valid_images_gather is [batchsize] wide. Every 8 corresponds to a timescale.
                    x_t, x_1_pred, valid_images_gather = x_t[0], x_1_pred[0], valid_images_gather[0] #-> (batch, H, W, C)
                    fig, axs = plt.subplots(8, 4*3, figsize=(30, 30))
                    
                    for j in range(min(4, valid_images_gather.shape[0] // 8)):
                        for k in range(8):
                            axs[k,3*j].imshow(process_img(valid_images_gather[j*8 + k]), vmin=0, vmax=1)
                            axs[k,3*j+1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
                            axs[k,3*j+2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
                    wandb.log({f'reconstruction_{dt_type}': wandb.Image(fig)}, step=step)
                    plt.close(fig)

        print("Denoising at N steps")

        denoise_timesteps_list = [1, 2, 4, 8, 16, 32]
        if FLAGS.model.denoise_timesteps == 128:
            denoise_timesteps_list.append(128)
        if FLAGS.model.cfg_scale != 0:
            denoise_timesteps_list.append('cfg')
        for denoise_timesteps in denoise_timesteps_list:
            do_cfg = False
            if denoise_timesteps == 'cfg':
                denoise_timesteps = denoise_timesteps_list[-2]
                do_cfg = True
            all_x = []
            delta_t = 1.0 / denoise_timesteps
            x = eps # [local_batch, ...]
            if use_gmm:
                x, sample_gmm_mu, sample_gmm_sigma = shard_data(x, eps_gmm_mu, eps_gmm_sigma)
            else:
                x = shard_data(x) # [batch, ...] (on all devices)
                sample_gmm_mu = None
                sample_gmm_sigma = None
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((eps.shape[0],), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if not do_cfg:
                    v = call_model(
                        train_state,
                        x,
                        t_vector,
                        dt_base,
                        visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond,
                        gmm_mu=sample_gmm_mu,
                        gmm_sigma=sample_gmm_sigma,
                    )
                else:
                    v_cond = call_model(train_state, x, t_vector, dt_base, visualize_labels, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)
                x = x + v * delta_t
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1) # [batch, timesteps, etc..] ->  # [devices, timesteps, batch, H, W, C]
            all_x = all_x[0]  # -> (timesteps, batch, H, W, C)
            all_x = np.transpose(all_x, (1, 0, 2, 3, 4))  # -> (batch, timesteps, H, W, C)
            all_x = all_x[:, -8:]
            if jax.process_index() == 0:
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(min(8, all_x.shape[1])):
                        axs[t, j].imshow(process_img(all_x[j, t]), vmin=0, vmax=1)
                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            flow_rows = []
            images_shape = batch_images.shape
            num_generations = 50048 #to match with paper's config
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
                delta_t = 1.0 / denoise_timesteps
                x_start = x
                path_length = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                prev_unit = None
                curvature_sum = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((images_shape[0], ), t)
                    dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                    if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                        dt_base = jnp.zeros_like(t_vector)
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    if cfg_scale == 1:
                        v = call_model(train_state, x, t_vector, dt_base, labels, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    elif cfg_scale == 0:
                        v = call_model(train_state, x, t_vector, dt_base, labels_uncond, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                    else:
                        v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                        v_pred_label = call_model(train_state, x, t_vector, dt_base, labels, gmm_mu=sample_gmm_mu, gmm_sigma=sample_gmm_sigma)
                        v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)
                    x_next = x + v * delta_t # Euler sampling.
                    segment = jnp.reshape(x_next - x, (images_shape[0], -1))
                    segment_length = jnp.linalg.norm(segment, axis=1)
                    path_length = path_length + segment_length
                    unit = segment / jnp.maximum(segment_length[:, None], 1e-8)
                    if prev_unit is not None:
                        curvature_sum = curvature_sum + jnp.linalg.norm(unit - prev_unit, axis=1)
                    prev_unit = unit
                    x = x_next
                endpoint = jnp.linalg.norm(jnp.reshape(x - x_start, (images_shape[0], -1)), axis=1)
                flow_rows.append({
                    'flow/path_length_mean': float(jax.device_get(jnp.mean(path_length))),
                    'flow/endpoint_displacement_mean': float(jax.device_get(jnp.mean(endpoint))),
                    'flow/straightness_ratio_mean': float(jax.device_get(jnp.mean(path_length / jnp.maximum(endpoint, 1e-8)))),
                    'flow/curvature_proxy_mean': float(jax.device_get(jnp.mean(curvature_sum / max(denoise_timesteps - 1, 1)))),
                })
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x) # Image is in [-1, 1] space.
                x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
            flow_metrics = {}
            if flow_rows:
                for name in flow_rows[0]:
                    flow_metrics[name] = float(np.mean([row[name] for row in flow_rows]))
            return activations, flow_metrics
        
        if FLAGS.fid_stats is not None:
            denoise_timesteps_list = _parse_eval_fid_timesteps(FLAGS)
            if FLAGS.model.cfg_scale != 0:
                denoise_timesteps_list.append('cfg')
            for denoise_timesteps in denoise_timesteps_list:
                if denoise_timesteps == 'cfg':
                    activations, flow_metrics = do_fid_calc(FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations, flow_metrics = do_fid_calc(1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)
                if jax.process_index() == 0:
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape((-1, activations.shape[-1]))
                    mu1 = np.mean(activations, axis=0)
                    sigma1 = np.cov(activations, rowvar=False)
                    fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(f"FID for denoise_timesteps {denoise_timesteps} is {fid}")
                    logged = {f'fid/timesteps/{denoise_timesteps}': fid, **flow_metrics}
                    logged.update({f'{k}/timesteps/{denoise_timesteps}': v for k, v in flow_metrics.items()})
                    wandb.log(logged, step=step)
                    eval_metrics.update({k: float(v) for k, v in logged.items()})
                    _append_metrics_jsonl(FLAGS.metrics_output_path, {'phase': 'eval_fid', 'step': int(step), **logged})
        return eval_metrics
