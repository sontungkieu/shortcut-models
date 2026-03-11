import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from functools import partial
from utils.sit_transport import get_transport_dt_base, sit_sample


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
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
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
        batch_labels_sharded, valid_labels_sharded = shard_data(
            batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(
            batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])  # Null token
        eps = jax.random.normal(key, batch_images.shape)

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

        @partial(jax.jit, static_argnums=(5,6))
        def call_model(train_state, images, t, dt, labels, use_ema=True, return_activations=False):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False, return_activations=return_activations)
            return output

        def sit_model_output(x, t_vector, labels_cond, cfg_scale, return_activations=False):
            dt_base = get_transport_dt_base(FLAGS.model['denoise_timesteps'], t_vector.shape[0])
            t_vector, dt_base = shard_data(t_vector, dt_base)
            if cfg_scale == 1:
                return call_model(train_state, x, t_vector, dt_base, labels_cond, return_activations=return_activations)
            if cfg_scale == 0:
                return call_model(train_state, x, t_vector, dt_base, labels_uncond, return_activations=return_activations)
            if return_activations:
                v_cond, logvars, activations = call_model(
                    train_state, x, t_vector, dt_base, labels_cond, return_activations=True)
                v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                return v_uncond + cfg_scale * (v_cond - v_uncond), logvars, activations
            v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
            v_cond = call_model(train_state, x, t_vector, dt_base, labels_cond)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        def sample_batch(x, labels, num_steps, cfg_scale, sample_key, return_history=False):
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
                    return_history=return_history,
                )

            delta_t = 1.0 / num_steps
            current_x = x
            history = [current_x] if return_history else None
            for ti in range(num_steps):
                t = ti / num_steps  # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((current_x.shape[0],), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(num_steps)
                if FLAGS.model.train_type == 'naive':
                    dt_base = jnp.ones_like(t_vector) * np.log2(FLAGS.model['denoise_timesteps'])
                if FLAGS.model.train_type == 'livereflow' and num_steps < 128:
                    dt_base = jnp.zeros_like(t_vector)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, current_x, t_vector, dt_base, labels)
                elif cfg_scale == 0:
                    v = call_model(train_state, current_x, t_vector, dt_base, labels_uncond)
                else:
                    v_cond = call_model(train_state, current_x, t_vector, dt_base, labels)
                    v_uncond = call_model(train_state, current_x, t_vector, dt_base, labels_uncond)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                if FLAGS.model.train_type == 'consistency':
                    eps_step = shard_data(jax.random.normal(jax.random.fold_in(sample_key, ti), current_x.shape))
                    x1pred = current_x + v * (1 - t)
                    current_x = x1pred * (t + delta_t) + eps_step * (1 - t - delta_t)
                else:
                    current_x = current_x + v * delta_t
                if return_history:
                    history.append(current_x)
            return history if return_history else current_x

        print("Training Loss per T.")
        if FLAGS.model['train_type'] == 'sit':
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            infos = None
            for t in np.arange(0, 32):
                t = t * (1.0 / 32)

                batch_images_n, batch_labels_n = next(dataset)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images_n = vae_encode(key, batch_images_n)
                batch_images_sharded, batch_labels_sharded = shard_data(
                    batch_images_n, batch_labels_n)
                _, info = update(train_state, train_state_teacher,
                                 batch_images_sharded, batch_labels_sharded, force_t=t)
                info = jax.experimental.multihost_utils.process_allgather(info)
                if infos is None:
                    infos = jax.tree_map(lambda x: [x], info)
                else:
                    infos = jax.tree_map(lambda x, y: y + [x], info, infos)
            time_axis = np.arange(0, 32) / 32
            ax.plot(time_axis, infos['loss'])
            ax.set_title("SiT")
            if jax.process_index() == 0:
                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)
                plt.close(fig)
        elif FLAGS.model.denoise_timesteps == 128:
            fig, axs = plt.subplots(5, 8, figsize=(15, 12))
            d_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            fig, axs = plt.subplots(3, 6, figsize=(15, 8))
            d_list = [0, 1, 2, 3, 4, 5]
        if FLAGS.model['train_type'] != 'sit':
            for d in d_list:
                infos = None
                for t in np.arange(0, 32):
                    t = t * (1.0 / 32)

                    batch_images_n, batch_labels_n = next(dataset)
                    if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                        batch_images_n = vae_encode(key, batch_images_n)
                    batch_images_sharded, batch_labels_sharded = shard_data(
                        batch_images_n, batch_labels_n)
                    _, info = update(train_state, train_state_teacher,
                                     batch_images_sharded, batch_labels_sharded, force_t=t, force_dt=d)
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
        if FLAGS.model['train_type'] == 'sit':
            print("Skipping one-step reconstruction plots for SiT because the transport path is not dt-indexed.")
        else:
            if 'latent' in FLAGS.dataset_name:
                eps = eps_valid
            for dt_type in ['flow', 'shortcut']:
                if len(jax.local_devices()) == 8:
                    if dt_type == 'flow':
                        t = jnp.arange(8) / 8  # between 0 and 0.875
                        t = jnp.tile(t, valid_images.shape[0] // 8)  # [batch, etc]
                        dt = 0
                        dt_base = jnp.ones_like(
                            t) * np.log2(FLAGS.model.denoise_timesteps)
                    elif dt_type == 'shortcut':
                        dt_base = jnp.array([0, 0, 0, 1, 2, 3, 4, 5])
                        if FLAGS.model.denoise_timesteps == 128:
                            dt_base = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
                        dt_base = jnp.tile(
                            dt_base, valid_images.shape[0] // 8)  # [batch, etc]
                        dt = 2.0 ** (-dt_base)
                        t = 1 - dt
                    eps_tile = jnp.repeat(eps, 8, axis=0)[:valid_images.shape[0]]
                    valid_images_tile = jnp.repeat(valid_images, 8, axis=0)[
                        :valid_images.shape[0]]
                    t_full = t[..., None, None, None]
                    x_t = (1 - (1 - 1e-5) * t_full) * \
                        eps_tile + t_full * valid_images_tile
                    x_t, t, dt_base = shard_data(x_t, t, dt_base)
                    v_pred, _, _ = call_model(
                        train_state,
                        x_t,
                        t,
                        dt_base,
                        valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond,
                        return_activations=True,
                    )
                    x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
                    x_t = jax.experimental.multihost_utils.process_allgather(x_t)
                    x_1_pred = jax.experimental.multihost_utils.process_allgather(
                        x_1_pred)
                    valid_images_gather = jax.experimental.multihost_utils.process_allgather(
                        shard_data(valid_images_tile))
                    if jax.process_index() == 0:
                        x_t = x_t[0]
                        x_1_pred = x_1_pred[0]
                        valid_images_gather = valid_images_gather[0]
                        fig, axs = plt.subplots(8, 4*3, figsize=(30, 30))

                        for j in range(min(4, valid_images_gather.shape[0] // 8)):
                            for k in range(8):
                                axs[k, 3*j].imshow(process_img(
                                    valid_images_gather[j*8 + k]), vmin=0, vmax=1)
                                axs[k, 3*j +
                                    1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
                                axs[k, 3*j +
                                    2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
                        wandb.log(
                            {f'reconstruction_{dt_type}': wandb.Image(fig)}, step=step)
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
            all_activations = {}
            x = eps  # [local_batch, ...]
            x = shard_data(x)  # [batch, ...] (on all devices)
            history = sample_batch(
                x,
                visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond,
                denoise_timesteps,
                FLAGS.model.cfg_scale if do_cfg else (1 if FLAGS.model.cfg_scale != 0 else 0),
                jax.random.fold_in(key, denoise_timesteps),
                return_history=True,
            )
            history = history[1:]
            capture_stride = max(1, len(history) // 8)
            for ti, x_step in enumerate(history):
                if denoise_timesteps <= 8 or ti % capture_stride == 0 or ti == len(history) - 1:
                    np_x = jax.experimental.multihost_utils.process_allgather(x_step)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1)
            all_x = all_x[0]
            all_x = np.transpose(all_x, (1, 0, 2, 3, 4))
            all_x = all_x[:, -8:]  # Last 8 timesteps
            if jax.process_index() == 0:
                num_viz_samples = min(8, all_x.shape[0])  # Limit samples
                num_viz_timesteps = min(8, all_x.shape[1])  # Limit timesteps
                fig, axs = plt.subplots(num_viz_timesteps, num_viz_samples, figsize=(
                    num_viz_samples * 3, num_viz_timesteps * 3))

                # Fix reshape: Xử lý single Axes (1x1 subplot)
                if num_viz_timesteps == 1 and num_viz_samples == 1:
                    # Single subplot: axs là Axes object, không cần reshape
                    pass
                elif num_viz_timesteps == 1:
                    # 1 row, multiple cols: axs là 1D array, reshape thành 2D (1, N)
                    axs = np.array(axs).reshape(1, -1)
                elif num_viz_samples == 1:
                    # Multiple rows, 1 col: axs là 2D với shape (M, 1), transpose nếu cần
                    axs = axs.reshape(-1, 1)

                for t in range(num_viz_timesteps):
                    for j in range(num_viz_samples):
                        sample_img = process_img(all_x[j, t])  # Single latent
                        if num_viz_timesteps == 1 and num_viz_samples == 1:
                            # Direct call cho single Axes
                            axs.imshow(sample_img, vmin=0, vmax=1)
                        else:
                            axs[t, j].imshow(sample_img, vmin=0, vmax=1)
                            axs[t, j].axis('off')
                            axs[t, j].set_title(f't={t}, sample={j}')
                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            images_shape = batch_images.shape
            num_generations = 4096
            print(
                f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
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
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                # [devices, batch//devices, 2048]
                acts = get_fid_activations(x)[..., 0, 0, :]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
            return activations

        if FLAGS.fid_stats is not None:
            denoise_timesteps_list = [1, 4, 32]
            if FLAGS.model.denoise_timesteps == 128:
                denoise_timesteps_list.append(128)
            if FLAGS.model.cfg_scale != 0:
                denoise_timesteps_list.append('cfg')
            for denoise_timesteps in denoise_timesteps_list:
                if denoise_timesteps == 'cfg':
                    activations = do_fid_calc(
                        FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations = do_fid_calc(
                        1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)
                if jax.process_index() == 0:
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape(
                        (-1, activations.shape[-1]))
                    mu1 = np.mean(activations, axis=0)
                    sigma1 = np.cov(activations, rowvar=False)
                    fid = fid_from_stats(
                        mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(
                        f"FID for denoise_timesteps {denoise_timesteps} is {fid}")
                    wandb.log(
                        {f'fid/timesteps/{denoise_timesteps}': fid}, step=step)

        if jax.process_index() == 0:
            for block_name, acts_list in all_activations.items():
                acts_arr = np.stack(acts_list, axis=1)
                print(f"acts_arr.shape: {acts_arr.shape} of block {block_name}")
                if acts_arr.ndim >= 3:
                    reduce_axes = tuple(range(2, acts_arr.ndim))  # tính norm theo các chiều sau (batch, timesteps, ...)
                    l2_norms = np.sqrt(np.sum(acts_arr * acts_arr, axis=reduce_axes))
                elif acts_arr.ndim == 2:
                    l2_norms = np.linalg.norm(acts_arr, axis=-1)
                else:
                    l2_norms = np.linalg.norm(acts_arr)


                num_viz_samples = min(8, l2_norms.shape[0])
                T = l2_norms.shape[1]

                table = wandb.Table(columns=["timestep", "l2", "sample"])
                for j in range(num_viz_samples):
                    for t in range(T):
                        table.add_data(t, float(l2_norms[j, t]), f"sample_{j}")

                chart = wandb.plot.line(
                    table, x="timestep", y="l2", stroke="sample",
                    title=f"{block_name} ({d_label})"
                )
                wandb.log({f"activations_l2/{block_name}/{d_label}": chart}, step=step)
