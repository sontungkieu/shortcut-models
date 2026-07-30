import jax
import jax.experimental
import json
import os
from pathlib import Path
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from functools import partial

from baselines.targets_gmm_tide import make_tide_source
from fid_repeat_utils import parse_eval_fid_seeds
from gmm_utils import infer_component_params, json_default, sample_prior_components
from metrics_io import append_metrics_csv


def _gmm_source_center_scale(FLAGS):
    if FLAGS.model.train_type == "gmm-centered":
        return float(FLAGS.model.gmm_source_center_scale)
    return None


def _gmm_tide_center_scale(FLAGS):
    if FLAGS.model.train_type == "gmm-tide":
        return float(FLAGS.model.gmm_source_center_scale)
    return 1.0


def _gmm_source_shift_mean(FLAGS):
    return bool(int(getattr(FLAGS.model, "gmm_source_shift_mean", 0)))


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
    return edges.astype(np.float32), schedule, power


def parse_trajectory_save_steps(value, denoise_timesteps):
    denoise_timesteps = int(denoise_timesteps)
    if denoise_timesteps <= 0:
        raise ValueError("trajectory_timesteps must be positive.")
    value = str(value or "").strip()
    if value:
        steps = []
        for token in value.split(","):
            token = token.strip()
            if token:
                steps.append(int(token))
    else:
        steps = np.rint(np.linspace(0, denoise_timesteps, 17)).astype(np.int32).tolist()
    steps.extend([0, denoise_timesteps])
    steps = sorted(set(steps))
    invalid = [step for step in steps if step < 0 or step > denoise_timesteps]
    if invalid:
        raise ValueError(
            f"trajectory_save_steps must lie in [0, {denoise_timesteps}], got {invalid}"
        )
    return steps


def compute_trajectory_metrics(states):
    """Compute raw-latent path diagnostics for states shaped [sample, time, ...]."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim < 3 or states.shape[1] < 2:
        raise ValueError("states must have shape [sample, time>=2, ...].")
    flat = states.reshape(states.shape[0], states.shape[1], -1)
    segments = np.diff(flat, axis=1)
    segment_lengths = np.linalg.norm(segments, axis=-1)
    path_length = np.sum(segment_lengths, axis=1)
    displacement = np.linalg.norm(flat[:, -1] - flat[:, 0], axis=-1)
    straightness_ratio = path_length / np.maximum(displacement, 1e-8)
    unit = segments / np.maximum(segment_lengths[..., None], 1e-8)
    if unit.shape[1] > 1:
        turning = np.linalg.norm(np.diff(unit, axis=1), axis=-1)
        curvature_proxy = np.mean(turning, axis=1)
    else:
        curvature_proxy = np.zeros((states.shape[0],), dtype=np.float32)
    per_sample = {
        "path_length": path_length.astype(np.float32),
        "endpoint_displacement": displacement.astype(np.float32),
        "straightness_ratio": straightness_ratio.astype(np.float32),
        "curvature_proxy": curvature_proxy.astype(np.float32),
    }
    summary = {}
    for name, values in per_sample.items():
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return per_sample, summary


def _gather_global_batch(value):
    gathered = jax.experimental.multihost_utils.process_allgather(value)
    array = np.asarray(jax.device_get(gathered))
    value_ndim = int(np.ndim(value))
    if array.ndim == value_ndim + 1:
        array = array.reshape((-1,) + array.shape[2:])
    return array


def _save_trajectory_contact_sheet(
    FLAGS,
    *,
    states,
    times,
    save_steps,
    vae_decode,
    shard_data,
    output_path,
    num_samples,
    max_columns=9,
):
    num_samples = min(int(num_samples), int(states.shape[0]))
    device_count = int(jax.device_count())
    num_samples = (num_samples // device_count) * device_count
    if num_samples <= 0:
        print(
            "Skipping trajectory contact sheet: trajectory_decode_samples must be "
            f"at least the global device count ({device_count})."
        )
        return None
    time_indices = np.unique(
        np.rint(np.linspace(0, states.shape[1] - 1, min(max_columns, states.shape[1]))).astype(np.int32)
    )
    decoded_columns = []
    for time_index in time_indices:
        batch = shard_data(jnp.asarray(states[:num_samples, time_index]))
        if FLAGS.model.use_stable_vae:
            if vae_decode is None:
                raise ValueError("Stable-VAE trajectory visualization requires vae_decode.")
            batch = vae_decode(batch)
        decoded = _gather_global_batch(batch)[:num_samples]
        decoded = np.clip(decoded * 0.5 + 0.5, 0.0, 1.0)
        decoded_columns.append(decoded)

    if jax.process_index() != 0:
        return None
    fig, axes = plt.subplots(
        num_samples,
        len(time_indices),
        figsize=(2.0 * len(time_indices), 2.0 * num_samples),
        squeeze=False,
    )
    for column, (time_index, decoded) in enumerate(zip(time_indices, decoded_columns)):
        for row in range(num_samples):
            axes[row, column].imshow(decoded[row])
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    f"step {int(save_steps[time_index])}\n"
                    f"$t={float(times[time_index]):.3f}$",
                    fontsize=9,
                )
    fig.suptitle("Learned denoising trajectory: decoded intermediate states", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def eval_denoising_trajectory(
    FLAGS,
    train_state,
    step,
    dataset,
    shard_data,
    vae_encode,
    vae_decode,
    *,
    gmm_state=None,
    router_state=None,
):
    """Run a small eval-only Euler solve and serialize the learned latent path."""
    if FLAGS.load_dir is None:
        raise ValueError("--mode=eval-trajectory requires --load_dir.")
    num_samples = int(FLAGS.trajectory_num_samples)
    denoise_timesteps = int(FLAGS.trajectory_timesteps)
    if num_samples <= 0:
        raise ValueError("trajectory_num_samples must be positive.")
    if num_samples % int(jax.device_count()) != 0:
        raise ValueError(
            "trajectory_num_samples must be divisible by the global device count "
            f"({jax.device_count()})."
        )
    save_steps = parse_trajectory_save_steps(
        FLAGS.trajectory_save_steps,
        denoise_timesteps,
    )
    output_path = Path(FLAGS.trajectory_output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("trajectory_output_path must end in .npz.")

    with jax.spmd_mode("allow_all"):
        use_tide = (
            FLAGS.model.train_type == "gmm-tide"
            and gmm_state is not None
            and router_state is not None
        )
        use_gmm = (
            FLAGS.model.train_type in ("naive", "gmm-centered", "gmm-tide")
            and gmm_state is not None
        )
        batch_images, _ = next(dataset)
        shape_key = jax.random.PRNGKey(int(FLAGS.trajectory_seed) + jax.process_index())
        if FLAGS.model.use_stable_vae and "latent" not in FLAGS.dataset_name:
            batch_images = vae_encode(shape_key, batch_images)
        if "latent" in FLAGS.dataset_name:
            batch_images = batch_images[..., batch_images.shape[-1] // 2 :]
        latent_shape = tuple(batch_images.shape[1:])
        images_shape = (num_samples,) + latent_shape

        key = jax.random.PRNGKey(int(FLAGS.trajectory_seed))
        key = jax.random.fold_in(key, jax.process_index())
        eps_key, label_key = jax.random.split(key)
        source_info = {}
        source_component_ids = np.full((num_samples,), -1, dtype=np.int32)
        if use_tide:
            x, sample_gmm_mu, sample_gmm_sigma, source_info = make_tide_source(
                eps_key,
                gmm_state,
                router_state,
                num_samples,
                latent_shape,
                topk=FLAGS.model.gmm_router_topk,
                temperature=FLAGS.model.gmm_router_temperature,
                gradient_mode=FLAGS.model.gmm_router_gradient_mode,
                gumbel_tau=FLAGS.model.gmm_router_gumbel_tau,
                source_mode=FLAGS.model.gmm_router_source_mode,
                routing_policy=FLAGS.model.gmm_router_routing_policy,
                shift_mixture_mean=_gmm_source_shift_mean(FLAGS),
                center_scale=_gmm_tide_center_scale(FLAGS),
            )
        elif use_gmm:
            x, sample_gmm_mu, sample_gmm_sigma, component_ids = sample_prior_components(
                eps_key,
                gmm_state,
                num_samples,
                latent_shape,
                center_scale=_gmm_source_center_scale(FLAGS),
            )
            source_component_ids = _gather_global_batch(shard_data(component_ids))[:num_samples]
        else:
            x = jax.random.normal(eps_key, images_shape)
            sample_gmm_mu = None
            sample_gmm_sigma = None

        labels = jax.random.randint(
            label_key,
            (num_samples,),
            0,
            int(FLAGS.model.num_classes),
        )
        labels_uncond = jnp.full(
            (num_samples,),
            int(FLAGS.model.num_classes),
            dtype=jnp.int32,
        )
        if use_gmm:
            x, labels, labels_uncond, sample_gmm_mu, sample_gmm_sigma = shard_data(
                x,
                labels,
                labels_uncond,
                sample_gmm_mu,
                sample_gmm_sigma,
            )
        else:
            x, labels, labels_uncond = shard_data(x, labels, labels_uncond)

        @partial(jax.jit, static_argnames=("use_ema",))
        def call_model(
            train_state_arg,
            images,
            t,
            dt,
            labels_arg,
            gmm_mu=None,
            gmm_sigma=None,
            use_ema=True,
        ):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state_arg.call_model_ema
            else:
                call_fn = train_state_arg.call_model
            return call_fn(
                images,
                t,
                dt,
                labels_arg,
                train=False,
                gmm_mu=gmm_mu,
                gmm_sigma=gmm_sigma,
            )

        t_edges, ode_schedule, ode_power = _ode_time_edges(FLAGS, denoise_timesteps)
        saved_states = {0: _gather_global_batch(x)[:num_samples]}
        cfg_scale = float(FLAGS.model.cfg_scale)
        for ti in tqdm.tqdm(range(denoise_timesteps), desc="Eval learned trajectory"):
            t = float(t_edges[ti])
            delta_t = float(t_edges[ti + 1] - t_edges[ti])
            t_vector = jnp.full((num_samples,), t, dtype=jnp.float32)
            dt_base = jnp.full(
                (num_samples,),
                np.log2(denoise_timesteps),
                dtype=jnp.float32,
            )
            if FLAGS.model.train_type == "livereflow" and denoise_timesteps < 128:
                dt_base = jnp.zeros_like(t_vector)
            t_vector, dt_base = shard_data(t_vector, dt_base)
            if cfg_scale == 1.0:
                velocity = call_model(
                    train_state,
                    x,
                    t_vector,
                    dt_base,
                    labels,
                    gmm_mu=sample_gmm_mu,
                    gmm_sigma=sample_gmm_sigma,
                )
            elif cfg_scale == 0.0:
                velocity = call_model(
                    train_state,
                    x,
                    t_vector,
                    dt_base,
                    labels_uncond,
                    gmm_mu=sample_gmm_mu,
                    gmm_sigma=sample_gmm_sigma,
                )
            else:
                velocity_uncond = call_model(
                    train_state,
                    x,
                    t_vector,
                    dt_base,
                    labels_uncond,
                    gmm_mu=sample_gmm_mu,
                    gmm_sigma=sample_gmm_sigma,
                )
                velocity_cond = call_model(
                    train_state,
                    x,
                    t_vector,
                    dt_base,
                    labels,
                    gmm_mu=sample_gmm_mu,
                    gmm_sigma=sample_gmm_sigma,
                )
                velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            x = x + velocity * delta_t
            step_after_update = ti + 1
            if step_after_update in save_steps:
                saved_states[step_after_update] = _gather_global_batch(x)[:num_samples]

        states = np.stack([saved_states[save_step] for save_step in save_steps], axis=1)
        times = t_edges[np.asarray(save_steps, dtype=np.int32)]
        labels_host = _gather_global_batch(labels)[:num_samples].astype(np.int32)
        source_mu_host = (
            _gather_global_batch(sample_gmm_mu)[:num_samples]
            if sample_gmm_mu is not None
            else np.empty((0,), dtype=np.float32)
        )
        source_sigma_host = (
            _gather_global_batch(sample_gmm_sigma)[:num_samples]
            if sample_gmm_sigma is not None
            else np.empty((0,), dtype=np.float32)
        )
        per_sample, metric_summary = compute_trajectory_metrics(states)
        scalar_source_info = {}
        for name, value in source_info.items():
            value = np.asarray(jax.device_get(value))
            if value.shape == ():
                scalar_source_info[name] = float(value)

        metadata = {
            "format_version": 1,
            "phase": "eval_trajectory",
            "run_name": str(FLAGS.wandb.name),
            "checkpoint_step": int(step),
            "trajectory_seed": int(FLAGS.trajectory_seed),
            "trajectory_num_samples": num_samples,
            "trajectory_timesteps": denoise_timesteps,
            "trajectory_save_steps": save_steps,
            "cfg_scale": cfg_scale,
            "train_type": str(FLAGS.model.train_type),
            "use_ema": bool(FLAGS.model.use_ema),
            "ode_schedule": ode_schedule,
            "ode_power": float(ode_power),
            "gmm_source_shift_mean": _gmm_source_shift_mean(FLAGS),
            "gmm_source_center_scale": (
                float(FLAGS.model.gmm_source_center_scale) if use_gmm else None
            ),
            "gmm_router_topk": int(FLAGS.model.gmm_router_topk) if use_tide else None,
            "gmm_router_temperature": (
                float(FLAGS.model.gmm_router_temperature) if use_tide else None
            ),
            "source_info": scalar_source_info,
            "metrics": metric_summary,
        }
        contact_sheet_path = output_path.with_name(output_path.stem + "_contact_sheet.png")
        _save_trajectory_contact_sheet(
            FLAGS,
            states=states,
            times=times,
            save_steps=save_steps,
            vae_decode=vae_decode,
            shard_data=shard_data,
            output_path=contact_sheet_path,
            num_samples=FLAGS.trajectory_decode_samples,
        )

        if jax.process_index() == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                states=states.astype(np.float32),
                times=np.asarray(times, dtype=np.float32),
                save_steps=np.asarray(save_steps, dtype=np.int32),
                labels=labels_host,
                source_mu=np.asarray(source_mu_host, dtype=np.float32),
                source_sigma=np.asarray(source_sigma_host, dtype=np.float32),
                source_component_ids=np.asarray(source_component_ids, dtype=np.int32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                **per_sample,
            )
            summary_path = output_path.with_name(output_path.stem + "_summary.json")
            summary_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n",
                encoding="utf-8",
            )
            payload = {
                "phase": "eval_trajectory",
                "step": int(step),
                "run_name": str(FLAGS.wandb.name),
                "trajectory_output_path": str(output_path),
                "trajectory_contact_sheet_path": str(contact_sheet_path),
                **{f"trajectory/{name}": value for name, value in metric_summary.items()},
            }
            _append_metrics_jsonl(FLAGS.metrics_output_path, payload)
            append_metrics_csv(FLAGS.metrics_output_path, payload)
            print("TRAJECTORY_EVAL_RESULT " + json.dumps(payload, sort_keys=True))
        return metadata


def eval_fid_repeats(
    FLAGS,
    train_state,
    step,
    dataset,
    shard_data,
    vae_encode,
    vae_decode,
    get_fid_activations,
    fid_from_stats,
    truth_fid_stats,
    *,
    eval_seeds,
    num_generations,
    gmm_state=None,
    router_state=None,
    log_wandb=False,
):
    eval_seeds = parse_eval_fid_seeds(eval_seeds)
    num_generations = int(num_generations)
    if num_generations <= 0:
        raise ValueError("eval_fid_generations must be positive.")
    if num_generations % int(FLAGS.batch_size) != 0:
        raise ValueError(
            "eval_fid_generations must be divisible by batch_size so every repeat "
            "uses exactly the requested sample count."
        )
    if get_fid_activations is None or fid_from_stats is None or truth_fid_stats is None:
        raise ValueError("Repeated FID evaluation requires --fid_stats.")

    with jax.spmd_mode('allow_all'):
        use_tide = FLAGS.model.train_type == 'gmm-tide' and gmm_state is not None and router_state is not None
        use_gmm = FLAGS.model.train_type in ('naive', 'gmm-centered', 'gmm-tide') and gmm_state is not None
        shape_key = jax.random.PRNGKey(eval_seeds[0] + jax.process_index())
        batch_images, batch_labels = next(dataset)
        if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
            batch_images = vae_encode(shape_key, batch_images)
        if 'latent' in FLAGS.dataset_name:
            batch_images = batch_images[..., batch_images.shape[-1] // 2:]
        images_shape = batch_images.shape
        labels_uncond = shard_data(
            jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']
        )

        @partial(jax.jit, static_argnames=("use_ema",))
        def call_model(train_state_arg, images, t, dt, labels, gmm_mu=None, gmm_sigma=None, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state_arg.call_model_ema
            else:
                call_fn = train_state_arg.call_model
            return call_fn(
                images,
                t,
                dt,
                labels,
                train=False,
                gmm_mu=gmm_mu,
                gmm_sigma=gmm_sigma,
            )

        def do_fid_calc(eval_seed, cfg_scale, denoise_timesteps):
            activations = []
            flow_rows = []
            t_edges, ode_schedule, ode_power = _ode_time_edges(FLAGS, denoise_timesteps)
            print(
                f"Calc repeated FID seed={eval_seed} CFG={cfg_scale} "
                f"timesteps={denoise_timesteps} generations={num_generations}"
            )
            for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                key = jax.random.PRNGKey(eval_seed)
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
                        routing_policy=FLAGS.model.gmm_router_routing_policy,
                        shift_mixture_mean=_gmm_source_shift_mean(FLAGS),
                        center_scale=_gmm_tide_center_scale(FLAGS),
                    )
                elif use_gmm:
                    x, sample_gmm_mu, sample_gmm_sigma, _ = sample_prior_components(
                        eps_key,
                        gmm_state,
                        images_shape[0],
                        images_shape[1:],
                        center_scale=_gmm_source_center_scale(FLAGS),
                    )
                else:
                    x = jax.random.normal(eps_key, images_shape)
                    sample_gmm_mu = None
                    sample_gmm_sigma = None
                labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                if use_gmm:
                    x, labels, sample_gmm_mu, sample_gmm_sigma = shard_data(
                        x,
                        labels,
                        sample_gmm_mu,
                        sample_gmm_sigma,
                    )
                else:
                    x, labels = shard_data(x, labels)

                x_start = x
                path_length = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                prev_unit = None
                curvature_sum = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                for ti in range(denoise_timesteps):
                    t = float(t_edges[ti])
                    delta_t = float(t_edges[ti + 1] - t_edges[ti])
                    t_vector = jnp.full((images_shape[0],), t)
                    dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                    if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                        dt_base = jnp.zeros_like(t_vector)
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    if cfg_scale == 1:
                        v = call_model(
                            train_state,
                            x,
                            t_vector,
                            dt_base,
                            labels,
                            gmm_mu=sample_gmm_mu,
                            gmm_sigma=sample_gmm_sigma,
                        )
                    elif cfg_scale == 0:
                        v = call_model(
                            train_state,
                            x,
                            t_vector,
                            dt_base,
                            labels_uncond,
                            gmm_mu=sample_gmm_mu,
                            gmm_sigma=sample_gmm_sigma,
                        )
                    else:
                        v_uncond = call_model(
                            train_state,
                            x,
                            t_vector,
                            dt_base,
                            labels_uncond,
                            gmm_mu=sample_gmm_mu,
                            gmm_sigma=sample_gmm_sigma,
                        )
                        v_cond = call_model(
                            train_state,
                            x,
                            t_vector,
                            dt_base,
                            labels,
                            gmm_mu=sample_gmm_mu,
                            gmm_sigma=sample_gmm_sigma,
                        )
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    x_next = x + v * delta_t
                    segment = jnp.reshape(x_next - x, (images_shape[0], -1))
                    segment_length = jnp.linalg.norm(segment, axis=1)
                    path_length = path_length + segment_length
                    unit = segment / jnp.maximum(segment_length[:, None], 1e-8)
                    if prev_unit is not None:
                        curvature_sum = curvature_sum + jnp.linalg.norm(unit - prev_unit, axis=1)
                    prev_unit = unit
                    x = x_next

                endpoint = jnp.linalg.norm(
                    jnp.reshape(x - x_start, (images_shape[0], -1)),
                    axis=1,
                )
                flow_rows.append(
                    {
                        'flow/path_length_mean': float(jax.device_get(jnp.mean(path_length))),
                        'flow/endpoint_displacement_mean': float(jax.device_get(jnp.mean(endpoint))),
                        'flow/straightness_ratio_mean': float(
                            jax.device_get(jnp.mean(path_length / jnp.maximum(endpoint, 1e-8)))
                        ),
                        'flow/curvature_proxy_mean': float(
                            jax.device_get(
                                jnp.mean(curvature_sum / max(denoise_timesteps - 1, 1))
                            )
                        ),
                        'flow/eval_ode_is_end_dense': float(
                            ode_schedule in ("end_dense", "end-dense", "power_end")
                        ),
                        'flow/eval_ode_power': float(ode_power),
                    }
                )
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)
                x = jax.image.resize(
                    x,
                    (x.shape[0], 299, 299, 3),
                    method='bilinear',
                    antialias=False,
                )
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                activations.append(np.array(acts))

            flow_metrics = {
                name: float(np.mean([row[name] for row in flow_rows]))
                for name in flow_rows[0]
            }
            return activations, flow_metrics

        evaluations = [
            (str(timesteps), 1 if FLAGS.model.cfg_scale != 0 else 0, timesteps)
            for timesteps in _parse_eval_fid_timesteps(FLAGS)
        ]
        if FLAGS.model.cfg_scale != 0:
            evaluations.append(('cfg', float(FLAGS.model.cfg_scale), int(FLAGS.model.denoise_timesteps)))

        records = []
        for repeat_index, eval_seed in enumerate(eval_seeds):
            for metric_suffix, cfg_scale, denoise_timesteps in evaluations:
                activations, flow_metrics = do_fid_calc(eval_seed, cfg_scale, denoise_timesteps)
                if jax.process_index() != 0:
                    continue
                activations = np.concatenate(activations, axis=0)
                activations = activations.reshape((-1, activations.shape[-1]))
                mu1 = np.mean(activations, axis=0)
                sigma1 = np.cov(activations, rowvar=False)
                fid = float(
                    fid_from_stats(
                        mu1,
                        sigma1,
                        truth_fid_stats['mu'],
                        truth_fid_stats['sigma'],
                    )
                )
                metric_name = f'fid/timesteps/{metric_suffix}'
                logged = {metric_name: fid, **flow_metrics}
                logged.update(
                    {f'{name}/timesteps/{metric_suffix}': value for name, value in flow_metrics.items()}
                )
                payload = {
                    'phase': 'eval_fid_repeat',
                    'step': int(step),
                    'run_name': str(FLAGS.wandb.name),
                    'eval_seed': int(eval_seed),
                    'eval_repeat_index': int(repeat_index),
                    'eval_fid_generations': int(num_generations),
                    'eval_cfg_scale': float(cfg_scale),
                    **logged,
                }
                _append_metrics_jsonl(FLAGS.metrics_output_path, payload)
                append_metrics_csv(FLAGS.metrics_output_path, payload)
                if log_wandb:
                    wandb.log(logged, step=step)
                result = {
                    'eval_seed': int(eval_seed),
                    'repeat_index': int(repeat_index),
                    'metric_name': metric_name,
                    'value': fid,
                    'step': int(step),
                    'num_generations': int(num_generations),
                    'cfg_scale': float(cfg_scale),
                    'flow_metrics': flow_metrics,
                }
                records.append(result)
                print("FID_REPEAT_RESULT " + json.dumps(result, sort_keys=True, default=json_default))
        return records


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
        use_gmm = FLAGS.model.train_type in ('naive', 'gmm-centered', 'gmm-tide') and gmm_state is not None
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
                gradient_mode=FLAGS.model.gmm_router_gradient_mode,
                gumbel_tau=FLAGS.model.gmm_router_gumbel_tau,
                source_mode=FLAGS.model.gmm_router_source_mode,
                routing_policy=FLAGS.model.gmm_router_routing_policy,
                shift_mixture_mean=_gmm_source_shift_mean(FLAGS),
                center_scale=_gmm_tide_center_scale(FLAGS),
            )
        elif use_gmm:
            eps, eps_gmm_mu, eps_gmm_sigma, _ = sample_prior_components(
                key,
                gmm_state,
                batch_images.shape[0],
                batch_images.shape[1:],
                center_scale=_gmm_source_center_scale(FLAGS),
            )
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
            _, _, _, gmm_mu, gmm_sigma = infer_component_params(
                gmm_state,
                jnp.asarray(latents),
                center_scale=_gmm_source_center_scale(FLAGS),
            )
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
            t_edges, _, _ = _ode_time_edges(FLAGS, denoise_timesteps)
            x = eps # [local_batch, ...]
            if use_gmm:
                x, sample_gmm_mu, sample_gmm_sigma = shard_data(x, eps_gmm_mu, eps_gmm_sigma)
            else:
                x = shard_data(x) # [batch, ...] (on all devices)
                sample_gmm_mu = None
                sample_gmm_sigma = None
            for ti in range(denoise_timesteps):
                t = float(t_edges[ti]) # From x_0 (noise) to x_1 (data)
                delta_t = float(t_edges[ti + 1] - t_edges[ti])
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
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == denoise_timesteps - 1:
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
            t_edges, ode_schedule, ode_power = _ode_time_edges(FLAGS, denoise_timesteps)
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
                        routing_policy=FLAGS.model.gmm_router_routing_policy,
                        shift_mixture_mean=_gmm_source_shift_mean(FLAGS),
                        center_scale=_gmm_tide_center_scale(FLAGS),
                    )
                elif use_gmm:
                    x, sample_gmm_mu, sample_gmm_sigma, _ = sample_prior_components(
                        eps_key,
                        gmm_state,
                        images_shape[0],
                        images_shape[1:],
                        center_scale=_gmm_source_center_scale(FLAGS),
                    )
                else:
                    x = jax.random.normal(eps_key, images_shape)
                    sample_gmm_mu = None
                    sample_gmm_sigma = None
                labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                if use_gmm:
                    x, labels, sample_gmm_mu, sample_gmm_sigma = shard_data(x, labels, sample_gmm_mu, sample_gmm_sigma)
                else:
                    x, labels = shard_data(x, labels)
                x_start = x
                path_length = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                prev_unit = None
                curvature_sum = jnp.zeros((images_shape[0],), dtype=jnp.float32)
                for ti in range(denoise_timesteps):
                    t = float(t_edges[ti]) # From x_0 (noise) to x_1 (data)
                    delta_t = float(t_edges[ti + 1] - t_edges[ti])
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
                    'flow/eval_ode_is_end_dense': float(ode_schedule in ("end_dense", "end-dense", "power_end")),
                    'flow/eval_ode_power': float(ode_power),
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
                    payload = {'phase': 'eval_fid', 'step': int(step), **logged}
                    _append_metrics_jsonl(FLAGS.metrics_output_path, payload)
                    append_metrics_csv(FLAGS.metrics_output_path, payload)
        return eval_metrics
