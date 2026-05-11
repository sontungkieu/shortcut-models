import json
import os
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from absl import app, flags
from ml_collections import config_flags

from gmm_router import GMMRouter, router_metrics, save_router_checkpoint
from gmm_utils import flatten_latents, json_default, load_gmm_stats, posterior_from_stats, sample_prior_components
from metrics_io import append_metrics_csv, clear_metrics_csv
from utils.datasets import get_dataset
from utils.stable_vae import StableVAE
from utils.wandb import default_wandb_config, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", "celebahq256", "TFDS dataset name.")
flags.DEFINE_string("tfds_data_dir", None, "Optional TFDS data_dir.")
flags.DEFINE_integer("batch_size", 64, "Router distillation batch size.")
flags.DEFINE_integer("seed", 10, "Random seed.")
flags.DEFINE_integer("debug_overfit", 0, "Use a tiny repeated dataset for debugging.")
flags.DEFINE_string("gmm_stats_path", "", "Input GMM stats .npz path.")
flags.DEFINE_string("router_save_path", "/kaggle/working/gmm_router.pkl", "Output router checkpoint path.")
flags.DEFINE_string("router_train_data_mode", "mix", "Router input mode: x1, x0, or mix.")
flags.DEFINE_float("router_mix_x1_prob", 0.5, "Probability of using x1 in mix mode.")
flags.DEFINE_string("router_target_type", "soft_kl", "Router target loss: soft_kl or hard_ce.")
flags.DEFINE_integer("router_max_steps", 10000, "Router optimizer steps.")
flags.DEFINE_integer("router_log_interval", 100, "Router metric logging interval.")
flags.DEFINE_integer("router_valid_interval", 1000, "Router validation interval.")
flags.DEFINE_integer("router_valid_batches", 8, "Validation batches per validation pass.")
flags.DEFINE_float("router_lr", 3e-4, "Router learning rate.")
flags.DEFINE_float("router_weight_decay", 1e-4, "Router AdamW weight decay.")
flags.DEFINE_integer("router_hidden_channels", 128, "Router first convolution width.")
flags.DEFINE_integer("router_mlp_hidden_size", 256, "Router hidden MLP width.")
flags.DEFINE_integer("router_depth", 3, "Router convolution depth.")
flags.DEFINE_bool("router_save_best", True, "Save the best validation-loss router instead of the last step.")
flags.DEFINE_string("metrics_output_path", None, "Optional JSONL path for router diagnostics.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "shortcut",
        "name": "gmm_router_{dataset_name}",
    }
)
config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)


def _append_jsonl(path: str, payload: Dict[str, object]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default))
        f.write("\n")


def _write_summary(path: str, payload: Dict[str, object]):
    if not path:
        return
    summary_path = path[:-6] + "_summary.json" if path.endswith(".jsonl") else path + ".summary.json"
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def _to_float_dict(metrics):
    out = {}
    for name, value in metrics.items():
        arr = np.asarray(value)
        if arr.shape == ():
            out[name] = float(arr)
    return out


def _add_router_overfit_metrics(log_metrics, train_metrics, valid_metrics, best_valid_loss, best_valid_step, step):
    eps = 1e-8
    gap_pairs = {
        "loss": ("loss", "valid_minus_train"),
        "kl_to_gmm": ("router/kl_to_gmm", "valid_minus_train"),
        "cross_entropy": ("router/cross_entropy", "valid_minus_train"),
        "top1_agreement": ("router/top1_agreement", "train_minus_valid"),
        "top1_prob_mean": ("router/top1_prob_mean", "train_minus_valid"),
        "usage_entropy_normalized": ("router/usage_entropy_normalized", "abs_gap"),
        "assign_max_frac": ("router/assign_max_frac", "valid_minus_train"),
        "num_unique_clusters": ("router/num_unique_clusters", "train_minus_valid"),
    }
    for short_name, (metric_name, mode) in gap_pairs.items():
        if metric_name not in train_metrics or metric_name not in valid_metrics:
            continue
        train_value = float(train_metrics[metric_name])
        valid_value = float(valid_metrics[metric_name])
        if mode == "train_minus_valid":
            gap = train_value - valid_value
        elif mode == "abs_gap":
            gap = abs(train_value - valid_value)
        else:
            gap = valid_value - train_value
        log_metrics[f"router_overfit/{short_name}_gap"] = gap
        if short_name in ("loss", "kl_to_gmm", "cross_entropy"):
            log_metrics[f"router_overfit/{short_name}_valid_to_train_ratio"] = valid_value / max(train_value, eps)
    log_metrics["router_overfit/best_valid_loss"] = float(best_valid_loss)
    log_metrics["router_overfit/best_valid_step"] = int(best_valid_step)
    log_metrics["router_overfit/steps_since_best_valid"] = int(step - best_valid_step)


def _encode_batch(vae_encode, key, dataset_name: str, batch_images):
    if "latent" in dataset_name:
        if batch_images.shape[-1] % 2 == 0 and batch_images.shape[-1] > 4:
            batch_images = batch_images[..., batch_images.shape[-1] // 2 :]
        return jnp.asarray(batch_images, dtype=jnp.float32)
    return vae_encode(key, batch_images)


def _router_inputs(mode: str, mix_x1_prob: float, key, gmm_state, x1):
    x0_key, mix_key = jax.random.split(key)
    x0, _, _, _ = sample_prior_components(x0_key, gmm_state, x1.shape[0], x1.shape[1:])
    if mode == "x1":
        return x1
    if mode == "x0":
        return x0
    if mode == "mix":
        keep_x1 = jax.random.bernoulli(
            mix_key,
            p=jnp.asarray(mix_x1_prob, dtype=jnp.float32),
            shape=(x1.shape[0],) + (1,) * (x1.ndim - 1),
        )
        return jnp.where(keep_x1, x1, x0)
    raise ValueError(f"Unknown router_train_data_mode {mode}")


def main(_):
    if not FLAGS.gmm_stats_path:
        raise ValueError("--gmm_stats_path is required")
    if FLAGS.router_train_data_mode not in ("x1", "x0", "mix"):
        raise ValueError("--router_train_data_mode must be x1, x0, or mix")
    if FLAGS.router_target_type not in ("soft_kl", "hard_ce"):
        raise ValueError("--router_target_type must be soft_kl or hard_ce")

    np.random.seed(FLAGS.seed)
    rng = jax.random.PRNGKey(FLAGS.seed)
    gmm_state = load_gmm_stats(FLAGS.gmm_stats_path)
    num_modes = int(gmm_state["pi"].shape[0])

    if jax.process_index() == 0:
        setup_wandb(
            {
                "dataset_name": FLAGS.dataset_name,
                "gmm_stats_path": FLAGS.gmm_stats_path,
                "router_train_data_mode": FLAGS.router_train_data_mode,
                "router_mix_x1_prob": FLAGS.router_mix_x1_prob,
                "router_target_type": FLAGS.router_target_type,
                "router_max_steps": FLAGS.router_max_steps,
                "router_lr": FLAGS.router_lr,
                "router_weight_decay": FLAGS.router_weight_decay,
                "router_hidden_channels": FLAGS.router_hidden_channels,
                "router_mlp_hidden_size": FLAGS.router_mlp_hidden_size,
                "router_depth": FLAGS.router_depth,
                "num_modes": num_modes,
            },
            **FLAGS.wandb,
        )
        if FLAGS.metrics_output_path:
            os.makedirs(os.path.dirname(FLAGS.metrics_output_path) or ".", exist_ok=True)
            open(FLAGS.metrics_output_path, "w", encoding="utf-8").close()
            clear_metrics_csv(FLAGS.metrics_output_path)

    dataset = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        True,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )
    dataset_valid = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        False,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )

    vae_encode = None
    if "latent" not in FLAGS.dataset_name:
        vae = StableVAE.create()
        vae_encode = jax.jit(vae.encode)

    example_images, _ = next(dataset)
    rng, encode_key = jax.random.split(rng)
    example_latents = _encode_batch(vae_encode, encode_key, FLAGS.dataset_name, example_images)
    latent_shape = tuple(example_latents.shape[1:])

    router_def = GMMRouter(
        num_modes=num_modes,
        hidden_channels=FLAGS.router_hidden_channels,
        mlp_hidden_size=FLAGS.router_mlp_hidden_size,
        depth=FLAGS.router_depth,
    )
    rng, init_key = jax.random.split(rng)
    params = router_def.init(init_key, jnp.zeros_like(example_latents), train=True)["params"]
    tx = optax.adamw(learning_rate=FLAGS.router_lr, weight_decay=FLAGS.router_weight_decay)
    opt_state = tx.init(params)

    def batch_loss(params, key, x1):
        x = _router_inputs(FLAGS.router_train_data_mode, FLAGS.router_mix_x1_prob, key, gmm_state, x1)
        q_target, _, _ = posterior_from_stats(gmm_state, flatten_latents(x))
        q_target = jax.lax.stop_gradient(q_target)
        logits, activations = router_def.apply(
            {"params": params},
            x,
            train=True,
            return_activations=True,
        )
        if FLAGS.router_target_type == "hard_ce":
            target_ids = jnp.argmax(q_target, axis=-1)
            loss_vec = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            loss = jnp.mean(loss_vec)
        else:
            log_pred = jax.nn.log_softmax(logits, axis=-1)
            q_safe = jnp.maximum(q_target, 1e-8)
            loss = jnp.mean(jnp.sum(q_target * (jnp.log(q_safe) - log_pred), axis=-1))
        metrics = {
            "loss": loss,
            "input_magnitude": jnp.sqrt(jnp.mean(jnp.square(x))),
            **router_metrics(logits, q_target),
            **{f"activations/{name}": jnp.sqrt(jnp.mean(jnp.square(value))) for name, value in activations.items()},
        }
        return loss, metrics

    @jax.jit
    def update_step(params, opt_state, rng, x1):
        rng, step_key = jax.random.split(rng)
        grads, metrics = jax.grad(batch_loss, has_aux=True)(params, step_key, x1)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        metrics["update_norm"] = optax.global_norm(updates)
        metrics["param_norm"] = optax.global_norm(params)
        return params, opt_state, rng, metrics

    @jax.jit
    def eval_step(params, rng, x1):
        _, metrics = batch_loss(params, rng, x1)
        return metrics

    def next_latents(dataset_iter, key):
        images, _ = next(dataset_iter)
        return _encode_batch(vae_encode, key, FLAGS.dataset_name, images)

    latest_metrics = {}
    best_valid_loss = float("inf")
    best_valid_step = 0
    best_params = params
    for step in range(1, FLAGS.router_max_steps + 1):
        rng, encode_key = jax.random.split(rng)
        x1 = next_latents(dataset, encode_key)
        params, opt_state, rng, metrics = update_step(params, opt_state, rng, x1)

        should_log = step == 1 or step % FLAGS.router_log_interval == 0
        should_valid = step == 1 or step % FLAGS.router_valid_interval == 0 or step == FLAGS.router_max_steps
        if should_log or should_valid:
            metrics_np = _to_float_dict(jax.device_get(metrics))
            log_metrics = {f"router_train/{k}": v for k, v in metrics_np.items()}

            if should_valid:
                valid_rows = []
                for valid_idx in range(max(int(FLAGS.router_valid_batches), 1)):
                    rng, valid_encode_key, valid_key = jax.random.split(rng, 3)
                    x1_valid = next_latents(dataset_valid, valid_encode_key)
                    valid_metrics = eval_step(params, jax.random.fold_in(valid_key, valid_idx), x1_valid)
                    valid_rows.append(_to_float_dict(jax.device_get(valid_metrics)))
                valid_mean = {
                    name: float(np.mean([row[name] for row in valid_rows if name in row]))
                    for name in valid_rows[0]
                }
                log_metrics.update({f"router_valid/{k}": v for k, v in valid_mean.items()})
                valid_loss = float(valid_mean.get("loss", float("inf")))
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_step = int(step)
                    best_params = params
                _add_router_overfit_metrics(
                    log_metrics,
                    metrics_np,
                    valid_mean,
                    best_valid_loss,
                    best_valid_step,
                    step,
                )

            latest_metrics = {"phase": "router", "step": int(step), **log_metrics}
            if jax.process_index() == 0:
                wandb.log(log_metrics, step=step)
                _append_jsonl(FLAGS.metrics_output_path, latest_metrics)
                append_metrics_csv(FLAGS.metrics_output_path, latest_metrics)
                _write_summary(FLAGS.metrics_output_path, latest_metrics)
                print(json.dumps(latest_metrics, sort_keys=True, default=json_default), flush=True)

    config = {
        "num_modes": num_modes,
        "hidden_channels": int(FLAGS.router_hidden_channels),
        "mlp_hidden_size": int(FLAGS.router_mlp_hidden_size),
        "depth": int(FLAGS.router_depth),
        "dataset_name": FLAGS.dataset_name,
        "latent_shape": list(latent_shape),
        "gmm_stats_path": FLAGS.gmm_stats_path,
        "router_train_data_mode": FLAGS.router_train_data_mode,
        "router_mix_x1_prob": float(FLAGS.router_mix_x1_prob),
        "router_target_type": FLAGS.router_target_type,
        "router_save_best": bool(FLAGS.router_save_best),
        "router_selected_step": int(best_valid_step if FLAGS.router_save_best and best_valid_step else FLAGS.router_max_steps),
        "router_best_valid_loss": float(best_valid_loss),
    }
    if jax.process_index() == 0:
        save_router_checkpoint(FLAGS.router_save_path, best_params if FLAGS.router_save_best else params, config)
        print(f"Saved GMM router to {FLAGS.router_save_path}", flush=True)
        if latest_metrics:
            wandb.summary.update(latest_metrics)


if __name__ == "__main__":
    app.run(main)
