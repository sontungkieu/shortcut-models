from functools import partial
import os
import time
from typing import Any

from absl import app, flags
import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import wandb

from dinov2_model import DINOv2ViT
from utils.checkpoint import Checkpoint
from utils.sharding import create_sharding
from utils.wandb import default_wandb_config, setup_wandb


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_name", "celebahq256", "TFDS dataset name.")
flags.DEFINE_string("tfds_data_dir", None, "TFDS data directory, e.g. Kaggle input path.")
flags.DEFINE_string("load_dir", None, "Checkpoint path to resume from.")
flags.DEFINE_string("save_dir", None, "Directory for step checkpoints.")
flags.DEFINE_integer("seed", 10, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Global batch size.")
flags.DEFINE_integer("max_steps", 100000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "DINO eval interval. Set <=0 to disable.")
flags.DEFINE_integer("eval_batches", 1, "Number of deterministic eval batches to average.")
flags.DEFINE_integer("demo_interval", 5000, "DINO demo image interval. Set <=0 to disable.")
flags.DEFINE_integer("demo_samples", 4, "Number of eval samples to render in DINO demos.")
flags.DEFINE_integer("save_interval", 5000, "Checkpoint interval.")
flags.DEFINE_integer("debug_overfit", 0, "Repeat a tiny input slice.")
flags.DEFINE_string(
    "fid_stats",
    None,
    "Unsupported for DINOv2 encoder training. Accepted only to make FID misuse explicit.",
)

model_config = ml_collections.ConfigDict(
    {
        "model_size": "vit_s",
        "image_size": 224,
        "local_crop_size": 98,
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "num_register_tokens": 4,
        "out_dim": 8192,
        "patch_out_dim": 8192,
        "global_crops": 2,
        "local_crops": 8,
        "global_scale_min": 0.4,
        "global_scale_max": 1.0,
        "local_scale_min": 0.05,
        "local_scale_max": 0.4,
        "mask_ratio": 0.4,
        "teacher_temp": 0.04,
        "student_temp": 0.1,
        "center_momentum": 0.9,
        "teacher_momentum": 0.996,
        "teacher_momentum_final": 1.0,
        "lr": 5e-4,
        "min_lr": 1e-6,
        "warmup_steps": 10000,
        "weight_decay": 0.04,
        "weight_decay_final": 0.4,
        "beta1": 0.9,
        "beta2": 0.95,
        "clip_grad_norm": 3.0,
        "ibot_loss_weight": 1.0,
        "sharding": "dp",
    }
)

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "shortcut",
        "name": "dinov2_jax_{dataset_name}",
    }
)

config_flags.DEFINE_config_dict("model", model_config, lock_config=False)
config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)


nonpytree_field = partial(flax.struct.field, pytree_node=False)


class DINOTrainState(flax.struct.PyTreeNode):
    rng: Any
    step: jnp.ndarray
    student_params: Any
    teacher_params: Any
    opt_state: Any
    center: jnp.ndarray
    patch_center: jnp.ndarray
    tx: Any = nonpytree_field()

    def save(self):
        return {
            "rng": self.rng,
            "step": self.step,
            "student_params": self.student_params,
            "teacher_params": self.teacher_params,
            "opt_state": self.opt_state,
            "center": self.center,
            "patch_center": self.patch_center,
        }

    def load(self, data):
        return self.replace(**data)


def _random_resized_crop(image, size, scale_min, scale_max):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    min_side = tf.minimum(height, width)
    scale = tf.random.uniform([], scale_min, scale_max)
    crop_side = tf.cast(tf.sqrt(scale) * tf.cast(min_side, tf.float32), tf.int32)
    crop_side = tf.maximum(crop_side, 1)
    offset_y = tf.random.uniform([], 0, height - crop_side + 1, dtype=tf.int32)
    offset_x = tf.random.uniform([], 0, width - crop_side + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_side, crop_side)
    image = tf.image.resize(image, (size, size), antialias=True)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _fixed_resized_crop(image, size, scale, offset_y_frac=0.5, offset_x_frac=0.5, flip=False):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    min_side = tf.minimum(height, width)
    crop_side = tf.cast(tf.sqrt(tf.constant(scale, dtype=tf.float32)) * tf.cast(min_side, tf.float32), tf.int32)
    crop_side = tf.clip_by_value(crop_side, 1, min_side)
    max_y = height - crop_side
    max_x = width - crop_side
    offset_y = tf.cast(tf.round(tf.cast(max_y, tf.float32) * offset_y_frac), tf.int32)
    offset_x = tf.cast(tf.round(tf.cast(max_x, tf.float32) * offset_x_frac), tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_side, crop_side)
    image = tf.image.resize(image, (size, size), antialias=True)
    if flip:
        image = tf.image.flip_left_right(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _normalize_image(image):
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return (image - mean) / std


def get_dinov2_dataset(dataset_name, batch_size, tfds_data_dir, config, debug_overfit=False, train=True):
    if dataset_name != "celebahq256":
        raise ValueError(f"Unsupported DINO dataset: {dataset_name}")

    def train_map_fn(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        global_crops = [
            _normalize_image(
                _random_resized_crop(
                    image,
                    config.image_size,
                    config.global_scale_min,
                    config.global_scale_max,
                )
            )
            for _ in range(config.global_crops)
        ]
        local_crops = [
            _normalize_image(
                _random_resized_crop(
                    image,
                    config.local_crop_size,
                    config.local_scale_min,
                    config.local_scale_max,
                )
            )
            for _ in range(config.local_crops)
        ]
        global_crops = tf.stack(global_crops, axis=0)
        local_crops = tf.stack(local_crops, axis=0)
        global_crops.set_shape((config.global_crops, config.image_size, config.image_size, 3))
        local_crops.set_shape((config.local_crops, config.local_crop_size, config.local_crop_size, 3))
        return {"global_crops": global_crops, "local_crops": local_crops}

    def eval_map_fn(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        global_scales = [
            1.0,
            max(float(config.global_scale_min), min(float(config.global_scale_max), 0.75)),
        ]
        global_crops = []
        for crop_idx in range(config.global_crops):
            scale = global_scales[crop_idx % len(global_scales)]
            global_crops.append(
                _normalize_image(
                    _fixed_resized_crop(
                        image,
                        config.image_size,
                        scale,
                        offset_y_frac=0.5,
                        offset_x_frac=0.5,
                        flip=bool(crop_idx % 2),
                    )
                )
            )

        anchors = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
        local_scale = 0.5 * (float(config.local_scale_min) + float(config.local_scale_max))
        local_crops = []
        for crop_idx in range(config.local_crops):
            offset_y_frac, offset_x_frac = anchors[crop_idx % len(anchors)]
            local_crops.append(
                _normalize_image(
                    _fixed_resized_crop(
                        image,
                        config.local_crop_size,
                        local_scale,
                        offset_y_frac=offset_y_frac,
                        offset_x_frac=offset_x_frac,
                        flip=bool(crop_idx % 2),
                    )
                )
            )

        global_crops = tf.stack(global_crops, axis=0)
        local_crops = tf.stack(local_crops, axis=0)
        global_crops.set_shape((config.global_crops, config.image_size, config.image_size, 3))
        local_crops.set_shape((config.local_crops, config.local_crop_size, config.local_crop_size, 3))
        return {"global_crops": global_crops, "local_crops": local_crops}

    split = tfds.split_for_jax_process("train", drop_remainder=True)
    dataset = tfds.load(dataset_name, split=split, data_dir=tfds_data_dir)
    dataset = dataset.map(train_map_fn if train else eval_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if debug_overfit:
        dataset = dataset.take(8).repeat()
    elif train:
        dataset = dataset.shuffle(10000, seed=42 + jax.process_index(), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return iter(tfds.as_numpy(dataset))


def make_model(config):
    if config.model_size != "vit_s":
        raise ValueError("Only model.model_size=vit_s is configured for this TPU v5e8 recipe.")
    return DINOv2ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_register_tokens=config.num_register_tokens,
        out_dim=config.out_dim,
        patch_out_dim=config.patch_out_dim,
    )


def make_schedules(config, max_steps):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=max_steps,
        end_value=config.min_lr,
    )
    wd_schedule = optax.cosine_decay_schedule(
        init_value=config.weight_decay,
        decay_steps=max_steps,
        alpha=config.weight_decay_final / config.weight_decay,
    )

    def teacher_momentum(step):
        step = jnp.minimum(step, max_steps)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * step / max_steps))
        return config.teacher_momentum_final - (
            config.teacher_momentum_final - config.teacher_momentum
        ) * cosine

    return lr_schedule, wd_schedule, teacher_momentum


def make_masks(key, global_crops, mask_ratio, patch_size):
    batch_size = global_crops.shape[0]
    num_global = global_crops.shape[1]
    grid = global_crops.shape[2] // patch_size
    num_patches = grid * grid
    masks = jax.random.uniform(key, (num_global, batch_size, num_patches)) < mask_ratio
    return masks


def dino_loss(teacher_logits, student_global_logits, student_local_logits, center, config):
    teacher_probs = jax.nn.softmax(
        (jax.lax.stop_gradient(teacher_logits) - center) / config.teacher_temp,
        axis=-1,
    )
    losses = []
    num_teacher = teacher_logits.shape[0]
    num_student_global = student_global_logits.shape[0]
    num_student_local = student_local_logits.shape[0]

    for teacher_idx in range(num_teacher):
        target = teacher_probs[teacher_idx]
        for student_idx in range(num_student_global):
            if student_idx == teacher_idx:
                continue
            log_probs = jax.nn.log_softmax(student_global_logits[student_idx] / config.student_temp, axis=-1)
            losses.append(-jnp.sum(target * log_probs, axis=-1).mean())
        for student_idx in range(num_student_local):
            log_probs = jax.nn.log_softmax(student_local_logits[student_idx] / config.student_temp, axis=-1)
            losses.append(-jnp.sum(target * log_probs, axis=-1).mean())

    return sum(losses) / len(losses)


def ibot_loss(teacher_patch_logits, student_patch_logits, masks, patch_center, config):
    teacher_probs = jax.nn.softmax(
        (jax.lax.stop_gradient(teacher_patch_logits) - patch_center) / config.teacher_temp,
        axis=-1,
    )
    log_probs = jax.nn.log_softmax(student_patch_logits / config.student_temp, axis=-1)
    token_loss = -jnp.sum(teacher_probs * log_probs, axis=-1)
    masks = masks.astype(jnp.float32)
    return jnp.sum(token_loss * masks) / jnp.maximum(jnp.sum(masks), 1.0)


def _denormalize_dino_images(images):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.clip(images * std + mean, 0.0, 1.0)


def _patch_activity_overlay(image, patch_activity):
    patch_count = patch_activity.shape[0]
    grid = int(np.sqrt(patch_count))
    if grid * grid != patch_count:
        return None
    heatmap = patch_activity.reshape(grid, grid)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    repeat_y = max(1, int(np.ceil(image.shape[0] / grid)))
    repeat_x = max(1, int(np.ceil(image.shape[1] / grid)))
    heatmap = np.repeat(np.repeat(heatmap, repeat_y, axis=0), repeat_x, axis=1)
    heatmap = heatmap[: image.shape[0], : image.shape[1]]
    heatmap_rgb = plt.get_cmap("magma")(heatmap)[..., :3]
    return np.clip(0.55 * image + 0.45 * heatmap_rgb, 0.0, 1.0)


def log_dinov2_demo(batch, patch_activity, step, config, save_dir, demo_samples):
    global_crops = _denormalize_dino_images(np.asarray(batch["global_crops"]))
    local_crops = _denormalize_dino_images(np.asarray(batch["local_crops"]))
    patch_activity = None if patch_activity is None else np.asarray(patch_activity)
    num_samples = min(int(demo_samples), global_crops.shape[0])
    num_global = min(config.global_crops, global_crops.shape[1])
    num_local = min(2, config.local_crops, local_crops.shape[1])
    has_heatmap = patch_activity is not None and patch_activity.size > 0 and patch_activity.shape[0] >= num_samples
    num_cols = num_global + num_local + int(has_heatmap)
    if num_samples == 0 or num_cols == 0:
        return

    fig, axs = plt.subplots(num_samples, num_cols, figsize=(3.0 * num_cols, 3.0 * num_samples), squeeze=False)
    for row in range(num_samples):
        col = 0
        for crop_idx in range(num_global):
            axs[row, col].imshow(global_crops[row, crop_idx], vmin=0, vmax=1)
            axs[row, col].set_title(f"global {crop_idx}")
            axs[row, col].axis("off")
            col += 1
        for crop_idx in range(num_local):
            axs[row, col].imshow(local_crops[row, crop_idx], vmin=0, vmax=1)
            axs[row, col].set_title(f"local {crop_idx}")
            axs[row, col].axis("off")
            col += 1
        if has_heatmap:
            overlay = _patch_activity_overlay(global_crops[row, 0], patch_activity[row])
            if overlay is None:
                axs[row, col].imshow(global_crops[row, 0], vmin=0, vmax=1)
            else:
                axs[row, col].imshow(overlay, vmin=0, vmax=1)
            axs[row, col].set_title("patch activity")
            axs[row, col].axis("off")

    fig.tight_layout()
    if save_dir is not None:
        demo_dir = os.path.join(save_dir, "demos")
        os.makedirs(demo_dir, exist_ok=True)
        fig.savefig(os.path.join(demo_dir, f"dinov2_demo_step_{step:08d}.png"), dpi=150)
    if wandb.run is not None:
        wandb.log({"dinov2/demo_crops_patch_activity": wandb.Image(fig)}, step=step)
    plt.close(fig)


def main(_):
    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch:", FLAGS.batch_size)
    print("Node Batch:", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)
        if FLAGS.fid_stats is not None:
            print(
                "DINOv2 is an encoder-only self-supervised model; --fid_stats is ignored. "
                "Use train.py/helper_eval.py for generator FID."
            )

    dataset = get_dinov2_dataset(
        FLAGS.dataset_name,
        local_batch_size,
        FLAGS.tfds_data_dir,
        FLAGS.model,
        FLAGS.debug_overfit,
        train=True,
    )
    eval_dataset = None
    if FLAGS.eval_interval > 0 or FLAGS.demo_interval > 0:
        eval_dataset = get_dinov2_dataset(
            FLAGS.dataset_name,
            local_batch_size,
            FLAGS.tfds_data_dir,
            FLAGS.model,
            FLAGS.debug_overfit,
            train=False,
        )
    example_batch = next(dataset)
    model_def = make_model(FLAGS.model)
    lr_schedule, wd_schedule, teacher_momentum_schedule = make_schedules(FLAGS.model, FLAGS.max_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.model.clip_grad_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            weight_decay=wd_schedule,
            b1=FLAGS.model.beta1,
            b2=FLAGS.model.beta2,
        ),
    )

    def init_state(rng):
        init_key, rng = jax.random.split(rng)
        example_images = jnp.zeros(
            (1, FLAGS.model.image_size, FLAGS.model.image_size, 3),
            dtype=jnp.float32,
        )
        example_mask = jnp.zeros(
            (1, (FLAGS.model.image_size // FLAGS.model.patch_size) ** 2),
            dtype=bool,
        )
        params = model_def.init(init_key, example_images, example_mask, True)["params"]
        opt_state = tx.init(params)
        return DINOTrainState(
            rng=rng,
            step=jnp.array(1, dtype=jnp.int32),
            student_params=params,
            teacher_params=params,
            opt_state=opt_state,
            center=jnp.zeros((1, FLAGS.model.out_dim), dtype=jnp.float32),
            patch_center=jnp.zeros((1, 1, FLAGS.model.patch_out_dim), dtype=jnp.float32),
            tx=tx,
        )

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init_state, rng)
    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(
        FLAGS.model.sharding,
        train_state_shape,
    )
    train_state = jax.jit(init_state, out_shardings=train_state_sharding)(rng)
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        train_state = train_state.load(cp.load_as_dict()["train_state"])
        train_state = jax.jit(lambda x: x, out_shardings=train_state_sharding)(train_state)
        start_step = int(jax.device_get(train_state.step))
        print("Loaded DINO checkpoint at step", start_step)

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, global_crops, local_crops):
        rng, mask_key = jax.random.split(train_state.rng)
        global_crops = jax.lax.with_sharding_constraint(global_crops, data_sharding)
        local_crops = jax.lax.with_sharding_constraint(local_crops, data_sharding)
        masks = make_masks(mask_key, global_crops, FLAGS.model.mask_ratio, FLAGS.model.patch_size)

        global_images = jnp.swapaxes(global_crops, 0, 1)
        local_images = jnp.swapaxes(local_crops, 0, 1)

        def teacher_apply(images):
            return model_def.apply(
                {"params": train_state.teacher_params},
                images,
                None,
                True,
            )

        teacher_cls, teacher_patch = jax.vmap(teacher_apply)(global_images)
        teacher_cls = jax.lax.stop_gradient(teacher_cls)
        teacher_patch = jax.lax.stop_gradient(teacher_patch)

        def loss_fn(student_params):
            def student_global_apply(images, mask):
                return model_def.apply(
                    {"params": student_params},
                    images,
                    mask,
                    True,
                )

            def student_local_apply(images):
                return model_def.apply(
                    {"params": student_params},
                    images,
                    None,
                    False,
                )

            student_cls, student_patch = jax.vmap(student_global_apply)(global_images, masks)
            student_local_cls = jax.vmap(student_local_apply)(local_images)

            loss_dino = dino_loss(
                teacher_cls,
                student_cls,
                student_local_cls,
                train_state.center,
                FLAGS.model,
            )
            loss_ibot = ibot_loss(
                teacher_patch,
                student_patch,
                masks,
                train_state.patch_center,
                FLAGS.model,
            )
            loss = loss_dino + FLAGS.model.ibot_loss_weight * loss_ibot
            info = {
                "loss": loss,
                "loss_dino": loss_dino,
                "loss_ibot": loss_ibot,
                "teacher_entropy": -jnp.mean(
                    jnp.sum(
                        jax.nn.softmax((teacher_cls - train_state.center) / FLAGS.model.teacher_temp, axis=-1)
                        * jax.nn.log_softmax((teacher_cls - train_state.center) / FLAGS.model.teacher_temp, axis=-1),
                        axis=-1,
                    )
                ),
            }
            return loss, info

        grads, info = jax.grad(loss_fn, has_aux=True)(train_state.student_params)
        updates, new_opt_state = train_state.tx.update(
            grads,
            train_state.opt_state,
            train_state.student_params,
        )
        new_student_params = optax.apply_updates(train_state.student_params, updates)
        momentum = teacher_momentum_schedule(train_state.step)
        new_teacher_params = jax.tree_util.tree_map(
            lambda student, teacher: teacher * momentum + student * (1.0 - momentum),
            new_student_params,
            train_state.teacher_params,
        )
        batch_center = jnp.mean(teacher_cls, axis=(0, 1))[None, :]
        batch_patch_center = jnp.mean(teacher_patch, axis=(0, 1, 2))[None, None, :]
        new_center = train_state.center * FLAGS.model.center_momentum + batch_center * (
            1.0 - FLAGS.model.center_momentum
        )
        new_patch_center = train_state.patch_center * FLAGS.model.center_momentum + batch_patch_center * (
            1.0 - FLAGS.model.center_momentum
        )
        new_state = train_state.replace(
            rng=rng,
            step=train_state.step + 1,
            student_params=new_student_params,
            teacher_params=new_teacher_params,
            opt_state=new_opt_state,
            center=new_center,
            patch_center=new_patch_center,
        )
        info = {
            **info,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(new_student_params),
            "lr": lr_schedule(train_state.step),
            "weight_decay": wd_schedule(train_state.step),
            "teacher_momentum": momentum,
            "mask_ratio": jnp.mean(masks.astype(jnp.float32)),
        }
        return new_state, info

    @partial(jax.jit, out_shardings=no_shard)
    def eval_step(train_state, global_crops, local_crops, mask_key):
        global_crops = jax.lax.with_sharding_constraint(global_crops, data_sharding)
        local_crops = jax.lax.with_sharding_constraint(local_crops, data_sharding)
        masks = make_masks(mask_key, global_crops, FLAGS.model.mask_ratio, FLAGS.model.patch_size)

        global_images = jnp.swapaxes(global_crops, 0, 1)
        local_images = jnp.swapaxes(local_crops, 0, 1)

        def teacher_apply(images):
            return model_def.apply(
                {"params": train_state.teacher_params},
                images,
                None,
                True,
            )

        teacher_cls, teacher_patch = jax.vmap(teacher_apply)(global_images)
        teacher_cls = jax.lax.stop_gradient(teacher_cls)
        teacher_patch = jax.lax.stop_gradient(teacher_patch)

        def student_global_apply(images, mask):
            return model_def.apply(
                {"params": train_state.student_params},
                images,
                mask,
                True,
            )

        def student_local_apply(images):
            return model_def.apply(
                {"params": train_state.student_params},
                images,
                None,
                False,
            )

        student_cls, student_patch = jax.vmap(student_global_apply)(global_images, masks)
        student_local_cls = jax.vmap(student_local_apply)(local_images)
        loss_dino = dino_loss(
            teacher_cls,
            student_cls,
            student_local_cls,
            train_state.center,
            FLAGS.model,
        )
        loss_ibot = ibot_loss(
            teacher_patch,
            student_patch,
            masks,
            train_state.patch_center,
            FLAGS.model,
        )
        loss = loss_dino + FLAGS.model.ibot_loss_weight * loss_ibot

        teacher_probs = jax.nn.softmax((teacher_cls - train_state.center) / FLAGS.model.teacher_temp, axis=-1)
        teacher_entropy = -jnp.mean(jnp.sum(teacher_probs * jnp.log(jnp.maximum(teacher_probs, 1e-8)), axis=-1))
        teacher_repr = teacher_cls.astype(jnp.float32)
        teacher_normed = teacher_repr / jnp.maximum(jnp.linalg.norm(teacher_repr, axis=-1, keepdims=True), 1e-6)
        view0 = teacher_normed[0]
        view1 = teacher_normed[1] if teacher_normed.shape[0] > 1 else teacher_normed[0]
        positive_cosine = jnp.mean(jnp.sum(view0 * view1, axis=-1))
        similarity = view0 @ view0.T
        offdiag_denominator = max(view0.shape[0] * (view0.shape[0] - 1), 1)
        offdiag_cosine = (jnp.sum(similarity) - jnp.trace(similarity)) / offdiag_denominator
        feature_std = jnp.mean(jnp.std(view0, axis=0))

        return {
            "loss": loss,
            "loss_dino": loss_dino,
            "loss_ibot": loss_ibot,
            "teacher_entropy": teacher_entropy,
            "positive_view_cosine": positive_cosine,
            "offdiag_view_cosine": offdiag_cosine,
            "feature_std": feature_std,
            "mask_ratio": jnp.mean(masks.astype(jnp.float32)),
        }

    @partial(jax.jit, out_shardings=data_sharding)
    def encode_patch_activity(train_state, global_crops):
        global_crops = jax.lax.with_sharding_constraint(global_crops, data_sharding)
        first_global = jnp.swapaxes(global_crops, 0, 1)[0]
        _, patch_logits = model_def.apply(
            {"params": train_state.teacher_params},
            first_global,
            None,
            True,
        )
        return jnp.sqrt(jnp.mean(jnp.square(patch_logits.astype(jnp.float32)), axis=-1))

    last_log_time = time.time()
    last_log_step = start_step - 1
    for step in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        batch = next(dataset)
        global_crops = shard_data(batch["global_crops"])
        local_crops = shard_data(batch["local_crops"])
        train_state, info = update(train_state, global_crops, local_crops)

        if step == 1 or step % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            metrics = {f"dinov2/{k}": float(np.asarray(v).mean()) for k, v in info.items()}
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            steps_elapsed = max(step - last_log_step, 1)
            metrics["dinov2/steps_per_sec"] = steps_elapsed / elapsed
            last_log_time = now
            last_log_step = step
            if jax.process_index() == 0:
                wandb.log(metrics, step=step)

        demo_batch = None
        should_eval = FLAGS.eval_interval > 0 and (step == 1 or step % FLAGS.eval_interval == 0)
        if should_eval and eval_dataset is not None:
            eval_infos = []
            for eval_idx in range(max(1, FLAGS.eval_batches)):
                eval_batch = next(eval_dataset)
                if demo_batch is None:
                    demo_batch = eval_batch
                eval_global_crops = shard_data(eval_batch["global_crops"])
                eval_local_crops = shard_data(eval_batch["local_crops"])
                mask_key = jax.random.fold_in(jax.random.PRNGKey(FLAGS.seed + 1000), step * 1000 + eval_idx)
                eval_info = eval_step(train_state, eval_global_crops, eval_local_crops, mask_key)
                eval_info = jax.device_get(eval_info)
                eval_infos.append({k: float(np.asarray(v).mean()) for k, v in eval_info.items()})
            eval_metrics = {
                f"dinov2/eval/{key}": float(np.mean([info[key] for info in eval_infos]))
                for key in eval_infos[0]
            }
            if jax.process_index() == 0:
                wandb.log(eval_metrics, step=step)

        should_demo = FLAGS.demo_interval > 0 and (step == 1 or step % FLAGS.demo_interval == 0)
        if should_demo and eval_dataset is not None:
            if demo_batch is None:
                demo_batch = next(eval_dataset)
            demo_global_crops = shard_data(demo_batch["global_crops"])
            patch_activity = encode_patch_activity(train_state, demo_global_crops)
            patch_activity = jax.device_get(global_to_local(patch_activity))
            patch_activity = np.asarray(patch_activity).reshape((-1, patch_activity.shape[-1]))
            if jax.process_index() == 0:
                log_dinov2_demo(
                    demo_batch,
                    patch_activity,
                    step,
                    FLAGS.model,
                    FLAGS.save_dir,
                    FLAGS.demo_samples,
                )

        if FLAGS.save_dir is not None and step % FLAGS.save_interval == 0:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(f"{FLAGS.save_dir}/step_{step:08d}", parallel=False)
                cp.train_state = train_state_gather
                cp.save()
                del cp
            del train_state_gather


if __name__ == "__main__":
    app.run(main)
