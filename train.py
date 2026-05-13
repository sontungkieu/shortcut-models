from typing import Any
import json
import os
from pathlib import Path
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from model import DiT
from gmm_router import load_router_state
from gmm_utils import load_gmm_stats, json_default
from helper_eval import eval_model
from helper_inference import do_inference
from metrics_io import append_metrics_csv

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_string('tfds_data_dir', None, 'Optional TFDS data_dir.')
flags.DEFINE_string('metrics_output_path', None, 'Optional JSONL path for lightweight train/eval diagnostics.')
flags.DEFINE_string('eval_fid_timesteps', '1,4,32', 'Comma-separated FID timestep list.')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('reset_step_on_load', 1, 'Reset optimizer/train step to zero after loading a checkpoint, as 1/0.')
flags.DEFINE_integer('batch_size', 32, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('mode', 'train', 'train or inference.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 768, # change this!
    'patch_size': 8, # change this!
    'depth': 2, # change this!
    'num_heads': 2, # change this!
    'mlp_ratio': 1, # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 1,
    'use_stable_vae': 1,
    'sharding': 'dp', # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 4, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut', # shortcut, naive, naive-gaussian, or gmm-tide.
    'gmm_stats_path': '',
    'gmm_cond_channels': 64,
    'gmm_router_path': '',
    'gmm_router_topk': 4,
    'gmm_router_temperature': 1.0,
    'gmm_router_update_policy': 'frozen',
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'shortcut_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)


def _append_metrics_jsonl(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default))
        f.write('\n')


def _write_summary_json(path, payload):
    if path is None:
        return
    if path.endswith('.jsonl'):
        summary_path = path[:-6] + '_summary.json'
    else:
        summary_path = path + '.summary.json'
    os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write('\n')


def _to_float_dict(metrics):
    out = {}
    for k, v in metrics.items():
        arr = np.asarray(v)
        if arr.shape == ():
            out[k] = float(arr)
    return out
    
##############################################
## Training Code.
##############################################
def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0 and FLAGS.mode == 'train':
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)
        
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit, data_dir=FLAGS.tfds_data_dir)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit, data_dir=FLAGS.tfds_data_dir)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    vae = None
    vae_encode = None
    vae_decode = None
    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        if 'latent' in FLAGS.dataset_name:
            example_obs = example_obs[:, :, :, example_obs.shape[-1] // 2:]
            example_obs_shape = example_obs.shape
        else:
            example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape = example_obs.shape
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = jax.jit(vae.encode)
        vae_decode = jax.jit(vae.decode)

    gmm_state = None
    router_state = None
    if FLAGS.model.train_type in ('naive', 'gmm-tide'):
        if not FLAGS.model.gmm_stats_path:
            raise ValueError(f'--model.train_type {FLAGS.model.train_type} requires --model.gmm_stats_path')
        gmm_state = load_gmm_stats(FLAGS.model.gmm_stats_path)
        print(f"Loaded GMM stats from {FLAGS.model.gmm_stats_path}")
    if FLAGS.model.train_type == 'gmm-tide':
        if not FLAGS.model.gmm_router_path:
            raise ValueError('--model.train_type gmm-tide requires --model.gmm_router_path')
        if FLAGS.model.gmm_router_update_policy != 'frozen':
            raise NotImplementedError('V1 gmm-tide supports only --model.gmm_router_update_policy frozen')
        if FLAGS.model.gmm_router_topk <= 0:
            raise ValueError('--model.gmm_router_topk must be positive')
        router_state = load_router_state(FLAGS.model.gmm_router_path)
        print(f"Loaded GMM router from {FLAGS.model.gmm_router_path}")

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network() 
        truth_fid_stats = np.load(FLAGS.fid_stats)
    else:
        get_fid_activations = None
        truth_fid_stats = None

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow')) else True,
        'gmm_cond_channels': FLAGS.model['gmm_cond_channels'],
    }
    model_def = DiT(**dit_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    if FLAGS.model.train_type in ('naive', 'gmm-tide'):
        print(tabulate_fn(
            example_obs,
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            jnp.zeros((1,), dtype=jnp.int32),
            gmm_mu=jnp.zeros(example_obs_shape, dtype=example_obs.dtype),
            gmm_sigma=jnp.ones(example_obs_shape, dtype=example_obs.dtype),
        ))
    else:
        print(tabulate_fn(example_obs, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)))

    if FLAGS.model.use_cosine:
        lr_schedule = optax.warmup_cosine_decay_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'], FLAGS.max_steps)
    elif FLAGS.model.warmup > 0:
        lr_schedule = optax.linear_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    else:
        lr_schedule = lambda x: FLAGS.model['lr']
    adam = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
    tx = optax.chain(adam)
    
    def init(rng):
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key, 'label_dropout': dropout_key, 'dropout': dropout2_key}
        if FLAGS.model.train_type in ('naive', 'gmm-tide'):
            params = model_def.init(
                model_rngs,
                example_obs,
                example_t,
                example_dt,
                example_label,
                gmm_mu=jnp.zeros_like(example_obs),
                gmm_sigma=jnp.ones_like(example_obs),
            )['params']
        else:
            params = model_def.init(model_rngs, example_obs, example_t, example_dt, example_label)['params']
        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
    jax.debug.visualize_array_sharding(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    jax.experimental.multihost_utils.assert_equal(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state'] # Debug

        def strip_process_axis(loaded, target):
            loaded = jax.device_get(loaded)
            if not hasattr(loaded, 'shape') or not hasattr(target, 'shape'):
                return loaded
            loaded_arr = np.asarray(loaded)
            target_shape = tuple(np.shape(target))
            if loaded_arr.shape == (1,) + target_shape:
                return loaded_arr[0]
            if loaded_arr.shape == (1,) and target_shape == ():
                return loaded_arr.reshape(()).item()
            return loaded

        if 'params' in replace_dict:
            replace_dict['params'] = jax.tree_map(strip_process_axis, replace_dict['params'], train_state.params)
        if 'params_ema' in replace_dict:
            replace_dict['params_ema'] = jax.tree_map(strip_process_axis, replace_dict['params_ema'], train_state.params_ema)
        if 'step' in replace_dict:
            step_arr = np.asarray(jax.device_get(replace_dict['step']))
            if step_arr.shape == (1,):
                replace_dict['step'] = int(step_arr[0])

        train_state = train_state.replace(**replace_dict)
        loaded_step = int(jax.device_get(train_state.step))
        if FLAGS.wandb.run_id != "None" or not bool(FLAGS.reset_step_on_load): # If we are continuing a run.
            start_step = loaded_step
        train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        if bool(FLAGS.reset_step_on_load):
            train_state = train_state.replace(step=0)
        del cp

    if FLAGS.model.train_type == 'progressive' or FLAGS.model.train_type == 'consistency-distillation':
        train_state_teacher = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
    else:
        train_state_teacher = None

    visualize_labels = example_labels
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(visualize_labels)
    imagenet_labels_path = Path('data/imagenet_labels.txt')
    if imagenet_labels_path.exists():
        imagenet_labels = imagenet_labels_path.read_text(encoding='utf-8').splitlines()
    else:
        imagenet_labels = [str(i) for i in range(int(FLAGS.model.num_classes) + 1)]

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, train_state_teacher, images, labels, force_t=-1, force_dt=-1):
        new_rng, targets_key, dropout_key, perm_key = jax.random.split(train_state.rng, 4)
        info = {}

        id_perm = jax.random.permutation(perm_key, images.shape[0])
        images = images[id_perm]
        labels = labels[id_perm]
        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0: # For unconditional generation.
            labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        gmm_mu = None
        gmm_sigma = None

        if FLAGS.model['train_type'] == 'naive':
            from baselines.targets_naive import get_targets
            x_t, v_t, t, dt_base, labels, info, gmm_mu, gmm_sigma = get_targets(
                FLAGS,
                targets_key,
                train_state,
                images,
                labels,
                force_t,
                force_dt,
                gmm_state=gmm_state,
            )
        elif FLAGS.model['train_type'] == 'gmm-tide':
            from baselines.targets_gmm_tide import get_targets
            x_t, v_t, t, dt_base, labels, info, gmm_mu, gmm_sigma = get_targets(
                FLAGS,
                targets_key,
                train_state,
                images,
                labels,
                force_t,
                force_dt,
                gmm_state=gmm_state,
                router_state=router_state,
            )
        elif FLAGS.model['train_type'] == 'naive-gaussian':
            from baselines.targets_naive_gaussian import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'shortcut':
            from targets_shortcut import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'progressive':
            from baselines.targets_progressive import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'consistency-distillation':
            from baselines.targets_consistency_distillation import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'consistency':
            from baselines.targets_consistency_training import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'livereflow':
            from baselines.targets_livereflow import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)

        def loss_fn(grad_params):
            v_prime, logvars, activations = train_state.call_model(
                x_t,
                t,
                dt_base,
                labels,
                train=True,
                rngs={'dropout': dropout_key},
                params=grad_params,
                return_activations=True,
                gmm_mu=gmm_mu,
                gmm_sigma=gmm_sigma,
            )
            residual = v_prime - v_t
            mse_v = jnp.mean(residual ** 2, axis=(1, 2, 3))
            loss = jnp.mean(mse_v)
            residual_mean = jnp.mean(residual, axis=0)
            residual_mean_sq = jnp.mean(jnp.square(residual_mean))
            residual_variance = jnp.mean(jnp.var(residual, axis=0))
            target_mean = jnp.mean(v_t, axis=0)
            pred_mean = jnp.mean(v_prime, axis=0)

            info = {
                'loss': loss,
                'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                'fm/loss_residual_variance': residual_variance,
                'fm/loss_residual_mean_sq': residual_mean_sq,
                'fm/loss_residual_decomp_sum': residual_variance + residual_mean_sq,
                'fm/loss_per_sample_variance': jnp.var(mse_v),
                'fm/loss_per_sample_std': jnp.sqrt(jnp.maximum(jnp.var(mse_v), 0.0)),
                'fm/loss_residual_variance_fraction': residual_variance / jnp.maximum(loss, 1e-8),
                'fm/loss_residual_mean_sq_fraction': residual_mean_sq / jnp.maximum(loss, 1e-8),
                'fm/target_variance': jnp.mean(jnp.var(v_t, axis=0)),
                'fm/target_mean_sq': jnp.mean(jnp.square(target_mean)),
                'fm/target_second_moment': jnp.mean(jnp.square(v_t)),
                'fm/pred_variance': jnp.mean(jnp.var(v_prime, axis=0)),
                'fm/pred_mean_sq': jnp.mean(jnp.square(pred_mean)),
                'fm/pred_second_moment': jnp.mean(jnp.square(v_prime)),
                **{'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
            }

            if FLAGS.model['train_type'] == 'shortcut' or FLAGS.model['train_type'] == 'livereflow':
                bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
                info['loss_flow'] = jnp.mean(mse_v[bootstrap_size:])
                info['loss_bootstrap'] = jnp.mean(mse_v[:bootstrap_size])
            
            return loss, info
        
        grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        info = {**info, **new_info}
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info
    
    if FLAGS.mode != 'train':
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, 
                       fid_from_stats, truth_fid_stats, gmm_state=gmm_state, router_state=router_state)
        return

    ###################################
    # Train Loop
    ###################################

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        # Sample data.
        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = shard_data(*next(dataset))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                vae_rng, vae_key = jax.random.split(vae_rng)
                batch_images = vae_encode(vae_key, batch_images)

        # Train update.
        train_state, update_info = update(train_state, train_state_teacher, batch_images, batch_labels)

        if i % FLAGS.log_interval == 0 or i == 1:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            valid_images, valid_labels = shard_data(*next(dataset_valid))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                valid_images = vae_encode(vae_rng, valid_images)
            _, valid_update_info = update(train_state, train_state_teacher, valid_images, valid_labels)
            valid_update_info = jax.device_get(valid_update_info)
            valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
            train_metrics['training/loss_valid'] = valid_update_info['loss']

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)
                json_metrics = _to_float_dict(train_metrics)
                payload = {'phase': 'train', 'step': int(i), **json_metrics}
                _append_metrics_jsonl(FLAGS.metrics_output_path, payload)
                append_metrics_csv(FLAGS.metrics_output_path, payload)
                _write_summary_json(FLAGS.metrics_output_path, payload)

        if FLAGS.model['train_type'] == 'progressive':
            num_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
            if i % (FLAGS.max_steps // num_sections) == 0:
                train_state_teacher = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)

        if i % FLAGS.eval_interval == 0:
            eval_metrics = eval_model(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, 
                       fid_from_stats, truth_fid_stats, gmm_state=gmm_state, router_state=router_state)
            if jax.process_index() == 0 and eval_metrics:
                payload = {'phase': 'eval', 'step': int(i), **eval_metrics}
                _append_metrics_jsonl(FLAGS.metrics_output_path, payload)
                append_metrics_csv(FLAGS.metrics_output_path, payload)
                _write_summary_json(FLAGS.metrics_output_path, payload)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(FLAGS.save_dir+str(train_state_gather.step+1), parallel=False)
                cp.train_state = train_state_gather
                cp.save()
                del cp
            del train_state_gather

if __name__ == '__main__':
    app.run(main)
