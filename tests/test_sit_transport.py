import types
import unittest

import jax
import jax.numpy as jnp

from baselines.targets_naive import get_targets as get_naive_targets
from baselines.targets_sit import get_targets as get_sit_targets
from targets_shortcut import get_targets as get_shortcut_targets
from utils.sit_transport import resolve_transport_eps, sit_sample


class DummyTrainState:
    def call_model(self, x, t, dt, labels, train=False):
        del t, dt, labels, train
        return jnp.zeros_like(x)

    def call_model_ema(self, x, t, dt, labels, train=False):
        del t, dt, labels, train
        return jnp.zeros_like(x)


def make_flags(train_type='sit', prediction='velocity'):
    model = {
        'class_dropout_prob': 0.1,
        'num_classes': 10,
        'denoise_timesteps': 128,
        'cfg_scale': 1.5,
        'bootstrap_cfg': 0,
        'bootstrap_every': 4,
        'bootstrap_ema': 1,
        'bootstrap_dt_bias': 0,
        'transport_path_type': 'linear',
        'transport_prediction': prediction,
        'transport_loss_weight': 'none',
        'transport_train_eps': None,
        'transport_sample_eps': None,
        'train_type': train_type,
    }
    return types.SimpleNamespace(
        batch_size=8,
        dataset_name='imagenet256',
        model=model,
    )


class SitTransportTest(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.images = jax.random.normal(self.key, (8, 4, 4, 2))
        self.labels = jnp.arange(8, dtype=jnp.int32) % 10
        self.train_state = DummyTrainState()

    def test_resolve_transport_eps_defaults(self):
        self.assertEqual(resolve_transport_eps('linear', 'velocity'), (0.0, 0.0))
        self.assertEqual(resolve_transport_eps('linear', 'noise'), (1e-3, 1e-3))
        self.assertEqual(resolve_transport_eps('vp', 'velocity'), (1e-5, 1e-3))

    def test_sit_targets_velocity(self):
        flags = make_flags(train_type='sit', prediction='velocity')
        x_t, target, t, dt_base, labels, sample_weight, info = get_sit_targets(
            flags, self.key, self.train_state, self.images, self.labels)
        self.assertEqual(x_t.shape, self.images.shape)
        self.assertEqual(target.shape, self.images.shape)
        self.assertEqual(t.shape, (8,))
        self.assertEqual(dt_base.shape, (8,))
        self.assertEqual(labels.shape, (8,))
        self.assertEqual(sample_weight.shape, (8,))
        self.assertTrue(jnp.all(jnp.isfinite(x_t)))
        self.assertTrue(jnp.all(jnp.isfinite(target)))
        self.assertTrue(jnp.all(jnp.isfinite(sample_weight)))
        self.assertIn('transport_weight_mean', info)

    def test_sit_targets_noise_and_score(self):
        for prediction in ('noise', 'score'):
            flags = make_flags(train_type='sit', prediction=prediction)
            x_t, target, _, _, _, sample_weight, _ = get_sit_targets(
                flags, self.key, self.train_state, self.images, self.labels)
            self.assertEqual(x_t.shape, self.images.shape)
            self.assertEqual(target.shape, self.images.shape)
            self.assertTrue(jnp.all(jnp.isfinite(target)))
            self.assertTrue(jnp.all(jnp.isfinite(sample_weight)))

    def test_naive_and_shortcut_target_interfaces(self):
        flags_naive = make_flags(train_type='naive')
        naive_outputs = get_naive_targets(
            flags_naive, self.key, self.train_state, self.images, self.labels)
        self.assertEqual(len(naive_outputs), 7)
        self.assertEqual(naive_outputs[5].shape, (8,))

        flags_shortcut = make_flags(train_type='shortcut')
        shortcut_outputs = get_shortcut_targets(
            flags_shortcut, self.key, self.train_state, self.images, self.labels)
        self.assertEqual(len(shortcut_outputs), 7)
        self.assertEqual(shortcut_outputs[5].shape, (8,))

    def test_sit_sampler_smoke(self):
        def model_fn(x, t_vector):
            del t_vector
            return jnp.zeros_like(x)

        x = jax.random.normal(self.key, (8, 4, 4, 2))
        ode_out = sit_sample(
            rng=self.key,
            x=x,
            model_fn=model_fn,
            num_steps=4,
            path_type='linear',
            prediction='velocity',
            train_eps=None,
            sample_eps=None,
            transport_type='ode',
            sampling_method='heun',
        )
        sde_out = sit_sample(
            rng=self.key,
            x=x,
            model_fn=model_fn,
            num_steps=4,
            path_type='linear',
            prediction='velocity',
            train_eps=None,
            sample_eps=None,
            transport_type='sde',
            sampling_method='euler',
            diffusion_form='sigma',
            last_step='mean',
        )
        self.assertTrue(jnp.all(jnp.isfinite(ode_out)))
        self.assertTrue(jnp.all(jnp.isfinite(sde_out)))


if __name__ == '__main__':
    unittest.main()
