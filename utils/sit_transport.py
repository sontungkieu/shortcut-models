import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp


VP_SIGMA_MIN = 0.1
VP_SIGMA_MAX = 20.0


def _normalize_path_type(path_type: str) -> str:
    normalized = path_type.lower()
    if normalized not in ("linear", "gvp", "vp"):
        raise ValueError(f"Unsupported transport path_type={path_type}")
    return normalized


def _normalize_prediction(prediction: str) -> str:
    normalized = prediction.lower()
    if normalized not in ("velocity", "score", "noise"):
        raise ValueError(f"Unsupported transport prediction={prediction}")
    return normalized


def _normalize_loss_weight(loss_weight: Optional[str]) -> str:
    if loss_weight is None:
        return "none"
    normalized = loss_weight.lower()
    if normalized not in ("none", "velocity", "likelihood"):
        raise ValueError(f"Unsupported transport loss_weight={loss_weight}")
    return normalized


def _normalize_transport_type(transport_type: str) -> str:
    normalized = transport_type.lower()
    if normalized not in ("ode", "sde"):
        raise ValueError(f"Unsupported transport type={transport_type}")
    return normalized


def _normalize_sampling_method(method: str) -> str:
    normalized = method.lower()
    if normalized not in ("euler", "heun"):
        raise ValueError(f"Unsupported sampling method={method}")
    return normalized


def _normalize_last_step(last_step: Optional[str]) -> Optional[str]:
    if last_step is None:
        return None
    normalized = last_step.lower()
    if normalized == "none":
        return None
    if normalized not in ("mean", "tweedie", "euler"):
        raise ValueError(f"Unsupported last step={last_step}")
    return normalized


def expand_t_like_x(t, x):
    dims = (1,) * (x.ndim - 1)
    return jnp.reshape(t, (t.shape[0],) + dims)


def resolve_transport_eps(
    path_type: str,
    prediction: str,
    train_eps: Optional[float] = None,
    sample_eps: Optional[float] = None,
):
    path_type = _normalize_path_type(path_type)
    prediction = _normalize_prediction(prediction)
    if path_type == "vp":
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif prediction != "velocity":
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0.0 if train_eps is None else train_eps
        sample_eps = 0.0 if sample_eps is None else sample_eps
    return float(train_eps), float(sample_eps)


def check_interval(
    path_type: str,
    prediction: str,
    train_eps: Optional[float],
    sample_eps: Optional[float],
    *,
    diffusion_form: str = "sbdm",
    sde: bool = False,
    reverse: bool = False,
    eval_mode: bool = False,
    last_step_size: float = 0.0,
):
    path_type = _normalize_path_type(path_type)
    prediction = _normalize_prediction(prediction)
    diffusion_form = diffusion_form.lower()
    train_eps, sample_eps = resolve_transport_eps(path_type, prediction, train_eps, sample_eps)

    t0 = 0.0
    t1 = 1.0
    eps = sample_eps if eval_mode else train_eps

    if path_type == "vp":
        if (not sde) or last_step_size == 0:
            t1 = 1.0 - eps
        else:
            t1 = 1.0 - last_step_size
    elif prediction != "velocity" or sde:
        if (diffusion_form == "sbdm" and sde) or prediction != "velocity":
            t0 = eps
        if (not sde) or last_step_size == 0:
            t1 = 1.0 - eps
        else:
            t1 = 1.0 - last_step_size

    if reverse:
        t0, t1 = 1.0 - t0, 1.0 - t1

    return float(t0), float(t1)


def compute_alpha_sigma(path_type: str, t):
    path_type = _normalize_path_type(path_type)
    t = jax.lax.convert_element_type(t, jnp.float32)

    if path_type == "linear":
        alpha_t = t
        d_alpha_t = jnp.ones_like(t)
        sigma_t = 1.0 - t
        d_sigma_t = -jnp.ones_like(t)
    elif path_type == "gvp":
        alpha_t = jnp.sin(t * math.pi / 2)
        d_alpha_t = (math.pi / 2) * jnp.cos(t * math.pi / 2)
        sigma_t = jnp.cos(t * math.pi / 2)
        d_sigma_t = -(math.pi / 2) * jnp.sin(t * math.pi / 2)
    else:
        log_mean_coeff = (
            -0.25 * ((1.0 - t) ** 2) * (VP_SIGMA_MAX - VP_SIGMA_MIN)
            - 0.5 * (1.0 - t) * VP_SIGMA_MIN
        )
        d_log_mean_coeff = (
            0.5 * (1.0 - t) * (VP_SIGMA_MAX - VP_SIGMA_MIN)
            + 0.5 * VP_SIGMA_MIN
        )
        alpha_t = jnp.exp(log_mean_coeff)
        d_alpha_t = alpha_t * d_log_mean_coeff
        exp_two_log_mean = jnp.exp(2.0 * log_mean_coeff)
        sigma_t = jnp.sqrt(jnp.maximum(1.0 - exp_two_log_mean, 1e-12))
        d_sigma_t = exp_two_log_mean * (2.0 * d_log_mean_coeff) / (-2.0 * sigma_t)

    return alpha_t, d_alpha_t, sigma_t, d_sigma_t


def compute_xt(path_type: str, t, x0, x1):
    alpha_t, _, sigma_t, _ = compute_alpha_sigma(path_type, t)
    alpha_t = expand_t_like_x(alpha_t, x1)
    sigma_t = expand_t_like_x(sigma_t, x1)
    return alpha_t * x1 + sigma_t * x0


def compute_ut(path_type: str, t, x0, x1):
    _, d_alpha_t, _, d_sigma_t = compute_alpha_sigma(path_type, t)
    d_alpha_t = expand_t_like_x(d_alpha_t, x1)
    d_sigma_t = expand_t_like_x(d_sigma_t, x1)
    return d_alpha_t * x1 + d_sigma_t * x0


def compute_drift(path_type: str, x, t):
    path_type = _normalize_path_type(path_type)
    t_like_x = expand_t_like_x(t, x)

    if path_type == "vp":
        beta_t = VP_SIGMA_MIN + (1.0 - t_like_x) * (VP_SIGMA_MAX - VP_SIGMA_MIN)
        return -0.5 * beta_t * x, beta_t / 2.0

    alpha_t, d_alpha_t, sigma_t, d_sigma_t = compute_alpha_sigma(path_type, t_like_x)
    alpha_ratio = d_alpha_t / jnp.clip(alpha_t, a_min=1e-12)
    drift = alpha_ratio * x
    diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
    return -drift, diffusion


def compute_diffusion(path_type: str, x, t, form: str = "constant", norm: float = 1.0):
    form = form.lower()
    t_like_x = expand_t_like_x(t, x)

    if form == "constant":
        return jnp.ones_like(x) * norm
    if form == "sbdm":
        return norm * compute_drift(path_type, x, t)[1]
    if form == "sigma":
        return norm * expand_t_like_x(compute_alpha_sigma(path_type, t)[2], x)
    if form == "linear":
        return norm * (1.0 - t_like_x)
    if form == "decreasing":
        return 0.25 * (norm * jnp.cos(math.pi * t_like_x) + 1.0) ** 2
    if form in ("increasing-decreasing", "inccreasing-decreasing"):
        return norm * (jnp.sin(math.pi * t_like_x) ** 2)
    raise ValueError(f"Unsupported diffusion form={form}")


def score_from_velocity(path_type: str, velocity, x, t):
    alpha_t, d_alpha_t, sigma_t, d_sigma_t = compute_alpha_sigma(path_type, expand_t_like_x(t, x))
    reverse_alpha_ratio = alpha_t / jnp.clip(d_alpha_t, a_min=1e-12)
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    return (reverse_alpha_ratio * velocity - x) / jnp.clip(var, a_min=1e-12)


def velocity_from_score(path_type: str, score, x, t):
    drift, diffusion = compute_drift(path_type, x, t)
    return diffusion * score - drift


def prediction_target_and_weight(path_type: str, prediction: str, loss_weight: Optional[str], t, x0, x1, xt):
    prediction = _normalize_prediction(prediction)
    loss_weight = _normalize_loss_weight(loss_weight)

    if prediction == "velocity":
        return compute_ut(path_type, t, x0, x1), jnp.ones((xt.shape[0],), dtype=jnp.float32)

    sigma_t = compute_alpha_sigma(path_type, t)[2]
    drift_var = compute_drift(path_type, xt, t)[1]
    sigma_sq = jnp.clip(sigma_t ** 2, a_min=1e-12)

    if loss_weight == "velocity":
        sample_weight = (drift_var.reshape((drift_var.shape[0], -1))[:, 0] ** 2) / sigma_sq
    elif loss_weight == "likelihood":
        sample_weight = drift_var.reshape((drift_var.shape[0], -1))[:, 0] / sigma_sq
    else:
        sample_weight = jnp.ones((xt.shape[0],), dtype=jnp.float32)

    if prediction == "noise":
        target = x0
    else:
        target = -x0 / jnp.clip(expand_t_like_x(sigma_t, x0), a_min=1e-12)

    return target, jax.lax.convert_element_type(sample_weight, jnp.float32)


def prediction_to_score(path_type: str, prediction: str, model_output, x, t):
    prediction = _normalize_prediction(prediction)
    if prediction == "score":
        return model_output
    if prediction == "noise":
        sigma_t = expand_t_like_x(compute_alpha_sigma(path_type, t)[2], x)
        return model_output / -jnp.clip(sigma_t, a_min=1e-12)
    return score_from_velocity(path_type, model_output, x, t)


def prediction_to_drift(path_type: str, prediction: str, model_output, x, t):
    prediction = _normalize_prediction(prediction)
    if prediction == "velocity":
        return model_output
    drift_mean, drift_var = compute_drift(path_type, x, t)
    if prediction == "noise":
        sigma_t = expand_t_like_x(compute_alpha_sigma(path_type, t)[2], x)
        score = model_output / -jnp.clip(sigma_t, a_min=1e-12)
    else:
        score = model_output
    return -drift_mean + drift_var * score


def get_transport_dt_base(denoise_timesteps: int, batch_size: int, force_dt: float = -1):
    dt_flow = int(round(math.log2(denoise_timesteps)))
    dt_base = jnp.ones((batch_size,), dtype=jnp.int32) * dt_flow
    if force_dt != -1:
        dt_base = jnp.ones((batch_size,), dtype=jnp.int32) * int(force_dt)
    return dt_base


def build_sit_training_batch(
    *,
    path_type: str,
    prediction: str,
    loss_weight: Optional[str],
    train_eps: Optional[float],
    sample_eps: Optional[float],
    dataset_name: str,
    denoise_timesteps: int,
    class_dropout_prob: float,
    num_classes: int,
    key,
    images,
    labels,
    force_t: float = -1,
    force_dt: float = -1,
):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, class_dropout_prob, (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, num_classes, labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == num_classes)

    if 'latent' in dataset_name:
        midpoint = images.shape[-1] // 2
        x0 = images[..., :midpoint]
        x1 = images[..., midpoint:]
    else:
        x1 = images
        x0 = jax.random.normal(noise_key, x1.shape)

    t0, t1 = check_interval(path_type, prediction, train_eps, sample_eps, eval_mode=False)
    t = jax.random.uniform(time_key, (x1.shape[0],), minval=t0, maxval=t1)
    if force_t != -1:
        t = jnp.ones((x1.shape[0],), dtype=jnp.float32) * force_t
    t = jax.lax.convert_element_type(t, jnp.float32)

    xt = compute_xt(path_type, t, x0, x1)
    target, sample_weight = prediction_target_and_weight(path_type, prediction, loss_weight, t, x0, x1, xt)
    dt_base = get_transport_dt_base(denoise_timesteps, xt.shape[0], force_dt=force_dt)

    info['transport_t_mean'] = jnp.mean(t)
    info['transport_t_min'] = jnp.min(t)
    info['transport_t_max'] = jnp.max(t)
    info['transport_weight_mean'] = jnp.mean(sample_weight)

    return xt, target, t, dt_base, labels_dropped, sample_weight, info


def sit_sample(
    *,
    rng,
    x,
    model_fn: Callable,
    num_steps: int,
    path_type: str,
    prediction: str,
    train_eps: Optional[float],
    sample_eps: Optional[float],
    transport_type: str = "ode",
    sampling_method: str = "heun",
    diffusion_form: str = "sigma",
    diffusion_norm: float = 1.0,
    last_step: Optional[str] = "mean",
    last_step_size: float = 0.04,
    return_history: bool = False,
):
    transport_type = _normalize_transport_type(transport_type)
    sampling_method = _normalize_sampling_method(sampling_method)
    last_step = _normalize_last_step(last_step)
    num_steps = max(int(num_steps), 1)

    path_type = _normalize_path_type(path_type)
    prediction = _normalize_prediction(prediction)

    def drift_fn(current_x, t_scalar):
        t_vector = jnp.full((current_x.shape[0],), t_scalar, dtype=jnp.float32)
        model_output = model_fn(current_x, t_vector)
        return prediction_to_drift(path_type, prediction, model_output, current_x, t_vector)

    def score_fn(current_x, t_scalar):
        t_vector = jnp.full((current_x.shape[0],), t_scalar, dtype=jnp.float32)
        model_output = model_fn(current_x, t_vector)
        return prediction_to_score(path_type, prediction, model_output, current_x, t_vector)

    history = [x] if return_history else None

    if transport_type == "ode":
        t0, t1 = check_interval(path_type, prediction, train_eps, sample_eps, eval_mode=True)
        times = jnp.linspace(t0, t1, num_steps + 1, dtype=jnp.float32)
        current_x = x
        for step_idx in range(num_steps):
            t_cur = float(times[step_idx])
            dt = float(times[step_idx + 1] - times[step_idx])
            if sampling_method == "euler":
                current_x = current_x + dt * drift_fn(current_x, t_cur)
            else:
                k1 = drift_fn(current_x, t_cur)
                predictor = current_x + dt * k1
                k2 = drift_fn(predictor, t_cur + dt)
                current_x = current_x + 0.5 * dt * (k1 + k2)
            if return_history:
                history.append(current_x)
        return history if return_history else current_x

    t0, t1 = check_interval(
        path_type,
        prediction,
        train_eps,
        sample_eps,
        diffusion_form=diffusion_form,
        sde=True,
        eval_mode=True,
        last_step_size=0.0 if last_step is None else last_step_size,
    )
    times = jnp.linspace(t0, t1, num_steps + 1, dtype=jnp.float32)
    current_x = x
    current_rng = rng

    for step_idx in range(num_steps):
        t_cur = float(times[step_idx])
        dt = float(times[step_idx + 1] - times[step_idx])
        t_vector = jnp.full((current_x.shape[0],), t_cur, dtype=jnp.float32)
        diffusion = compute_diffusion(path_type, current_x, t_vector, form=diffusion_form, norm=diffusion_norm)

        current_rng, noise_rng = jax.random.split(current_rng)
        noise = jax.random.normal(noise_rng, current_x.shape) * math.sqrt(dt)

        if sampling_method == "euler":
            sde_drift = drift_fn(current_x, t_cur) + diffusion * score_fn(current_x, t_cur)
            current_x = current_x + sde_drift * dt + jnp.sqrt(2.0 * diffusion) * noise
        else:
            x_hat = current_x + jnp.sqrt(2.0 * diffusion) * noise
            k1 = drift_fn(x_hat, t_cur) + diffusion * score_fn(x_hat, t_cur)
            x_predictor = x_hat + dt * k1
            t_next = min(t_cur + dt, 1.0)
            t_next_vector = jnp.full((x_predictor.shape[0],), t_next, dtype=jnp.float32)
            diffusion_next = compute_diffusion(path_type, x_predictor, t_next_vector, form=diffusion_form, norm=diffusion_norm)
            k2 = drift_fn(x_predictor, t_next) + diffusion_next * score_fn(x_predictor, t_next)
            current_x = x_hat + 0.5 * dt * (k1 + k2)

        if return_history:
            history.append(current_x)

    if last_step is not None:
        t_vector = jnp.full((current_x.shape[0],), t1, dtype=jnp.float32)
        if last_step == "mean":
            diffusion = compute_diffusion(path_type, current_x, t_vector, form=diffusion_form, norm=diffusion_norm)
            current_x = current_x + (drift_fn(current_x, t1) + diffusion * score_fn(current_x, t1)) * last_step_size
        elif last_step == "tweedie":
            alpha_t = expand_t_like_x(compute_alpha_sigma(path_type, t_vector)[0], current_x)
            sigma_t = expand_t_like_x(compute_alpha_sigma(path_type, t_vector)[2], current_x)
            current_x = current_x / jnp.clip(alpha_t, a_min=1e-12) + (
                (sigma_t ** 2) / jnp.clip(alpha_t, a_min=1e-12)
            ) * score_fn(current_x, t1)
        else:
            current_x = current_x + drift_fn(current_x, t1) * last_step_size

        if return_history:
            history.append(current_x)

    return history if return_history else current_x
