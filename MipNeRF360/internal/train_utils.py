# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple, Mapping

from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import flax
from flax import traverse_util
import optax


def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
  return tree_sum(jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree))


def tree_norm(tree):
  return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0)


def tree_len(tree):
  return tree_sum(
      jax.tree_util.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), tree))


def summarize_tree(tree, fn, ancestry=(), max_depth=3):
  """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
  stats = {}
  for k, v in tree.items():
    name = ancestry + (k,)
    stats['/'.join(name)] = fn(v)
    if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
      stats.update(summarize_tree(v, fn, ancestry=name, max_depth=max_depth))
  return stats


def compute_data_loss(batch, rays, renderings, config, use_static_mask):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  loss_dict = {}
  stats = collections.defaultdict(lambda: [])
  static_mask = (rays.static_mask >= 0.5).astype(batch.rgb.dtype)

  for rendering in renderings:
    if use_static_mask:
      lossmult = jnp.broadcast_to(static_mask, batch.rgb[..., :3].shape)
      lossmult = static_mask + (1 - static_mask) * config.withmask_transient_weight
    else:
      # lossmult can be used to apply a weight to each ray in the batch.
      # For example: masking out rays, applying the Bayer mosaic mask, upweighting
      # rays from lower resolution images and so on.
      lossmult = rays.lossmult
      lossmult = jnp.broadcast_to(lossmult, batch.rgb[..., :3].shape)
      if config.disable_multiscale_loss:
        lossmult = jnp.ones_like(lossmult)

    resid_sq = (rendering['rgb'] - batch.rgb[..., :3])**2
    denom = jnp.maximum(lossmult.sum(), jnp.finfo(lossmult.dtype).eps)
    stats['mses'].append((lossmult * resid_sq).sum() / denom)

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq
    elif config.data_loss_type == 'charb':
      # Charbonnier loss.
      data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
    else:
      assert False
    data_losses.append((lossmult * data_loss).sum() / denom)

  data_losses = jnp.array(data_losses)
  loss_dict['data'] = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss_dict, stats


def compute_robustnerf_loss(batch, renderings, inlier_thresholds, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  loss_dict = {}
  stats = collections.defaultdict(lambda: [])

  for i, rendering in enumerate(renderings):
    resid_sq = (rendering['rgb'] - batch.rgb[..., :3])**2
    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq
    elif config.data_loss_type == 'charb':
      # Charbonnier loss.
      data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
    else:
      assert False
    
    errors = jax.lax.stop_gradient(jnp.sqrt(resid_sq))
    robust_mask, robust_stats = robustnerf_mask(errors, inlier_thresholds[i], config)
    robust_mask = jax.lax.stop_gradient(robust_mask)
    for key in robust_stats:
      stats[f'robust_{key}'].append(robust_stats[key])
    
    lossmult = jnp.broadcast_to(robust_mask, data_loss.shape)
    denom = jnp.maximum(lossmult.sum(), jnp.finfo(lossmult.dtype).eps)
    stats['mses'].append((lossmult * resid_sq).sum() / denom)
    data_losses.append((lossmult * data_loss).sum() / denom)

  data_losses = jnp.array(data_losses)
  loss_dict['data'] = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss_dict, stats


def compute_nerfw_loss(batch, renderings, ray_historys, config):
  data_losses = []
  loss_dict = {}
  stats = collections.defaultdict(lambda: [])

  beta = renderings[-1]['uncertainty']
  density = ray_historys[-1]['density_transient']

  for i, rendering in enumerate(renderings):
    pred_rgb = rendering['rgb_combined' if 'rgb_combined' in rendering.keys() else 'rgb']
    resid_sq = (pred_rgb - batch.rgb[..., :3])**2

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq
    elif config.data_loss_type == 'charb':
      # Charbonnier loss.
      data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
    else:
      assert False
    if i == len(renderings)-1:
      loss_dict['beta'] = config.nerfw_beta_loss_mult * jnp.log(beta).mean() + config.nerfw_beta_loss_bias
      data_loss = data_loss / (2 * beta**2)
      loss_dict['density'] = config.nerfw_density_loss_mult * density.mean()

    data_losses.append(data_loss.mean())
    stats['mses'].append(resid_sq.mean())

  data_losses = jnp.array(data_losses)
  loss_dict['data'] = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss_dict, stats
    

def compute_hanerf_loss(batch, renderings, train_frac, config):
  data_losses = []
  loss_dict = {}
  stats = collections.defaultdict(lambda: [])

  mask_size_loss_mult = jnp.maximum(
    config.hanerf_mask_size_loss_mult_min,
    config.hanerf_mask_size_loss_mult_max * jnp.exp(-train_frac * config.max_steps * config.hanerf_mask_size_loss_mult_k)
  )
  implicit_mask = renderings[-1]['implicit_mask']
  stats['implicit_mask'].append(implicit_mask.mean())

  for i in range(len(renderings)):
    rendering = renderings[i]
    resid_sq = (rendering['rgb'] - batch.rgb[..., :3])**2

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq
    elif config.data_loss_type == 'charb':
      # Charbonnier loss.
      data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
    else:
      assert False
    
    if i == len(renderings) - 1:
      data_loss = (1. - implicit_mask) * data_loss
      loss_dict['mask_size'] = mask_size_loss_mult * (implicit_mask**2).mean()
    else:
      data_loss = (1. - jax.lax.stop_gradient(implicit_mask)) * data_loss

    data_losses.append(data_loss.mean())
    stats['mses'].append(resid_sq.mean())

  data_losses = jnp.array(data_losses)
  loss_dict['data'] = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss_dict, stats


def interlevel_loss(ray_history, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  last_ray_results = ray_history[-1]
  c = jax.lax.stop_gradient(last_ray_results['sdist'])
  w = jax.lax.stop_gradient(last_ray_results['weights'])
  loss_interlevel = 0.
  for ray_results in ray_history[:-1]:
    cp = ray_results['sdist']
    wp = ray_results['weights']
    loss_interlevel += jnp.mean(stepfun.lossfun_outer(c, w, cp, wp))
  return config.interlevel_loss_mult * loss_interlevel


def distortion_loss(ray_history, config):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = last_ray_results['sdist']
  w = last_ray_results['weights']
  loss = jnp.mean(stepfun.lossfun_distortion(c, w))
  return config.distortion_loss_mult * loss


def robustnerf_mask(
    errors: jnp.ndarray, inlier_thresholds: jnp.ndarray, config
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
  """Computes RobustNeRF mask.

  Args:
    errors: f32[n,h,w,c]. Per-subpixel errors.
    inlier_threshold: f32[]. Upper bound on per-pixel loss to use to determine
      if a pixel is an inlier or not.
    config: Config object.

  Returns:
    mask: f32[?,?,?,?]. Binary mask that broadcasts to shape [n,h,w,c].
    stats: { str: f32[] }. Statistics to pass on.
  """
  epsilon = 1e-3
  dtype = errors.dtype
  error_per_pixel = jnp.mean(errors, axis=-1, keepdims=True)  # f32[n,h,w,1]
  next_inlier_threshold = jnp.quantile(
      error_per_pixel, config.robustnerf_inlier_quantile
  )
  mask = jnp.ones_like(error_per_pixel, dtype=dtype)
  stats = {'inlier_threshold': next_inlier_threshold}

  assert (
      config.robustnerf_inner_patch_size <= config.patch_size
  ), 'patch_size must be larger than robustnerf_inner_patch_size.'

  # 1.0 for inlier pixels.
  curr_inlier_threshold = inlier_thresholds[0]
  is_inlier_loss = (error_per_pixel < curr_inlier_threshold).astype(dtype)
  stats['is_inlier_loss'] = jnp.mean(is_inlier_loss)

  # Apply 3x3 box filter.
  f = config.robustnerf_smoothed_filter_size
  window = jnp.ones((1, 1, f, f)) / (f * f)
  has_inlier_neighbors = jax.lax.conv(
      jnp.transpose(is_inlier_loss, [0, 3, 1, 2]), window, (1, 1), 'SAME'
  )
  has_inlier_neighbors = jnp.transpose(has_inlier_neighbors, [0, 2, 3, 1])

  # Binarize after smoothing.
  has_inlier_neighbors = (
      has_inlier_neighbors > 1 - config.robustnerf_smoothed_inlier_quantile
  ).astype(dtype)
  stats['has_inlier_neighbors'] = jnp.mean(has_inlier_neighbors)

  # Construct binary mask for inner pixels. The entire inner patch is either
  # active or inactive.
  inner_patch_mask = _robustnerf_inner_patch_mask(
      config.robustnerf_inner_patch_size, config.patch_size
  )
  is_inlier_patch = jnp.mean(
      is_inlier_loss, axis=[1, 2], keepdims=True
  )  # f32[n,1,1,1]
  is_inlier_patch = (
      is_inlier_patch > 1 - config.robustnerf_inner_patch_inlier_quantile
  ).astype(dtype)
  is_inlier_patch = is_inlier_patch * inner_patch_mask
  stats['is_inlier_patch'] = jnp.mean(is_inlier_patch)

  # A pixel is an inlier if it is an inlier according to any of the above
  # criteria.
  mask = (
      is_inlier_patch + has_inlier_neighbors + is_inlier_loss > epsilon
  ).astype(dtype)

  stats['mask'] = jnp.mean(mask)
  return mask, stats


def _robustnerf_inner_patch_mask(
    inner_patch_size, outer_patch_size, *, dtype=jnp.float32
):
  """Constructs binary mask for inner patch.

  Args:
    inner_patch_size: Size of the (square) inside patch.
    outer_patch_size: Size of the (square) outer patch.
    dtype: dtype for result

  Returns:
    Binary mask of shape (1, outer_patch_size, outer_patch_size, 1). Mask is
      1.0 for the center (inner_patch_size, inner_patch_size) square and 0.0
      elsewhere.
  """
  pad_size_lower = (outer_patch_size - inner_patch_size) // 2
  pad_size_upper = outer_patch_size - (inner_patch_size + pad_size_lower)
  mask = jnp.pad(
      jnp.ones((1, inner_patch_size, inner_patch_size, 1), dtype=dtype),
      (
          (0, 0),  # batch
          (pad_size_lower, pad_size_upper),  # height
          (pad_size_lower, pad_size_upper),  # width
          (0, 0),  # channels
      ),
  )
  return mask


def clip_gradients(grad, config):
  """Clips gradients of each MLP individually based on norm and max value."""
  # Clip the gradients of each MLP individually.
  grad_clipped = {'params': {}}
  for k, g in grad['params'].items():
    # Clip by value.
    if config.grad_max_val > 0:
      g = jax.tree_util.tree_map(
          lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), g)

    # Then clip by norm.
    if config.grad_max_norm > 0:
      mult = jnp.minimum(
          1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + tree_norm(g)))
      g = jax.tree_util.tree_map(lambda z: mult * z, g)  # pylint:disable=cell-var-from-loop

    grad_clipped['params'][k] = g
  grad = type(grad)(grad_clipped)
  return grad


def create_train_step(model: models.Model,
                      config: configs.Config,
                      is_finetune: bool):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.
    dataset: Training dataset.

  Returns:
    pmap'ed training function.
  """

  def train_step(
      rng,
      state,
      batch,
      train_frac,
      inlier_thresholds
  ):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      train_frac: float, the fraction of training that is complete.
      inlier_thresholds: jnp.ndarray

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    rng, key = random.split(rng)

    def loss_fn(variables):
      rays = batch.rays

      renderings, ray_history = model.apply(
          variables,
          key if config.randomized else None,
          rays,
          train_frac=train_frac,
          compute_extras=False,
          zero_glo=False,
          zero_tra=False)

      if is_finetune or config.transient_type is None:
        losses, stats = compute_data_loss(batch, rays, renderings, config, False)
      elif config.transient_type == 'withmask':
        losses, stats = compute_data_loss(batch, rays, renderings, config, True)
      elif config.transient_type == 'robustnerf':
        losses, stats = compute_robustnerf_loss(batch, renderings, inlier_thresholds, config)
      elif config.transient_type == 'nerfw':
        losses, stats = compute_nerfw_loss(batch, renderings, ray_history, config)
      elif config.transient_type == 'hanerf':
        losses, stats = compute_hanerf_loss(batch, renderings, train_frac, config)
      else:
        raise ValueError()

      if not is_finetune:
        if config.interlevel_loss_mult > 0:
          losses['interlevel'] = interlevel_loss(ray_history, config)

        if config.distortion_loss_mult > 0:
          losses['distortion'] = distortion_loss(ray_history, config)

      stats['weight_l2s'] = summarize_tree(variables['params'], tree_norm_sq)

      if not is_finetune and config.weight_decay_mults:
        it = config.weight_decay_mults.items
        losses['weight'] = jnp.sum(
            jnp.array([m * stats['weight_l2s'][k] for k, m in it()]))

      stats['loss'] = jnp.sum(jnp.array(list(losses.values())))
      stats['losses'] = losses

      return stats['loss'], stats

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, stats), grad = loss_grad_fn(state.params)

    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    grad = pmean(grad)
    stats = pmean(stats)

    stats['grad_norms'] = summarize_tree(grad['params'], tree_norm)
    stats['grad_maxes'] = summarize_tree(grad['params'], tree_abs_max)

    grad = clip_gradients(grad, config)

    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

    new_state = state.apply_gradients(grads=grad)

    opt_delta = jax.tree_util.tree_map(lambda x, y: x - y, new_state,
                                       state).params['params']
    stats['opt_update_norms'] = summarize_tree(opt_delta, tree_norm)
    stats['opt_update_maxes'] = summarize_tree(opt_delta, tree_abs_max)

    stats['psnrs'] = image.mse_to_psnr(stats['mses'])
    stats['psnr'] = stats['psnrs'][-1]
    return new_state, stats, rng

  train_pstep = jax.pmap(
      train_step,
      axis_name='batch',
      in_axes=(0, 0, 0, None, None),
      donate_argnums=(0, 1))
  return train_pstep


def create_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict) -> Tuple[TrainState, Callable[[int], float]]:
  """Creates optax optimizer for model training."""
  adam_kwargs = {
      'b1': config.adam_beta1,
      'b2': config.adam_beta2,
      'eps': config.adam_eps,
  }
  lr_kwargs = {
      'max_steps': config.max_steps,
      'lr_delay_steps': config.lr_delay_steps,
      'lr_delay_mult': config.lr_delay_mult,
  }

  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs)

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
  tx = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

  return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def create_finetune_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict) -> Tuple[TrainState, Callable[[int], float]]:
  """Creates optax optimizer for model training."""
  adam_kwargs = {
      'b1': config.finetune_adam_beta1,
      'b2': config.finetune_adam_beta2,
      'eps': config.finetune_adam_eps,
  }
  lr_kwargs = {
      'max_steps': config.finetune_max_steps,
      'lr_delay_steps': config.finetune_lr_delay_steps,
      'lr_delay_mult': config.finetune_lr_delay_mult,
  }

  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs)

  lr_fn_main = get_lr_fn(config.finetune_lr_init, config.finetune_lr_final)
  adam = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)
  partition_optimizers = {'trainable': adam, 'frozen': optax.set_to_zero()}
  # param_partitions = flax.core.freeze(traverse_util.path_aware_map(
  #   lambda path, v: 'trainable' if 'embedding' in path else 'frozen', variables)) # incorrect when flax > 0.7.0
  param_partitions = traverse_util.path_aware_map(
    lambda path, v: 'trainable' if 'embedding' in path else 'frozen', variables)
  tx = optax.multi_transform(partition_optimizers, param_partitions)

  # visualize a subset of the param_partitions structure
  if False: # jax.process_index() == 0:
    flat = list(traverse_util.flatten_dict(param_partitions).items())
    # print(flax.core.freeze(traverse_util.unflatten_dict(dict(flat)))) # incorrect when flax > 0.7.0
    print(traverse_util.unflatten_dict(dict(flat)))

  return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def create_render_fn(model: models.Model, config):
  """Creates pmap'ed function for full image rendering."""

  def render_eval_fn(variables, train_frac, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            train_frac=train_frac,
            compute_extras=True,
            zero_glo=config.enable_render_zero_glo,
            zero_tra=config.enable_render_zero_tra),
        axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, None, 0),
      axis_name='batch',
  )
  return render_eval_pfn


def setup_model(
    config: configs.Config,
    rng: jnp.array
) -> Tuple[models.Model, TrainState, Callable[
    [FrozenVariableDict, jnp.array, utils.Rays],
    MutableMapping[Text, Any]], Callable[
        [jnp.array, TrainState, utils.Batch, Optional[Tuple[Any, ...]], float],
        Tuple[TrainState, Dict[Text, Any], jnp.array]], Callable[[int], float]]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  dummy_rays = utils.dummy_rays()
  model, variables = models.construct_model(rng, dummy_rays, config)

  state, lr_fn = create_optimizer(config, variables)
  render_eval_pfn = create_render_fn(model, config)
  train_pstep = create_train_step(model, config, False)

  return model, state, render_eval_pfn, train_pstep, lr_fn


def setup_finetune_model(
    config: configs.Config,
    model: models.Model,
    state: TrainState
):
  variables = state.params
  new_state, lr_fn = create_finetune_optimizer(config, variables)
  train_pstep = create_train_step(model, config, True)
  
  return new_state, train_pstep, lr_fn