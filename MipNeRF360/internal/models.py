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

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple

from flax import linen as nn
import gin
from internal import configs
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import render
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp

gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(coord.contract, module='coord')


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = random.split(rng)
  return key, rng


@gin.configurable
class Model(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""
  config: Any = None  # A Config class, must be set upon construction.
  num_prop_samples: int = 64  # The number of samples for each proposal level.
  num_nerf_samples: int = 32  # The number of samples the final nerf level.
  num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
  bg_intensity_range: Tuple[float] = (1., 1.)  # The range of background colors.
  anneal_slope: float = 10  # Higher = more rapid annealing.
  stop_level_grad: bool = True  # If True, don't backprop across levels.
  use_viewdirs: bool = True  # If True, use view directions as input.
  raydist_fn: Callable[..., Any] = None  # The curve used for ray dists.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  disable_integration: bool = False  # If True, use PE instead of IPE.
  single_jitter: bool = True  # If True, jitter whole rays instead of samples.
  dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
  dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
  num_glo_features: int = 0  # GLO vector length, disabled if 0.
  num_transient_features: int = 0 # Transient vector length, disabled if 0.
  num_embeddings: int = 3500  # Upper bound on max number of train images.
  near_anneal_rate: Optional[float] = None  # How fast to anneal in near bound.
  near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
  resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
  use_gpu_resampling: bool = False  # Use gather ops for faster GPU resampling.
  opaque_background: bool = False  # If true, make the background opaque.
  beta_min: float = 0.03 # beta for nerf-w

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac,
      compute_extras,
      zero_glo,
      zero_tra,
  ):
    """The mip-NeRF Model.

    Args:
      rng: random number generator (or None for deterministic output).
      rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """

    if self.config.transient_type in [None, 'withmask', 'robustnerf']:
      assert self.num_transient_features == 0
    elif self.config.transient_type in ['nerfw', 'hanerf']:
      assert self.num_transient_features > 0
    else:
      raise ValueError()
    # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
    # being regularized.
    nerf_mlp = NerfMLP(disable_transient=(self.config.transient_type!='nerfw'))
    prop_mlp = PropMLP(disable_transient=True)

    implicit_mask = ImplicitMask() if self.config.transient_type=='hanerf' else None

    if self.num_glo_features > 0:
      if not zero_glo:
        # Construct/grab GLO vectors for the cameras of each input ray.
        glo_vecs = GloEmbed(self.num_embeddings, self.num_glo_features)
        embed_idx = rays.embed_idx[..., 0]
        glo_vec = glo_vecs(embed_idx)
      else:
        glo_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_glo_features,))
    else:
      glo_vec = None

    if self.num_transient_features > 0:
      if not zero_tra:
        # Construct/grab Transient vectors for the cameras of each input ray.
        tra_vecs = TransientEmbed(self.num_embeddings, self.num_transient_features)
        embed_idx = rays.embed_idx[..., 0]
        tra_vec = tra_vecs(embed_idx)
      else:
        tra_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_transient_features,))
    else:
      tra_vec = None

    # Define the mapping from normalized to metric ray distance.
    _, s_to_t = coord.construct_ray_warps(self.raydist_fn, rays.near, rays.far)

    # Initialize the range of (normalized) distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    # `near_anneal_rate` can be used to anneal in the near bound at the start
    # of training, eg. 0.1 anneals in the bound over the first 10% of training.
    if self.near_anneal_rate is None:
      init_s_near = 0.
    else:
      init_s_near = jnp.clip(1 - train_frac / self.near_anneal_rate, 0,
                             self.near_anneal_init)
    init_s_far = 1.
    sdist = jnp.concatenate([
        jnp.full_like(rays.near, init_s_near),
        jnp.full_like(rays.far, init_s_far)
    ],
                            axis=-1)
    weights = jnp.ones_like(rays.near)
    prod_num_samples = 1

    ray_history = []
    renderings = []
    for i_level in range(self.num_levels):
      is_prop = i_level < (self.num_levels - 1)
      num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

      # Dilate by some multiple of the expected span of each current interval,
      # with some bias added in.
      dilation = self.dilation_bias + self.dilation_multiplier * (
          init_s_far - init_s_near) / prod_num_samples

      # Record the product of the number of samples seen so far.
      prod_num_samples *= num_samples

      # After the first level (where dilation would be a no-op) optionally
      # dilate the interval weights along each ray slightly so that they're
      # overestimates, which can reduce aliasing.
      use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
      if i_level > 0 and use_dilation:
        sdist, weights = stepfun.max_dilate_weights(
            sdist,
            weights,
            dilation,
            domain=(init_s_near, init_s_far),
            renormalize=True)
        sdist = sdist[..., 1:-1]
        weights = weights[..., 1:-1]

      # Optionally anneal the weights as a function of training iteration.
      if self.anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, self.anneal_slope)
      else:
        anneal = 1.

      # A slightly more stable way to compute weights**anneal. If the distance
      # between adjacent intervals is zero then its weight is fixed to 0.
      logits_resample = jnp.where(
          sdist[..., 1:] > sdist[..., :-1],
          anneal * jnp.log(weights + self.resample_padding), -jnp.inf)

      # Draw sampled intervals from each ray's current weights.
      key, rng = random_split(rng)
      sdist = stepfun.sample_intervals(
          key,
          sdist,
          logits_resample,
          num_samples,
          single_jitter=self.single_jitter,
          domain=(init_s_near, init_s_far),
          use_gpu_resampling=self.use_gpu_resampling)

      # Optimization will usually go nonlinear if you propagate gradients
      # through sampling.
      if self.stop_level_grad:
        sdist = jax.lax.stop_gradient(sdist)

      # Convert normalized distances to metric distances.
      tdist = s_to_t(sdist)

      # Cast our rays, by turning our distance intervals into Gaussians.
      gaussians = render.cast_rays(
          tdist,
          rays.origins,
          rays.directions,
          rays.radii,
          self.ray_shape,
          diag=False)

      if self.disable_integration:
        # Setting the covariance of our Gaussian samples to 0 disables the
        # "integrated" part of integrated positional encoding.
        gaussians = (gaussians[0], jnp.zeros_like(gaussians[1]))

      # Push our Gaussians through one of our two MLPs.
      mlp = prop_mlp if is_prop else nerf_mlp
      key, rng = random_split(rng)
      ray_results = mlp(
          key,
          gaussians,
          viewdirs=rays.viewdirs if self.use_viewdirs else None,
          glo_vec=None if is_prop else glo_vec,
          tra_vec=None if is_prop else tra_vec
      )

      # Get the weights used by volumetric rendering (and our other losses).
      weights = render.compute_alpha_weights(
          ray_results['density'],
          tdist,
          rays.directions,
          opaque_background=self.opaque_background,
      )[0]

      # Define or sample the background color for each ray.
      if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
        # If the min and max of the range are equal, just take it.
        bg_rgbs = self.bg_intensity_range[0]
      elif rng is None:
        # If rendering is deterministic, use the midpoint of the range.
        bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
      else:
        # Sample RGB values from the range for each ray.
        key, rng = random_split(rng)
        bg_rgbs = random.uniform(
            key,
            shape=weights.shape[:-1] + (3,),
            minval=self.bg_intensity_range[0],
            maxval=self.bg_intensity_range[1])

      # Render each ray.
      rendering = render.volumetric_rendering(
          ray_results['rgb'],
          weights,
          tdist,
          bg_rgbs,
          rays.far,
          compute_extras,
          extras=None
      )

      if compute_extras:
        # Collect some rays to visualize directly. By naming these quantities
        # with `ray_` they get treated differently downstream --- they're
        # treated as bags of rays, rather than image chunks.
        n = self.config.vis_num_rays
        rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
        rendering['ray_weights'] = (
            weights.reshape([-1, weights.shape[-1]])[:n, :])
        rgb = ray_results['rgb']
        rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]

      if 'density_transient' in ray_results.keys():
        weights_static_partial, weights_transient_partial, weights_combined = render.compute_dual_alpha_weights(
          ray_results['density'],
          ray_results['density_transient'],
          tdist,
          rays.directions,
          opaque_background=self.opaque_background,
        )
        rendering['rgb_combined'], rendering['rgb_static'], rendering['rgb_transient'] = \
          render.volumetric_rendering_combined_color(
            ray_results['rgb'], ray_results['rgb_transient'], bg_rgbs,
            weights_static_partial, weights_transient_partial, weights_combined,
        )
        
        weights_transient = render.compute_alpha_weights(
          ray_results['density_transient'],
          tdist,
          rays.directions,
          opaque_background=self.opaque_background,
        )[0]

        rendering['uncertainty'] = \
          (weights_transient[..., None] * ray_results['uncertainty']).sum(axis=-2) + self.beta_min

      renderings.append(rendering)
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['weights'] = jnp.copy(weights)
      ray_history.append(ray_results)

    if compute_extras:
      # Because the proposal network doesn't produce meaningful colors, for
      # easier visualization we replace their colors with the final average
      # color.
      weights = [r['ray_weights'] for r in renderings]
      rgbs = [r['ray_rgbs'] for r in renderings]
      final_rgb = jnp.sum(rgbs[-1] * weights[-1][..., None], axis=-2)
      avg_rgbs = [
          jnp.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
      ]
      for i in range(len(avg_rgbs)):
        renderings[i]['ray_rgbs'] = avg_rgbs[i]

    if implicit_mask is not None:
      renderings[-1]['implicit_mask'] = implicit_mask(rays.pix_coords, tra_vec)

    return renderings, ray_history


def construct_model(rng, rays, config):
  """Construct a mip-NeRF 360 model.

  Args:
    rng: jnp.ndarray. Random number generator.
    rays: an example of input Rays.
    config: A Config class.

  Returns:
    model: initialized nn.Module, a NeRF model with parameters.
    init_variables: flax.Module.state, initialized NeRF model parameters.
  """
  # Grab just 10 rays, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10],
                               rays)
  model = Model(config=config)
  init_variables = model.init(
      rng,  # The RNG used by flax to initialize random weights.
      rng=None,  # The RNG used by sampling within the model.
      rays=ray,
      train_frac=1.,
      compute_extras=False,
      zero_glo=model.num_glo_features == 0,
      zero_tra=model.num_transient_features == 0)
  return model, init_variables


class MLP(nn.Module):
  """A PosEnc MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  bottleneck_width: int = 256  # The width of the bottleneck vector.
  net_depth_viewdirs: int = 1  # The depth of the second part of MLP.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  net_depth_transient: int = 4 # The depth of the transient MLP.
  net_width_transient: int = 128 # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
  skip_layer_transient: int = 4 # Add a skip connection to transient MLP every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
  density_activation: Callable[..., Any] = nn.softplus  # Density activation.
  density_bias: float = -1.  # Shift added to raw densities pre-activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
  rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  uncertainty_activation: Callable[..., Any] = nn.softplus  # uncertainty activation.
  disable_rgb: bool = False  # If True don't output RGB.
  disable_transient: bool = True
  warp_fn: Callable[..., Any] = None
  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).

  def setup(self):
    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)).T

    # Precompute and define viewdir encoding function.
    def dir_enc_fn(direction):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

    self.dir_enc_fn = dir_enc_fn

  @nn.compact
  def __call__(self,
               rng,
               gaussians,
               viewdirs=None,
               glo_vec=None,
               tra_vec=None):
    """Evaluate the MLP.

    Args:
      rng: jnp.ndarray. Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: jnp.ndarray(float32), [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      glo_vec: [..., num_glo_features], The GLO vector for each ray
      tra_vec: [..., num_tra_features], The transient vector for each ray

    Returns:
      rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
      density: jnp.ndarray(float32), with a shape of [...].
    """

    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)())

    density_key, rng = random_split(rng)

    def predict_density(means, covs):
      """Helper function to output density."""
      # Encode input positions

      if self.warp_fn is not None:
        means, covs = coord.track_linearize(self.warp_fn, means, covs)

      lifted_means, lifted_vars = (
          coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
      x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                   self.min_deg_point, self.max_deg_point)

      inputs = x
      # Evaluate network to produce the output density.
      for i in range(self.net_depth):
        x = dense_layer(self.net_width)(x)
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)
      raw_density = dense_layer(1)(x)[..., 0]  # Hardcoded to a single channel.
      # Add noise to regularize the density predictions if needed.
      if (density_key is not None) and (self.density_noise > 0):
        raw_density += self.density_noise * random.normal(
            density_key, raw_density.shape)
      return raw_density, x

    means, covs = gaussians
    raw_density, x = predict_density(means, covs)

    # Apply bias and activation to raw density
    density = self.density_activation(raw_density + self.density_bias)

    if self.disable_rgb:
      rgb = jnp.zeros_like(means)
    else:
      if viewdirs is not None:
        # Output of the first part of MLP.
        if self.bottleneck_width > 0:
          bottleneck = dense_layer(self.bottleneck_width)(x)

          # Add bottleneck noise.
          if (rng is not None) and (self.bottleneck_noise > 0):
            key, rng = random_split(rng)
            bottleneck += self.bottleneck_noise * random.normal(
                key, bottleneck.shape)

          x = [bottleneck]
        else:
          x = []

        # Encode view directions.
        dir_enc = self.dir_enc_fn(viewdirs)

        dir_enc = jnp.broadcast_to(
            dir_enc[..., None, :],
            bottleneck.shape[:-1] + (dir_enc.shape[-1],))

        # Append view (or reflection) direction encoding to bottleneck vector.
        x.append(dir_enc)

        # Append GLO vector if used.
        if glo_vec is not None:
          glo_vec = jnp.broadcast_to(glo_vec[..., None, :],
                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
          x.append(glo_vec)

        # Concatenate bottleneck, directional encoding, and GLO.
        x = jnp.concatenate(x, axis=-1)

        # Output of the second part of MLP.
        inputs = x
        for i in range(self.net_depth_viewdirs):
          x = dense_layer(self.net_width_viewdirs)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_dir == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

      rgb = self.rgb_activation(self.rgb_premultiplier *
                                dense_layer(self.num_rgb_channels)(x) +
                                self.rgb_bias)

      # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

      if tra_vec is not None and not self.disable_transient:
        tra_vec = jnp.broadcast_to(tra_vec[..., None, :],
                                   bottleneck.shape[:-1] + tra_vec.shape[-1:])
        x = jnp.concatenate([bottleneck, tra_vec], axis=-1)
        inputs = x
        for i in range(self.net_depth_transient):
          x = dense_layer(self.net_width_transient)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_transient == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)
        raw_density_transient = dense_layer(1)(x)[..., 0]  # Hardcoded to a single channel.
        # Apply bias and activation to raw density
        density_transient = self.density_activation(raw_density_transient + self.density_bias)
        rgb_transient = self.rgb_activation(self.rgb_premultiplier *
                                            dense_layer(self.num_rgb_channels)(x) +
                                            self.rgb_bias)
        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb_transient = rgb_transient * (1 + 2 * self.rgb_padding) - self.rgb_padding
        uncertainty = self.uncertainty_activation(dense_layer(1)(x))

    outputs = dict(
        density=density,
        rgb=rgb,
    )
    if tra_vec is not None and not self.disable_transient:
      outputs['density_transient'] = density_transient
      outputs['rgb_transient'] = rgb_transient
      outputs['uncertainty'] = uncertainty

    return outputs


@gin.configurable
class NerfMLP(MLP):
  pass


@gin.configurable
class PropMLP(MLP):
  pass

class TransientEmbed(nn.Embed):
  pass

class GloEmbed(nn.Embed):
  pass

def render_image(render_fn: Callable[[jnp.array, utils.Rays],
                                     Tuple[List[Mapping[Text, jnp.ndarray]],
                                           List[Tuple[jnp.ndarray, ...]]]],
                 rays: utils.Rays,
                 rng: jnp.array,
                 config: configs.Config,
                 verbose: bool = True) -> MutableMapping[Text, Any]:
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rng, rays) -> pytree.
    rays: a `Rays` pytree, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    config: A Config class.
    verbose: print progress indicators.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays.origins.shape[:2]
  num_rays = height * width
  rays = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.process_index()
  chunks = []
  idx0s = range(0, num_rays, config.render_chunk_size)
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
      print(f'Rendering chunk {i_chunk}/{len(idx0s)-1}')
    chunk_rays = (
        jax.tree_util.tree_map(
            lambda r: r[idx0:idx0 + config.render_chunk_size], rays))
    actual_chunk_size = chunk_rays.origins.shape[0]
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_util.tree_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by host_count.
    rays_per_host = chunk_rays.origins.shape[0] // jax.process_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = jax.tree_util.tree_map(lambda r: utils.shard(r[start:stop]),
                                        chunk_rays)
    chunk_renderings, _ = render_fn(rng, chunk_rays)

    # Unshard the renderings.
    chunk_renderings = jax.tree_util.tree_map(
        lambda v: utils.unshard(v[0], padding), chunk_renderings)

    # Gather the final pass for 2D buffers and all passes for ray bundles.
    chunk_rendering = chunk_renderings[-1]
    for k in chunk_renderings[0]:
      if k.startswith('ray_'):
        chunk_rendering[k] = [r[k] for r in chunk_renderings]

    chunks.append(chunk_rendering)

  # Concatenate all chunks within each leaf of a single pytree.
  rendering = (
      jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *chunks))
  for k, z in rendering.items():
    if not k.startswith('ray_'):
      # Reshape 2D buffers into original image shape.
      rendering[k] = z.reshape((height, width) + z.shape[1:])

  # After all of the ray bundles have been concatenated together, extract a
  # new random bundle (deterministically) from the concatenation that is the
  # same size as one of the individual bundles.
  keys = [k for k in rendering if k.startswith('ray_')]
  if keys:
    num_rays = rendering[keys[0]][0].shape[0]
    ray_idx = random.permutation(random.PRNGKey(0), num_rays)
    ray_idx = ray_idx[:config.vis_num_rays]
    for k in keys:
      rendering[k] = [r[ray_idx] for r in rendering[k]]

  return rendering

class ImplicitMask(nn.Module):
  net_depth: int = 4   # The depth of the MLP.
  net_width: int = 256 # The width of the MLP.
  net_activation: Callable[..., Any] = nn.relu # The activation function.
  deg_coord: int = 10  # Degree of encoding for pixel coordinate
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  def setup(self):
    self.enc_fn = lambda coords: coord.pos_enc(
      coords, min_deg=0, max_deg=self.deg_coord, append_identity=True
    )
  
  @nn.compact
  def __call__(self, pix_coords, tra_vec):
    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)())
    
    coords_enc = self.enc_fn(pix_coords)
    x = jnp.concatenate([coords_enc, tra_vec], axis=-1)
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
    
    out = nn.sigmoid(dense_layer(1)(x))
    return out