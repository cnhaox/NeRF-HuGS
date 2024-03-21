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

"""Different datasets implementation plus a general port for all the datasets."""

import abc
import copy
import json
import os
from os import path
import queue
import threading
from typing import Mapping, Optional, Sequence, Text, Tuple, Union, List
from pathlib import Path

import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import utils
import jax
from jax.tree_util import tree_map
import numpy as np
from PIL import Image
import pandas as pd

# This is ugly, but it works.
import sys
sys.path.insert(0,'internal/pycolmap')
sys.path.insert(0,'internal/pycolmap/pycolmap')
import pycolmap


def load_dataset(
  split, 
  is_training, 
  sample_from_half_image, 
  batch_size,
  patch_size,
  patch_dilation,
  image_num_per_batch,
  train_dir, 
  config
):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'blender': Blender,
      'llff': LLFF,
      'tat_nerfpp': TanksAndTemplesNerfPP,
      'tat_fvs': TanksAndTemplesFVS,
      'dtu': DTU,
      'kubric': Kubric,
      'phototourism': Phototourism,
      'distractor': Distractor,
  }
  return dataset_dict[config.dataset_loader](
    split, 
    is_training, 
    sample_from_half_image, 
    batch_size,
    patch_size,
    patch_dilation,
    image_num_per_batch,
    train_dir, 
    config
  )


class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

  def process(
      self
  ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, 
             Sequence[Optional[Mapping[Text, float]]], 
             Sequence[camera_utils.ProjectionType], np.ndarray]:
    """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocams, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocams: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
      camtypes
      pts3d
    """

    self.load_cameras()
    self.load_images()
    self.load_points3D()

    # Extract extrinsic matrices in world-to-camera format.
    imdata = self.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    # Extract intrinsics and distortion parameters and camtype
    camdata = self.cameras
    pixtocams = []
    distortion_params = []
    camtypes = []
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)

      cam = camdata[im.camera_id]
      # Extract focal lengths and principal point parameters.
      fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
      pixtocams.append(np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy)))

      type_ = cam.camera_type
      if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
        params = None
        camtype = camera_utils.ProjectionType.PERSPECTIVE

      elif type_ == 1 or type_ == 'PINHOLE':
        params = None
        camtype = camera_utils.ProjectionType.PERSPECTIVE

      if type_ == 2 or type_ == 'SIMPLE_RADIAL':
        params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
        params['k1'] = cam.k1
        camtype = camera_utils.ProjectionType.PERSPECTIVE

      elif type_ == 3 or type_ == 'RADIAL':
        params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
        params['k1'] = cam.k1
        params['k2'] = cam.k2   
        camtype = camera_utils.ProjectionType.PERSPECTIVE

      elif type_ == 4 or type_ == 'OPENCV':
        params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
        params['k1'] = cam.k1
        params['k2'] = cam.k2
        params['p1'] = cam.p1
        params['p2'] = cam.p2
        camtype = camera_utils.ProjectionType.PERSPECTIVE

      elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
        params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
        params['k1'] = cam.k1
        params['k2'] = cam.k2
        params['k3'] = cam.k3
        params['k4'] = cam.k4
        camtype = camera_utils.ProjectionType.FISHEYE
      
      distortion_params.append(params)
      camtypes.append(camtype)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    pixtocams = np.stack(pixtocams, axis=0)
    # Extract pts3d
    pts3d = self.points3D
    
    return names, poses, pixtocams, distortion_params, camtypes, pts3d


def load_blender_posedata(data_dir, split=None):
  """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
  suffix = '' if split is None else f'_{split}'
  pose_file = path.join(data_dir, f'transforms{suffix}.json')
  with utils.open_file(pose_file, 'r') as fp:
    meta = json.load(fp)
  names = []
  poses = []
  for _, frame in enumerate(meta['frames']):
    filepath = os.path.join(data_dir, frame['file_path'])
    if utils.file_exists(filepath):
      names.append(frame['file_path'].split('/')[-1])
      poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
  poses = np.stack(poses, axis=0)

  w = meta['w']
  h = meta['h']
  cx = meta['cx'] if 'cx' in meta else w / 2.
  cy = meta['cy'] if 'cy' in meta else h / 2.
  if 'fl_x' in meta:
    fx = meta['fl_x']
  else:
    fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
  if 'fl_y' in meta:
    fy = meta['fl_y']
  else:
    fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))
  pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
  coeffs = ['k1', 'k2', 'p1', 'p2']
  if not any([c in meta for c in coeffs]):
    params = None
  else:
    params = {c: (meta[c] if c in meta else 0.) for c in coeffs}
  camtype = camera_utils.ProjectionType.PERSPECTIVE
  return names, poses, pixtocam, params, camtype


class Dataset(threading.Thread, metaclass=abc.ABCMeta):
  """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    distortion_params: dict, the camera distortion model parameters.
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    near: float, near plane value for rays.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

  def __init__(self,
               split: str,
               is_training: bool, 
               sample_from_half_image: bool, 
               batch_size: int,
               patch_size: int,
               patch_dilation: int,
               image_num_per_batch: int,
               data_dir: str,
               config: configs.Config):
    super().__init__()

    # Initialize attributes
    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self._patch_size = np.maximum(patch_size, 1)
    self._batch_size = batch_size // jax.process_count()
    self._image_num_per_batch = image_num_per_batch // jax.process_count()
    self._patch_dilation = patch_dilation
    if self._image_num_per_batch * self._patch_size**2 > self._batch_size:
      raise ValueError(f'Image size {self._image_num_per_batch} * Patch size {self._patch_size}^2 too large for ' +
                       f'per-process batch size {self._batch_size}')
    self._test_camera_idx = 0
    self._render_spherical = False

    self.split = utils.DataSplit(split)
    self.is_training = is_training
    self.sample_from_half_image = sample_from_half_image
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.render_path = config.render_path
    self.distortion_params = None
    self.poses = None
    self.pixtocam_ndc = None
    self.camtypes = None
    self.pts3d = None

    # Providing type comments for these attributes, they must be correctly
    # initialized by _load_renderings() (see docstring) in any subclass.
    self.images: List[np.ndarray] = None
    self.static_masks: List[np.ndarray] = None
    self.focals: np.ndarray = None
    self.heights: np.ndarray = None
    self.widths: np.ndarray = None
    self.nears: List[np.ndarray] = None
    self.fars: List[np.ndarray] = None
    self.embed_idxs: np.ndarray = None
    self.camtoworlds: np.ndarray = None
    self.pixtocams: np.ndarray = None

    # Load data from disk using provided config parameters.
    self._load_renderings(config)

    if self.render_path:
      raise NotImplementedError()
      if config.render_path_file is not None:
        with utils.open_file(config.render_path_file, 'rb') as fp:
          render_poses = np.load(fp)
        self.camtoworlds = render_poses
        n_examples = self.camtoworlds.shape[0]
      else:
        n_examples = self.camtoworlds.shape[0]
      if config.render_resolution is not None:
        width, height = config.render_resolution
        self.widths = np.array([width for _ in range(n_examples)])
        self.heights = np.array([height for _ in range(n_examples)])
      else:
        self.widths = np.array([self.widths[0] for _ in range(n_examples)])
        self.heights = np.array([self.heights[0] for _ in range(n_examples)])
      if config.render_focal is not None:
        focal = config.render_focal
        self.focals = np.array([focal for _ in range(n_examples)])
      else:
        self.focals = np.array([self.focals[0] for _ in range(n_examples)])
      if config.render_camtype is not None:
        if config.render_camtype == 'pano':
          self._render_spherical = True
          self.camtypes = [camera_utils.ProjectionType.PERSPECTIVE for _ in range(n_examples)]
        else:
          self.camtypes = [camera_utils.ProjectionType(config.render_camtype) for _ in range(n_examples)]
      else:
        self.camtypes = [self.camtypes[0] for _ in range(n_examples)]
      if config.render_embed_idx is not None:
        embed_idx = config.render_embed_idx
        self.embed_idxs = np.array([embed_idx for _ in range(n_examples)])
      else:
        self.embed_idxs = np.array([self.embed_idxs[0] for _ in range(n_examples)])

      if self.pts3d is None:
        # use default near and far
        self.nears = [np.ones((self.heights[i], self.widths[i], 1), dtype=np.float32)*self.near for i in range(n_examples)]
        self.fars = [np.ones((self.heights[i], self.widths[i], 1), dtype=np.float32)*self.far for i in range(n_examples)]
      else:
        raise NotImplementedError()

      self.static_masks = [np.ones_like(self.nears[i]) for i in range(n_examples)]
      self.distortion_params = [None for _ in range(n_examples)]
      self.pixtocams = np.stack([
        camera_utils.get_pixtocam(self.focals[i], self.widths[i], self.heights[i])
        for i in range(n_examples)
      ], axis=0)

    self._n_examples = self.camtoworlds.shape[0]

    self.cameras = (self.pixtocams,
                    self.camtoworlds,
                    self.pixtocam_ndc)

    # Seed the queue with one batch to avoid race condition.
    if self.is_training:
      self._next_fn = self._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self._queue.get()
    if self.is_training:
      return utils.shard(x)
    else:
      # Do NOT move test `rays` to device, since it may be very large.
      return x

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    if self.is_training:
      return utils.shard(x)
    else:
      return jax.device_put(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())

  @property
  def size(self):
    return self._n_examples

  @abc.abstractmethod
  def _load_renderings(self, config):
    """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

  def _make_ray_batch(self,
                      pix_x_int: np.ndarray,
                      pix_y_int: np.ndarray,
                      cam_idx: np.int32,
                      lossmult: Optional[np.ndarray] = None
                      ) -> utils.Batch:
    """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
        'static_mask': self.static_masks[cam_idx][pix_y_int, pix_x_int],
        'near': self.nears[cam_idx][pix_y_int, pix_x_int],
        'far': self.fars[cam_idx][pix_y_int, pix_x_int],
        'cam_idx': broadcast_scalar(cam_idx),
        'embed_idx': broadcast_scalar(self.embed_idxs[cam_idx])
    }

    pixels = utils.Pixels(pix_x_int, pix_y_int, **ray_kwargs)
    distortion_params = self.distortion_params[cam_idx]
    camtype = self.camtypes[cam_idx]
    rays = camera_utils.cast_ray_batch(
      self.cameras, pixels, self.heights, self.widths, distortion_params, camtype, xnp=np
    )

    # Create data batch.
    batch = {}
    batch['rays'] = rays
    if not self.render_path:
      batch['rgb'] = self.images[cam_idx][pix_y_int, pix_x_int]
    return utils.Batch(**batch)

  def _next_train(self) -> utils.Batch:
    """Sample next training batch (random rays)."""
    # We assume all images in the dataset are the same resolution, so we can use
    # the same width/height for sampling all pixels coordinates in the batch.
    # Batch/patch sampling parameters.
    num_patches_per_image = (self._batch_size // self._image_num_per_batch) // self._patch_size ** 2
    lower_border = 0
    upper_border = 0 + (self._patch_size - 1) * self._patch_dilation
    # Add patch coordinate offsets.
    # Shape will broadcast to (num_patches, _patch_size, _patch_size).
    patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
        self._patch_size, self._patch_size)

    batch = None
    for _ in range(self._image_num_per_batch):
      # Random camera indices.
      cam_idx = np.random.randint(0, self._n_examples)
      height, width = self.heights[cam_idx], self.widths[cam_idx]
      if self.sample_from_half_image: width = width // 2
      # Random pixel patch x-coordinates.
      pix_x_int = np.random.randint(lower_border, width - upper_border,
                                    (num_patches_per_image, 1, 1))
      # Random pixel patch y-coordinates.
      pix_y_int = np.random.randint(lower_border, height - upper_border,
                                    (num_patches_per_image, 1, 1))
      pix_x_int = pix_x_int + patch_dx_int * self._patch_dilation
      pix_y_int = pix_y_int + patch_dy_int * self._patch_dilation

      new_batch = self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)
      if batch is None:
        batch = new_batch
      else:
        batch = tree_map(
          lambda x, y: np.concatenate([x, y], axis=0), batch, new_batch
        )
    return batch

  def generate_ray_batch(self, cam_idx: int) -> utils.Batch:
    """Generate ray batch for a specified camera in the dataset."""
    if self._render_spherical:
      camtoworld = self.camtoworlds[cam_idx]
      rays = camera_utils.cast_spherical_rays(
          camtoworld, self.heights[cam_idx], self.widths[cam_idx], self.near, self.far, xnp=np)
      return utils.Batch(rays=rays)
    else:
      # Generate rays for all pixels in the image.
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.widths[cam_idx], self.heights[cam_idx])
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

  def _next_test(self) -> utils.Batch:
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples
    return self.generate_ray_batch(cam_idx)


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')

    embed_offsets = {}
    embed_offset = 0
    for val in ['train', 'test']:
      pose_file = path.join(self.data_dir, f'transforms_{val}.json')
      with utils.open_file(pose_file, 'r') as fp:
        meta = json.load(fp)
        embed_offsets[val] = embed_offset
        embed_offset += len(meta['frames'])

    pose_file = path.join(self.data_dir, f'transforms_{self.split.value}.json')
    with utils.open_file(pose_file, 'r') as fp:
      meta = json.load(fp)

    self.images = []
    self.static_masks = []
    self.nears = []
    self.fars = []
    self.focals = []
    self.heights = []
    self.widths = []
    self.embed_idxs = []
    self.camtoworlds = []
    self.pixtocams = []
    self.distortion_params = []
    self.camtypes = []

    static_mask_dir = os.path.join(self.data_dir, config.static_mask_dir_name)
    if not utils.file_exists(static_mask_dir):
      print(f"{static_mask_dir} does not exist.")
    for img_idx, frame in enumerate(meta['frames']):
      fprefix = os.path.join(self.data_dir, frame['file_path'])

      def get_img(f, fprefix=fprefix):
        image = utils.load_img(fprefix + f)
        if config.factor > 1:
          image = lib_image.downsample(image, config.factor)
        return image

      image = get_img('.png') / 255.
      rgb, alpha = image[..., :3], image[..., -1:]
      image = rgb * alpha + (1. - alpha) # Use a white background.
      height, width = image.shape[:2]

      static_mask_path = os.path.join(static_mask_dir, f"{frame['file_path']}.png")
      if utils.file_exists(static_mask_path):
        static_mask = utils.load_img(static_mask_path) / 255.
        if static_mask.shape[0] != height or static_mask.shape[1] != width:
          static_mask = cv2.resize(static_mask, (width, height))
      else:
        static_mask = np.ones_like(image)
      
      self.images.append(image)
      self.static_masks.append(static_mask[...,:1].reshape(height, width, 1))
      self.nears.append(np.ones((height, width, 1), dtype=np.float32) * self.near)
      self.fars.append(np.ones((height, width, 1), dtype=np.float32) * self.far)
      focal = .5 * width / np.tan(.5 * float(meta['camera_angle_x']))
      self.focals.append(focal)
      self.heights.append(height)
      self.widths.append(width)
      self.embed_idxs.append(embed_offsets[self.split.value] + img_idx)
      self.camtoworlds.append(np.array(frame['transform_matrix'], dtype=np.float32))
      self.pixtocams.append(camera_utils.get_pixtocam(focal, width, height))
      self.distortion_params.append(None)
      self.camtypes.append(camera_utils.ProjectionType.PERSPECTIVE)

    self.heights = np.array(self.heights)
    self.widths = np.array(self.widths)
    self.focals = np.array(self.focals)
    self.embed_idxs = np.array(self.embed_idxs)
    self.camtoworlds = np.stack(self.camtoworlds, axis=0)
    self.pixtocams = np.stack(self.pixtocams, axis=0)


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    # Set up scaling factor.
    image_dir_suffix = ''
    # Use downsampling factor
    if config.factor > 0:
      image_dir_suffix = f'_{config.factor}'
      factor = config.factor
    else:
      factor = 1

    # Copy COLMAP data to local disk for faster loading.
    colmap_dir = os.path.join(self.data_dir, 'sparse/0/')

    # Load poses.
    if utils.file_exists(colmap_dir):
      pose_data = NeRFSceneManager(colmap_dir).process()
    else:
      # Attempt to load Blender/NGP format if COLMAP data not present.
      raise NotImplementedError()
      pose_data = load_blender_posedata(self.data_dir)
    image_names, poses, pixtocams, distortion_params, camtypes, _ = pose_data

    # Previous NeRF results were generated with images sorted by filename,
    # use this flag to ensure metrics are reported on the same test set.
    if config.load_alphabetical:
      inds = np.argsort(image_names)
      image_names = [image_names[i] for i in inds]
      poses = poses[inds]
      pixtocams = pixtocams[inds]
      distortion_params = [distortion_params[i] for i in inds]
      camtypes = [camtypes[i] for i in inds]

    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocams = pixtocams @ np.diag([factor, factor, 1.])
    self.pixtocams = pixtocams.astype(np.float32)
    self.focals = 1. / self.pixtocams[:, 0, 0]
    self.distortion_params = distortion_params
    self.camtypes = camtypes

    # Load images.
    colmap_image_dir = os.path.join(self.data_dir, 'images')
    image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
      if not utils.file_exists(d):
        raise ValueError(f'Image folder {d} does not exist.')
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(utils.listdir(colmap_image_dir))
    image_files = sorted(utils.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f])
                   for f in image_names]
    images = [utils.load_img(x) / 255. for x in image_paths]
    
    # Load static mask
    static_mask_dir = os.path.join(self.data_dir, config.static_mask_dir_name)
    if not utils.file_exists(static_mask_dir):
      print(f"{static_mask_dir} does not exist. ")
      static_masks = [np.ones_like(img) for img in images]
    else:
      static_masks = []
      for idx, f in enumerate(image_names):
        height, width = images[idx].shape[:2]
        static_mask_path = os.path.join(
          static_mask_dir, f"{Path(colmap_to_image[f]).stem}.png"
        )
        static_mask = utils.load_img(static_mask_path) / 255.
        if static_mask.shape[0] != height or static_mask.shape[1] != width:
          static_mask = cv2.resize(static_mask, (width, height))
        static_masks.append(static_mask[..., :1].reshape(height, width, 1))

    # Load bounds if possible (only used in forward facing scenes).
    posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
    if utils.file_exists(posefile):
      with utils.open_file(posefile, 'rb') as fp:
        poses_arr = np.load(fp)
      bounds = poses_arr[:, -2:]
    else:
      bounds = np.array([0.01, 1.])
    self.colmap_to_world_transform = np.eye(4)

    # Separate out 360 versus forward facing scenes.
    if config.forward_facing:
      # Set the projective matrix defining the NDC transformation.
      self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
      # Rescale according to a default bd factor.
      scale = 1. / (bounds.min() * .75)
      poses[:, :3, 3] *= scale
      self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
      bounds *= scale
      # Recenter poses.
      poses, transform = camera_utils.recenter_poses(poses)
      self.colmap_to_world_transform = (
          transform @ self.colmap_to_world_transform)
      # Forward-facing spiral render path.
      self.render_poses = camera_utils.generate_spiral_path(
          poses, bounds, n_frames=config.render_path_frames)
    else:
      # Rotate/scale poses to align ground with xy plane and fit to unit cube.
      poses, transform = camera_utils.transform_poses_pca(poses)
      self.colmap_to_world_transform = transform
      if config.render_spline_keyframes is not None:
        rets = camera_utils.create_render_spline_path(config, image_names, poses)
        self.spline_indices, self.render_poses = rets
      else:
        # Automatically generated inward-facing elliptical render path.
        self.render_poses = camera_utils.generate_ellipse_path(
            poses,
            n_frames=config.render_path_frames,
            z_variation=config.z_variation,
            z_phase=config.z_phase)

    self.poses = poses

    # Select the split.
    all_indices = np.arange(poses.shape[0])
    if config.llff_use_all_images_for_training:
      train_indices = all_indices
    else:
      train_indices = all_indices[all_indices % config.llffhold != 0]
    split_indices = {
        utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        utils.DataSplit.TRAIN: train_indices,
    }
    indices = split_indices[self.split]
    # All per-image quantities must be re-indexed using the split indices.
    poses = poses[indices]
    self.pixtocams = self.pixtocams[indices]
    self.focals = self.focals[indices]
    self.distortion_params = [self.distortion_params[i] for i in indices]
    self.camtypes = [self.camtypes[i] for i in indices]
    self.embed_idxs = np.array(indices)
    image_names = [image_names[i] for i in indices]
    self.images = [images[i] for i in indices]
    self.static_masks = [static_masks[i] for i in indices]

    self.heights = []
    self.widths = []
    self.nears = []
    self.fars = []
    for img in self.images:
      self.heights.append(img.shape[0])
      self.widths.append(img.shape[1])
      self.nears.append(np.ones((*img.shape[:2], 1), dtype=np.float32) * self.near)
      self.fars.append(np.ones((*img.shape[:2], 1), dtype=np.float32) * self.far)

    self.heights = np.array(self.heights)
    self.widths = np.array(self.widths)
    self.camtoworlds = self.render_poses if config.render_path else poses


class TanksAndTemplesNerfPP(Dataset):
  """Subset of Tanks and Temples Dataset as processed by NeRF++."""

  def _load_renderings(self, config):
    raise NotImplementedError()
    """Load images from disk."""
    if config.render_path:
      split_str = 'camera_path'
    else:
      split_str = self.split.value

    basedir = os.path.join(self.data_dir, split_str)

    def load_files(dirname, load_fn, shape=None):
      files = [
          os.path.join(basedir, dirname, f)
          for f in sorted(utils.listdir(os.path.join(basedir, dirname)))
      ]
      mats = np.array([load_fn(utils.open_file(f, 'rb')) for f in files])
      if shape is not None:
        mats = mats.reshape(mats.shape[:1] + shape)
      return mats

    poses = load_files('pose', np.loadtxt, (4, 4))
    # Flip Y and Z axes to get correct coordinate frame.
    poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

    # For now, ignore all but the first focal length in intrinsics
    intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

    if not config.render_path:
      images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.
      self.images = images
      self.height, self.width = self.images.shape[1:3]

    else:
      # Hack to grab the image resolution from a test image
      d = os.path.join(self.data_dir, 'test', 'rgb')
      f = os.path.join(d, sorted(utils.listdir(d))[0])
      shape = utils.load_img(f).shape
      self.height, self.width = shape[:2]
      self.images = None

    self.camtoworlds = poses
    self.focal = intrinsics[0, 0, 0]
    self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                               self.height)


class TanksAndTemplesFVS(Dataset):
  """Subset of Tanks and Temples Dataset as processed by Free View Synthesis."""

  def _load_renderings(self, config):
    raise NotImplementedError()
    """Load images from disk."""
    render_only = config.render_path and self.split == utils.DataSplit.TEST

    basedir = os.path.join(self.data_dir, 'dense')
    sizes = [f for f in sorted(utils.listdir(basedir)) if f.startswith('ibr3d')]
    sizes = sizes[::-1]

    if config.factor >= len(sizes):
      raise ValueError(f'Factor {config.factor} larger than {len(sizes)}')

    basedir = os.path.join(basedir, sizes[config.factor])
    open_fn = lambda f: utils.open_file(os.path.join(basedir, f), 'rb')

    files = [f for f in sorted(utils.listdir(basedir)) if f.startswith('im_')]
    if render_only:
      files = files[:1]
    images = np.array([np.array(Image.open(open_fn(f))) for f in files]) / 255.

    names = ['Ks', 'Rs', 'ts']
    intrinsics, rot, trans = (np.load(open_fn(f'{n}.npy')) for n in names)

    # Convert poses from colmap world-to-cam into our cam-to-world.
    w2c = np.concatenate([rot, trans[..., None]], axis=-1)
    c2w_colmap = np.linalg.inv(camera_utils.pad_poses(w2c))[:, :3, :4]
    c2w = c2w_colmap @ np.diag(np.array([1, -1, -1, 1]))

    # Reorient poses so z-axis is up
    poses, _ = camera_utils.transform_poses_pca(c2w)
    self.poses = poses

    self.images = images
    self.height, self.width = self.images.shape[1:3]
    self.camtoworlds = poses
    # For now, ignore all but the first focal length in intrinsics
    self.focal = intrinsics[0, 0, 0]
    self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                               self.height)

    if render_only:
      render_path = camera_utils.generate_ellipse_path(
          poses,
          config.render_path_frames,
          z_variation=config.z_variation,
          z_phase=config.z_phase)
      self.images = None
      self.camtoworlds = render_path
      self.render_poses = render_path
    else:
      # Select the split.
      all_indices = np.arange(images.shape[0])
      indices = {
          utils.DataSplit.TEST:
              all_indices[all_indices % config.llffhold == 0],
          utils.DataSplit.TRAIN:
              all_indices[all_indices % config.llffhold != 0],
      }[self.split]

      self.images = self.images[indices]
      self.camtoworlds = self.camtoworlds[indices]


class DTU(Dataset):
  """DTU Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    raise NotImplementedError()
    if config.render_path:
      raise ValueError('render_path cannot be used for the DTU dataset.')

    images = []
    pixtocams = []
    camtoworlds = []

    # Find out whether the particular scan has 49 or 65 images.
    n_images = len(utils.listdir(self.data_dir)) // 8

    # Loop over all images.
    for i in range(1, n_images + 1):
      # Set light condition string accordingly.
      if config.dtu_light_cond < 7:
        light_str = f'{config.dtu_light_cond}_r' + ('5000'
                                                    if i < 50 else '7000')
      else:
        light_str = 'max'

      # Load image.
      fname = os.path.join(self.data_dir, f'rect_{i:03d}_{light_str}.png')
      image = utils.load_img(fname) / 255.
      if config.factor > 1:
        image = lib_image.downsample(image, config.factor)
      images.append(image)

      # Load projection matrix from file.
      fname = path.join(self.data_dir, f'../../cal18/pos_{i:03d}.txt')
      with utils.open_file(fname, 'rb') as f:
        projection = np.loadtxt(f, dtype=np.float32)

      # Decompose projection matrix into pose and camera matrix.
      camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
      camera_mat = camera_mat / camera_mat[2, 2]
      pose = np.eye(4, dtype=np.float32)
      pose[:3, :3] = rot_mat.transpose()
      pose[:3, 3] = (t[:3] / t[3])[:, 0]
      pose = pose[:3]
      camtoworlds.append(pose)

      if config.factor > 0:
        # Scale camera matrix according to downsampling factor.
        camera_mat = np.diag([1. / config.factor, 1. / config.factor, 1.
                             ]).astype(np.float32) @ camera_mat
      pixtocams.append(np.linalg.inv(camera_mat))

    pixtocams = np.stack(pixtocams)
    camtoworlds = np.stack(camtoworlds)
    images = np.stack(images)

    def rescale_poses(poses):
      """Rescales camera poses according to maximum x/y/z value."""
      s = np.max(np.abs(poses[:, :3, -1]))
      out = np.copy(poses)
      out[:, :3, -1] /= s
      return out

    # Center and scale poses.
    camtoworlds, _ = camera_utils.recenter_poses(camtoworlds)
    camtoworlds = rescale_poses(camtoworlds)
    # Flip y and z axes to get poses in OpenGL coordinate system.
    camtoworlds = camtoworlds @ np.diag([1., -1., -1., 1.]).astype(np.float32)

    all_indices = np.arange(images.shape[0])
    split_indices = {
        utils.DataSplit.TEST: all_indices[all_indices % config.dtuhold == 0],
        utils.DataSplit.TRAIN: all_indices[all_indices % config.dtuhold != 0],
    }
    indices = split_indices[self.split]

    self.images = images[indices]
    self.height, self.width = images.shape[1:3]
    self.camtoworlds = camtoworlds[indices]
    self.pixtocams = pixtocams[indices]


class Kubric(Dataset):
  """Kubric Dataset"""
  def _load_renderings(self, config):
    if config.factor > 0:
      factor = config.factor
    else:
      factor = 1

    with open(os.path.join(self.data_dir, 'scene_gt.json'), 'r') as f:
      scene_json = json.load(f)
      scene_center = np.array(scene_json['center'])
      scene_scale = scene_json['scale']
      self.scale_factor = scene_scale
      scene_near = scene_json['near']
      scene_far = scene_json['far'] * 1.2 # original far is not enough
    
    with open(os.path.join(self.data_dir,"dataset.json"), 'r') as f:
      dataset_json = json.load(f)
      train_image_names = dataset_json['train_ids']
      train_image_names = [str(i) for i in train_image_names]
    with open(os.path.join(self.data_dir, "freeze-test/dataset.json"), 'r') as f:
      dataset_json = json.load(f)
      val_image_names = dataset_json['val_ids']
      val_image_names = [str(i) for i in val_image_names]
    
    if self.split == utils.DataSplit.TRAIN:
      image_dir = os.path.join(self.data_dir, f'rgb/{factor}x')
      static_mask_dir = os.path.join(self.data_dir, config.static_mask_dir_name)
      camera_dir = os.path.join(self.data_dir, 'camera-gt')
      image_names = train_image_names
      embed_offset = 0
    elif self.split == utils.DataSplit.TEST:
      image_dir = os.path.join(self.data_dir, f'freeze-test/static-rgb/{factor}x')
      static_mask_dir = os.path.join(self.data_dir, f'freeze-test/{config.static_mask_dir_name}')
      camera_dir = os.path.join(self.data_dir, 'freeze-test/camera-gt')
      image_names = val_image_names
      embed_offset = len(train_image_names)
    
    images = []
    static_masks = []
    nears = []
    fars = []
    focals = []
    heights = []
    widths = []
    embed_idxs = []
    camtoworlds = []
    pixtocams = []
    distortion_params = []
    camtypes = []

    if not utils.file_exists(static_mask_dir):
      print(f"{static_mask_dir} does not exist. ")
    for i, image_name in enumerate(image_names):
      with open(os.path.join(camera_dir, f'{image_name}.json'), 'r') as f:
        camera_json = json.load(f)
      
      orientation = np.asarray(camera_json['orientation'])
      position = np.asarray(camera_json['position'])
      focal_length = camera_json['focal_length']
      principal_point = np.asarray(camera_json['principal_point'])
      skew = camera_json['skew']
      pixel_aspect_ratio = camera_json['pixel_aspect_ratio']
      radial_distortion = np.asarray(camera_json['radial_distortion'])
      tangential_distortion = np.asarray(camera_json['tangential_distortion'])

      scale_factor_x = focal_length
      scale_factor_y = focal_length * pixel_aspect_ratio
      pixtocam = np.array([
          [1 / scale_factor_x, - skew / scale_factor_x, - principal_point[0] / scale_factor_x],
          [                 0,      1 / scale_factor_y, - principal_point[1] / scale_factor_y],
          [                 0,                       0,                                     1]
      ], dtype=np.float32)
      if factor > 1:
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])
      
      distortion_param = {
        'k1': radial_distortion[0], 'k2': radial_distortion[1], 'k3': radial_distortion[2],
        'p1': tangential_distortion[0], 'p2': tangential_distortion[1]
      }

      camtoworld = np.concatenate([orientation.T, position.reshape(3,1)], axis=1) # (3,4)
      # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
      camtoworld = camtoworld @ np.diag([1, -1, -1, 1])
      # recenter and rescale
      camtoworld[:3, 3] -= scene_center
      camtoworld[:3, 3] *= scene_scale

      # load image
      image = utils.load_img(os.path.join(image_dir, f"{image_name}.png")) / 255.
      if image.shape[-1]==4:
        image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:]) # Use a white background.
      height, width = image.shape[:2]

      static_mask_path = os.path.join(static_mask_dir, f"{image_name}.png")
      if utils.file_exists(static_mask_path):
        static_mask = utils.load_img(static_mask_path) / 255.
        if static_mask.shape[0] != height or static_mask.shape[1] != width:
          static_mask = cv2.resize(static_mask, (width, height))
      else:
        static_mask = np.ones_like(image[..., :3])

      images.append(image)
      static_masks.append(static_mask[..., :1].reshape(height, width, 1))
      nears.append(np.ones((height, width, 1), dtype=np.float32) * scene_near)
      fars.append(np.ones((height, width, 1), dtype=np.float32) * scene_far)
      heights.append(height)
      widths.append(width)
      embed_idxs.append(embed_offset + i)
      focals.append(focal_length / factor)
      distortion_params.append(distortion_param)
      camtypes.append(camera_utils.ProjectionType.PERSPECTIVE)
      camtoworlds.append(camtoworld)
      pixtocams.append(pixtocam)

    self.images = images
    self.static_masks = static_masks
    self.nears = nears
    self.fars = fars
    self.heights = np.array(heights)
    self.widths = np.array(widths)
    self.focals = np.array(focals, dtype=np.float32)
    self.embed_idxs = np.array(embed_idxs)
    self.distortion_params = distortion_params
    self.camtypes = camtypes

    self.camtoworlds = np.stack(camtoworlds, axis=0)
    self.pixtocams = np.stack(pixtocams, axis=0)


PHOTOTOURISM_BOUND_DICT = {
  'brandenburg_gate': 24,
  'sacre_coeur': 11,
  'taj_mahal': 16,
  'trevi_fountain': 35
}

class Phototourism(Dataset):
  """Phototourism Dataset"""

  def _load_renderings(self, config):
    # Use downsampling factor
    if config.factor > 0:
      factor = config.factor
    else:
      factor = 1
    
    # Copy COLMAP data to local disk for faster loading.
    colmap_dir = os.path.join(self.data_dir, 'dense/sparse')

    pose_data = NeRFSceneManager(colmap_dir).process()
    image_names, poses, pixtocams, distortion_params, camtypes, pts3d = pose_data

    # read all files in the tsv first (split to train and test later)
    split_file = list(Path(self.data_dir).glob("*.tsv"))[0]
    split_data = pd.read_csv(split_file, sep='\t')
    # the split
    train_image_names = []
    test_image_names = []
    for i in range(len(split_data)):
      if split_data['split'][i]=='train': train_image_names.append(split_data['filename'][i])
      elif split_data['split'][i]=='test': test_image_names.append(split_data['filename'][i])
    all_image_names = train_image_names + test_image_names
    if self.split == utils.DataSplit.TRAIN:
      selected_image_names = train_image_names
    elif self.split==utils.DataSplit.TEST:
      selected_image_names = test_image_names
    else:
      raise NotImplementedError()
    
    # select the related data in tsv
    poses_, pixtocams_, distortion_params_, camtypes_ = [], [], [], []
    for image_name in all_image_names:
      image_idx = image_names.index(image_name)
      poses_.append(poses[image_idx])
      pixtocams_.append(pixtocams[image_idx])
      distortion_params_.append(distortion_params[image_idx])
      camtypes_.append(camtypes[image_idx])
    del image_names, poses, pixtocams, distortion_params, camtypes
    image_names = all_image_names
    poses = np.stack(poses_, axis=0)
    pixtocams = np.stack(pixtocams_, axis=0)
    distortion_params = distortion_params_
    camtypes = camtypes_
    
    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocams = pixtocams @ np.diag([factor, factor, 1.])
    pixtocams = pixtocams.astype(np.float32)
    focals = 1. / pixtocams[:, 0, 0]

    # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    poses, transform = camera_utils.recenter_poses(poses)
    pts3d = np.concatenate([pts3d, np.ones_like(pts3d[..., :1])], axis=-1)
    pts3d = pts3d @ transform.T
    # use object as center
    points_center = pts3d[:,:3].mean(0)
    center_transform = np.eye(4)
    center_transform[:3, 3] = - points_center
    poses = camera_utils.unpad_poses(center_transform @ camera_utils.pad_poses(poses))
    pts3d = pts3d @ center_transform.T
    transform = center_transform @ transform

    bound = PHOTOTOURISM_BOUND_DICT[Path(self.data_dir).name]
    scale_factor = 2 / bound
    new_transform = np.diag([scale_factor, scale_factor, scale_factor, 1])
    poses[..., :3, 3] *= scale_factor
    pts3d = pts3d @ new_transform.T
    transform = new_transform @ transform

    self.colmap_to_world_transform = transform
    self.poses = poses
    self.pts3d = pts3d

    # load images and related data
    self.images = []
    self.static_masks = []
    self.nears = []
    self.fars = []
    self.focals = []
    self.heights = []
    self.widths = []
    self.embed_idxs = []
    self.camtoworlds = []
    self.pixtocams = []
    self.distortion_params = []
    self.camtypes = []

    image_dir = os.path.join(self.data_dir, 'dense/images')
    static_mask_dir = os.path.join(self.data_dir, f'dense/{config.static_mask_dir_name}')
    if not utils.file_exists(static_mask_dir):
      print(f"{static_mask_dir} does not exist. ")

    for i, image_name in enumerate(selected_image_names):
      image_idx = image_names.index(image_name)
      image_path = os.path.join(image_dir, image_name)
      image = utils.load_img(image_path) / 255.
      height, width = image.shape[:2]

      static_mask_path = os.path.join(static_mask_dir, f"{image_name.split('.')[0]}.png")
      if utils.file_exists(static_mask_path):
        static_mask = utils.load_img(static_mask_path) / 255.
      else:
        static_mask = np.ones_like(image)

      if factor > 1:
        height, width = height//factor, width//factor
        image = cv2.resize(image, (width, height))
      if static_mask.shape[0] != height or static_mask.shape[1] != width:
        static_mask = cv2.resize(static_mask, (width, height))

      # use pts3d to compute near and far
      pose = camera_utils.pad_poses(poses[image_idx]) @ np.diag([1,-1,-1,1]) # use colmap coordinate system
      w2c = np.linalg.inv(pose)
      pts_cam = (pts3d @ w2c.T)[:, :3] # xyz in the idx-th cam coordinate
      pts_cam = pts_cam[pts_cam[:, 2]>0] # filter out points that lie behind the 
      near = np.percentile(pts_cam[:, 2], 0.1)
      far = np.percentile(pts_cam[:, 2], 99.9)

      self.images.append(image.reshape(height, width, 3))
      self.static_masks.append(static_mask[..., :1].reshape(height, width, 1))
      self.nears.append(np.ones((height, width, 1), dtype=np.float32) * near)
      self.fars.append(np.ones((height, width, 1), dtype=np.float32) * far)
      self.focals.append(focals[image_idx])
      self.heights.append(height)
      self.widths.append(width)
      self.embed_idxs.append(image_idx)
      self.camtoworlds.append(poses[image_idx])
      self.pixtocams.append(pixtocams[image_idx])
      self.distortion_params.append(distortion_params[image_idx])
      self.camtypes.append(camtypes[image_idx])

    self.focals = np.array(self.focals)
    self.heights = np.array(self.heights)
    self.widths = np.array(self.widths)
    self.embed_idxs = np.array(self.embed_idxs)
    self.camtoworlds = np.stack(self.camtoworlds, axis=0)
    self.pixtocams = np.stack(self.pixtocams, axis=0)


class Distractor(Dataset):
  """Dataset from RobustNeRF"""
  def _load_renderings(self, config):
    # Set up scaling factor.
    image_dir_suffix = ''
    if config.factor > 0:
      image_dir_suffix = f'_{config.factor}'
      factor = config.factor
    else:
      factor = 1
    
    # Copy COLMAP data to local disk for faster loading.
    colmap_dir = os.path.join(self.data_dir, '0/sparse/0')

    pose_data = NeRFSceneManager(colmap_dir).process()
    image_names, poses, pixtocams, distortion_params, camtypes, pts3d = pose_data

    # read all files in the json first (split to train and test later)
    with open(os.path.join(self.data_dir, "0/data_split.json")) as fp:
      split_data = json.load(fp)
    # the split
    train_image_names = split_data['train']
    test_image_names = split_data['test']
    all_image_names = train_image_names + test_image_names
    if self.split==utils.DataSplit.TRAIN:
      selected_image_names = train_image_names
    elif self.split==utils.DataSplit.TEST:
      selected_image_names = test_image_names
    else:
      raise NotImplementedError()
    
    # select the related data
    poses_, pixtocams_, distortion_params_, camtypes_ = [], [], [], []
    for image_name in all_image_names:
      image_idx = image_names.index(image_name)
      poses_.append(poses[image_idx])
      pixtocams_.append(pixtocams[image_idx])
      distortion_params_.append(distortion_params[image_idx])
      camtypes_.append(camtypes[image_idx])
    del image_names, poses, pixtocams, distortion_params, camtypes
    image_names = all_image_names
    poses = np.stack(poses_, axis=0)
    pixtocams = np.stack(pixtocams_, axis=0)
    distortion_params = distortion_params_
    camtypes = camtypes_
    
    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocams = pixtocams @ np.diag([factor, factor, 1.])
    pixtocams = pixtocams.astype(np.float32)
    focals = 1. / pixtocams[:, 0, 0]

    # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    poses, transform = camera_utils.transform_poses_pca(poses)
    pts3d = np.concatenate([pts3d, np.ones_like(pts3d[..., :1])], axis=-1)
    pts3d = pts3d @ transform.T
    # use object as center
    points_center = pts3d[:,:3].mean(0)
    center_transform = np.eye(4)
    center_transform[:3, 3] = - points_center
    poses = camera_utils.unpad_poses(center_transform @ camera_utils.pad_poses(poses))
    pts3d = pts3d @ center_transform.T
    transform = center_transform @ transform
    # make camera into unit cube again
    scale_factor = 1. / np.max(np.abs(poses[:, :3, 3]))
    poses[:, :3, 3] *= scale_factor
    pts3d[:, :3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    self.colmap_to_world_transform = transform
    self.poses = poses
    self.pts3d = pts3d

    # load images and related data
    self.images = []
    self.static_masks = []
    self.nears = []
    self.fars = []
    self.focals = []
    self.heights = []
    self.widths = []
    self.embed_idxs = []
    self.camtoworlds = []
    self.pixtocams = []
    self.distortion_params = []
    self.camtypes = []

    image_dir = os.path.join(self.data_dir, f'0/images{image_dir_suffix}')
    static_mask_dir = os.path.join(self.data_dir, f'0/{config.static_mask_dir_name}')
    if not utils.file_exists(static_mask_dir):
      print(f"{static_mask_dir} does not exist. ")

    for i, image_name in enumerate(selected_image_names):
      image_idx = image_names.index(image_name)
      image_path = os.path.join(image_dir, image_name)
      image = utils.load_img(image_path) / 255.
      height, width = image.shape[:2]

      static_mask_path = os.path.join(static_mask_dir, f"{image_name.split('.')[0]}.png")
      if utils.file_exists(static_mask_path):
        static_mask = utils.load_img(static_mask_path) / 255.
        if static_mask.shape[0] != height or static_mask.shape[1] != width:
          static_mask = cv2.resize(static_mask, (width, height))
      else:
        static_mask = np.ones_like(image)

      # use pts3d to compute near and far
      pose = camera_utils.pad_poses(poses[image_idx]) @ np.diag([1,-1,-1,1]) # use colmap coordinate system
      w2c = np.linalg.inv(pose)
      pts_cam = (pts3d @ w2c.T)[:, :3] # xyz in the idx-th cam coordinate
      pts_cam = pts_cam[pts_cam[:, 2]>=0] # filter out points that lie behind the 
      pts_uv = (pts_cam @ np.linalg.inv(pixtocams[image_idx]).T) / np.maximum(pts_cam[:, 2:], np.finfo(pts_cam.dtype).eps)
      is_in_cone = (pts_uv[:, 0] <= width) * (pts_uv[:, 0] >= 0) \
                  * (pts_uv[:, 1] <= height) * (pts_uv[:, 1] >= 0)
      pts_cam = pts_cam[is_in_cone]
      near = np.percentile(pts_cam[:, 2], 0.1) * 0.8 # min(np.percentile(pts_cam[:, 2], 0.1) * 0.8, self.near)
      far = self.far

      self.images.append(image.reshape(height, width, 3))
      self.static_masks.append(static_mask[..., :1].reshape(height, width, 1))
      self.nears.append(np.ones((height, width, 1), dtype=np.float32) * near)
      self.fars.append(np.ones((height, width, 1), dtype=np.float32) * far)
      self.focals.append(focals[image_idx])
      self.heights.append(height)
      self.widths.append(width)
      self.embed_idxs.append(image_idx)
      self.camtoworlds.append(poses[image_idx])
      self.pixtocams.append(pixtocams[image_idx])
      self.distortion_params.append(distortion_params[image_idx])
      self.camtypes.append(camtypes[image_idx])

    self.focals = np.array(self.focals)
    self.heights = np.array(self.heights)
    self.widths = np.array(self.widths)
    self.embed_idxs = np.array(self.embed_idxs)
    self.camtoworlds = np.stack(self.camtoworlds, axis=0)
    self.pixtocams = np.stack(self.pixtocams, axis=0)