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

"""Utility functions."""

import enum
import os
from typing import Any, Dict, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import ExifTags
from PIL import Image

_Array = Union[np.ndarray, jnp.ndarray]


@flax.struct.dataclass
class Pixels:
  """All tensors must have the same num_dims and first n-1 dims must match."""
  pix_x_int: _Array
  pix_y_int: _Array
  lossmult: _Array
  static_mask: _Array
  near: _Array
  far: _Array
  embed_idx: _Array
  cam_idx: _Array


@flax.struct.dataclass
class Rays:
  """All tensors must have the same num_dims and first n-1 dims must match."""
  pix_coords: _Array
  origins: _Array
  directions: _Array
  viewdirs: _Array
  radii: _Array
  lossmult: _Array
  static_mask: _Array
  near: _Array
  far: _Array
  embed_idx: _Array
  cam_idx: _Array


# Dummy Rays object that can be used to initialize NeRF model.
def dummy_rays() -> Rays:
  data_fn = lambda n: jnp.zeros((1, n))
  return Rays(
      pix_coords=data_fn(2),
      origins=data_fn(3),
      directions=data_fn(3),
      viewdirs=data_fn(3),
      radii=data_fn(1),
      lossmult=data_fn(1),
      static_mask=data_fn(1),
      near=data_fn(1),
      far=data_fn(1),
      embed_idx=data_fn(1).astype(jnp.int32),
      cam_idx=data_fn(1).astype(jnp.int32))


@flax.struct.dataclass
class Batch:
  """Data batch for NeRF training or testing."""
  rays: Union[Pixels, Rays]
  rgb: Optional[_Array] = None


class DataSplit(enum.Enum):
  """Dataset split."""
  TRAIN = 'train'
  TEST = 'test'


class BatchingMethod(enum.Enum):
  """Draw rays randomly from a single image or all images, in each batch."""
  ALL_IMAGES = 'all_images'
  SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return os.path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return os.path.isdir(pth)


def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_util.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def load_exif(pth: str) -> Dict[str, Any]:
  """Load EXIF data for an image."""
  with open_file(pth, 'rb') as f:
    image_pil = Image.open(f)
    exif_pil = image_pil._getexif()  # pylint: disable=protected-access
    if exif_pil is not None:
      exif = {
          ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS
      }
    else:
      exif = {}
  return exif


def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
