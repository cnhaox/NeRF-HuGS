from typing import Dict, Optional, Tuple, List, Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.distributed as dist

from utils.config_utils import BaseConfig
from utils.ray_utils import intersect_aabb
from utils.camera_utils import pixels_to_rays, ProjectionType

class BaseDataset(Dataset):
    def __init__(
        self,
        data_split: str,
        training: bool,
        batch_size: int,
        patch_size: int,
        patch_dilation: int,
        num_img_per_batch: int,
        data_dir: str,
        sample_from_half_image: bool,
        config: BaseConfig,
        record_func: callable = print,
    ) -> None:
        super().__init__()
        if config.rescale_scene or config.enable_clip_near_far:
            assert config.bound is not None
        if config.enable_scene_contraction:
            assert config.bound == 1, "Set bound to be 1 first. It will be changed to 2 later."
            assert config.rescale_scene
            assert not config.enable_clip_near_far

        # Initialize training attributes
        self.batch_size: int = batch_size
        self.patch_size: int = patch_size
        self.patch_dilation: int = patch_dilation
        self.num_img_per_batch: int = num_img_per_batch
        self.num_patch_per_batch = self.batch_size // self.patch_size**2
        self.num_patch_per_img = self.num_patch_per_batch // self.num_img_per_batch

        self.sample_from_half_image: bool = sample_from_half_image

        self.split: str = data_split
        self.training: bool = training
        self.data_dir = Path(data_dir)
        self.downsample_factor: int = config.downsample_factor
        self.near: Optional[float] = config.near
        self.far: Optional[float] = config.far
        self.render_path: bool = config.render_path
        self.bound: Optional[float] = config.bound
        self.rescale_scene: bool = config.rescale_scene
        self.enable_ndc: bool = config.enable_ndc

        self.background_color_type: str = config.train_background_color if self.training \
                                    else config.test_background_color
        
        # Attention! We use the same opengl camera coordinates (right, up, back) as original NeRF.
        self.scale_factor: float = 1.0
        self.transform: Tensor = torch.eye(4)            # (4, 4)
        self.image_names: Optional[List[str]] = None     # 
        self.n_samples: Optional[int] = None             # N_images
        self.images: Optional[List[Tensor]] = None       # (h, w, 3/4) * N_images
        self.freqs: Optional[List[Tensor]] = None        # (h, w, 1) * N_images
        self.static_masks: Optional[List[Tensor]] = None # (h, w, 1) * N_images
        self.nears: Optional[List[Tensor]] = None        # (h, w, 1) * N_images
        self.fars: Optional[List[Tensor]] = None         # (h, w, 1) * N_images

        self.heights: Optional[Tensor] = None            # (N_images, 1)
        self.widths: Optional[Tensor] = None             # (N_images, 1)
        self.embed_idxs: Optional[Tensor] = None         # (N_images, 1)
        self.cam2worlds: Optional[Tensor] = None         # (N_images, 4, 4)
        self.pix2cams: Optional[Tensor] = None           # (N_images, 3, 3)
        self.pix2cam_ndc: Optional[Tensor] = None        # (3, 3)

        self.distortion_params: Optional[List[Optional[Mapping[str, float]]]] = None # _ * N_images
        self.camtypes: Optional[List[ProjectionType]] = None # _ * N_images
        
        # Load data from disk using provided config parameters.
        self.load_renderings(config, record_func)
        if self.render_path:
            raise NotImplementedError()
        
        if config.enable_clip_near_far:
            self.clip_near_far()
            record_func("Near and far have be cliped into the cube. ")
        if config.enable_scene_contraction:
            self.bound = 2
        
        record_func(f"Scale factor of dataset: {self.scale_factor:.3f}")
        record_func(f"Bound of dataset: {self.bound:.3f}")
        record_func(f"Size of dataset: {self.n_samples}")
        nears = np.array([near.mean() for near in self.nears])
        record_func(f"Near of dataset: min={nears.min():.3f}, mean={nears.mean():.3f}, max={nears.max():.3f}")
        fars = np.array([far.mean() for far in self.fars])
        record_func(f"Far of dataset: min={fars.min():.3f}, mean={fars.mean():.3f}, max={fars.max():.3f}")

    def __len__(self) -> int:
        if self.training:
            return 1000 * self.num_img_per_batch
        else:
            return self.n_samples


    def load_renderings(self, config: BaseConfig, record_func: callable) -> None:
        raise NotImplementedError()
    
    
    def clip_near_far(self):
        # use bounding box to clip near and far
        aabb = torch.tensor([[-1,-1,-1],[1,1,1]], dtype=torch.float32) * self.bound
        for index in range(self.n_samples):
            height = self.heights[index].item()
            width = self.widths[index].item()
            pix_x_int, pix_y_int = torch.meshgrid(
                torch.arange(width), torch.arange(height), indexing='xy'
            )
            origins, directions, _ = pixels_to_rays(
                pix_x_int.reshape(-1),
                pix_y_int.reshape(-1),
                self.pix2cams[index], 
                self.cam2worlds[index], 
                self.distortion_params[index],
                self.pix2cam_ndc,
                self.camtypes[index]
            ) # (-1, 3)
            _, near_box, far_box = intersect_aabb(aabb, origins, directions)
            near_box = near_box.reshape(height, width, 1)
            far_box = far_box.reshape(height, width, 1)
            near = torch.maximum(near_box, self.nears[index])
            far = torch.minimum(far_box, self.fars[index])
            far = torch.maximum(near, far)
            self.nears[index] = near
            self.fars[index] = far


    def train_sample(self) -> Tuple[int, Tensor, Tensor]:
        img_idx = torch.randint(0, self.n_samples, (1, )).item()

        height, width = self.images[img_idx].shape[:2]
        if self.sample_from_half_image: width = width // 2
        upper = (self.patch_size - 1) * self.patch_dilation + 1 - 1
        pix_x_int = torch.randint(0, width - upper, (self.num_patch_per_img,1,1))
        pix_y_int = torch.randint(0, height - upper, (self.num_patch_per_img,1,1))

        patch_x_offset, patch_y_offset = torch.meshgrid(
            torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='xy'
        )

        pix_x_int = pix_x_int + patch_x_offset * self.patch_dilation # (num_patch, patch_size, patch_size)
        pix_y_int = pix_y_int + patch_y_offset * self.patch_dilation # (num_patch, patch_size, patch_size)

        return img_idx, pix_x_int, pix_y_int


    def test_sample(self, index) -> Tuple[int, Tensor, Tensor]:
        height = self.heights[index].item()
        width = self.widths[index].item()
        pix_x_int, pix_y_int = torch.meshgrid(
            torch.arange(width), torch.arange(height), indexing='xy'
        )

        return index, pix_x_int[None, ...], pix_y_int[None, ...]


    def __getitem__(self, index) -> dict:
        if self.training: 
            idx, pix_x_int, pix_y_int = self.train_sample()
        else:
            idx, pix_x_int, pix_y_int = self.test_sample(index)
        # idx, pix_x_int, pix_y_int = self.test_sample(0)

        data_shape = pix_x_int.shape
        pix_x_int = pix_x_int.reshape(-1)
        pix_y_int = pix_y_int.reshape(-1)
        cam2world = self.cam2worlds[idx]
        pix2cam = self.pix2cams[idx]
        distortion_param = self.distortion_params[idx]
        camtype = self.camtypes[idx]
        origins, directions, viewdirs = pixels_to_rays(
            pix_x_int, 
            pix_y_int, 
            pix2cam, 
            cam2world, 
            distortion_param,
            self.pix2cam_ndc,
            camtype
        ) # (patch_size, patch_size, 3) or (patch_size^2, 3)
        coords = torch.stack([
            (pix_x_int.to(torch.float32) + 0.5) / self.widths[idx].item(), 
            (pix_y_int.to(torch.float32) + 0.5) / self.heights[idx].item()
        ], dim = -1)
        nears = self.nears[idx][pix_y_int, pix_x_int]
        fars = self.fars[idx][pix_y_int, pix_x_int]
        embed_idxs = torch.ones((*pix_x_int.shape, 1), dtype=torch.int) * self.embed_idxs[idx]
        
        if self.background_color_type == 'white':
            bg_rgb = torch.ones_like(origins).to(dtype=torch.float32)
        elif self.background_color_type == 'gray':
            bg_rgb = torch.ones_like(origins).to(dtype=torch.float32) * 0.5
        elif self.background_color_type == 'black':
            bg_rgb = torch.zeros_like(origins).to(dtype=torch.float32)
        elif self.background_color_type == 'random':
            bg_rgb = torch.rand_like(origins).to(dtype=torch.float32)
        else:
            raise ValueError()
        
        batch = {
            'coord': coords.reshape(*data_shape, 2),         # (num_patch, patch_size, patch_size, 2)
            'origin': origins.reshape(*data_shape, 3),       # (num_patch, patch_size, patch_size, 3)
            'direction': directions.reshape(*data_shape, 3), # (num_patch, patch_size, patch_size, 3)
            'viewdir': viewdirs.reshape(*data_shape, 3),     # (num_patch, patch_size, patch_size, 3)
            'bg_rgb': bg_rgb.reshape(*data_shape, 3),        # (num_patch, patch_size, patch_size, 3)
            'embed_idx': embed_idxs.reshape(*data_shape, 1), # (num_patch, patch_size, patch_size, 1)
            'near': nears.reshape(*data_shape, 1),           # (num_patch, patch_size, patch_size, 1)
            'far': fars.reshape(*data_shape, 1),             # (num_patch, patch_size, patch_size, 1)
        }

        if self.images is not None:
            batch['rgb'] = self.images[idx][pix_y_int, pix_x_int].reshape(*data_shape, -1) # (num_patch, patch_size, patch_size, 3/4)
            if batch['rgb'].shape[-1]==4:
                batch['rgb'] = batch['rgb'][..., :3] * batch['rgb'][..., -1:] + batch['bg_rgb'][..., :3] * (1 - batch['rgb'][..., -1:])
        # if self.freqs is not None:
        #     batch['freq'] = self.freqs[idx][pix_y_int, pix_x_int].reshape(*data_shape, 1) # (num_patch, patch_size, patch_size, 1)
        if self.static_masks is not None:
            batch['static_mask'] = self.static_masks[idx][pix_y_int, pix_x_int].reshape(*data_shape, 1) # (num_patch, patch_size, patch_size, 1)

        return batch