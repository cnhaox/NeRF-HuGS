from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
import os

import numpy as np
import cv2

import torch
from torch import Tensor

from datasets.base import BaseDataset
from datasets.colmap_utils import NeRFSceneManager
from utils.camera_utils import get_pixtocam, intrinsic_matrix, recenter_poses
from utils.config_utils import BaseConfig
from utils.image_utils import load_image


class LLFFDataset(BaseDataset):
    """LLFF Dataset."""
    
    def load_renderings(self, config: BaseConfig) -> None:
        # set up scaling factor
        image_dir_suffix = ''
        # use downsampling factor
        if self.downsample_factor > 1:
            image_dir_suffix = f'_{self.downsample_factor}'
            factor = self.downsample_factor
        else:
            factor = 1
        
        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.data_dir, 'sparse/0/')
        image_names, poses, pixtocams, distortion_params, camtypes, _ = NeRFSceneManager(colmap_dir).process()
        # use 1st camera parameter for all cameras
        pixtocam = pixtocams[0]
        distortion_param = distortion_params[0]
        camtype = camtypes[0]

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        if config.load_alphabetical:
            inds = np.argsort(image_names)
            image_names = [image_names[i] for i in inds]
            poses = poses[inds]

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])

        if self.enable_ndc:
            # Load bounds if possible (only used in forward facing scenes).
            posefile = self.data_dir / 'poses_bounds.npy'
            if posefile.exists():
                with open(posefile, 'rb') as fp:
                    poses_arr = np.load(fp)
                bounds = poses_arr[:, -2:]
            else:
                bounds = np.array([0.01, 1.])

            # Set the projective matrix defining the NDC transformation.
            self.pix2cam_ndc = pixtocam.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = recenter_poses(poses)
            colmap_to_world_transform = transform @ colmap_to_world_transform
        else:
            # Recenter poses.
            poses, transform = recenter_poses(poses)
            scale_factor = 1. / np.max(np.abs(poses[:, :3, 3]))
            poses[:, :3, 3] *= scale_factor

        # Load images.
        colmap_image_dir = self.data_dir / 'images'
        image_dir = self.data_dir / f'images{image_dir_suffix}'
        for d in [image_dir, colmap_image_dir]:
            if not d.exists():
                raise ValueError(f'Image folder {d} does not exist.')
        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(os.listdir(colmap_image_dir))
        image_files = sorted(os.listdir(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [image_dir / colmap_to_image[f] for f in image_names]
        image_names = []

        # Select the split
        all_indices = np.arange(len(image_paths))
        if config.llff_use_all_images_for_training:
            train_indices = all_indices
        else:
            train_indices = all_indices[all_indices % config.llffhold != 0]
        test_indices = all_indices[all_indices % config.llffhold == 0]
        if self.split=='train': indices = train_indices
        elif self.split=='test': indices = test_indices

        images = []
        static_masks = []
        nears = []
        fars = []

        heights = []
        widths = []
        embed_idxs = []
        cam2worlds = []
        pix2cams = []

        self.distortion_params = []
        self.camtypes = []

        self.n_samples = len(indices)
        for i, idx in enumerate(indices):
            image_file = image_paths[idx]
            image_names.append(image_paths[idx].stem)
            image = load_image(image_file)
            height, width = image.shape[:2]

            static_mask_file = self.data_dir / "static_masks" / f"{Path(image_file).stem}.png"
            if static_mask_file.exists():
                static_mask = load_image(static_mask_file)[..., :1]
                static_mask = cv2.resize(static_mask, (width, height))
            else:
                static_mask = np.ones_like(image[..., :1])
            
            images.append(torch.tensor(image.reshape(height, width, 3), dtype=torch.float32))
            static_masks.append(torch.tensor(static_mask.reshape(height, width, 1), dtype=torch.float32))
            nears.append(torch.ones_like(static_masks[-1]) * self.near)
            fars.append(torch.ones_like(static_masks[-1]) * self.far)
            
            heights.append(height)
            widths.append(width)
            embed_idxs.append(idx)
            cam2worlds.append(torch.tensor(poses[idx], dtype=torch.float32))
            pix2cams.append(torch.tensor(pixtocam, dtype=torch.float32))

            self.distortion_params.append(distortion_param)
            self.camtypes.append(camtype)
        
        self.images = images
        self.image_names = image_names
        self.static_masks = static_masks
        self.nears = nears
        self.fars = fars

        self.heights = torch.tensor(heights, dtype=torch.int).reshape(-1, 1)
        self.widths = torch.tensor(widths, dtype=torch.int).reshape(-1, 1)
        self.embed_idxs = torch.tensor(embed_idxs, dtype=torch.int).reshape(-1, 1)
        self.cam2worlds = torch.stack(cam2worlds, dim=0)
        self.pix2cams = torch.stack(pix2cams, dim=0)