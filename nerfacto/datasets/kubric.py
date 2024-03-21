from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
import os

import numpy as np
import cv2
import pandas as pd

import torch
from torch import Tensor

from datasets.base import BaseDataset
from datasets.colmap_utils import NeRFSceneManager

from utils.image_utils import load_image
from utils.config_utils import BaseConfig
from utils.camera_utils import  ProjectionType


class KubricDataset(BaseDataset):
    """Kubric Dataset."""

    def load_renderings(self, config: BaseConfig, record_func: callable) -> None:
        record_func("Near and far will be computed according to scene_gt.json. ")
        with open(self.data_dir / 'scene_gt.json', 'r') as f:
            scene_json = json.load(f)
            scene_center = np.array(scene_json['center'])
            scene_scale = scene_json['scale']
            self.scale_factor = scene_scale
            scene_near = scene_json['near']
            scene_far = scene_json['far'] * 1.2 # original far is not enough

        with open(self.data_dir / "dataset.json", 'r') as f:
            dataset_json = json.load(f)
            train_image_names = dataset_json['train_ids']
            train_image_names = [str(i) for i in train_image_names]
        with open(self.data_dir / "freeze-test" / "dataset.json", 'r') as f:
            dataset_json = json.load(f)
            val_image_names = dataset_json['val_ids']
            val_image_names = [str(i) for i in val_image_names]

        if self.split == 'train':
            rgb_dir = self.data_dir / 'rgb' / f'{self.downsample_factor}x'
            static_mask_dir = self.data_dir / config.static_mask_dir
            camera_dir = self.data_dir / 'camera-gt'
            image_names = train_image_names
            embed_offset = 0
        elif self.split == 'test':
            rgb_dir = self.data_dir / 'freeze-test' / 'static-rgb' / f'{self.downsample_factor}x'
            static_mask_dir = self.data_dir / 'freeze-test' / config.static_mask_dir
            camera_dir = self.data_dir / 'freeze-test' / 'camera-gt'
            image_names = val_image_names
            embed_offset = len(train_image_names)
        if not static_mask_dir.exists():
            record_func(f"{str(static_mask_dir)} does not exist. Use default setting. ")
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

        self.n_samples = len(image_names)
        self.image_names = []
        for i, image_name in enumerate(image_names):
            with open(camera_dir / f"{image_name}.json", 'r') as f:
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

            if self.downsample_factor > 1:
                pixtocam = pixtocam @ np.diag([self.downsample_factor, self.downsample_factor, 1.])
                
            distortion_param = {
                'k1': radial_distortion[0], 'k2': radial_distortion[1], 'k3': radial_distortion[2],
                'p1': tangential_distortion[0], 'p2': tangential_distortion[1]
            }

            camtoworld = np.eye(4)
            camtoworld[:3,:4] = np.concatenate([orientation.T, position.reshape(3,1)], axis=1)
            # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
            camtoworld = camtoworld @ np.diag([1, -1, -1, 1])
            # recenter and rescale
            camtoworld[:3, 3] -= scene_center
            camtoworld[:3, 3] *= scene_scale

            # load image
            image = load_image(rgb_dir / f"{image_name}.png")
            height, width = image.shape[:2]

            static_mask_path = static_mask_dir / f"{image_name}.png"
            if static_mask_path.exists():
                static_mask = load_image(static_mask_path)[..., :1]
                if static_mask.shape[0] != height or static_mask.shape[1] != width:
                    static_mask = cv2.resize(static_mask, (width, height)).reshape(height, width, 1)
            else:
                static_mask = np.ones_like(image[..., :1])

            near = np.ones_like(image[..., :1]) * scene_near
            far = np.ones_like(image[..., :1]) * scene_far

            images.append(torch.tensor(image.reshape(height, width, -1), dtype=torch.float32))
            static_masks.append(torch.tensor(static_mask.reshape(height, width, 1), dtype=torch.float32))
            nears.append(torch.tensor(near.reshape(height, width, 1), dtype=torch.float32))
            fars.append(torch.tensor(far.reshape(height, width, 1), dtype=torch.float32))

            heights.append(height)
            widths.append(width)
            embed_idxs.append(embed_offset + i)
            cam2worlds.append(torch.tensor(camtoworld, dtype=torch.float32))
            pix2cams.append(torch.tensor(pixtocam, dtype=torch.float32))

            self.image_names.append(image_name)
            self.distortion_params.append(distortion_param)
            self.camtypes.append(ProjectionType.PERSPECTIVE)
        
        self.images = images
        self.static_masks = static_masks
        self.nears = nears
        self.fars = fars

        self.heights = torch.tensor(heights, dtype=torch.int).reshape(-1, 1)
        self.widths = torch.tensor(widths, dtype=torch.int).reshape(-1, 1)
        self.embed_idxs = torch.tensor(embed_idxs, dtype=torch.int).reshape(-1, 1)
        self.cam2worlds = torch.stack(cam2worlds, dim=0)
        self.pix2cams = torch.stack(pix2cams, dim=0)