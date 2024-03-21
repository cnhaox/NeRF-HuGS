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
from utils.camera_utils import transform_poses_pca


class DistractorDataset(BaseDataset):
    """Distractor Dataset."""

    def load_renderings(self, config: BaseConfig, record_func) -> None:
        if self.near is None:
            record_func("Near will be computed according to COLMAP file. ")
        if self.far is None:
            record_func("Far will be computed according to COLMAP file. ")
        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = self.data_dir / "0" / "sparse" / "0"
        image_names, poses, pixtocams, distortion_params, camtypes, pts3d = NeRFSceneManager(str(colmap_dir)).process()

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        factor = self.downsample_factor
        suffix = f'_{self.downsample_factor}' if self.downsample_factor > 1 else ''
        pixtocams = pixtocams @ np.diag([factor, factor, 1.])

        # read image_names in the json first (split to train and test later)
        with open(self.data_dir / "0/data_split.json") as fp:
            split_data = json.load(fp)
        # the split
        train_image_names = split_data['train']
        test_image_names = split_data['test']
        all_image_names = train_image_names + test_image_names
        if self.split=='train':
            selected_image_names = train_image_names
        elif self.split=='test':
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

        # Recenter poses.
        poses, transform = transform_poses_pca(poses, True)
        pts3d = np.concatenate([pts3d, np.ones_like(pts3d[..., :1])], axis=-1)
        pts3d = pts3d @ transform.T
        # use mean of pts3d as center
        shift = - np.mean(pts3d[..., :3], axis=0)
        new_transform = np.eye(4)
        new_transform[:3, 3] = shift
        poses = new_transform @ poses
        pts3d = pts3d @ new_transform.T
        transform = new_transform @ transform
        # make camera into [-1,1]^3 cube again
        scale_factor = 1. / np.max(np.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor
        pts3d[:, :3] *= scale_factor
        transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

        self.pts3d = pts3d
        self.transform = transform

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

        image_dir = self.data_dir / f"0/images{suffix}"
        static_mask_dir = self.data_dir / f"0/{config.static_mask_dir}"
        if not static_mask_dir.exists():
            record_func(f"{str(static_mask_dir)} not exist. ")
        self.image_names = selected_image_names
        for i, image_name in enumerate(selected_image_names):
            image_idx = image_names.index(image_name)
            image_path = image_dir / image_name
            image = load_image(image_path)
            height, width = image.shape[:2]

            static_mask_path = static_mask_dir / f"{image_name.split('.')[0]}.png"
            if static_mask_path.exists():
                static_mask = load_image(static_mask_path)
                if static_mask.shape[0] != height or static_mask.shape[1] != width:
                    static_mask = cv2.resize(static_mask, (width, height))
                static_mask = static_mask[..., :1]
            else:
                static_mask = np.ones_like(image[..., :1])

            cam2world = torch.tensor(poses[image_idx], dtype=torch.float32)
            pix2cam = torch.tensor(pixtocams[image_idx], dtype=torch.float32)

            # use pts3d to compute near and far
            pose = poses[image_idx] @ np.diag([1,-1,-1,1]) # use colmap coordinate system
            w2c = np.linalg.inv(pose)
            pts_cam = (self.pts3d @ w2c.T)[:, :3] # xyz in the idx-th cam coordinate
            pts_cam = pts_cam[pts_cam[:, 2]>=0] # filter out points that lie behind the cam
            pts_uv = (pts_cam @ np.linalg.inv(pixtocams[image_idx]).T) / np.maximum(pts_cam[:, 2:], np.finfo(pts_cam.dtype).eps)
            is_in_cone = (pts_uv[:, 0] <= width) * (pts_uv[:, 0] >= 0) \
                        * (pts_uv[:, 1] <= height) * (pts_uv[:, 1] >= 0)
            pts_cam = pts_cam[is_in_cone]
            near_3d = np.percentile(pts_cam[:, 2], 0.1) * 0.8
            far_3d = np.percentile(pts_cam[:, 2], 99.9) * 1.2
            near = near_3d if self.near is None else min(near_3d, self.near)
            far = far_3d if self.far is None else self.far
            nears.append(torch.ones((height, width, 1), dtype=torch.float32) * near)
            fars.append(torch.ones((height, width, 1), dtype=torch.float32) * far)

            images.append(torch.tensor(image.reshape(height, width, 3), dtype=torch.float32))
            static_masks.append(torch.tensor(static_mask.reshape(height, width, 1), dtype=torch.float32))
            
            heights.append(height)
            widths.append(width)
            embed_idxs.append(image_idx)
            cam2worlds.append(cam2world)
            pix2cams.append(pix2cam)

            self.distortion_params.append(distortion_params[image_idx])
            self.camtypes.append(camtypes[image_idx])

        self.images = images
        self.image_names = [Path(image_name).stem for image_name in self.image_names]
        self.static_masks = static_masks
        self.nears = nears
        self.fars = fars

        self.heights = torch.tensor(heights, dtype=torch.int).reshape(-1, 1)
        self.widths = torch.tensor(widths, dtype=torch.int).reshape(-1, 1)
        self.embed_idxs = torch.tensor(embed_idxs, dtype=torch.int).reshape(-1, 1)
        self.cam2worlds = torch.stack(cam2worlds, dim=0)
        self.pix2cams = torch.stack(pix2cams, dim=0)
        self.n_samples = self.cam2worlds.shape[0]