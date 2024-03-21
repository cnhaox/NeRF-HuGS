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
from utils.camera_utils import recenter_poses

bound_dict = {
    'brandenburg_gate': 24,
    'sacre_coeur': 11,
    'taj_mahal': 16,
    'trevi_fountain': 35
}

class PhototourismDataset(BaseDataset):
    """Phototourism Dataset."""

    def load_renderings(self, config: BaseConfig, record_func) -> None:
        record_func("Near and far will be computed according to COLMAP file. ")
        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.data_dir, 'dense/sparse')
        image_names, poses, pixtocams, distortion_params, camtypes, pts3d = NeRFSceneManager(colmap_dir).process()

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        factor = self.downsample_factor
        pixtocams = pixtocams @ np.diag([factor, factor, 1.])

        # read all files in the tsv first (split to train and test later)
        split_file = list(self.data_dir.glob("*.tsv"))[0]
        split_data = pd.read_csv(split_file, sep='\t')
        # the split
        train_image_names = []
        test_image_names = []
        for i in range(len(split_data)):
            if split_data['split'][i]=='train': train_image_names.append(split_data['filename'][i])
            elif split_data['split'][i]=='test': test_image_names.append(split_data['filename'][i])
        all_image_names = train_image_names + test_image_names
        if self.split=='train':
            selected_image_names = train_image_names
        elif self.split=='test':
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

        # Recenter poses.
        poses, transform = recenter_poses(poses)
        pts3d = np.concatenate([
            pts3d, np.ones_like(pts3d[..., :1])
        ], axis=-1)
        pts3d = pts3d @ transform.T
        # use object as center
        shift = - np.mean(pts3d[..., :3], axis=0)
        new_transform = np.eye(4)
        new_transform[:3, 3] = shift
        poses = new_transform @ poses
        pts3d = pts3d @ new_transform.T
        transform = new_transform @ transform

        if self.rescale_scene:
            bound = bound_dict[self.data_dir.name]
            scale_factor = self.bound / bound
            new_transform = np.diag([scale_factor, scale_factor, scale_factor, 1])
            # poses = new_transform @ poses # don't use it. Original poses[..., :3,:3] can keep the direction to be unit
            poses[..., :3, 3] *= scale_factor
            self.scale_factor = scale_factor
            pts3d = pts3d @ new_transform.T
            transform = new_transform @ transform

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

        self.n_samples = len(selected_image_names)
        self.image_names = selected_image_names
        image_dir = self.data_dir / "dense" / "images"
        static_mask_dir = self.data_dir / "dense" / config.static_mask_dir
        if not static_mask_dir.exists():
            record_func(f"{str(static_mask_dir)} does not exist. Use default setting. ")
        for i, image_name in enumerate(selected_image_names):
            image_idx = image_names.index(image_name)
            image_path = image_dir / image_name
            image = load_image(image_path)
            static_mask_path = static_mask_dir / f"{image_name.split('.')[0]}.png"
            if static_mask_path.exists():
                static_mask = load_image(static_mask_path)[..., :1]
            else:
                static_mask = np.ones_like(image[..., :1])
            
            height, width = image.shape[:2]
            if self.downsample_factor > 1:
                height = height // self.downsample_factor
                width = width // self.downsample_factor
                image = cv2.resize(image, (width, height))
            if static_mask.shape[0] != height or static_mask.shape[1] != width:
                static_mask = cv2.resize(static_mask, (width, height))

            cam2world = torch.tensor(poses[image_idx], dtype=torch.float32)
            pix2cam = torch.tensor(pixtocams[image_idx], dtype=torch.float32)
            
            # use pts3d to compute near and far
            pose = poses[image_idx] @ np.diag([1,-1,-1,1]) # use colmap coordinate system
            w2c = np.linalg.inv(pose)
            pts_cam = (self.pts3d @ w2c.T)[:, :3] # xyz in the idx-th cam coordinate
            pts_cam = pts_cam[pts_cam[:, 2]>0] # filter out points that lie behind the cam
            near = np.percentile(pts_cam[:, 2], 0.1)
            far = np.percentile(pts_cam[:, 2], 99.9)
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