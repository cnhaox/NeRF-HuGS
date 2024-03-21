from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json

import numpy as np
import cv2

import torch
from torch import Tensor

from datasets.base import BaseDataset
from utils.camera_utils import get_pixtocam, ProjectionType
from utils.config_utils import BaseConfig
from utils.image_utils import load_image

# aabb data comes from nsvf dataset
aabb_dict = {
    'chair': [-0.9128502130508422, -0.8927719712257385, -1.1939758777618408, 
               0.6871498107910157,  0.7072280526161194,  1.2060242176055909],
    'drums': [-1.3144566535949707, -0.932932686805725, -0.7522089004516601, 
               1.085543441772461, 1.067067313194275, 0.8477911233901978],
    'ficus': [-0.6518060386180877, -1.0935752511024475, -1.2943775177001953, 
               0.5481940090656281, 0.5064247727394104, 1.1056225776672364],
    'hotdog': [-1.2112753582000733, -1.2783885192871094, -0.4059063982963562, 
                1.1887247371673584, 1.1216115760803222, 0.39409361362457274],
    'lego': [-0.8325289607048034, -1.3345391273498535, -0.8325301527976989,
              0.7674710631370545, 1.0654609680175782, 1.167469847202301],
    'materials': [-1.3345369815826416, -0.993173611164093, -0.5112450242042541,
                   1.06546311378479, 1.006826388835907, 0.28875498771667485],
    'mic': [-1.4349385023117065, -1.0935752511024475, -0.9530120015144348, 
             0.5650614976882935, 0.9064247488975525, 1.0469879984855652],
    'ship': [-1.3791147232055665, -1.3791175842285157, -0.7325300931930542,
              1.4208852291107177, 1.4208823680877685, 0.46746995449066164]
}

class BlenderDataset(BaseDataset):
    """Blender Dataset."""
    
    def load_renderings(self, config: BaseConfig, record_func: callable) -> None:
        assert not config.enable_scene_contraction
        pose_file = self.data_dir / f"transforms_{self.split}.json" # f"transforms_train.json" # 
        embed_indices_file = self.data_dir / f"embed_indices.json"
        with open(pose_file, 'r') as fp:
            meta = json.load(fp)
        embed_indices_dict = None
        if embed_indices_file.exists():
            with open(embed_indices_file, 'r') as fp:
                embed_indices_dict = json.load(fp)
        
        static_mask_dir = self.data_dir / "static_masks"
        if not static_mask_dir.exists():
            record_func(f"{static_mask_dir} does not exist. Use default setting. ")
        
        images = []
        image_names = []
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

        self.n_samples = len(meta['frames'])
        for idx, frame in enumerate(meta['frames']):
            image_file = self.data_dir / f"{frame['file_path']}.png"
            image = load_image(image_file)
            image_names.append(frame['file_path'])
            height, width = image.shape[:2]

            static_mask_file = static_mask_dir / f"{frame['file_path']}.png"
            if static_mask_file.exists():
                static_mask = load_image(static_mask_file)
                static_mask = static_mask[..., :1]
            else:
                static_mask = image[..., 3:]
            
            if self.downsample_factor > 1:
                height = height // self.downsample_factor
                width = width // self.downsample_factor
                image = cv2.resize(image, (width, height))
                static_mask = cv2.resize(static_mask, (width, height))
            
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            
            # get intrinsic matrix and cam_directions
            focal = .5 * width / np.tan(.5 * float(meta['camera_angle_x']))
            intrinsic_inv = get_pixtocam(focal, width, height)
            # get embed_idx
            embed_idx = idx if embed_indices_dict is None \
                        else embed_indices_dict[frame['file_path']]
            
            # image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:]) # Use a white background.
            images.append(torch.tensor(image.reshape(height, width, -1), dtype=torch.float32))
            static_masks.append(torch.tensor(static_mask.reshape(height, width, 1), dtype=torch.float32))
            nears.append(torch.ones_like(static_masks[-1]) * self.near)
            fars.append(torch.ones_like(static_masks[-1]) * self.far)

            heights.append(height)
            widths.append(width)
            embed_idxs.append(embed_idx)
            cam2worlds.append(torch.tensor(c2w, dtype=torch.float32))
            pix2cams.append(torch.tensor(intrinsic_inv, dtype=torch.float32))

            self.distortion_params.append(None)
            self.camtypes.append(ProjectionType.PERSPECTIVE)
        
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
        
        if self.rescale_scene:
            # shift and rescale the camera's positions into the bounding box
            data_type = self.data_dir.stem
            aabb = torch.tensor(aabb_dict[data_type], dtype=torch.float32).reshape(2,3)
            shift = -torch.mean(aabb, dim=0)
        
            # slightly larger
            aabb = 1.05 * (aabb + shift)
            if data_type=='lego': aabb *= 1.1
            elif data_type=='mic': aabb *= 1.2
            scale_factor = self.bound / torch.abs(aabb).max().item()
            self.transform[:3, 3] = shift
            self.transform = torch.diag(torch.tensor([scale_factor, scale_factor, scale_factor, 1.], dtype=torch.float32)) @ self.transform
        
            self.cam2worlds[..., :3, 3] += shift
            self.cam2worlds[..., :3, 3] *= scale_factor
            self.scale_factor = scale_factor