from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
import yaml
from pathlib import Path

from models import model_config_dict

@dataclass
class BaseConfig:
    seed: int = 7
    enable_amp: bool = True
    # data
    dataset_type: str = 'blender'
    static_mask_dir: str = 'static_masks'
    downsample_factor: int = 1
    bound: Optional[float] = None
    rescale_scene: bool = False
    enable_scene_contraction: bool = False
    near: float = 0.0
    far: float = 1.0
    enable_clip_near_far: bool = False
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    llff_use_all_images_for_training: bool = False # If true, use all input images for training.
    enable_ndc: bool = False # only used in llff
    load_alphabetical: bool = True # llff: Load images in COLMAP vs alphabetical
    render_path: bool = False
    train_background_color: str = 'random'
    test_background_color: str = 'white'
    # model
    model_type: str = 'nerfacto'
    render_chunk_size: int = 16384  # Chunk size for whole-image renderings.
    # used by train.py
    batch_size: int = 8192
    patch_size: int = 1
    patch_dilation: int = 1
    num_img_per_batch: int = 16
    num_steps: int = 50000
    warmup_steps: int = 3000
    lr_init: float = 1e-2
    opt_betas: Tuple[float, float] = (0.9, 0.999)
    opt_eps: float = 1e-8
    lr_final: float = 1e-3
    lr_decay_mult: float = 1e-8
    eval_render_every: int = 10000 # Steps between test set renders when training
    eval_images_num: int = 4
    save_eval_render: bool = False
    use_eval_lpips: bool = True
    save_weight_every: int = 10000

    finetune_enable: bool = False
    finetune_init_parameters: bool = False
    finetune_batch_size: int = 8192
    finetune_patch_size: int = 1
    finetune_patch_dilation: int = 1
    finetune_num_img_per_batch: int = 16
    finetune_num_steps: int = 50000
    finetune_warmup_steps: int = 3000
    finetune_params: Optional[Tuple[str, ...]] = None
    finetune_lr_init: float = 1e-2
    finetune_opt_betas: Tuple[float, float] = (0.9, 0.999)
    finetune_opt_eps: float = 1e-8
    finetune_lr_final: float = 1e-3
    finetune_lr_decay_mult: float = 1e-8

    # use in test.py
    save_test_render: bool = True
    

def load_configs(config_path):
    with open(config_path, "r") as f:
        config_dict: dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if 'base' in config_dict.keys():
        config = BaseConfig(**config_dict['base'])
    else:
        config = BaseConfig()
    
    if 'model' in config_dict.keys():
        model_config = model_config_dict[config.model_type](**config_dict['model'])
    else:
        model_config = model_config_dict[config.model_type]()
    
    return config, model_config

def save_configs(config_path, config, model_config):
    save_dict = {
        'base': asdict(config),
        'model': asdict(model_config)
    }
    with open(config_path, "w") as f:
        yaml.dump(save_dict, f)
    