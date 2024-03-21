import collections
import json
from pathlib import Path
from typing import List, Optional
from argparse import ArgumentParser

import torch
import numpy as np
import cv2
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfacto.utils.image_utils import load_image
IMAGE_TYPES = ['whole', 'half_right', 'half_left']
tensor2float_fn = lambda x: {key:value.item() for key, value in x.items()}

@torch.no_grad()
def main(experiment_dir: str, 
         scene_names: List[str], 
         image_type: str,
         is_save: bool,
         output_dir: str,
         device: str):
    experiment_path = Path(experiment_dir)
    output_path = Path(output_dir)
    results = collections.defaultdict(lambda: {})
    experiment_mean_result = collections.defaultdict(lambda: [])

    metrics = MetricCollection({
        'psnr': PeakSignalNoiseRatio(data_range=1),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1),
        'lpips': LearnedPerceptualImagePatchSimilarity(normalize=True, net_type='alex')
    }).to(device=device)

    for scene_name in scene_names:
        scene_dir = experiment_path / scene_name
        pred_dir = scene_dir / "test_preds"
        scene_mean_result = collections.defaultdict(lambda: [])
        for gt_path in sorted(pred_dir.glob("*_gt.png")):
            image_name = gt_path.stem[:-3]
            pred_path = pred_dir / f"{image_name}_color.png"

            pred = torch.tensor(
                load_image(pred_path)[..., :3], 
                dtype=torch.float32, device=device
            ).unsqueeze(0).permute(0,3,1,2).clip(0., 1.)
            gt = torch.tensor(
                load_image(gt_path)[..., :3], 
                dtype=torch.float32, device=device
            ).unsqueeze(0).permute(0,3,1,2).clip(0., 1.)
            if image_type == 'whole':
                pass
            elif image_type == 'half_left':
                pred = pred[..., :pred.shape[-1]//2]
                gt = gt[..., :gt.shape[-1]//2]
            elif image_type == 'half_right':
                pred = pred[..., pred.shape[-1]//2:]
                gt = gt[..., gt.shape[-1]//2:]

            results[scene_name][image_name] = tensor2float_fn(metrics(pred, gt))
            for key, val in results[scene_name][image_name].items():
                scene_mean_result[key].append(val)

        results[scene_name]['mean'] = {
            key: np.mean(val) for key, val in scene_mean_result.items()
        }
        for key, val in results[scene_name]['mean'].items():
            experiment_mean_result[key].append(val)

    results['mean'] = {
        key: np.mean(val) for key, val in experiment_mean_result.items()
    }

    # print results
    suffix_lengths = np.array([len(scene_name) for scene_name in results.keys()], dtype=int)
    suffix_lengths = np.max(suffix_lengths) - suffix_lengths
    for i, scene_name in enumerate(results.keys()):
        tmp = results['mean'] if scene_name=='mean' else results[scene_name]['mean']
        str_list = []
        for key in ['psnr', 'ssim', 'lpips']:
            str_list.append(f"{key}={tmp[key]:.2f}" if key=='psnr' else f"{key}={tmp[key]:.3f}")
        print(f"{scene_name}: {' '*suffix_lengths[i]}{', '.join(str_list)}")
    
    if is_save:
        if not output_path.exists():
            output_path.mkdir(parents=True)
        with open(output_path / f"metrics_results.json", "w") as fp:
            json.dump(results, fp, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser(description="Compute metrics for the images. ")
    parser.add_argument('--experiment_dir', type=str, help="Experiment folder path. ")
    parser.add_argument('--scene_names', nargs='+', type=str, help="List of scene names. ")
    parser.add_argument('--output_dir', type=str, default='output_metrics', help="Output folder path. ")
    parser.add_argument('--save', action='store_true', help="Whether to save detailed results in output_dir. ")
    parser.add_argument('--image_type', type=str, choices=IMAGE_TYPES, default=IMAGE_TYPES[0], help="Part of the image used. (Default: 'whole')")
    parser.add_argument('--device', type=str, default="cuda", help="Pytorch device. (Default: 'cuda')")
    args = parser.parse_args()

    main(
        args.experiment_dir, args.scene_names, args.image_type, 
        args.save, args.output_dir, args.device
    )