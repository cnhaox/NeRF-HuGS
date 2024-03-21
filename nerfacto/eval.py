import argparse
from pathlib import Path
import collections
import json

import numpy as np
from tqdm import tqdm

import torch
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import dataset_dict
from models import model_dict, criterion_dict
from utils.config_utils import load_configs
from utils.utils import init_seed, seed_worker, State, to_device
from utils.checkpoint_utils import load_weights
from utils.image_utils import save_image, depth2img_v2, color_correct


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help="The path of the hparams yaml file. ")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--eval_data', type=str, choices=['train', 'test'])
    parser.add_argument('--original_name', action="store_true")
    parser.add_argument('--only_pred_gt', action="store_true")

    return parser.parse_args()


tensor2float_fn = lambda x: {key:value.item() for key, value in x.items()}


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config, model_config = load_configs(args.config)

    save_dir = Path(args.save_dir)
    log_dir = save_dir / "logs"
    result_dir = save_dir / f"{args.eval_data}_preds"
    for dir in [save_dir, log_dir, result_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # prepare data
    test_dataset = dataset_dict[config.dataset_type](
        data_split=args.eval_data, training=False, 
        batch_size=config.batch_size, 
        patch_size=config.patch_size,
        patch_dilation=config.patch_dilation,
        num_img_per_batch=config.num_img_per_batch, 
        sample_from_half_image=False,
        data_dir=args.data_dir, config=config
    )
    metrics_dict = {
        'psnr': PeakSignalNoiseRatio(data_range=1),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1),
        'lpips': LearnedPerceptualImagePatchSimilarity(normalize=True, net_type='alex')
    }
    metrics = MetricCollection(metrics_dict).to(device)
    results = {}
    total_results = collections.defaultdict(lambda: [])

    # prepare model
    model = model_dict[config.model_type](
        model_config, 
        test_dataset.bound, 
        config.enable_amp,
        config.enable_scene_contraction
    ).to(device)
    # load weight
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_paths = list(checkpoint_dir.glob('finetune_checkpoint_*.ckpt'))
    if len(checkpoint_paths) == 0:
        checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.ckpt'))
    
    if len(checkpoint_paths)==0:
        print(f"Checkpoint not found in {checkpoint_dir} ")
        return
    checkpoint_path = str(sorted(checkpoint_paths)[-1])
    state = load_weights(checkpoint_path, model, device=device)
    print(f"Use checkpoint: {checkpoint_path} ")

    model.eval()
    test_count = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_dataset))
        for test_idx, batch in enumerate(test_dataset):
            batch = to_device(batch, device) # (1, h, w, -1)
            data_shape = batch['origin'].shape[1:-1]
            save_name = test_dataset.image_names[test_idx] if args.original_name \
                        else f"{test_idx:04d}"
            for key in batch.keys():
                batch[key] = batch[key].view(-1, batch[key].shape[-1])
            with torch.autocast(device_type='cuda', enabled=config.enable_amp):
                outputs = model(
                    batch=batch, curr_step=state.step, perturb=False, chunk_size=config.render_chunk_size
                )
            
            pred = torch.clip(outputs['rgb'].reshape(*data_shape, 3), 0, 1)
            gt = torch.clip(batch['rgb'].reshape(*data_shape, 3), 0, 1)
            pred_cc = torch.tensor(color_correct(pred.cpu().numpy(), gt.cpu().numpy())).to(pred)

            pred = pred[None, ...].permute(0, 3, 1, 2)
            pred_cc = pred_cc[None, ...].permute(0, 3, 1, 2)
            gt = gt[None, ...].permute(0, 3, 1, 2)

            results[save_name] = {}
            for key, val in tensor2float_fn(metrics(pred, gt)).items():
                results[save_name][f'{key}_raw'] = val
                total_results[f'{key}_raw'].append(val)
            for key, val in tensor2float_fn(metrics(pred_cc, gt)).items():
                results[save_name][f'{key}_cc'] = val
                total_results[f'{key}_cc'].append(val)
            metrics.reset()
        
            if config.save_test_render:
                pred = pred.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                pred_cc = pred_cc.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                save_image(pred, result_dir / f"{save_name}_color.png")
                save_image(gt, result_dir / f"{save_name}_gt.png")

                if not args.only_pred_gt:
                    save_image(pred_cc, result_dir / f"{save_name}_colorcc.png")
                    depth = outputs['depth'].reshape(*data_shape).cpu().numpy()
                    acc = outputs['accumulation'].reshape(*data_shape).cpu().numpy()
                    depth_vis = depth2img_v2(depth, acc)
                    save_image(depth_vis, result_dir / f"{save_name}_depth.png")
                    for key in outputs.keys():
                        if 'rgb' in key and key != 'rgb':
                            img = outputs[key].reshape(*data_shape, 3).cpu().numpy()
                            save_image(img, result_dir / f"{save_name}_{key}.png")

            pbar.update(1)
            test_count += 1
        
        metrics.reset()
        pbar.close()
        
        results['mean'] = {key: np.mean(val) for key, val in total_results.items()}
        with open(log_dir / f"eval_{args.eval_data}_results.json", "w") as fp:
            json.dump(results, fp, indent=4)
        print(results['mean'])
            
if __name__ == '__main__':
    args = get_args()
    main(args)