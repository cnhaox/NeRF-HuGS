import argparse
from pathlib import Path
import collections

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.optim import Adam
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import dataset_dict
from models import model_dict, criterion_dict
from utils.config_utils import load_configs, save_configs
from utils.record_utils import Recorder
from utils.lr_scheduler_utils import get_warmup_decay_scheduler
from utils.checkpoint_utils import load_snapshot, save_snapshot
from utils.utils import init_seed, seed_worker, to_device, State
from utils.image_utils import mse_to_psnr, save_image, depth2img, color_correct


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help="Path of the hparams yaml file.")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    return parser.parse_args()


def main(args):
    config, model_config = load_configs(args.config)

    init_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = f"cuda:0"

    # prepare recorder
    save_dir = Path(args.save_dir)
    log_dir = save_dir / "logs"
    checkpoint_dir = save_dir / "checkpoints"
    result_dir = save_dir / "eval_results"
    for dir in [save_dir, log_dir, checkpoint_dir, result_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    save_configs(save_dir / "config.yml", config, model_config)
    recorder = Recorder(log_dir)

    # prepare dataset and dataloader  
    train_dataset, train_loader = None, None
    recorder.print("Loading test dataset ...")  
    eval_dataset = dataset_dict[config.dataset_type](
        data_split='test', training=False,
        batch_size=config.batch_size, 
        patch_size=config.patch_size,
        patch_dilation=config.patch_dilation,
        num_img_per_batch=config.num_img_per_batch,
        sample_from_half_image=False,
        data_dir=args.data_dir, 
        config=config,
        record_func=recorder.print,
    )
    
    # prepare metrics
    eval_metrics_dict = {
        'psnr': PeakSignalNoiseRatio(data_range=1),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1)
    }
    if config.use_eval_lpips: 
        eval_metrics_dict['lpips'] = LearnedPerceptualImagePatchSimilarity(normalize=True)
    eval_metrics = MetricCollection(eval_metrics_dict).to(device)
    
    # prepare model and criterion
    model = model_dict[config.model_type](
        model_config, 
        eval_dataset.bound, 
        config.enable_amp,
        config.enable_scene_contraction
    ).to(device)
    criterion = criterion_dict[config.model_type](model)
    params_dict = model.get_params_dict()
    optimizer, scheduler, scaler = None, None, None
    num_epochs = 0

    # load weight
    checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.ckpt'))
    if len(checkpoint_paths) == 0:
        recorder.print("Checkpoint file not found. Init parameters. ")
        state = State()
    else:
        checkpoint_path = str(sorted(checkpoint_paths)[-1])
        recorder.print(f"Use checkpoint file: {checkpoint_path}")
        state = load_snapshot(checkpoint_path, model, optimizer, scheduler, scaler, device)
    recorder.print(f"Num of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    for train_stage in ['train', 'finetune']:
        del train_dataset, train_loader, optimizer, scheduler, scaler
        if train_stage=='train':
            recorder.print("Start train. ")
            is_finetune = False
            train_dataset_args = {
                'data_split': 'train',
                'batch_size': config.batch_size,
                'patch_size': config.patch_size,
                'patch_dilation': config.patch_dilation,
                'num_img_per_batch': config.num_img_per_batch,
                'sample_from_half_image': False,
            }
            trainable_params = list(params_dict.keys())
            lr_init = config.lr_init
            lr_final = config.lr_final
            lr_decay_mult = config.lr_decay_mult
            opt_betas = config.opt_betas
            opt_eps = config.opt_eps
            warmup_steps = config.warmup_steps
            num_steps = config.num_steps
        else:
            recorder.print("Start finetune. ")
            is_finetune = True
            train_dataset_args = {
                'data_split': 'test',
                'batch_size': config.finetune_batch_size,
                'patch_size': config.finetune_patch_size,
                'patch_dilation': config.finetune_patch_dilation,
                'num_img_per_batch': config.finetune_num_img_per_batch,
                'sample_from_half_image': True,
            }
            trainable_params = config.finetune_params
            lr_init = config.finetune_lr_init
            lr_final = config.finetune_lr_final
            lr_decay_mult = config.finetune_lr_decay_mult
            opt_betas = config.finetune_opt_betas
            opt_eps = config.finetune_opt_eps
            warmup_steps = config.finetune_warmup_steps
            num_steps = config.finetune_num_steps

        # prepare train dataset and dataloader
        recorder.print("Loading train dataset ...")
        train_dataset = dataset_dict[config.dataset_type](
            training=True, data_dir=args.data_dir, 
            config=config, record_func=recorder.print, 
            **train_dataset_args
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_dataset.num_img_per_batch,
            shuffle=True, num_workers=8, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True
        )

        # prepare optimizer, scheduler and scaler
        params_list = []
        params_name_list = []
        for key in trainable_params:
            params_list.append({'params': params_dict[key], 'lr': lr_init})
            params_name_list.append(key)
        optimizer = Adam(params_list, betas=opt_betas, eps=opt_eps)
        scheduler = get_warmup_decay_scheduler(
            optimizer, lr_init, lr_final, lr_decay_mult, warmup_steps, num_steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=config.enable_amp)

        num_epochs += num_steps//len(train_loader) + min(num_steps%len(train_loader), 1)
        start_epoch = state.epoch
        extra_infos = {}

        for _ in range(start_epoch, num_epochs):
            state.epoch += 1
            epoch_result = collections.defaultdict(lambda: [])

            pbar = tqdm(total=len(train_loader))
            pbar.set_description(f"train {state.epoch:3d}/{num_epochs:d}")
        
            model.train()
            for train_idx, batch in enumerate(train_loader):
                state.step += 1
                if train_idx%100==0:
                    for i in range(len(optimizer.param_groups)):
                        recorder.writer.add_scalar(f"train/lr_{params_name_list[i]}", optimizer.param_groups[i]['lr'], state.step)

                batch = to_device(batch, device) # (n_img, n_patch, patch_size, patch_size, -1)
                data_shape = (
                    batch['origin'].shape[0] * batch['origin'].shape[1],
                    batch['origin'].shape[2], batch['origin'].shape[3]
                ) # (n_patch, patch_size, patch_size)
                for key in batch.keys():
                    batch[key] = batch[key].view(-1, batch[key].shape[-1])

                optimizer.zero_grad()
                extra_infos['curr_step'] = state.step
                extra_infos['curr_frac'] = state.step / config.num_steps
                with torch.autocast(device_type='cuda', enabled=config.enable_amp):
                    outputs = model(batch=batch, curr_step=state.step, perturb=True)
                    loss, info_dict, extra_infos = criterion(
                        outputs=outputs, batch=batch, data_shape=data_shape,
                        is_finetune=is_finetune, extra_infos=extra_infos
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                for key in list(info_dict.keys()):
                    if key[:3] == 'mse':
                        info_dict[f'psnr{key[3:]}'] = mse_to_psnr(info_dict[key])
                info_dict['loss'] = loss.detach()
                for key in info_dict.keys(): 
                    epoch_result[key].append(info_dict[key].item())
                    
                pbar.set_postfix(
                    avg_loss=np.mean(epoch_result['loss']),
                    avg_psnr=np.mean(epoch_result['psnr']),
                    loss=epoch_result['loss'][-1], psnr=epoch_result['psnr'][-1]
                )
                pbar.update(1)
                for key in epoch_result.keys():
                    recorder.writer.add_scalar(f"train/{key}", epoch_result[key][-1], state.step)
            
            pbar.close()
            info_strs = []
            for key in epoch_result.keys():
                val = np.mean(epoch_result[key])
                recorder.writer.add_scalar(f"train/avg_{key}", val, state.step)
                info_strs.append(f"avg_{key}={val:.3e}")
            recorder.print(f"train {state.epoch:3d}/{num_epochs:d}: {', '.join(info_strs)}")
            
            # test-set evaluation
            if config.eval_render_every > 0 \
                and state.epoch%(config.eval_render_every//len(train_loader))==0:

                del batch, outputs
                torch.cuda.empty_cache()

                pbar = tqdm(total=config.eval_images_num)
                pbar.set_description(f"eval {state.epoch:3d}/{num_epochs:d}")
            
                model.eval()
                results = collections.defaultdict(lambda: [])
                with torch.no_grad():
                    for idx_bias in range(config.eval_images_num):
                        eval_idx = (state.next_eval_idx + idx_bias) % len(eval_dataset)
                        batch = to_device(eval_dataset[eval_idx], device) # (1, h, w, -1)
                        data_shape = batch['origin'].shape[1:-1]
                        for key in batch.keys():
                            batch[key] = batch[key].view(-1, batch[key].shape[-1])
                
                        with torch.autocast(device_type='cuda', enabled=config.enable_amp):
                            outputs = model(
                                batch=batch, curr_step=state.step, perturb=False, chunk_size=config.render_chunk_size
                            )
                        pred = torch.clip(outputs['rgb'].reshape(*data_shape, 3), 0., 1.)
                        gt = torch.clip(batch['rgb'].reshape(*data_shape, 3), 0., 1.)
                        pred_cc = torch.tensor(color_correct(pred.cpu().numpy(), gt.cpu().numpy())).to(pred)

                        pred = pred[None, ...].permute(0, 3, 1, 2)
                        pred_cc = pred_cc[None, ...].permute(0, 3, 1, 2)
                        gt = gt[None, ...].permute(0, 3, 1, 2)
                        
                        metric_results = {
                            'raw': eval_metrics(pred, gt),
                            'cc': eval_metrics(pred_cc, gt),
                        }
                        for key in metric_results.keys():
                            for sub_key in metric_results[key].keys():
                                results[f'{key}_{sub_key}'].append(metric_results[key][sub_key].item())
                
                        if config.save_eval_render:
                            pred = pred.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                            pred_cc = pred_cc.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                            gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                            save_image(pred, result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_pred.png")
                            save_image(pred_cc, result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_pred_cc.png")
                            save_image(gt, result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_gt.png")
                            for key in outputs.keys():
                                if 'rgb' in key and key != 'rgb':
                                    img = outputs[key].reshape(*data_shape, 3).cpu().numpy()
                                    save_image(img, result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_{key}.png")
                                if 'depth' in key:
                                    depth = outputs[key].reshape(*data_shape).cpu().numpy()
                                    save_image(depth2img(depth), result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_{key}.png")
                                if 'accumulation' in key or 'mask' in key:
                                    data = outputs[key].reshape(*data_shape, 1).cpu().numpy()
                                    data = np.repeat(data, repeats=3, axis=-1)
                                    save_image(data, result_dir / f"epoch{state.epoch:03d}_idx{eval_idx:04d}_{key}.png")
                        pbar.update(1)
   
                eval_metrics.reset()
                state.next_eval_idx += max(config.eval_images_num//2, 1)
                state.next_eval_idx = state.next_eval_idx % len(eval_dataset)

                pbar.close()
                info_strs = []
                for key in results.keys():
                    val = np.mean(results[key])
                    info_strs.append(f"avg_{key}={val:.3e}")
                    recorder.writer.add_scalar(f"eval/{key}", val, state.step)
                recorder.print(f"eval {state.epoch:3d}/{num_epochs:d}: {', '.join(info_strs)}")

            if config.save_weight_every > 0 and state.epoch%(config.save_weight_every//len(train_loader))==0:
                filename = f"checkpoint_{state.step:08d}.ckpt"
                if is_finetune: filename = "finetune_" + filename
                filename = str(checkpoint_dir / filename)
                save_snapshot(filename, state, model, optimizer, scheduler, scaler)
                recorder.print(f"Checkpoint {filename} saved.")

        if state.epoch%(config.save_weight_every//len(train_loader)) != 0:
            filename = f"checkpoint_{state.step:08d}.ckpt"
            if is_finetune: filename = "finetune_" + filename
            filename = str(checkpoint_dir / filename)
            save_snapshot(filename, state, model, optimizer, scheduler, scaler)
            recorder.print(f"Checkpoint {filename} saved.")


if __name__ == '__main__':
    args = get_args()
    main(args)