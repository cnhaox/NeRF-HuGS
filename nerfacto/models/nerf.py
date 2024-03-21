from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

from models.custom_functions import pos_enc, trunc_exp, spatial_distortion_norm2
from utils.utils import split_tensor_data, merge_tensor_data
from utils.ray_utils import sample_intervals, density_to_weight, dual_density_to_weight, render_features, render_combined_features, render_depth
from utils.loss_utils import get_robustnerf_mask

@dataclass
class ModelConfig:
    net_depth: int = 8
    """The depth of the first part of MLP."""
    net_width: int = 256
    """The width of the first part of MLP."""
    bottleneck_width: int = 256
    """The width of the bottleneck vector."""
    net_depth_viewdirs: int = 1
    """The depth of the second part of ML."""
    net_width_viewdirs: int = 128
    """The width of the second part of MLP."""
    net_depth_transient: int = 4
    """The depth of the transient ML."""
    net_width_transient: int = 128
    """The width of the transient MLP."""
    net_activation: str = 'relu'
    """The activation function."""
    min_deg_point: int = 0
    """Min degree of positional encoding for 3D points."""
    max_deg_point: int = 12
    """Max degree of positional encoding for 3D points."""
    skip_layer: int = 4
    """Add a skip connection to the output of every N layers."""
    skip_layer_dir: int = 4
    """Add a skip connection to 2nd MLP every N layers."""
    skip_layer_transient: int = 4
    """Add a skip connection to transient MLP every N layers."""
    deg_view: int = 4
    """Degree of encoding for viewdirs or refdirs."""
    bottleneck_noise: float = 0.0
    """Std. deviation of noise added to bottleneck."""
    density_activation: str = 'softplus'
    """Density activation."""
    density_bias: float = -1.
    """Shift added to raw densities pre-activation."""
    density_noise: float = 0.
    """Standard deviation of noise added to raw density."""
    rgb_premultiplier: float = 1.
    """Premultiplier on RGB before activation."""
    rgb_activation: str = 'sigmoid'
    """The RGB activation."""
    rgb_bias: float = 0.
    """The shift added to raw colors pre-activation."""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs."""
    beta_min: float = 0.1

    transient_type: Optional[str] = None # None, 'robustnerf', 'nerfw', 'hanerf', 'withmask'
    num_embedding: int = 3500
    use_appearance_embedding: bool = False
    use_transient_embedding: bool = False
    appearance_embedding_dim: int = 32
    """The dim of appearance embedding."""
    transient_embedding_dim: int = 16
    """The dim of transient embedding."""
    eval_embedding: str='average'
    """Whether to use average appearance embedding or zeros for inference."""

    net_depth_implicit: int = 4
    """The depth of the implicitMLP."""
    net_width_implicit: int = 256
    """The width of the implicitMLP."""
    deg_implicit: int = 10
    """Degree of encoding for implicitMLP."""

    num_coarse_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the coarse nerf network."""
    num_fine_nerf_samples_per_ray: int = 128
    """Number of samples per ray for the fine nerf network."""
    proposal_initial_sampler: str = 'uniform'
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    use_single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""
    opaque_background: bool = False

    # setting of loss functions
    rgb_loss_type: str = 'mse'
    rgb_charb_loss_padding: float = 0.001
    coarse_rgb_loss_mult: float = 1.0
    fine_rgb_loss_mult: float = 1.0

    # nerfw loss
    nerfw_beta_loss_mult: float = 1.0
    nerfw_beta_loss_bias: float = 3.0
    nerfw_density_loss_mult: float = 0.01

    # hanerf loss
    hanerf_mask_size_loss_mult_min: float = 6e-3
    hanerf_mask_size_loss_mult_max: float = 5e-2
    hanerf_mask_size_loss_mult_k: float = 1e-3

    # robust loss
    robustnerf_inlier_quantile: float = 0.8
    robustnerf_smoothed_filter_size: int = 3
    robustnerf_smoothed_inlier_quantile: float = 0.5
    robustnerf_inner_patch_size: int = 8
    robustnerf_inner_patch_inlier_quantile: float = 0.4

    # withmask loss
    withmask_transient_weight: float = 0.


class Model(nn.Module):
    def __init__(
        self, 
        config: ModelConfig,
        bound: Optional[float],
        enable_amp: bool,
        enable_scene_contraction: bool,
    ) -> None:
        super().__init__()

        self.config = config
        self.bound = bound # not used in nerf
        self.enable_amp = enable_amp
        self.enable_scene_contraction = enable_scene_contraction
        if self.enable_scene_contraction:
            scene_contraction = lambda x: spatial_distortion_norm2(x)
        else:
            scene_contraction = None
        
        if self.config.transient_type in ['nerfw', 'hanerf']:
            assert self.config.transient_embedding_dim > 0 \
                and self.config.use_transient_embedding
        else:
            assert not self.config.use_transient_embedding

        # embedding
        self.use_appearance_embedding = self.config.use_appearance_embedding
        if self.use_appearance_embedding:
            self.embedding_appearance = nn.Embedding(
                self.config.num_embedding, self.config.appearance_embedding_dim
            )
            appearance_embedding_dim = self.config.appearance_embedding_dim
        else:
            self.embedding_appearance = None
            appearance_embedding_dim = 0
        
        self.use_transient_embedding = self.config.use_transient_embedding
        if self.use_transient_embedding:
            self.embedding_transient = nn.Embedding(
                self.config.num_embedding, self.config.transient_embedding_dim
            )
            transient_embedding_dim = self.config.transient_embedding_dim
        else:
            self.embedding_transient = None
            transient_embedding_dim = 0
        
        # Fields
        model_arg = {
            'net_depth': self.config.net_depth,
            'net_width': self.config.net_width,
            'bottleneck_width': self.config.bottleneck_width,
            'appearance_embedding_dim': appearance_embedding_dim,
            'net_depth_viewdirs': self.config.net_depth_viewdirs,
            'net_width_viewdirs': self.config.net_width_viewdirs,
            'net_depth_transient': self.config.net_depth_transient,
            'net_width_transient': self.config.net_width_transient,
            'net_activation': self.config.net_activation,
            'min_deg_point': self.config.min_deg_point,
            'max_deg_point': self.config.max_deg_point,
            'skip_layer': self.config.skip_layer,
            'skip_layer_dir': self.config.skip_layer_dir,
            'skip_layer_transient': self.config.skip_layer_transient,
            'deg_view': self.config.deg_view,
            'bottleneck_noise': self.config.bottleneck_noise,
            'density_activation': self.config.density_activation,
            'density_bias': self.config.density_bias,
            'density_noise': self.config.density_noise,
            'rgb_premultiplier': self.config.rgb_premultiplier,
            'rgb_activation': self.config.rgb_activation,
            'rgb_bias': self.config.rgb_bias,
            'rgb_padding': self.config.rgb_padding,
            'spatial_distortion': scene_contraction,
        }
        tmp_dim = transient_embedding_dim if self.config.transient_type=='nerfw' else 0
        self.field = nn.ModuleDict({
            'coarse': MLP(transient_embedding_dim=0, **model_arg),
            'fine': MLP(transient_embedding_dim=tmp_dim, **model_arg)
        })

        if self.config.transient_type == 'hanerf':
            self.implicit_mask = ImplicitMask(
                net_depth=self.config.net_depth_implicit,
                net_width=self.config.net_width_implicit,
                transient_embedding_dim=transient_embedding_dim,
                net_activation=self.config.net_activation,
                deg_coord=self.config.deg_implicit
            )
        else:
            self.implicit_mask = None

        if self.config.proposal_initial_sampler == 'piecewise':
            self.spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
            self.spacing_fn_inv = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))
        elif self.config.proposal_initial_sampler == 'uniform':
            self.spacing_fn = lambda x: x
            self.spacing_fn_inv = lambda x: x
        elif self.config.proposal_initial_sampler == 'reciprocal':
            self.spacing_fn = lambda x: torch.reciprocal(x)
            self.spacing_fn_inv = lambda x: torch.reciprocal(x)
        else:
            raise ValueError(f"Sampler does not support {self.config.proposal_initial_sampler}. ")

    def construct_ray_warps(self, t_near: Tensor, t_far: Tensor):
        s_near = self.spacing_fn(t_near)
        s_far = self.spacing_fn(t_far)
        t_to_s = lambda t: (self.spacing_fn(t) - s_near) / (s_far - s_near)
        s_to_t = lambda s: self.spacing_fn_inv(s * s_far + (1 - s) * s_near)
        return t_to_s, s_to_t

    def get_params_dict(self) -> Dict[str, List[Parameter]]:
        params_dict = {
            'field': list(self.field.parameters())
        }
        if self.embedding_appearance is not None:
            params_dict['appearance_embedding'] = \
                list(self.embedding_appearance.parameters())
        if self.embedding_transient is not None:
            params_dict['transient_embedding'] = \
                list(self.embedding_transient.parameters())
        if self.implicit_mask is not None:
            params_dict['implicit_mask'] = \
                list(self.implicit_mask.parameters())
        return params_dict

    def get_embedding(self, embedding: nn.Embedding, embed_idxs: Tensor) -> Tensor:
        embedding_dim = embedding.weight.shape[-1]
        device = embed_idxs.device
        if self.training:
            embed = embedding(embed_idxs)
        else:
            if self.config.eval_embedding=='average':
                embed = torch.ones(
                    (*embed_idxs.shape, embedding_dim), device=device
                ) * embedding.weight.mean(dim=0)
            elif self.config.eval_embedding=='zero':
                embed = torch.zeros(
                    (*embed_idxs.shape, embedding_dim), device=device
                )
            elif self.config.eval_embedding=='original':
                embed = embedding(embed_idxs)
            else:
                raise NotImplementedError(f"{self.config.eval_embedding} is not supported.")
        return embed

    def forward_rays(self, rays: Dict[str, Tensor], curr_step: int, perturb: bool) -> dict:
        _, s_to_t = self.construct_ray_warps(rays['near'], rays['far'])

        spacing_bins = torch.cat(
            [torch.zeros_like(rays['near']), torch.ones_like(rays['far'])], 
            dim=-1
        )
        weights = torch.ones_like(rays['near'])
        spacing_domain = (0., 1.)

        outputs = {}
        for field_type in ['coarse', 'fine']:
            num_samples = self.config.num_coarse_nerf_samples_per_ray if field_type=='coarse' \
                else self.config.num_fine_nerf_samples_per_ray
            with torch.no_grad():
                anneal = 1.
                padding = 0.
                spacing_bins_ = sample_intervals(
                    spacing_bins, weights, anneal, padding, num_samples, 
                    perturb, self.config.use_single_jitter, spacing_domain
                )
                if field_type=='coarse':
                    spacing_bins = spacing_bins_
                else:
                    centers = (spacing_bins[..., 1:] + spacing_bins[..., :-1]) / 2
                    centers_ = (spacing_bins_[..., 1:] + spacing_bins_[..., :-1]) / 2
                    centers, _ = torch.sort(torch.cat([centers, centers_], -1), -1)
                    mid = (centers[..., 1:] + centers[..., :-1]) / 2
                    spacing_bins = torch.cat([
                        (2 * centers[..., :1] - mid[..., :1]).clamp_min(spacing_domain[0]),
                        mid,
                        (2 * centers[..., -1:] - mid[..., -1:]).clamp_max(spacing_domain[1])
                    ], dim=-1)
                euclidean_bins = s_to_t(spacing_bins)

            # prepare inputs of field 
            rays_t = (euclidean_bins[..., 1:] + euclidean_bins[..., :-1]) / 2 # (num_rays, num_samples)
            positions = rays['origin'].unsqueeze(1) + rays['direction'].unsqueeze(1) * rays_t.unsqueeze(2) # (num_rays, num_samples, 3)
            viewdirs = rays['viewdir'].unsqueeze(1).expand_as(positions)
            embed_idxs = rays['embed_idx'].expand_as(positions[..., 0])

            data_shape = positions.shape[:-1]
            positions = positions.contiguous().view(-1, 3)
            viewdirs = viewdirs.contiguous().view(-1, 3)
            embed_idxs = embed_idxs.contiguous().view(-1)

            embedded_appearance = None if not self.use_appearance_embedding \
                else self.get_embedding(self.embedding_appearance, embed_idxs).contiguous()
            embedded_transient = None if field_type=='coarse' or self.config.transient_type != 'nerfw' \
                else self.get_embedding(self.embedding_transient, embed_idxs).contiguous()
            
            field_outputs = self.field[field_type](
                positions, viewdirs, embedded_appearance, embedded_transient
            )
            for key in field_outputs.keys(): 
                field_outputs[key] = field_outputs[key].view(*data_shape, -1)
            
            weights, _, _ = density_to_weight(
                field_outputs['density'][..., 0], euclidean_bins, 
                rays['direction'], self.config.opaque_background
            )
            accumulated_weight = torch.sum(weights, dim=-1)
            color = render_features(weights, field_outputs['rgb'], rays['bg_rgb'], False)
            depth = render_depth(weights, euclidean_bins)
            
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            outputs[f'rgb{suffix}'] = color
            outputs[f'depth{suffix}'] = depth
            outputs[f'accumulation{suffix}'] = accumulated_weight

            if 'density_transient' in field_outputs.keys():                
                weights_static_partial, weights_transient_partial, weights_combined = dual_density_to_weight(
                    field_outputs['density'][..., 0],
                    field_outputs['density_transient'][..., 0],
                    euclidean_bins, 
                    rays['direction'], 
                    self.config.opaque_background
                )
                rgb_combined, rgb_static_partial, rgb_transient_partial = render_combined_features(
                    weights_static_partial, weights_transient_partial, weights_combined,
                    field_outputs['rgb'], field_outputs['rgb_transient'], rays['bg_rgb'], False
                )
                
                weights_transient, _, _ = density_to_weight(
                    field_outputs['density_transient'][..., 0], euclidean_bins,
                    rays['direction'], self.config.opaque_background
                )
                uncertainty = render_features(
                    weights_transient, field_outputs['uncertainty'], None, False
                ) + self.config.beta_min
                depth_combined = render_depth(weights_combined, euclidean_bins)
                outputs[f'rgb_static{suffix}'] = rgb_static_partial
                outputs[f'rgb_transient{suffix}'] = rgb_transient_partial
                outputs[f'rgb_combined{suffix}'] = rgb_combined
                outputs[f'uncertainty{suffix}'] = uncertainty
                outputs[f'accumulation_transient_partial{suffix}'] = torch.sum(weights_transient_partial, dim=-1)
                outputs[f'depth_combined{suffix}'] = depth_combined
                
                if self.training:
                    outputs[f'density_transient{suffix}'] = field_outputs['density_transient'][..., 0]
        
        if self.implicit_mask is not None:
            coords = rays['coord'] # (n_rays, 2)
            embed_idxs = rays['embed_idx'][..., 0] # (n_rays, )
            embedded_transient = self.get_embedding(self.embedding_transient, embed_idxs).contiguous()
            mask = self.implicit_mask(coords, embedded_transient)
            outputs['implicit_mask'] = mask
        
        return outputs

    def forward(self, batch: Dict[str, Tensor], curr_step: int, perturb: bool, chunk_size: Optional[int] = None) -> dict:
        if self.training:
            outputs = self.forward_rays(batch, curr_step, perturb)
        else:
            batch_list = split_tensor_data(batch, chunk_size)
            outputs_list = list()
            for sub_batch in batch_list:
                outputs_list.append(self.forward_rays(sub_batch, curr_step, perturb))
            outputs = merge_tensor_data(outputs_list)

        return outputs


class Loss(nn.Module):
    def __init__(
        self,
        model: Model
    ) -> None:
        super().__init__()
        self.config = model.config
        if self.config.rgb_loss_type == 'mse':
            # Mean-squared error (L2) loss.
            self.rgb_loss = \
                lambda resid_sq: resid_sq
        elif self.config.rgb_loss_type == 'charb':
            # Charbonnier loss.
            self.rgb_loss = \
                lambda resid_sq: torch.sqrt(resid_sq + self.config.rgb_charb_loss_padding**2)
        else:
            raise NotImplementedError()

    def compute_data_loss(
        self,
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        extra_infos: dict,
    ):
        loss_list = []
        info_dict = {}
        gt_rgb = batch['rgb'].reshape(*data_shape, -1)

        for field_type in ['coarse', 'fine']:
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            if field_type == 'coarse':
                rgb_loss_mult = self.config.coarse_rgb_loss_mult
            else:
                rgb_loss_mult = self.config.fine_rgb_loss_mult

            pred_rgb = outputs[f'rgb{suffix}'].reshape(*data_shape, -1)
            resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
            rgb_loss = rgb_loss_mult * self.rgb_loss(resid_sq).mean()
            loss_list.append(rgb_loss)
            info_dict[f'rgb_loss{suffix}'] = rgb_loss.detach()
            info_dict[f'mse{suffix}'] = resid_sq.detach().mean()
        
        return sum(loss_list), info_dict, extra_infos

    def compute_withmask_loss(
        self,
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        extra_infos: dict,
    ):
        loss_list = []
        info_dict = {}
        gt_rgb = batch['rgb'].reshape(*data_shape, -1)
        static_mask = (batch['static_mask'].reshape(*data_shape, 1) >= 0.5).to(gt_rgb)

        for field_type in ['coarse', 'fine']:
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            if field_type == 'coarse':
                rgb_loss_mult = self.config.coarse_rgb_loss_mult
            else:
                rgb_loss_mult = self.config.fine_rgb_loss_mult
            pred_rgb = outputs[f'rgb{suffix}'].reshape(*data_shape, -1)
            resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)

            lossmult = static_mask + (1 - static_mask) * self.config.withmask_transient_weight
            lossmult = lossmult.broadcast_to(resid_sq.shape)
            denom = lossmult.sum().clamp_min(torch.finfo(lossmult.dtype).eps)
            rgb_loss = rgb_loss_mult * ((lossmult * self.rgb_loss(resid_sq)).sum() / denom)

            loss_list.append(rgb_loss)
            info_dict[f'rgb_loss{suffix}'] = rgb_loss.detach()
            info_dict[f'mse{suffix}'] = (lossmult * resid_sq.detach()).sum() / denom
        
        return sum(loss_list), info_dict, extra_infos

    def compute_robustnerf_loss(
        self,
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        extra_infos: dict
    ):
        loss_list = []
        info_dict = {}
        gt_rgb = batch['rgb'].reshape(*data_shape, -1)

        for field_type in ['coarse', 'fine']:
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            if field_type == 'coarse':
                rgb_loss_mult = self.config.coarse_rgb_loss_mult
            else:
                rgb_loss_mult = self.config.fine_rgb_loss_mult

            pred_rgb = outputs[f'rgb{suffix}'].reshape(*data_shape, -1)
            resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
            rgb_loss = self.rgb_loss(resid_sq)
            errors = resid_sq.detach()

            robust_mask, robust_info_dict, extra_infos = get_robustnerf_mask(
                errors=errors, 
                suffix=suffix,
                extra_infos=extra_infos,
                inlier_quantile=self.config.robustnerf_inlier_quantile,
                smoothed_filter_size=self.config.robustnerf_smoothed_filter_size,
                smoothed_inlier_quantile=self.config.robustnerf_smoothed_inlier_quantile,
                inner_patch_size=self.config.robustnerf_inner_patch_size,
                inner_patch_inlier_quantile=self.config.robustnerf_inner_patch_inlier_quantile,
            )
            for key in robust_info_dict.keys(): 
                info_dict[key] = robust_info_dict[key]
            
            lossmult = torch.broadcast_to(robust_mask, rgb_loss.shape)
            denom = lossmult.sum().clamp_min(torch.finfo(lossmult.dtype).eps)
            rgb_loss = rgb_loss_mult * ((lossmult * rgb_loss).sum() / denom)
            loss_list.append(rgb_loss)
            info_dict[f'rgb_loss{suffix}'] = rgb_loss.detach()
            info_dict[f'mse{suffix}'] = (lossmult * resid_sq.detach()).sum() / denom
        
        return sum(loss_list), info_dict, extra_infos

    def compute_nerfw_loss(
        self,
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        extra_infos: dict,
    ):
        loss_list = []
        info_dict = {}
        gt_rgb = batch['rgb'].reshape(*data_shape, -1)

        for field_type in ['coarse', 'fine']:
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            if field_type == 'coarse':
                rgb_loss_mult = self.config.coarse_rgb_loss_mult
            else:
                rgb_loss_mult = self.config.fine_rgb_loss_mult
            
            pred_type = f'rgb_combined{suffix}'
            if pred_type not in outputs.keys():
                pred_type = f'rgb{suffix}'
            pred_rgb = outputs[pred_type].reshape(*data_shape, -1)
            resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
            rgb_loss = self.rgb_loss(resid_sq)

            if f'uncertainty{suffix}' in outputs.keys():
                beta = outputs[f'uncertainty{suffix}'].reshape(*data_shape, -1)
                rgb_loss = rgb_loss / (2 * beta**2)
                
                beta_loss = self.config.nerfw_beta_loss_mult * torch.log(beta).mean() + self.config.nerfw_beta_loss_bias # add bias to avoid loss < 0
                loss_list.append(beta_loss)
                info_dict[f'beta_loss{suffix}'] = beta_loss.detach()
            
            if f'density_transient{suffix}' in outputs.keys():
                density = outputs[f'density_transient{suffix}']
                density_loss = self.config.nerfw_density_loss_mult * density.mean()
                loss_list.append(density_loss)
                info_dict[f'density_loss{suffix}'] = density_loss.detach()

            rgb_loss = rgb_loss_mult * rgb_loss.mean()
            loss_list.append(rgb_loss)
            info_dict[f'rgb_loss{suffix}'] = rgb_loss.detach()
            info_dict[f'mse{suffix}'] = resid_sq.detach().mean()
        
        return sum(loss_list), info_dict, extra_infos

    def compute_hanerf_loss(
        self,
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        extra_infos: dict,
    ):
        loss_list = []
        info_dict = {}
        curr_step = extra_infos['curr_step']

        gt_rgb = batch['rgb'].reshape(*data_shape, -1)
        static_mask = (batch['static_mask'].reshape(*data_shape, 1) >= 0.5).to(gt_rgb)

        mask_size_loss_mult = max(
            self.config.hanerf_mask_size_loss_mult_min,
            self.config.hanerf_mask_size_loss_mult_max * np.exp(-curr_step * self.config.hanerf_mask_size_loss_mult_k)
        )

        for field_type in ['coarse', 'fine']:
            suffix = '' if field_type == 'fine' else f'_{field_type}'
            if field_type == 'coarse':
                rgb_loss_mult = self.config.coarse_rgb_loss_mult
            else:
                rgb_loss_mult = self.config.fine_rgb_loss_mult
            
            pred_rgb = outputs[f'rgb{suffix}'].reshape(*data_shape, -1)
            resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
            rgb_loss = self.rgb_loss(resid_sq)

            if 'implicit_mask' in outputs.keys():
                implicit_mask = outputs['implicit_mask'].reshape(*data_shape, -1)
                if field_type == 'coarse':
                    rgb_loss = (1 - implicit_mask.detach()) * rgb_loss
                else:
                    rgb_loss = (1 - implicit_mask) * rgb_loss
                    info_dict['implicit_mask'] = implicit_mask.mean().detach()
                    mask_size_loss = mask_size_loss_mult * (implicit_mask**2).mean()
                    loss_list.append(mask_size_loss)
                    info_dict['mask_size_loss'] = mask_size_loss.detach()

            rgb_loss = rgb_loss_mult * rgb_loss.mean()
            loss_list.append(rgb_loss)
            info_dict[f'rgb_loss{suffix}'] = rgb_loss.detach()
            info_dict[f'mse{suffix}'] = resid_sq.detach().mean()
        
        return sum(loss_list), info_dict, extra_infos

    def forward(
        self, 
        outputs: Dict[str, Tensor], 
        batch: Dict[str, Tensor], 
        data_shape,
        is_finetune: bool,
        extra_infos: dict,
    ):  
        if is_finetune or self.config.transient_type is None:
            loss, info_dict, extra_infos = self.compute_data_loss(
                outputs, batch, data_shape, extra_infos
            )
        elif self.config.transient_type == 'robustnerf':
            loss, info_dict, extra_infos = self.compute_robustnerf_loss(
                outputs, batch, data_shape, extra_infos
            )
        elif self.config.transient_type == 'nerfw':
            loss, info_dict, extra_infos = self.compute_nerfw_loss(
                outputs, batch, data_shape, extra_infos
            )
        elif self.config.transient_type == 'hanerf':
            loss, info_dict, extra_infos = self.compute_hanerf_loss(
                outputs, batch, data_shape, extra_infos
            )
        else:
            raise NotImplementedError()
        
        return loss, info_dict, extra_infos


class MLP(nn.Module):
    def __init__(
        self, 
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        bottleneck_width: int = 256,  # The width of the bottleneck vector.
        appearance_embedding_dim: int = 32, # The dim of appearance embedding.
        transient_embedding_dim: int = 16, # The dim of transient embedding.
        net_depth_viewdirs: int = 1,  # The depth of the second part of ML.
        net_width_viewdirs: int = 128,  # The width of the second part of MLP.
        net_depth_transient: int = 4,  # The depth of the transient ML.
        net_width_transient: int = 128,  # The width of the transient MLP.
        net_activation: str = 'relu',  # The activation function.
        min_deg_point: int = 0,  # Min degree of positional encoding for 3D points.
        max_deg_point: int = 12,  # Max degree of positional encoding for 3D points.
        skip_layer: int = 4,  # Add a skip connection to the output of every N layers.
        skip_layer_dir: int = 4,  # Add a skip connection to 2nd MLP every N layers.
        skip_layer_transient: int = 4,  # Add a skip connection to transient MLP every N layers.
        deg_view: int = 4,  # Degree of encoding for viewdirs or refdirs.
        bottleneck_noise: float = 0.0,  # Std. deviation of noise added to bottleneck.
        density_activation: str = 'softplus',  # Density activation.
        density_bias: float = -1.,  # Shift added to raw densities pre-activation.
        density_noise: float = 0.,  # Standard deviation of noise added to raw density.
        rgb_premultiplier: float = 1.,  # Premultiplier on RGB before activation.
        rgb_activation: str = 'sigmoid',  # The RGB activation.
        rgb_bias: float = 0.,  # The shift added to raw colors pre-activation.
        rgb_padding: float = 0.001,  # Padding added to the RGB outputs.
        spatial_distortion:  Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.bottleneck_width = bottleneck_width
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim
        self.net_depth_viewdirs = net_depth_viewdirs
        self.net_width_viewdirs = net_width_viewdirs
        self.net_depth_transient = net_depth_transient
        self.net_width_transient = net_width_transient
        self.net_activation = net_activation
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.skip_layer = skip_layer
        self.skip_layer_dir = skip_layer_dir
        self.skip_layer_transient = skip_layer_transient
        self.deg_view = deg_view
        self.bottleneck_noise = bottleneck_noise

        self.density_bias = density_bias
        self.density_noise = density_noise
        if density_activation == 'relu':
            self.density_activation = \
                lambda raw_density: F.relu(raw_density + self.density_bias)
        elif density_activation == 'softplus':
            self.density_activation = \
                lambda raw_density: F.softplus(raw_density + self.density_bias)
        elif density_activation == 'trunc_exp':
            self.density_activation = \
                lambda raw_density: trunc_exp(raw_density + self.density_bias)
        else: raise NotImplementedError()

        self.rgb_premultiplier = rgb_premultiplier
        self.rgb_bias = rgb_bias
        self.rgb_padding = rgb_padding
        if rgb_activation == 'sigmoid':
            self.rgb_activation = \
                lambda raw_rgb: F.sigmoid(self.rgb_premultiplier * raw_rgb + self.rgb_bias)
        else: raise NotImplementedError()

        self.spatial_distortion = spatial_distortion

        self.pos_enc_fn = \
            lambda pos: pos_enc(pos, self.min_deg_point, self.max_deg_point, True)
        pos_samples = self.pos_enc_fn(torch.rand((1,3), dtype=torch.float32))

        mlp_base = []
        in_dim = pos_samples.shape[-1]
        last_dim = in_dim
        sub_mlp = []
        for i in range(self.net_depth):
            lin = nn.Linear(last_dim, self.net_width)
            torch.nn.init.kaiming_uniform_(lin.weight)
            sub_mlp.append(lin)
            if self.net_activation == 'relu': sub_mlp.append(nn.ReLU())
            elif self.net_activation == 'softplus': sub_mlp.append(nn.Softplus())
            else: raise NotImplementedError()        
            if i % self.skip_layer == 0 and i > 0:
                last_dim = self.net_width + in_dim
                mlp_base.append(nn.Sequential(*sub_mlp))
                sub_mlp = []
            else:
                last_dim = self.net_width
        if len(sub_mlp) > 0: mlp_base.append(nn.Sequential(*sub_mlp))
        self.mlp_base = nn.ModuleList(mlp_base)
        self.mlp_density = nn.Linear(last_dim, 1)
        torch.nn.init.kaiming_uniform_(self.mlp_density.weight)

        if self.bottleneck_width > 0:
            self.mlp_bottleneck = nn.Linear(self.net_width, self.bottleneck_width)

        in_dim = self.bottleneck_width if self.bottleneck_width > 0 else 0
        self.dir_enc_fn = \
            lambda dir: pos_enc(dir, 0, self.deg_view, True)
        dir_samples = self.dir_enc_fn(torch.rand((1,3), dtype=torch.float32))
        in_dim += dir_samples.shape[-1]
        in_dim += self.appearance_embedding_dim

        mlp_head = []
        last_dim = in_dim
        sub_mlp = []
        for i in range(self.net_depth_viewdirs):
            lin = nn.Linear(last_dim, self.net_width_viewdirs)
            torch.nn.init.kaiming_uniform_(lin.weight)
            sub_mlp.append(lin)
            if self.net_activation == 'relu': sub_mlp.append(nn.ReLU())
            elif self.net_activation == 'softplus': sub_mlp.append(nn.Softplus())
            else: raise NotImplementedError()
            if i % self.skip_layer_dir == 0 and i > 0:
                last_dim = self.net_width_viewdirs + in_dim
                mlp_head.append(nn.Sequential(*sub_mlp))
                sub_mlp = []
            else:
                last_dim = self.net_width_viewdirs
        if len(sub_mlp) > 0: mlp_head.append(nn.Sequential(*sub_mlp))
        self.mlp_head = nn.ModuleList(mlp_head)
        self.mlp_rgb = nn.Linear(last_dim, 3)
        torch.nn.init.kaiming_uniform_(self.mlp_rgb.weight)

        if self.transient_embedding_dim > 0:
            in_dim = self.bottleneck_width if self.bottleneck_width > 0 else 0
            in_dim += self.transient_embedding_dim
            mlp_transient = []
            last_dim = in_dim
            sub_mlp = []
            for i in range(self.net_depth_transient):
                lin = nn.Linear(last_dim, self.net_width_transient)
                torch.nn.init.kaiming_uniform_(lin.weight)
                sub_mlp.append(lin)
                if self.net_activation == 'relu': sub_mlp.append(nn.ReLU())
                elif self.net_activation == 'softplus': sub_mlp.append(nn.Softplus())
                else: raise NotImplementedError()
                if i % self.skip_layer_transient == 0 and i > 0:
                    last_dim = self.net_width_transient + in_dim
                    mlp_transient.append(nn.Sequential(*sub_mlp))
                    sub_mlp = []
                else:
                    last_dim = self.net_width_transient
            if len(sub_mlp) > 0: mlp_transient.append(nn.Sequential(*sub_mlp))
            self.mlp_transient = nn.ModuleList(mlp_transient)
            self.mlp_density_transient = nn.Linear(last_dim, 1)
            torch.nn.init.kaiming_uniform_(self.mlp_density_transient.weight)
            self.mlp_rgb_transient = nn.Linear(last_dim, 3)
            torch.nn.init.kaiming_uniform_(self.mlp_rgb_transient.weight)
            self.mlp_uncertainty = nn.Linear(last_dim, 1)
            torch.nn.init.kaiming_uniform_(self.mlp_uncertainty.weight)

    def density(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        positions: (n, 3)
        '''
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        x = self.pos_enc_fn(positions)

        inputs = x
        for i in range(len(self.mlp_base)):
            if i > 0: x = torch.cat([x, inputs], dim=-1)
            x = self.mlp_base[i](x)
        
        raw_density = self.mlp_density(x).to(positions)
        if self.training and self.density_noise > 0:
            raw_density = raw_density + self.density_noise * torch.randn_like(raw_density)
        density = self.density_activation(raw_density)
        return density, x

    def forward(
        self, positions: Tensor, viewdirs: Tensor, 
        embedded_appearance: Optional[Tensor],
        embedded_transient: Optional[Tensor]
    ) -> Dict[str, Tensor]:
        '''
        positions: (n, 3)
        viewdirs: (n, 3)
        embedded_appearance: None or (n, appearance_embedding_dim)
        '''
        outputs = {}
        density, x = self.density(positions)

        bottleneck = self.mlp_bottleneck(x)
        if self.training and self.bottleneck_noise > 0:
            bottleneck = bottleneck + self.bottleneck_noise * torch.randn_like(bottleneck)
        
        x = [bottleneck, self.dir_enc_fn(viewdirs)]
        if embedded_appearance is not None: x.append(embedded_appearance)
        x = torch.cat(x, dim=-1)
        inputs = x
        for i in range(len(self.mlp_head)):
            if i > 0: x = torch.cat([x, inputs], dim=-1)
            x = self.mlp_head[i](x)
        raw_rgb = self.mlp_rgb(x).to(positions)
        rgb = self.rgb_activation(raw_rgb) * (1 + 2 * self.rgb_padding) - self.rgb_padding

        outputs = {
            'density': density,
            'rgb': rgb
        }

        if self.transient_embedding_dim > 0:
            x = torch.cat([bottleneck, embedded_transient], dim=-1)
            inputs = x
            for i in range(len(self.mlp_transient)):
                if i > 0: x = torch.cat([x, inputs], dim=-1)
                x = self.mlp_transient[i](x)
            
            raw_density_transient = self.mlp_density_transient(x).to(positions)
            if self.training and self.density_noise > 0:
                raw_density_transient = raw_density_transient + self.density_noise * torch.randn_like(raw_density_transient)
            density_transient = self.density_activation(raw_density_transient)
            
            raw_rgb_transient = self.mlp_rgb_transient(x).to(positions)
            rgb_transient = self.rgb_activation(raw_rgb_transient) * (1 + 2 * self.rgb_padding) - self.rgb_padding
            
            uncertainty = F.softplus(self.mlp_uncertainty(x).to(positions))

            outputs['density_transient'] = density_transient
            outputs['rgb_transient'] = rgb_transient
            outputs['uncertainty'] = uncertainty

        return outputs
    

class ImplicitMask(nn.Module):
    def __init__(
        self,
        net_depth: int = 4, # The depth of the MLP.
        net_width: int = 256, # The width of the MLP.
        transient_embedding_dim: int = 128, # The dim of appearance embedding.
        net_activation: str = 'relu',  # The activation function.
        deg_coord: int = 10,  # Max degree of positional encoding for 2D coordinate.
    ) -> None:
        super().__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.transient_embedding_dim = transient_embedding_dim
        self.net_activation = net_activation
        self.deg_coord = deg_coord
        
        self.coord_enc_fn = \
            lambda coord: pos_enc(coord, 0, self.deg_coord, True)
        coord_samples = self.coord_enc_fn(torch.rand((1,2), dtype=torch.float32))

        mlp_base = []
        in_dim = coord_samples.shape[-1] + transient_embedding_dim
        last_dim = in_dim
        for i in range(self.net_depth):
            lin = nn.Linear(last_dim, self.net_width)
            torch.nn.init.kaiming_uniform_(lin.weight)
            mlp_base.append(lin)
            if self.net_activation == 'relu': 
                mlp_base.append(nn.ReLU())
            elif self.net_activation == 'softplus': 
                mlp_base.append(nn.Softplus())
            else: 
                raise NotImplementedError()
            last_dim = self.net_width
        lin = nn.Linear(last_dim, 1)
        torch.nn.init.kaiming_uniform_(lin.weight)
        mlp_base.append(lin)
        mlp_base.append(nn.Sigmoid())
        self.mlp_base = nn.Sequential(*mlp_base)

    def forward(self, coordinates: Tensor, embedded_transient: Tensor) -> Tensor:
        '''
        coordinates: (n, 2)
        embedded_transient: None or (n, transient_embedding_dim)
        '''
        x = self.coord_enc_fn(coordinates)
        out = self.mlp_base(
            torch.cat([x, embedded_transient], dim=-1)
        ).to(coordinates)

        return out