from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import tinycudann as tcnn

from models.custom_functions import spatial_distortion, trunc_exp, spatial_distortion_norm2
from utils.utils import split_tensor_data, merge_tensor_data
from utils.ray_utils import sample_intervals, density_to_weight, dual_density_to_weight, render_features, render_combined_features, render_depth
from utils.loss_utils import interlevel_loss, distortion_loss, get_robustnerf_mask

@dataclass
class ModelConfig:
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    geo_feat_dim: int = 15
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 128
    density_activation: str = 'trunc_exp' # 'trunc_exp', 'softplus'
    enable_tcnn_mlp: bool = True
    beta_min: float = 0.1

    transient_type: Optional[str] = None # None, 'robustnerf', 'nerfw', 'hanerf', 'withmask'
    num_embedding: int = 3500
    use_appearance_embedding: bool = False
    use_transient_embedding: bool = False
    appearance_embedding_dim: int = 32
    transient_embedding_dim: int = 16
    eval_embedding: str='average'
    """Whether to use average appearance embedding or zeros for inference."""

    num_levels_implicit: int = 8
    base_res_implicit: int = 16
    max_res_implicit: int = 1024
    log2_hashmap_size_implicit: int = 17
    features_per_level_implicit: int = 2
    hidden_dim_implicit: int = 128

    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Optional[str] = None
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    proposal_histogram_padding: float = 0.01
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights. Set it <= 0 to disable."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    opaque_background: bool = False

    # setting of loss functions
    rgb_loss_type: str = 'mse'
    rgb_charb_loss_padding: float = 0.001
    rgb_loss_mult: float = 1.0
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
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
        self.bound = bound
        self.enable_amp = enable_amp
        self.enable_scene_contraction = enable_scene_contraction
        assert self.bound is not None
        if self.enable_scene_contraction:
            assert self.bound == 2.0, f"When using scene contraction, bound should be set to 2, but got {self.bound}"
            # scene_contraction = lambda x: spatial_distortion(x, float("inf"))
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
        self.field = NerfactoField(
            bound=self.bound,
            num_levels=self.config.num_levels,
            base_res=self.config.base_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            hidden_dim=self.config.hidden_dim,
            geo_feat_dim=self.config.geo_feat_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            density_activation=self.config.density_activation,
            enable_tcnn_mlp=self.config.enable_tcnn_mlp,
            enable_amp=self.enable_amp,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim if self.config.transient_type=='nerfw' else 0,
            spatial_distortion=scene_contraction
        )

        # proposal networks
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                bound=self.bound,
                density_activation=self.config.density_activation,
                enable_tcnn_mlp=self.config.enable_tcnn_mlp,
                enable_amp=self.enable_amp,
                spatial_distortion=scene_contraction,
                **prop_net_args
            )
            self.proposal_networks.append(network)
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    bound=self.bound,
                    density_activation=self.config.density_activation,
                    enable_tcnn_mlp=self.config.enable_tcnn_mlp,
                    enable_amp=self.enable_amp,
                    spatial_distortion=scene_contraction,
                    **prop_net_args
                )
                self.proposal_networks.append(network)

        if self.config.transient_type == 'hanerf':
            self.implicit_mask = ImplicitMask(
                num_levels=self.config.num_levels_implicit,
                base_res=self.config.base_res_implicit,
                max_res=self.config.max_res_implicit,
                log2_hashmap_size=self.config.log2_hashmap_size_implicit,
                hidden_dim=self.config.hidden_dim_implicit,
                features_per_level=self.config.features_per_level_implicit,
                transient_embedding_dim=transient_embedding_dim,
                enable_tcnn_mlp=self.config.enable_tcnn_mlp,
                enable_amp=self.enable_amp,
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
            'field': list(self.field.parameters()),
            'proposal': list(self.proposal_networks.parameters())
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

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(curr_step / N, 0, 1)
            bias = lambda x, s:  (s * x) / ((s - 1) * x + 1)
            anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
        else:
            anneal = 1.0
        
        proposal_update_interval = int(np.clip(
            np.interp(curr_step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1, self.config.proposal_update_every
        ))
        enable_proposal_update = ((curr_step % proposal_update_interval) == 0)
        
        weights_list = []
        spacing_bins_list = []
        euclidean_bins_list = []
        spacing_bins = torch.cat(
            [torch.zeros_like(rays['near']), torch.ones_like(rays['far'])], 
            dim=-1
        )
        weights = torch.ones_like(rays['near'])
        spacing_domain = (0., 1.)
        outputs = {}

        for i_level in range(self.config.num_proposal_iterations + 1):
            is_prop = i_level < self.config.num_proposal_iterations
            num_samples = self.config.num_proposal_samples_per_ray[i_level] if is_prop \
                            else self.config.num_nerf_samples_per_ray
            
            with torch.no_grad():
                padding = self.config.proposal_histogram_padding
                spacing_bins = sample_intervals(
                    spacing_bins, weights, anneal, padding, num_samples, 
                    perturb, self.config.use_single_jitter, spacing_domain
                )

            euclidean_bins = s_to_t(spacing_bins)
            rays_t = (euclidean_bins[..., 1:] + euclidean_bins[..., :-1]) / 2 # (num_rays, num_samples)
            positions = rays['origin'].unsqueeze(1) + rays['direction'].unsqueeze(1) * rays_t.unsqueeze(2) # (num_rays, num_samples, 3)
            data_shape = positions.shape[:-1]
            
            if is_prop:
                net_idx = 0 if self.config.use_same_proposal_network else i_level
                positions = positions.contiguous().view(-1, 3)
                viewdirs = None
                embedded_appearance = None
                embedded_transient = None
                with torch.set_grad_enabled(enable_proposal_update):
                    field_outputs = self.proposal_networks[net_idx](positions, viewdirs, embedded_appearance, embedded_transient)
            else:
                viewdirs = rays['viewdir'].unsqueeze(1).expand_as(positions)
                embed_idxs = rays['embed_idx'].expand_as(positions[..., 0])

                positions = positions.contiguous().view(-1, 3)
                viewdirs = viewdirs.contiguous().view(-1, 3)
                embed_idxs = embed_idxs.contiguous().view(-1)
                embedded_appearance = None if not self.use_appearance_embedding \
                    else self.get_embedding(self.embedding_appearance, embed_idxs).contiguous()
                embedded_transient = None if self.config.transient_type != 'nerfw' \
                    else self.get_embedding(self.embedding_transient, embed_idxs).contiguous()
                field_outputs = self.field(positions, viewdirs, embedded_appearance, embedded_transient)
            
            for key in field_outputs.keys(): 
                field_outputs[key] = field_outputs[key].view(*data_shape, -1)
            
            weights, _, _ = density_to_weight(
                field_outputs['density'][..., 0], euclidean_bins, 
                rays['direction'], self.config.opaque_background
            )
            weights_list.append(weights)
            spacing_bins_list.append(spacing_bins)
            euclidean_bins_list.append(euclidean_bins)

            suffix = f'_prop_{i_level}' if is_prop else ''
            accumulated_weight = torch.sum(weights, dim=-1)

            if 'rgb' in field_outputs.keys():
                outputs[f'rgb{suffix}'] = render_features(weights, field_outputs['rgb'], rays['bg_rgb'], False)
            outputs[f'depth{suffix}'] = render_depth(weights, euclidean_bins)
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
                outputs[f'rgb_static{output_type}'] = rgb_static_partial
                outputs[f'rgb_transient{output_type}'] = rgb_transient_partial
                outputs[f'rgb_combined{output_type}'] = rgb_combined
                outputs[f'uncertainty{output_type}'] = uncertainty
                outputs[f'depth_combined{output_type}'] = depth_combined
                
                if self.training:
                    outputs[f'density_transient{output_type}'] = field_outputs['density_transient'][..., 0]

        if self.implicit_mask is not None:
            coords = rays['coord'] # (n_rays, 2)
            embed_idxs = rays['embed_idx'][..., 0] # (n_rays, )
            embedded_transient = self.get_embedding(self.embedding_transient, embed_idxs).contiguous()
            mask = self.implicit_mask(coords, embedded_transient)
            outputs['implicit_mask'] = mask

        if self.training:
            outputs['weights_list'] = weights_list
            outputs['spacing_bins_list'] = spacing_bins_list
        
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
        pred_rgb = outputs['rgb'].reshape(*data_shape, -1)
        resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
        rgb_loss = self.config.rgb_loss_mult * self.rgb_loss(resid_sq).mean()
        loss_list.append(rgb_loss)
        info_dict[f'rgb_loss'] = rgb_loss.detach()
        info_dict[f'mse'] = resid_sq.detach().mean()
        
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
        pred_rgb = outputs['rgb'].reshape(*data_shape, -1)
        static_mask = (batch['static_mask'].reshape(*data_shape, 1) >= 0.5).to(gt_rgb)
        lossmult = static_mask + (1 - static_mask) * self.config.withmask_transient_weight
        
        resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
        lossmult = lossmult.broadcast_to(resid_sq.shape)
        denom = lossmult.sum().clamp_min(torch.finfo(lossmult.dtype).eps)
        rgb_loss = self.config.rgb_loss_mult * ((lossmult * self.rgb_loss(resid_sq)).sum() / denom)
        loss_list.append(rgb_loss)
        info_dict[f'rgb_loss'] = rgb_loss.detach()
        info_dict[f'mse'] = (lossmult * resid_sq.detach()).sum() / denom
        
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
        pred_rgb = outputs['rgb'].reshape(*data_shape, -1)
        resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
        rgb_loss = self.rgb_loss(resid_sq)
        errors = resid_sq.detach()
            
        robust_mask, robust_info_dict, extra_infos = get_robustnerf_mask(
            errors=errors, 
            suffix=None,
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
        rgb_loss = self.config.rgb_loss_mult * ((lossmult * rgb_loss).sum() / denom)
        loss_list.append(rgb_loss)
        info_dict['rgb_loss'] = rgb_loss.detach()
        info_dict['mse'] = ((lossmult * resid_sq).sum() / denom).detach()
        
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
        pred_rgb = outputs['rgb_combined'].reshape(*data_shape, -1)
        resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
        rgb_loss = self.rgb_loss(resid_sq)

        beta = outputs['uncertainty'].reshape(*data_shape, -1)
        rgb_loss = rgb_loss / (2 * beta**2)
        
        beta_loss = self.config.nerfw_beta_loss_mult * torch.log(beta).mean() + self.config.nerfw_beta_loss_bias # add bias to avoid loss < 0
        loss_list.append(beta_loss)
        info_dict[f'beta_loss'] = beta_loss.detach()
            
        density = outputs[f'density_transient']
        density_loss = self.config.nerfw_density_loss_mult * density.mean()
        loss_list.append(density_loss)
        info_dict[f'density_loss'] = density_loss.detach()

        rgb_loss = self.config.rgb_loss_mult * rgb_loss.mean()
        loss_list.append(rgb_loss)
        info_dict['rgb_loss'] = rgb_loss.detach()
        info_dict['mse'] = resid_sq.detach().mean()
        
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
        mask_size_loss_mult = max(
            self.config.hanerf_mask_size_loss_mult_min,
            self.config.hanerf_mask_size_loss_mult_max * np.exp(-curr_step * self.config.hanerf_mask_size_loss_mult_k)
        )

        gt_rgb = batch['rgb'].reshape(*data_shape, -1)
        pred_rgb = outputs['rgb'].reshape(*data_shape, -1)
        resid_sq = (pred_rgb - gt_rgb)**2 # (n_patch, patch_size, patch_size, -1)
        rgb_loss = self.rgb_loss(resid_sq)

        implicit_mask = outputs['implicit_mask'].reshape(*data_shape, -1)
        rgb_loss = (1 - implicit_mask) * rgb_loss
        info_dict['implicit_mask'] = implicit_mask.mean().detach()
        mask_size_loss = mask_size_loss_mult * (implicit_mask**2).mean()
        loss_list.append(mask_size_loss)
        info_dict['mask_size_loss'] = mask_size_loss.detach()

        rgb_loss = self.config.rgb_loss_mult * rgb_loss.mean()
        loss_list.append(rgb_loss)
        info_dict['rgb_loss'] = rgb_loss.detach()
        info_dict['mse'] = resid_sq.detach().mean()
        
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
        elif self.config.transient_type == 'withmask':
            loss, info_dict, extra_infos = self.compute_withmask_loss(
                outputs, batch, data_shape, extra_infos
            )
        else:
            raise NotImplementedError()

        if self.config.interlevel_loss_mult > 0:
            interlevel_loss_ = self.config.interlevel_loss_mult * \
                interlevel_loss(outputs['weights_list'], outputs['spacing_bins_list'])
            loss = loss + interlevel_loss_
            info_dict['interlevel_loss'] = interlevel_loss_.detach()
        if self.config.distortion_loss_mult > 0:
            distortion_loss_ = self.config.distortion_loss_mult * \
                distortion_loss(outputs['weights_list'], outputs['spacing_bins_list'])
            loss = loss + distortion_loss_
            info_dict['distortion_loss'] = distortion_loss_.detach()
        
        return loss, info_dict, extra_infos


class NerfactoField(nn.Module):
    def __init__(
        self, 
        bound: float,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        num_layers_transient: int = 3,
        hidden_dim_transient: int = 128,
        density_activation: str = 'trunc_exp',
        density_bias: float = -1.,
        rgb_bias: float = 0.,
        enable_tcnn_mlp: bool = True,
        enable_amp: bool = True,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        spatial_distortion: Optional[Callable] = None,
    ) -> None:
        
        super().__init__()  

        self.bound = bound
        self.geo_feat_dim = geo_feat_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_transient = num_layers_transient
        self.hidden_dim_transient = hidden_dim_transient
        self.enable_tcnn_mlp = enable_tcnn_mlp
        self.enable_amp = enable_amp
        self.density_bias = density_bias
        self.rgb_bias = rgb_bias

        self.register_buffer("base_res", torch.tensor(base_res))
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim

        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
            dtype=None if self.enable_amp else torch.float32
        )

        if density_activation == 'trunc_exp':
            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = lambda raw_density: F.softplus(raw_density + self.density_bias)
        else: 
            raise NotImplementedError()
        self.rgb_activation = lambda raw_rgb: F.sigmoid(raw_rgb + self.rgb_bias)
        
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        if self.enable_tcnn_mlp:
            assert self.enable_amp
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1 + self.geo_feat_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
                network_config={
                    "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )

            self.mlp_head = tcnn.Network(
                n_input_dims=self.direction_encoder.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP" if hidden_dim_color <= 128 else "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )

            if self.transient_embedding_dim > 0:
                self.mlp_transient = tcnn.Network(
                    n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                    n_output_dims=5,
                    network_config={
                        "otype": "FullyFusedMLP" if hidden_dim_color <= 128 else "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim_transient,
                        "n_hidden_layers": num_layers_transient - 1,
                    },
                )
        else:
            grid_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
                dtype=None if self.enable_amp else torch.float32
            )
            
            mlp_base = [grid_encoder]
            in_dim = grid_encoder.n_output_dims
            last_dim = in_dim
            for i in range(self.num_layers - 1):
                lin = nn.Linear(last_dim, self.hidden_dim)
                torch.nn.init.kaiming_uniform_(lin.weight)
                mlp_base.append(lin)
                mlp_base.append(nn.ReLU())
                last_dim = self.hidden_dim
            lin = nn.Linear(last_dim, 1 + self.geo_feat_dim)
            torch.nn.init.kaiming_uniform_(lin.weight)
            mlp_base.append(lin)
            self.mlp_base = nn.Sequential(*mlp_base)

            mlp_head = []
            in_dim = self.direction_encoder.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
            last_dim = in_dim
            for i in range(self.num_layers_color - 1):
                lin = nn.Linear(last_dim, self.hidden_dim_color)
                torch.nn.init.kaiming_uniform_(lin.weight)
                mlp_head.append(lin)
                mlp_head.append(nn.ReLU())
                last_dim = self.hidden_dim_color
            lin = nn.Linear(last_dim, 3)
            torch.nn.init.kaiming_uniform_(lin.weight)
            mlp_head.append(lin)
            self.mlp_head = nn.Sequential(*mlp_head)

            if self.transient_embedding_dim > 0:
                mlp_transient = []
                in_dim = self.geo_feat_dim + self.transient_embedding_dim
                last_dim = in_dim
                for i in range(self.num_layers_transient - 1):
                    lin = nn.Linear(last_dim, self.hidden_dim_transient)
                    torch.nn.init.kaiming_uniform_(lin.weight)
                    mlp_transient.append(lin)
                    mlp_transient.append(nn.ReLU())
                    last_dim = self.hidden_dim_transient
                lin = nn.Linear(last_dim, 5)
                torch.nn.init.kaiming_uniform_(lin.weight)
                mlp_transient.append(lin)
                self.mlp_transient = nn.Sequential(*mlp_transient)


    def density(self, positions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        positions: (n, 3)
        '''
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = (positions + self.bound) / (2 * self.bound)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions >= 0.0) & (positions <= 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        
        x = self.mlp_base(positions)
        
        raw_density, base_mlp_out = torch.split(x, [1, self.geo_feat_dim], dim=-1)
        density = self.density_activation(raw_density.to(positions))
        
        density = density * selector[..., None]
        return density, base_mlp_out, selector


    def forward(
        self, 
        positions: Tensor, 
        viewdirs: Tensor, 
        embedded_appearance: Optional[Tensor],
        embedded_transient: Optional[Tensor]
    ) -> Dict[str, Tensor]:
        '''
        positions: (n, 3)
        viewdirs: (n, 3)
        embedded_appearance: None or (n, appearance_embedding_dim)
        '''
        density, geo_feat, selector = self.density(positions)

        viewdirs = (viewdirs + 1.0) / 2.0
        d = self.direction_encoder(viewdirs)

        h = torch.cat([d, geo_feat], dim=-1)
        if self.appearance_embedding_dim > 0:
            h = torch.cat([h, embedded_appearance], dim=-1)
        raw_rgb = self.mlp_head(h).to(positions)
        rgb = self.rgb_activation(raw_rgb)

        outputs = {
            'rgb': rgb,
            'density': density
        }

        if self.transient_embedding_dim > 0:
            x = torch.cat([geo_feat, embedded_transient], dim=-1)
            out = self.mlp_transient(x).to(positions)
            outputs['density_transient'] = self.density_activation(out[..., :1]) * selector[..., None]
            outputs['rgb_transient'] = self.rgb_activation(out[..., 1:4])
            outputs['uncertainty'] = F.softplus(out[..., 4:])

        return outputs
    

class HashMLPDensityField(nn.Module):
    def __init__(
        self,
        bound: float,
        num_levels: int = 8,
        base_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        density_activation: str = 'trunc_exp',
        density_bias: float = -1.,
        enable_tcnn_mlp: bool = True,
        enable_amp: bool = True,
        spatial_distortion: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        self.bound = bound
        self.spatial_distortion = spatial_distortion
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.enable_tcnn_mlp = enable_tcnn_mlp
        self.enable_amp = enable_amp
        self.density_bias = density_bias
        
        self.register_buffer("base_res", torch.tensor(base_res))
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        if density_activation == 'trunc_exp':
            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = lambda raw_density: F.softplus(raw_density + self.density_bias)
        else: 
            raise NotImplementedError()
        
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        if self.enable_tcnn_mlp:
            assert self.enable_amp
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
                network_config={
                    "otype": "FullyFusedMLP"  if hidden_dim <= 128 else "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )
        else:
            grid_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
                dtype=None if self.enable_amp else torch.float32
            )

            mlp_base = [grid_encoder]
            in_dim = grid_encoder.n_output_dims
            last_dim = in_dim
            for i in range(self.num_layers - 1):
                lin = nn.Linear(last_dim, self.hidden_dim)
                torch.nn.init.kaiming_uniform_(lin.weight)
                mlp_base.append(lin)
                mlp_base.append(nn.ReLU())
                last_dim = self.hidden_dim
            lin = nn.Linear(last_dim, 1)
            torch.nn.init.kaiming_uniform_(lin.weight)
            mlp_base.append(lin)
            self.mlp_base = nn.Sequential(*mlp_base)


    def density(self, positions: Tensor) -> Tensor:
        '''
        positions: (n, 3)
        '''
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = (positions + self.bound) / (2 * self.bound)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions >= 0.0) & (positions <= 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        density_before_activation = self.mlp_base(positions).to(positions)
        density = self.density_activation(density_before_activation)
        
        density = density * selector[..., None]
        return density
    

    def forward(
        self, 
        positions: Tensor, 
        viewdirs: Tensor, 
        embedded_appearance: Optional[Tensor],
        embedded_transient: Optional[Tensor]
    ) -> Dict[str, Tensor]:
        '''
        positions: (n, 3)
        viewdirs: (n, 3)
        embedded_appearance: None or (n, appearance_embedding_dim)
        '''
        density = self.density(positions)

        outputs = {
            'density': density
        }

        return outputs


class ImplicitMask(nn.Module):
    def __init__(
        self,
        num_levels: int = 8,
        base_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 17,
        features_per_level: int = 2,
        num_layers: int = 3,
        hidden_dim: int = 128,
        transient_embedding_dim: int = 128,
        enable_tcnn_mlp: bool = True,
        enable_amp: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.transient_embedding_dim = transient_embedding_dim
        self.enable_tcnn_mlp = enable_tcnn_mlp
        self.enable_amp = enable_amp
        
        self.register_buffer("base_res", torch.tensor(base_res))
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            dtype=None if self.enable_amp else torch.float32
        )
        in_dim = self.grid_encoder.n_output_dims + self.transient_embedding_dim
        if self.enable_tcnn_mlp:
            assert self.enable_amp
            self.mlp_base = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP"  if hidden_dim <= 128 else "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )
        else:
            mlp_base = []
            last_dim = in_dim
            for i in range(self.num_layers - 1):
                lin = nn.Linear(last_dim, self.hidden_dim)
                torch.nn.init.kaiming_uniform_(lin.weight)
                mlp_base.append(lin)
                mlp_base.append(nn.ReLU())
                last_dim = self.hidden_dim
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
        x = self.grid_encoder(coordinates)
        out = self.mlp_base(
            torch.cat([x, embedded_transient], dim=-1)
        ).to(coordinates)

        return out