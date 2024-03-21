from typing import List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
EPS = 1.0e-7

def outer(t0_starts: Tensor, t0_ends: Tensor, t1_starts: Tensor, t1_ends: Tensor, y1: Tensor) -> Tensor:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(t: Tensor, w: Tensor, t_env: Tensor, w_env: Tensor) -> Tensor:
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def interlevel_loss(weights_list: List[Tensor], spacing_bins_list: List[Tensor]) -> Tensor:
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = spacing_bins_list[-1].detach()
    w = weights_list[-1].detach()
    loss_interlevel = 0.0
    for spacing_bins, weights in zip(spacing_bins_list[:-1], weights_list[:-1]):
        cp = spacing_bins  # (num_rays, num_samples + 1)
        wp = weights  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


# Verified
def lossfun_distortion(t: Tensor, w: Tensor) -> Tensor:
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list: List[Tensor], spacing_bins_list: List[Tensor]) -> Tensor:
    """From mipnerf360"""
    c = spacing_bins_list[-1]
    w = weights_list[-1]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


@torch.no_grad()
def get_robustnerf_mask(
    errors: Tensor, 
    suffix: Optional[str],
    extra_infos: dict,
    inlier_quantile: float,
    smoothed_filter_size: int,
    smoothed_inlier_quantile: float,
    inner_patch_size: int,
    inner_patch_inlier_quantile: float,
) -> Tuple[Tensor, dict, dict]:
    '''
    errors: (n, patch_size, patch_size, 3)
    '''
    epsilon = 1e-3
    error_per_pixel = torch.mean(errors, dim=-1, keepdim=True) # (n,h,w,1)
    patch_size = error_per_pixel.shape[1:3]
    assert inner_patch_size <= min(patch_size), \
        'patch_size must be larger than robustnerf_inner_patch_size.'
    info_dict = {}
    if suffix is None: 
        suffix = ''

    # 1.0 for inlier pixels.
    next_inlier_threshold = torch.quantile(error_per_pixel, inlier_quantile)
    curr_inlier_threshold = 1. if f'inlier_threshold{suffix}' not in extra_infos.keys() \
                            else extra_infos[f'inlier_threshold{suffix}']
    info_dict[f'inlier_threshold{suffix}'] = next_inlier_threshold
    is_inlier_loss = (error_per_pixel < curr_inlier_threshold).to(error_per_pixel)
    extra_infos[f'inlier_threshold{suffix}'] = next_inlier_threshold.item()
    info_dict[f'is_inlier_loss{suffix}'] = is_inlier_loss.mean()

    # Apply filter
    f = smoothed_filter_size
    kernel = torch.ones(1,1,f,f).to(error_per_pixel) / (f * f)
    has_inlier_neighbors = F.conv2d(
        is_inlier_loss.permute(0,3,1,2), kernel, padding='same'
    ).permute(0,2,3,1)

    # Binarize after smoothing.
    has_inlier_neighbors = (
        has_inlier_neighbors > 1. - smoothed_inlier_quantile
    ).to(error_per_pixel)
    info_dict[f'has_inlier_neighbors{suffix}'] = has_inlier_neighbors.mean()
            
    # Construct binary mask for inner pixels. The entire inner patch is either
    # active or inactive.
    inner_patch_mask = torch.zeros_like(error_per_pixel)
    h_start = (patch_size[0] - inner_patch_size) // 2
    w_start = (patch_size[1] - inner_patch_size) // 2
    inner_patch_mask[:, h_start:h_start+inner_patch_size, w_start:w_start+inner_patch_size, :] = 1
    is_inlier_patch = torch.mean(is_inlier_loss, dim=(1,2), keepdim=True) # (n,1,1,1)
    is_inlier_patch = (is_inlier_patch > 1. - inner_patch_inlier_quantile).to(error_per_pixel)
    is_inlier_patch = is_inlier_patch * inner_patch_mask
    info_dict[f'is_inlier_patch{suffix}'] = is_inlier_patch.mean()

    # A pixel is an inlier if it is an inlier according to any of the above
    # criteria.
    mask = (
        is_inlier_patch + has_inlier_neighbors + is_inlier_loss > epsilon
    ).to(error_per_pixel)
    info_dict[f'robust_mask{suffix}'] = mask.mean()
        
    return mask, info_dict, extra_infos