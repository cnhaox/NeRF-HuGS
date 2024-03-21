from typing import Tuple, Optional, List
import torch
from torch import Tensor
from torch.types import Device

@torch.cuda.amp.autocast(enabled=False)
def intersect_aabb(aabb: Tensor, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    '''
    aabb: (2, 3)
    '''
    # avoid divide by zero
    inv_d = 1.0 / torch.where(
        torch.abs(rays_d) >= torch.finfo(rays_d.dtype).eps, 
        rays_d, torch.finfo(rays_d.dtype).eps
    )
    t = (aabb.unsqueeze(0) - rays_o.unsqueeze(1)) * inv_d.unsqueeze(1) # (N, 2, 3)
    near = torch.max(torch.min(t, dim=1).values, dim=-1, keepdim=True).values # (N, 1)
    far = torch.min(torch.max(t, dim=1).values, dim=-1, keepdim=True).values # (N, 1)
    is_intersect = (near <= far)
    return is_intersect, near, far


@torch.cuda.amp.autocast(enabled=False)
def intersect_sphere(center: Tensor, radius: float, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    '''
    center: (3)
    rays_o: (n, 3)
    rays_d: (n, 3)
    '''
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2 * torch.sum(rays_d * (rays_o - center), dim=-1, keepdim=True)
    c = torch.sum((rays_o - center)**2, dim=-1, keepdim=True) - radius**2
    tmp = b**2 - 4*a*c
    is_intersect = (tmp >= 0)
    tmp = torch.sqrt(torch.where(tmp>=0, tmp, 0))
    near = (-b - tmp) / (2*a)
    far = (-b + tmp) / (2*a)
    return is_intersect, near, far
def alpha_to_weight(alphas: Tensor) -> Tensor:
    alphas = alphas.squeeze(-1) # (n_ray, r_sample)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alphas[..., :1]), 1.0 - alphas + 1e-7], -1), -1
    )
    weights = alphas * transmittance[..., :-1]
    return weights


@torch.cuda.amp.autocast(enabled=False)
def uniform_sample(
    num_rays: int, num_samples: int, perturb: bool, single_jitter: bool, device: Device
) -> Tensor:
    """Uniform sampling from [0, 1]"""
    bins = torch.linspace(0.0, 1.0, num_samples+1, device=device)[None, ...]  # [1, num_samples+1]
    bins = bins.repeat(num_rays, 1)
    if perturb:
        if single_jitter:
            rand_ = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
        else:
            rand_ = torch.rand((num_rays, num_samples+1), dtype=bins.dtype, device=bins.device)
        bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
        bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
        bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
        bins = bin_lower + (bin_upper - bin_lower) * rand_
    return bins


@torch.cuda.amp.autocast(enabled=False)
def pdf_sample(
    spacing_bins: Tensor, weights: Tensor, num_samples: int, perturb: bool, single_jitter: bool
) -> Tensor:
    num_bins = num_samples + 1
    # Add small offset to rays with zero weight to prevent NaNs
    eps = 1e-5
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding

    pdf = weights / weights_sum
    cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if perturb:
        # Stratified samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
        if single_jitter:
            rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
        else:
            rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
        u = u + rand
    else:
        # Uniform samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u + 1.0 / (2 * num_bins)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.clamp(inds - 1, 0, spacing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, spacing_bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(spacing_bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(spacing_bins, -1, above)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    new_bins = bins_g0 + t * (bins_g1 - bins_g0)

    return new_bins

@torch.cuda.amp.autocast(enabled=False)
def sample(
    spacing_bins: Tensor, 
    weights: Tensor, 
    anneal: float,
    padding: float,
    num_samples: int, 
    perturb: bool, 
    single_jitter: bool, 
    deterministic_center: bool
) -> Tensor:
    # used in jaxnerf / nerfstudio
    # eps = 1e-5
    # annealed_weights = weights + padding
    # if anneal != 1:
    #     annealed_weights = torch.pow(annealed_weights, anneal)
    # weight_sum = torch.sum(annealed_weights, dim=-1, keepdim=True)
    # new_padding = (eps - weight_sum).clamp_min(0)
    # annealed_weights = annealed_weights + new_padding / annealed_weights.shape[-1]
    # weight_sum += new_padding
    # pdf = annealed_weights / weight_sum

    device = spacing_bins.device
    dtype = spacing_bins.dtype
    eps = torch.finfo(dtype).eps

    # used in mipnerf360
    weights_logit = torch.where(
        spacing_bins[..., 1:] > spacing_bins[..., :-1],
        anneal * torch.log(weights + padding), -torch.inf
    )
    # avoid weights_logit on one ray is all -inf, which leads to NaNs in softmax
    weights_logit[(weights_logit<=-torch.inf).all(dim=-1)] = 1
    pdf = torch.softmax(weights_logit, dim=-1)
    
    cdf = torch.cumsum(pdf[..., :-1], dim=-1).clamp_max(1)
    cdf = torch.cat([torch.zeros_like(pdf[...,:1]), cdf, torch.ones_like(pdf[...,:1])], dim=-1)

    
    if perturb:
        u_max = eps + (1. - eps) / num_samples
        max_jitter = (1. - u_max) / (num_samples -1) - eps
        d = 1 if single_jitter else num_samples
        u = torch.linspace(0, 1. - u_max, num_samples, dtype=dtype, device=device) \
            + torch.rand((*cdf.shape[:-1], d), dtype=dtype, device=device) * max_jitter
    else:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = torch.linspace(pad, 1. - pad - eps, num_samples, dtype=dtype, device=device)
        else:
            u = torch.linspace(0, 1. - eps, num_samples, dtype=dtype, device=device)
        u = torch.broadcast_to(u, (*cdf.shape[:-1], num_samples))

    # # # # # # # # 
    # # Originally used in jaxnerf and mipnerf360
    # # too slow and take up too much memory!
    # mask = u[..., None, :] >= cdf[..., :, None]
    # def find_interval(x):
    #     # Grab the value where `mask` switches from True to False, and vice versa.
    #     # This approach takes advantage of the fact that `x` is sorted.
    #     x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2).values
    #     x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2).values
    #     return x0, x1
    # bins_g0, bins_g1 = find_interval(spacing_bins)
    # cdf_g0, cdf_g1 = find_interval(cdf)
    # t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    # samples = bins_g0 + t * (bins_g1 - bins_g0)
    # # # # # # # #

    cdf = cdf.contiguous()
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.clamp(inds - 1, 0, spacing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, spacing_bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(spacing_bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(spacing_bins, -1, above)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    
    return samples


@torch.cuda.amp.autocast(enabled=False)
def sample_intervals(
    spacing_bins: Tensor, 
    weights: Tensor, 
    anneal: float,
    padding: float,
    num_samples: int, 
    perturb: bool, 
    single_jitter: bool, 
    domain: Tuple[float, float],
) -> Tensor:
    centers = sample(
        spacing_bins, weights, anneal, padding, 
        num_samples, perturb, single_jitter, True
    )
    # The intervals we return will span the midpoints of each adjacent sample.
    mid = (centers[..., 1:] + centers[..., :-1]) / 2

    # Each first/last fencepost is the reflection of the first/last midpoint
    # around the first/last sampled center. We clamp to the limits of the input
    # domain, provided by the caller.
    minval, maxval = domain
    first = (2 * centers[..., :1] - mid[..., :1]).clamp_min(minval)
    last = (2 * centers[..., -1:] - mid[..., -1:]).clamp_max(maxval)

    new_bins = torch.cat([first, mid, last], dim=-1)
    return new_bins
    

def density_to_weight(densities: Tensor, euclidean_bins: Tensor, directions: Tensor, opaque_background=False) -> Tensor:
    '''
    deltas: [..., num_samples]
    densities: [..., num_samples]
    '''
    euclidean_deltas = euclidean_bins[..., 1:] - euclidean_bins[..., :1]
    deltas = euclidean_deltas * torch.linalg.norm(directions[..., None, :], dim=-1)
    density_delta = densities * deltas
    if opaque_background:
        # Equivalent to making the final euclidean-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1], torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)
    
    alphas = 1 - torch.exp(-density_delta)

    transmittance = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1)) # [..., num_samples]

    weights = alphas * transmittance  # [..., num_samples]
    weights = torch.nan_to_num(weights)
    return weights, alphas, transmittance


def dual_density_to_weight(
    densities1: Tensor, densities2: Tensor,
    euclidean_bins: Tensor, directions: Tensor, opaque_background=False
) -> Tensor:
    '''
    deltas: [..., num_samples]
    densities: [..., num_samples]
    '''
    euclidean_deltas = euclidean_bins[..., 1:] - euclidean_bins[..., :1]
    deltas = euclidean_deltas * torch.linalg.norm(directions[..., None, :], dim=-1)
    density_delta_1 = densities1 * deltas
    density_delta_2 = densities2 * deltas
    density_delta = (densities1 + densities2) * deltas
    if opaque_background:
        # Equivalent to making the final euclidean-interval infinitely wide.
        density_delta_1 = torch.cat([
            density_delta_1[..., :-1], 
            torch.full_like(density_delta_1[..., -1:], torch.inf)
        ], dim=-1)
        density_delta_2 = torch.cat([
            density_delta_2[..., :-1], 
            torch.full_like(density_delta_2[..., -1:], torch.inf)
        ], dim=-1)
        density_delta = torch.cat([
            density_delta[..., :-1], 
            torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)
    
    alphas_1 = 1 - torch.exp(-density_delta_1)
    alphas_2 = 1 - torch.exp(-density_delta_2)
    alphas = 1 - torch.exp(-density_delta)
    
    transmittance = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1)) # [..., num_samples]

    weights_1 = torch.nan_to_num(alphas_1 * transmittance)  # [..., num_samples]
    weights_2 = torch.nan_to_num(alphas_2 * transmittance)  # [..., num_samples]
    weights =torch.nan_to_num(alphas * transmittance) # [..., num_samples]
    return weights_1, weights_2, weights


def render_features(
    weights: Tensor, 
    features: Tensor, 
    background_features: Optional[Tensor], 
    require_detach: bool
) -> Tensor:
    weights_ = weights[..., None].detach() if require_detach \
        else weights[..., None]
    feats = torch.sum(weights_ * features, dim=-2)
    if background_features is not None:
        acc = torch.sum(weights_, dim=-2)
        bg_acc = (1. - acc).clamp_min(0)
        feats = feats + background_features * bg_acc
    
    return feats


def render_combined_features(
    weights_static: Tensor,
    weights_transient: Tensor,
    weights_combined: Optional[Tensor],
    features_static: Tensor,
    features_transient: Tensor,
    background_features: Optional[Tensor],
    require_detach: bool
) -> Tensor:
    weights_static_ = weights_static[..., None].detach() if require_detach \
        else weights_static[..., None]
    feats_static = torch.sum(weights_static_ * features_static, dim=-2)

    weights_transient_ = weights_transient[..., None].detach() if require_detach \
        else weights_transient[..., None]
    feats_transient = torch.sum(weights_transient_ * features_transient, dim=-2)
    
    feats = feats_static + feats_transient
    if background_features is not None and weights_combined is not None:
        weights_ = weights_combined[..., None].detach() if require_detach \
            else weights_combined[..., None]
        acc = torch.sum(weights_, dim=-2)
        bg_acc = (1. - acc).clamp_min(0)
        feats = feats + background_features * bg_acc
        
    return feats, feats_static, feats_transient


def render_depth(weights: Tensor, euclidean_bins: Tensor) -> Tensor:
    steps = (euclidean_bins[..., 1:] + euclidean_bins[..., :-1]) / 2
    acc = torch.sum(weights, dim=-1)
    acc = torch.where(acc > 0, acc, torch.finfo(acc.dtype).eps)
    depth = torch.sum(weights * steps, dim=-1) / acc
    depth = torch.clip(depth, 0.0, steps.max())
    return depth