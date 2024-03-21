from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_fwd, custom_bwd

@torch.cuda.amp.autocast(enabled=False)
def spatial_distortion(positions: Tensor, order: Optional[Union[float, int]] = None) -> Tensor:
    mag = torch.linalg.norm(positions, ord=order, dim=-1)[..., None]
    return torch.where(mag < 1, positions, (2 - (1 / mag)) * (positions / mag))


@torch.cuda.amp.autocast(enabled=False)
def spatial_distortion_norm2(x: Tensor) -> Tensor:
  """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
  eps = torch.finfo(x.dtype).eps
  # Clamping to eps prevents non-finite gradients when x == 0.
  x_mag_sq = torch.sum(x**2, axis=-1, keepdims=True).clamp_min(eps)
  z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
  return z


class SceneContraction(nn.Module):
    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order
    
    def forward(self, positions):
        new_positions = spatial_distortion(positions, self.order)

        return new_positions


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
    

trunc_exp = TruncExp.apply
"""Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
gradients."""


@torch.cuda.amp.autocast(enabled=False)
def pos_enc(x: Tensor, min_deg: int, max_deg: int, append_identity=True) -> Tensor:
    """The positional encoding used by the original NeRF paper."""
    scales = (2**torch.arange(min_deg, max_deg)).to(x)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*x.shape[:-1], -1)
    feat = torch.sin(torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], axis=-1))
    if append_identity:
        feat = torch.cat([x, feat], dim=-1)
    
    return feat