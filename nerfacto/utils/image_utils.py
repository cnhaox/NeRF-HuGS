import numpy as np
import cv2
import torch
from torch import Tensor
from matplotlib import cm

def mse_to_psnr(mse: Tensor) -> Tensor:
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * torch.log(mse)

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
        value: A 1D image.
        weight: A weight map, in [0, 1].
        colormap: A colormap function.
        lo: The lower bound to use when rendering, if None then use a percentile.
        hi: The upper bound to use when rendering, if None then use a percentile.
        percentile: What percentile of the value map to crop to when automatically
            generating `lo` and `hi`. Depends on `weight` as well as `value'.
        curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
            before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
        modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
            `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
        matte_background: If True, matte the image over a checkerboard.

    Returns:
        A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
        value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        if len(value.shape) != 3:
            raise ValueError(f'value must have 3 dims but has {len(value.shape)}')
        if value.shape[-1] != 3:
            raise ValueError(
                f'value must have 3 channels but has {len(value.shape[-1])}')
        colorized = value

    return matte(colorized, weight) if matte_background else colorized

def depth2img(depth, min_depth = None, max_depth = None):
    if min_depth is None: min_depth = depth.min()
    if max_depth is None: max_depth = depth.max()
    t = max_depth - min_depth
    if t <= 0: t = np.finfo(depth.dtype).eps
    depth = (depth-min_depth)/t
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    depth_img =depth_img.astype(np.float32) / 255.
    return depth_img

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

def depth2img_v2(depth: np.ndarray, acc: np.ndarray) -> np.ndarray:
    min_depth, max_depth = depth.min(), depth.max()
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth_vis = visualize_cmap(
        depth, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn
    )
    return depth_vis


def load_image(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    return image

def load_mask(filename):
    mask = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    mask = mask[..., :1].astype(np.float32) / 255.
    return mask

def save_image(image: np.ndarray, filename):
    image = (np.clip(image, 0, 1)*255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image)


def color_correct(img, ref, num_iters=5, eps=0.5/255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(np.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = np.concatenate(a_mat, axis=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = np.where(mask[:, None], a_mat, 0)
            mb = np.where(mask, b, 0)
            # Solve the linear system. We're using the np.lstsq instead of jnp because
            # it's significantly more stable in this case, for some reason.
            w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert np.all(np.isfinite(w))
            warp.append(w)
        warp = np.stack(warp, axis=-1)
        # Apply the warp to update img_mat.
        img_mat = np.clip(
            np.matmul(a_mat, warp), 0, 1)
    corrected_img = np.reshape(img_mat, img.shape)
    return corrected_img