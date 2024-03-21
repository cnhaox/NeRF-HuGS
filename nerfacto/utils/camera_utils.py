import enum
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Mapping


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0, 1.]], 
        dtype=np.float32
    )

def average_poses(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 4, 4)

    Outputs:
        poses_centered: (N_images, 4, 4) the centered poses
        transform: (4, 4)
    """
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg = np.concatenate([
        pose_avg, np.array([0.,0.,0.,1.]).reshape(1,4)
    ], axis=0)
    transform = np.linalg.inv(pose_avg)
    poses_centered = transform @ poses # (N_images, 4, 4)
    return poses_centered, transform


def transform_poses_pca(poses: np.ndarray, enable_rescale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 4, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    poses_recentered = transform @ poses
    
    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1, 1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if enable_rescale:
        # Just make sure it's it in the [-1, 1]^3 cube
        scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def compute_transform_from_endpoints(endpoints: np.ndarray):
    '''
    endpoints: (n, 3)
    '''
    transform = np.eye(4)
    center = np.mean(endpoints, axis=0)
    transform[:3, 3] = -center
    distances = np.linalg.norm(endpoints - center.reshape(1, 3), axis=1)
    vec = endpoints[np.argmax(distances)] - center
    scale_factor = 1. / np.linalg.norm(vec)
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    vec = vec * scale_factor

    # Rotate around the X-axis
    x, y, z = vec[0], vec[1], vec[2]
    sin_ = y / np.sqrt(y**2+z**2)
    cos_ = z / np.sqrt(y**2+z**2)
    matrix_x = np.array([
        [1.,   0.,    0.],
        [0., cos_, -sin_],
        [0., sin_,  cos_]
    ])
    if y < 0:
        matrix_x = matrix_x.T
    vec = matrix_x @ vec

    # Rotate around the Y-axis
    x, y, z = vec[0], vec[1], vec[2]
    sin_ = x / np.sqrt(x**2+z**2)
    cos_ = z / np.sqrt(x**2+z**2)
    matrix_y = np.array([
        [ cos_, 0., sin_],
        [   0., 1.,   0.],
        [-sin_, 0., cos_]
    ])
    if x > 0:
        matrix_y = matrix_y.T
    vec = matrix_y @ vec

    rotate_matrix = np.eye(4)
    rotate_matrix[:3, :3] = matrix_y @ matrix_x
    transform = rotate_matrix @ transform

    return transform


def convert_to_ndc(origins: Tensor, directions: Tensor, pix2cam: Tensor, near: float = 1.) -> Tuple[Tensor, Tensor]:
    # Shift ray origins to near plane, such that oz = -near.
    # This makes the new near bound equal to 0.
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = torch.split(directions, 1, dim=-1) # (..., 1)
    ox, oy, oz = torch.split(origins, 1, dim=-1) # (..., 1)

    xmult = 1. / pix2cam[0, 2]  # Equal to -2. * focal / cx
    ymult = 1. / pix2cam[1, 2]  # Equal to -2. * focal / cy

    # Perspective projection into NDC for the t = 0 near points
    #   origins + 0 * directions
    origins_ndc = torch.cat([xmult * ox / oz, ymult * oy / oz, - torch.ones_like(oz)], dim=-1)
    # Perspective projection into NDC for the t = infinity far points
    #   origins + infinity * directions
    infinity_ndc = torch.cat([xmult * dx / dz, ymult * dy / dz, torch.ones_like(oz)], dim=-1)
    # directions_ndc points from origins_ndc to infinity_ndc
    directions_ndc = infinity_ndc - origins_ndc

    return origins_ndc, directions_ndc


def get_pixtocam(focal: float, width: float, height: float):
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width*.5, height*.5)
    return np.linalg.inv(camtopix)


def get_intrinsic_matrix(focal: float, width: float, height: float) -> np.ndarray:
    """Intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width*.5, height*.5)
    return camtopix


def _compute_residual_and_jacobian(
    x: Tensor,
    y: Tensor,
    xd: Tensor,
    yd: Tensor,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3  + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: Tensor,
    yd: Tensor,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    k4: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10
) -> Tuple[Tensor, Tensor]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = xd.clone()
    y = yd.clone()

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps, x_numerator / denominator,
            torch.zeros_like(denominator)
        )
        step_y = torch.where(
            torch.abs(denominator) > eps, y_numerator / denominator,
            torch.zeros_like(denominator)
        )

        x = x + step_x
        y = y + step_y

    return x, y


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""
    PERSPECTIVE = 'perspective'
    FISHEYE = 'fisheye'


@torch.cuda.amp.autocast(dtype=torch.float32)
def pixels_to_rays(
    pix_x_int: Tensor, 
    pix_y_int: Tensor,
    pix2cam: Tensor,
    cam2world: Tensor,
    distortion_param: Optional[Mapping[str, float]],
    pix2cam_ndc: Optional[Tensor],
    camtype: ProjectionType = ProjectionType.PERSPECTIVE,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    pix_x_int = pix_x_int.to(torch.float32)
    pix_y_int = pix_y_int.to(torch.float32)
    # Must add half pixel offset to shoot rays through pixel centers.
    pixel_dirs = torch.stack(
        [pix_x_int + 0.5, pix_y_int + 0.5, torch.ones_like(pix_x_int)], 
        dim=-1
    ) # (n, 3) or (n, n, 3)

    # Apply inverse intrinsic matrices.
    camera_dirs = torch.matmul(pix2cam, pixel_dirs[..., None])[..., 0] # (n, 3) or (n, n, 3)
    
    if distortion_param is not None:
        # Correct for distortion.
        x, y = _radial_and_tangential_undistort(
            camera_dirs[..., 0], camera_dirs[..., 1], **distortion_param,
        )
        camera_dirs = torch.stack([x, y, torch.ones_like(x)], -1)

    if camtype == ProjectionType.FISHEYE:
        theta = torch.sqrt(torch.sum(torch.square(camera_dirs[..., :2]), dim=-1))
        theta = torch.minimum(torch.pi, theta)

        sin_theta_over_theta = torch.sin(theta) / theta
        camera_dirs = torch.stack([
            camera_dirs[..., 0] * sin_theta_over_theta,
            camera_dirs[..., 1] * sin_theta_over_theta,
            torch.cos(theta),
        ], axis=-1)
    
    # Flip from OpenCV to OpenGL coordinate system.
    camera_dirs = torch.matmul(camera_dirs, torch.diag(torch.tensor([1., -1., -1.])).to(camera_dirs))

    # Apply camera rotation matrices.
    directions = torch.matmul(cam2world[:3, :3], camera_dirs[..., None])[..., 0] # (n, 3) or (n, n, 3)
    
    origins = cam2world[:3, 3].expand_as(directions)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)

    if pix2cam_ndc is not None:
        # Convert ray origins and directions into projective NDC space.
        origins, directions = convert_to_ndc(origins, directions, pix2cam_ndc)

    return origins, directions, viewdirs