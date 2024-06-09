import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import yaml

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.amg import calculate_stability_score
from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary


@dataclass
class SegmentConfig:
    sam_points_per_side: int = 64
    """The number of points to be sampled along one side of the image. 
    The total number of points is points_per_side**2."""
    sam_pred_iou_thresh: float = 0.8
    """Sets the number of points run simultaneously by the model. 
    Higher numbers may be faster but use more GPU memory."""
    sam_stability_score_thresh: float = 0.9
    """A filtering threshold in [0,1], using the stability of the mask 
    under changes to the cutoff used to binarize the model's mask predictions."""
    sam_crop_n_layers: int = 1
    """If >0, mask prediction will be run again on crops of the image. 
    Sets the number of layers to run, where each layer has 2**i_layer number of image crops."""
    sam_crop_n_points_downscale_factor: int = 2
    """The number of points-per-side sampled in layer n 
    is scaled down by crop_n_points_downscale_factor**n."""
    sam_min_mask_region_area: int = 50
    """If >0, postprocessing will be applied to remove disconnected regions 
    and holes in masks with area smaller than min_mask_region_area. Requires opencv."""

    # parameters that control SfM-based segmentation
    sfm_point_count_threshold: int = 5
    """Threshold $\mathcal{T}_{SfM}$ that filter feature points based on the number of matching points $n_i^j$."""
    sfm_delete_outlier_points: bool = False
    """Whether to delete outlier points (far away from others) after filtering."""
    sfm_cluster_num: int = -1
    """The cluster number of K-Means applying on the static feature points. 
    Set -1 to disable K-Means."""
    sfm_points_per_mask: int = 1
    """The number of points required for each mask."""
    sfm_seg_batch_size: int = 128
    """Batch size of masks."""
    sfm_use_highest_iou: bool = True
    """Whether to use the mask w/ highest iou as the final result."""
    sfm_pred_iou_thresh: float = 0.8
    sfm_stability_score_offset: float = 1.0
    sfm_stability_score_thresh: float = 0.92

    residual_quantile_upper: float = 0.95

    smooth_kernel_size: int = 7
    erode_kernel_size: int = 5


def load_image(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    return image


def save_image(image: np.ndarray, filename):
    image = (np.clip(image, 0, 1)*255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image)


def delete_same_points(point_coords: torch.Tensor, distance_threshold: float = 0.1) -> torch.Tensor:
    """
    Delete points that are too close or the same.

    args:
        point_coords: (n, 2)
    """
    selected = torch.zeros(point_coords.shape[:1], dtype=bool, device=point_coords.device)
    selected[0] = True
    for i in range(1, point_coords.shape[0]):
        valid_coords = point_coords[selected]
        min_distance = torch.linalg.norm(point_coords[i] - valid_coords, dim=-1).min()
        selected[i] = (min_distance >= distance_threshold)
    return point_coords[selected]


def delete_outlier_points(point_coords: torch.Tensor, std_coefficient: float = 3.0) -> torch.Tensor:
    """
    args:
        point_coords: (n, 2)
    """
    point_num = point_coords.shape[0]
    distances = torch.linalg.norm(point_coords.unsqueeze(1) - point_coords.unsqueeze(0), dim=-1) # (n, n)
    distances = distances[~torch.eye(point_num, dtype=bool)].reshape(point_num, point_num-1) # (n, n-1)
    threshold = distances.mean() + std_coefficient * distances.std()
    selected = (distances.min(dim=-1).values < threshold)
    return point_coords[selected]


def fill_gap(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    args:
        mask with gaps: (height, width, 1)
    return:
        mask without gaps: (height, width, 1)
    """
    height, width = mask.shape[:2]
    if kernel_size%2==0: kernel_size += 1
    pad_size = kernel_size // 2

    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask) / (kernel_size**2)
    mask_padded = F.pad(mask.reshape(1, 1, height, width), (pad_size, )*4, mode='reflect')
    mask_smooth = F.conv2d(mask_padded, kernel, padding='valid').reshape(mask.shape)

    return ((mask_smooth + mask) >= 0.5).to(mask)


def intersect_masks(coarse_mask: torch.Tensor, 
                    fine_masks: torch.Tensor, 
                    intersect_threshold: float) -> torch.Tensor:
    """
    Given a coarse mask and a set of fine masks, return the fine mask based on the intersection.
    
    args:
        coarse_mask: (h, w, 1)
        masks: (n, h, w, 1)
    return:
        fine_mask: (h, w, 1)
    """
    selected = (
        torch.sum(coarse_mask[None, ...] * fine_masks, dim=(1,2,3)) / torch.sum(fine_masks, dim=(1,2,3))
    ) >= intersect_threshold # (n,)
    fine_mask = torch.zeros_like(coarse_mask) if selected.sum()==0 \
                else (fine_masks[selected].sum(dim=0) >= 0.5).to(coarse_mask.dtype)
    return fine_mask


def visualize_mask(image: torch.Tensor, 
                   mask: torch.Tensor, 
                   random_color: bool = False) -> torch.Tensor:
    """
    args:
        image: (h, w, 3)
        mask: (h, w, 1)
    """
    color = torch.rand(3) if random_color else torch.tensor([30/255, 144/255, 255/255])
    color = color.reshape(1, 1, 3).to(image)
    image_masked = torch.clip(
        mask * (0.35 * color + 0.65 * image) + (1 - mask) * image, 0., 1.
    )
    return image_masked


def visualize_points(image: torch.Tensor, 
                     point_coords: torch.Tensor, 
                     point_labels: torch.Tensor, 
                     radius: int = 10) -> torch.Tensor:
    """
    args:
        image: (h, w, 3)
        point_coords: (n, 2)
        point_labels: (n, )
    """
    image_np = image.cpu().numpy()
    point_coords_np = point_coords.cpu().numpy().reshape(-1, 2).astype(int)
    point_labels_np = point_labels.cpu().numpy().reshape(-1)
    for i in range(point_coords_np.shape[0]):
        # positive -> green, negative -> red
        color = (0, 1, 0) if point_labels_np[i]==1 else (1, 0, 0)
        image_np = cv2.circle(image_np, tuple(point_coords_np[i]), radius, color, -1)
    return torch.tensor(image_np).to(image)


def main(image_path: str,
         colmap_path: str,
         sam_model_type: str,
         sam_checkpoint_path: str,
         output_path: str,
         config: SegmentConfig):
    # load segment anything model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.sam_points_per_side,
        pred_iou_thresh=config.sam_pred_iou_thresh,
        stability_score_thresh=config.sam_stability_score_thresh,
        crop_n_layers=config.sam_crop_n_layers,
        crop_n_points_downscale_factor=config.sam_crop_n_points_downscale_factor,
        min_mask_region_area=config.sam_min_mask_region_area, # Requires open-cv to run post-processing
    )
    predictor = SamPredictor(sam)

    # load data
    image_dir = Path(image_path)
    gt_paths = sorted(image_dir.glob("*_gt.png"))

    colmap_dir = Path(colmap_path)
    imgdata = read_images_binary(str(colmap_dir / 'images.bin'))
    ptsdata = read_points3d_binary(str(colmap_dir / 'points3D.bin'))
    camdata = read_cameras_binary(str(colmap_dir / 'cameras.bin'))
    imageName2Id = {Path(imgdata[key].name).stem: key for key in imgdata.keys()}

    output_dirs = {'base': Path(output_path)}
    output_dirs['visualization'] = output_dirs['base'] / 'visualizations'
    output_dirs['static_mask'] = output_dirs['base'] / 'static_masks'
    for key in output_dirs.keys(): 
        output_dirs[key].mkdir(parents=True, exist_ok=True)

    image_num = len(gt_paths)
    pbar = tqdm(total=image_num)
    
    for gt_path in gt_paths:
        image_name = gt_path.stem[:-3]
        pred_path = image_dir / f"{image_name}_color.png"
        
        pred = torch.tensor(load_image(pred_path)) # (h, w, 3)
        gt = torch.tensor(load_image(gt_path))     # (h, w, 3)
        height, width = pred.shape[:2]
        # visualize
        visualization = torch.zeros((height * 2, width * 8, 3)).to(pred)
        visualization[height*0:height*1, width*0:width*1] = gt
        visualization[height*0:height*1, width*1:width*2] = pred

        # use SAM to get fine masks
        sam_image = (gt.cpu().numpy() * 255).astype(np.uint8)
        sam_results = sorted(mask_generator.generate(sam_image), key=(lambda x: x['area']), reverse=True)
        # some pixels are not included in sam_results, so fix them
        sam_index_mask = np.ones((height, width, 1), dtype=int) * -1
        for index, sam_result in enumerate(sam_results):
            sam_index_mask[sam_result['segmentation']] = index
        index_end = len(sam_results)
        index = index_end
        for ij in range(height*width):
            i, j = ij // width, ij % width
            if sam_index_mask[i, j] != -1: continue
            neighbors = []
            if i > 0 and sam_index_mask[i-1, j] >= index_end: neighbors.append(sam_index_mask[i-1, j])
            if j > 0 and sam_index_mask[i, j-1] >= index_end: neighbors.append(sam_index_mask[i, j-1])
            if i > 0 and j > 0 and sam_index_mask[i-1, j-1] >= index_end: neighbors.append(sam_index_mask[i-1, j-1])
            neighbors = np.unique(neighbors)
            if len(neighbors) == 0:
                sam_index_mask[i, j] = index
                index += 1
            else:
                sam_index_mask[i, j] = neighbors[0]
                for neighbor in neighbors[1:]:
                    sam_index_mask[sam_index_mask==neighbor] = neighbors[0]
        alive_indexs = sorted(np.unique(sam_index_mask))
        sam_masks = []
        sam_mask_vis = torch.zeros((height, width, 3)).to(pred)
        for mask_index in alive_indexs:
            sam_masks.append(torch.tensor((sam_index_mask==mask_index).astype(np.float32)).to(pred))
            sam_mask_vis = sam_mask_vis + sam_masks[-1] * torch.rand(1, 1, 3)
        sam_masks = torch.stack(sam_masks, dim=0) # (n, h, w, 1)
        visualization[height*0:height*1, width*2:width*3] = 0.65 * gt + 0.35 * sam_mask_vis

        # compute color residual
        residual = (pred - gt).abs().mean(dim=-1, keepdim=True)         # (h, w, 1)
        residual_mask_base = (residual <= residual.mean()).to(residual) # (h, w, 1)
        residual_mask_upper = (
            residual <= torch.quantile(residual, config.residual_quantile_upper)
        ).to(residual) # (h, w, 1)
        # residual.mean() may be larger than quantile, so take the union
        residual_mask_upper = ((residual_mask_base + residual_mask_upper) >= 0.5).to(residual)
        # visualize
        visualization[height*1:height*2, width*0:width*1] = (
            (residual - residual.min()) / (residual.max() - residual.min())
        ).expand_as(pred)
        visualization[height*1:height*2, width*1:width*2] = residual_mask_base.expand_as(pred)
        visualization[height*1:height*2, width*2:width*3] = residual_mask_upper.expand_as(pred)

        # residual base/upper + sam
        residual_mask_base_sam = intersect_masks(residual_mask_base, sam_masks, 0.5)
        residual_mask_base_sam = fill_gap(residual_mask_base_sam, 5)
        visualization[height*0:height*1, width*3:width*4] = visualize_mask(gt, residual_mask_base_sam)
        visualization[height*1:height*2, width*3:width*4] = residual_mask_base_sam.expand_as(pred)
        residual_mask_upper_sam = intersect_masks(residual_mask_upper, sam_masks, 0.5)
        residual_mask_upper_sam = fill_gap(residual_mask_base_sam, 5)
        visualization[height*0:height*1, width*4:width*5] = visualize_mask(gt, residual_mask_upper_sam)
        visualization[height*1:height*2, width*4:width*5] = residual_mask_upper_sam.expand_as(pred)

        # filter feature points
        colmap_id = imageName2Id[image_name]
        feature_points = []
        for i in range(len(imgdata[colmap_id].point3D_ids)):
            xy = imgdata[colmap_id].xys[i]
            point3D_id = imgdata[colmap_id].point3D_ids[i]
            camera_id = imgdata[colmap_id].camera_id
            downsample_factor_h = height / camdata[camera_id].height
            downsample_factor_w = width / camdata[camera_id].width
            if config.sfm_point_count_threshold == 0 \
                or (point3D_id != -1 and len(ptsdata[point3D_id].image_ids) >= config.sfm_point_count_threshold):
                xy = [
                    np.clip(xy[0] * downsample_factor_w, 0, width),
                    np.clip(xy[1] * downsample_factor_h, 0, height)
                ]
                feature_points.append(xy)
        feature_points = torch.tensor(feature_points, dtype=torch.float32)
        
        # get sfm mask
        sfm_mask = torch.zeros(height, width).to(pred)
        if feature_points.shape[0] > 0:
            feature_points = delete_same_points(feature_points)
            if config.sfm_delete_outlier_points:
                feature_points = delete_outlier_points(feature_points)
            # use KMeans to get fewer points
            n_clusters = config.sfm_cluster_num
            if n_clusters > 0 and n_clusters < feature_points.shape[0]:
                algorithm = KMeans(
                    n_clusters=n_clusters, n_init='auto', random_state=0
                ).fit(feature_points.cpu().numpy())
                clusters = torch.tensor(algorithm.cluster_centers_).to(feature_points) # (n_clusters, 2)
                distances = torch.linalg.norm(clusters.unsqueeze(0) - feature_points.unsqueeze(1), dim=-1) # (n_clusters, n_points)
                input_points = torch.unique(torch.argsort(distances, dim=-1)[..., 0])
                del algorithm, clusters, distances
            else:
                input_points = feature_points
            # use multiple neighbor point to get one mask
            if config.sfm_points_per_mask > 1:
                distances = torch.linalg.norm(input_points.unsqueeze(0) - feature_points.unsqueeze(1), dim=-1) # (n_input, n_points)
                selected = torch.argsort(distances, dim=-1)[..., :config.sfm_points_per_mask]
                input_points = feature_points[selected] # (n_input, n_p, 2)
            else:
                input_points = input_points.unsqueeze(1) # (n_input, 1, 2)
            input_labels = torch.ones(input_points.shape[:-1], dtype=torch.int, device=input_points.device)

            predictor.set_image(sam_image)
            input_points_transformed = torch.as_tensor(
                predictor.transform.apply_coords(input_points.cpu().numpy(), predictor.original_size),
                dtype=torch.float, device=predictor.device
            )
            input_labels_transformed = input_labels.to(device=predictor.device)
            for start_idx in range(0, input_points_transformed.shape[0], config.sfm_seg_batch_size):
                end_idx = min(start_idx + config.sfm_seg_batch_size, input_points_transformed.shape[0])
                masks, iou_preds, _ = predictor.predict_torch(
                    input_points_transformed[start_idx:end_idx],
                    input_labels_transformed[start_idx:end_idx],
                    multimask_output = True,
                    return_logits = True
                )
                if config.sfm_use_highest_iou:
                    keep_mask = torch.argmax(iou_preds, dim=-1)
                    masks = masks[torch.arange(masks.shape[0]).to(keep_mask), keep_mask]
                    iou_preds = iou_preds[torch.arange(iou_preds.shape[0]).to(keep_mask), keep_mask]
                else:
                    masks = masks.flatten(0,1)
                    iou_preds = iou_preds.flatten(0,1)
                
                if config.sfm_pred_iou_thresh > 0:
                    keep_mask = iou_preds > config.sfm_pred_iou_thresh
                    masks = masks[keep_mask]
                    iou_preds = iou_preds[keep_mask]
                
                stability_score = calculate_stability_score(
                    masks, predictor.model.mask_threshold, config.sfm_stability_score_offset
                )
                if config.sfm_stability_score_thresh > 0:
                    keep_mask = stability_score >= config.sfm_stability_score_thresh
                    masks = masks[keep_mask]
                    iou_preds = iou_preds[keep_mask]

                masks = masks > predictor.model.mask_threshold
                sfm_mask = sfm_mask + masks.sum(dim=0).to(sfm_mask)
            
            predictor.reset_image()
            sfm_mask = fill_gap((sfm_mask[..., None] >= 0.5).to(sfm_mask), 5)
            sfm_mask_vis = visualize_mask(gt, sfm_mask)
            sfm_mask_vis = visualize_points(sfm_mask_vis, input_points, input_labels, min(height, width)//100)
        else:
            sfm_mask_vis = visualize_mask(gt, sfm_mask)
        visualization[height*0:height*1, width*5:width*6] = sfm_mask_vis
        visualization[height*1:height*2, width*5:width*6] = sfm_mask.expand_as(pred)

        # sfm + residual
        sfm_residual_mask = ((sfm_mask + residual_mask_base) * residual_mask_upper >= 0.5).to(pred)
        visualization[height*0:height*1, width*6:width*7] = visualize_mask(gt, sfm_residual_mask)
        visualization[height*1:height*2, width*6:width*7] = sfm_residual_mask

        # sfm + residual + sam
        if config.smooth_kernel_size > 0:
            f = config.smooth_kernel_size
            kernel = torch.ones(1, 1, f, f).to(sfm_residual_mask) / (f**2)
            sfm_residual_mask_smooth = (F.conv2d(
                sfm_residual_mask.unsqueeze(0).permute(0,3,1,2), kernel, padding='same'
            ).permute(0,2,3,1)[0]).to(sfm_residual_mask)
            sfm_residual_mask = ((sfm_residual_mask + sfm_residual_mask_smooth) >= 0.5).to(sfm_residual_mask)
        sfm_residual_sam_mask = intersect_masks(sfm_residual_mask, sam_masks, 0.5)
        
        if config.erode_kernel_size > 0:
            erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (config.erode_kernel_size, config.erode_kernel_size)
            )
            sfm_residual_sam_mask = cv2.erode(
                sfm_residual_sam_mask.cpu().numpy(), erode_kernel
            ).reshape(sfm_residual_mask.shape)
            sfm_residual_sam_mask = torch.tensor((sfm_residual_sam_mask>=0.5)).to(gt)
        visualization[height*0:height*1, width*7:width*8] = visualize_mask(gt, sfm_residual_sam_mask)
        visualization[height*1:height*2, width*7:width*8] = sfm_residual_sam_mask.expand_as(gt)

        save_image(
            visualization.cpu().numpy(), 
            output_dirs['visualization'] / f"{image_name}.png"
        )
        save_image(
            sfm_residual_sam_mask.expand_as(gt).cpu().numpy(), 
            output_dirs['static_mask'] / f"{image_name}.png"
        )
        pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--colmap', type=str)
    parser.add_argument('--sam_model', type=str, choices=['default', 'vit_h', 'vit_l', 'vit_b'], default='vit_h')
    parser.add_argument('--sam_checkpoint', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config_dict = yaml.load(fp.read(), Loader=yaml.FullLoader)
        config = SegmentConfig(**config_dict)

    main(
        args.images, args.colmap, 
        args.sam_model, args.sam_checkpoint, 
        args.output, config
    )