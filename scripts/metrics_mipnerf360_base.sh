#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
TORCH_ENV=nerfhugs_torch
#### distractor
EXPERIMENT_DIR=MipNeRF360/nerf_results/distractor_1024_glo4_base
SCENES=(t_balloon_statue and-bot crab yoda)
IMAGE_TYPE=whole
#### kubric
# EXPERIMENT_DIR=MipNeRF360/nerf_results/kubric_1024_base
# SCENES=(kubric_car kubric_cars kubric_bag kubric_chair kubric_pillow)
# IMAGE_TYPE=whole
#### phototourism
# EXPERIMENT_DIR=MipNeRF360/nerf_results/phototourism_1024_base
# SCENES=(brandenburg_gate sacre_coeur taj_mahal trevi_fountain)
# IMAGE_TYPE=half_right

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${TORCH_ENV}"
python metrics.py --experiment_dir "${EXPERIMENT_DIR}" \
                  --scene_names "${SCENES[@]}" \
                  --image_type "${IMAGE_TYPE}" \
                  --device cuda
