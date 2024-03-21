#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
TORCH_ENV=nerfhugs_torch
#### distractor
DATA_DIR=/mnt/data-ssd/RobustNerf
CONFIG_NAME=distractor_nerfacto_base
SCENES=(and-bot crab t_balloon_statue yoda)
#### kubric
# DATA_DIR=/mnt/data-ssd/kubric_dataset
# CONFIG_NAME=kubric_nerfacto_base
# SCENES=(kubric_car kubric_cars kubric_bag kubric_chair kubric_pillow)
#### phototourism
# DATA_DIR=/mnt/data-ssd/phototourism
# CONFIG_NAME=phototourism_nerfacto_base
# SCENES=(brandenburg_gate sacre_coeur taj_mahal trevi_fountain)


source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${TORCH_ENV}"
pushd "./nerfacto"
for SCENE in "${SCENES[@]}"
do
  SCENE_DIR="${DATA_DIR}"/"${SCENE}"
  CHECKPOINT_DIR=./nerf_results/"$CONFIG_NAME"/"$SCENE"
  python train.py \
    --config "configs/${CONFIG_NAME}.yml" \
    --data_dir "${SCENE_DIR}" \
    --save_dir "${CHECKPOINT_DIR}"
done
popd