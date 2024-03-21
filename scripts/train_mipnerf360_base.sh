#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
JAX_ENV=nerfhugs_jax
#### distractor
DATA_DIR=/mnt/data-ssd/RobustNerf
CONFIG_NAME=distractor_1024_glo4_base
SCENES=(and-bot crab t_balloon_statue yoda)
#### kubric
# DATA_DIR=/mnt/data-ssd/kubric_dataset
# CONFIG_NAME=kubric_1024_base
# SCENES=(kubric_car kubric_cars kubric_bag kubric_chair kubric_pillow)
#### phototourism
# DATA_DIR=/mnt/data-ssd/phototourism
# CONFIG_NAME=phototourism_1024_base
# SCENES=(brandenburg_gate sacre_coeur taj_mahal trevi_fountain)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${JAX_ENV}"
pushd "./MipNeRF360"
for SCENE in "${SCENES[@]}"
do
  SCENE_DIR="${DATA_DIR}"/"${SCENE}"
  CHECKPOINT_DIR=./nerf_results/"$EXPERIMENT"/"$SCENE"
  python -m train \
    --gin_configs="configs/${CONFIG_NAME}.gin" \
    --gin_bindings="Config.data_dir = '${SCENE_DIR}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
    --logtostderr
done
popd