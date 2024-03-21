#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
TORCH_ENV=nerfhugs_torch
DATA_DIR=/mnt/data-ssd/kubric_dataset
SCENE=kubric_car
COLMAP_DIR=/mnt/data-ssd/d2nerf_data/"${SCENE}"_colmap/sparse/0 # /mnt/data-ssd/kubric_colmap/"$SCENE"/sparse/0
SAM_TYPE=vit_h
SAM_PATH=/mnt/data-ssd/code/segment-anything/sam_vit_h_4b8939.pth
NERF_CONFIG_NAME=kubric_nerfacto_gen_mask
HUGS_CONFIG_NAME=kubric


source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${TORCH_ENV}"
pushd "./nerfacto"
SCENE_DIR="${DATA_DIR}"/"${SCENE}"
CHECKPOINT_DIR=$(pwd)/nerf_results/"$NERF_CONFIG_NAME"/"$SCENE"
python train.py \
  --config "configs/${NERF_CONFIG_NAME}.yml" \
  --data_dir "${SCENE_DIR}" \
  --save_dir "${CHECKPOINT_DIR}"
python eval.py \
  --config "configs/${NERF_CONFIG_NAME}.yml" \
  --data_dir "${SCENE_DIR}" \
  --save_dir "${CHECKPOINT_DIR}" \
  --eval_data "train" \
  --original_name \
  --only_pred_gt
popd
pushd "./HuGS"
python generate_static_mask.py \
  --images "${CHECKPOINT_DIR}/train_preds" \
  --colmap "${COLMAP_DIR}" \
  --sam_model "${SAM_TYPE}" \
  --sam_checkpoint "${SAM_PATH}" \
  --output "${CHECKPOINT_DIR}/masks" \
  --config "configs/${HUGS_CONFIG_NAME}.yml"
popd