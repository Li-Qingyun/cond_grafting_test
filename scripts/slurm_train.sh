PARTITION_NAME=$1

PYTHONPATH='.'$PYTHONPATH  mim train mmdet configs/grafting_conditional_detr_r50_dc5_8x2_50e_coco.py \
  --launcher slurm --gpus 8 --gpus-per-node 8 --partition ${PARTITION_NAME}
