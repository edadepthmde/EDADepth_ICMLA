# ------------------------------------------------------------------------------
# The script is from ECoDepth (https://github.com/Aradhye2002/EcoDepth/).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nproc_per_node=1 \
    test.py \
    --dataset kitti \
    --data_path ./data/kitti/test/ \
    --flip_test  \
    --kitti_crop garg_crop \
    --do_kb_crop True \
    --kitti_split_to_half True \
    --no_of_classes 50 \
    --save_visualize \
    --save_eval_pngs \
    --exp_name testing_kitti_beit2 \
    --max_depth 80.0 \
    --max_depth_eval 80.0 \
    --min_depth_eval 1e-3 \
    --median_scaling False \
    --batch_size 1 \
    --ckpt_dir ./checkpoints_depth/kitti.ckpt \
