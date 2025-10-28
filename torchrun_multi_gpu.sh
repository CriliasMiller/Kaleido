#!/bin/bash

if [ -z "$MLP_WORKER_NUM" ]; then
    MLP_WORKER_NUM=1   # 节点数
    MLP_GPU=8          # 每节点 GPU 数
fi

if [ -z "$MLP_WORKER_0_HOST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=35446
    NODE_RANK=0
else
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=$MLP_WORKER_0_PORT
    NODE_RANK=${MLP_ROLE_INDEX:-0} # 当前节点rank
fi

export MASTER_ADDR MASTER_PORT

# torchrun 启动命令
torchrun \
    --nnodes $MLP_WORKER_NUM \
    --nproc_per_node $MLP_GPU \
    --node_rank $NODE_RANK \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    sample_video.py \
    --base configs/video_model/dit_crossattn_14B_wanvae.yaml \
    configs/sampling/sample_wanvae_concat_14b.yaml \
    --seed $RANDOM