#!/bin/bash  

# 配置参数  
NUM_GPUS=8  
NNODES=2  
ADDR="node-0"  
PORT=12345  

# Rank 0 最后运行
echo "Starting training on local machine with rank 0"
cd /tmp/LLaVa_NeXT
bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh \
    --nproc_per_node "${NUM_GPUS}" \
    --nnodes "${NNODES}" \
    --node_rank 0 \
    --master_addr "${ADDR}" \
    --master_port "${PORT}"

echo "All training tasks have been started!"