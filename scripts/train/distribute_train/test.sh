#!/bin/bash  

# 配置参数  
NUM_GPUS=8  
NNODES=16  
ADDR=node-0  
PORT=12345  

# 遍历每个节点并通过SSH启动训练  
for ((RANK=0; RANK<NNODES; RANK++)); do  
    NODE="node-$RANK"  
    echo "Starting training on $NODE with rank $RANK"  
      
    bash << EOF  
    cd /tmp/LLaVa_NeXT  
    bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh \
        --nproc_per_node "${NUM_GPUS}" \
        --nnodes "${NNODES}" \
        --node_rank "${RANK}" \
        --master_addr "${ADDR}" \
        --master_port "${PORT}"  
EOF
  
done
