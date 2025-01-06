#!/bin/bash  

# 配置参数  
NUM_GPUS=8  
NNODES=2  
ADDR="node-0"  
PORT=12345  

# 遍历每个节点并启动训练  
for ((RANK=1; RANK<NNODES; RANK++)); do  
    NODE="node-$RANK"  
    echo "Starting training on $NODE with rank $RANK"  

    # 其他 Rank 节点通过 SSH 后台运行，并使用 nohup 确保不会阻塞
    ssh $NODE bash << EOF &
    cd /tmp/LLaVa_NeXT
    nohup bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh \
        --nproc_per_node "${NUM_GPUS}" \
        --nnodes "${NNODES}" \
        --node_rank "${RANK}" \
        --master_addr "${ADDR}" \
        --master_port "${PORT}" > train_rank_${RANK}.log 2>&1 &
EOF
done



