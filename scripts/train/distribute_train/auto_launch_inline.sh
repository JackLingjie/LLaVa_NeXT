#!/bin/bash  

# 配置参数  
NUM_GPUS=8  
NNODES=2  
ADDR="node-0"  
PORT=12345  
NODES_FILE="scripts/train/config/nodes.txt"  # 节点列表文件

# 确保 nodes.txt 文件存在
if [ ! -f "$NODES_FILE" ]; then
    echo "Error: $NODES_FILE not found!"
    exit 1
fi

# 使用 pdsh 分布式启动训练
pdsh --inline -R ssh -w ^$NODES_FILE "
    cd /tmp/LLaVa_NeXT &&
    bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh \
        --nproc_per_node ${NUM_GPUS} \
        --nnodes ${NNODES} \
        --node_rank \$(hostname | grep -o '[0-9]*$') \
        --master_addr ${ADDR} \
        --master_port ${PORT}
"


echo "All nodes have been started!"
