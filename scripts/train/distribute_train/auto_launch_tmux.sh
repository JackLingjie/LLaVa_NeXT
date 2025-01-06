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
for ((RANK=0; RANK<NNODES; RANK++)); do
    NODE="node-$RANK"
    tmux new-session -d -s train_$RANK "ssh $NODE bash -c 'cd /tmp/LLaVa_NeXT && bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh --nproc_per_node ${NUM_GPUS} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${ADDR} --master_port ${PORT}'"
done


echo "All nodes have been started!"
