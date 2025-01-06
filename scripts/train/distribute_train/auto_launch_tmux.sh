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

# 使用 tmux 分布式启动训练
for ((RANK=0; RANK<NNODES; RANK++)); do
    NODE="node-$RANK"
    SESSION_NAME="train_$RANK"
    echo "Starting training on $NODE with rank $RANK in tmux session $SESSION_NAME"

    # 检查是否存在同名 tmux 会话
    tmux has-session -t $SESSION_NAME 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Session $SESSION_NAME already exists. Skipping..."
        continue
    fi

    # 创建新的 tmux 会话
    tmux new-session -d -s $SESSION_NAME "
        ssh $NODE bash -c '
            cd /tmp/LLaVa_NeXT &&
            pwd &&
            ls &&
            bash scripts/train/distribute_train/finetune_siglip_qwen2_mid_stage_multinode_base.sh \
                --nproc_per_node ${NUM_GPUS} \
                --nnodes ${NNODES} \
                --node_rank ${RANK} \
                --master_addr ${ADDR} \
                --master_port ${PORT}
        '
    "

    # 确认会话创建成功
    if [ $? -ne 0 ]; then
        echo "Failed to create tmux session $SESSION_NAME"
        exit 1
    fi
done

echo "All nodes have been started! Use 'tmux ls' to check sessions."
