#!/bin/bash

# 节点列表文件路径
NODES_FILE="scripts/train/config/nodes_3.txt"  # 节点列表文件

# 确保节点列表文件存在
if [ ! -f "$NODES_FILE" ]; then
    echo "Error: $NODES_FILE not found!"
    exit 1
fi

# 检查节点文件内容是否包含合法节点名称
if ! grep -q "node-" "$NODES_FILE"; then
    echo "Error: $NODES_FILE does not contain valid node names!"
    exit 1
fi

# 使用 pdsh 在所有节点上执行 git pull
pdsh -R ssh -w ^$NODES_FILE bash -c "' 
set -e  # 如果有任何命令失败，退出脚本

# 切换到 LLaVa_NeXT 目录
cd /tmp/LLaVa_NeXT || { echo \"Directory /tmp/LLaVa_NeXT not found!\"; exit 1; }

# 拉取最新代码
echo \"Pulling latest code...\"
git pull || { echo \"Failed to pull latest code!\"; exit 1; }
'"

echo "Git pull completed on all nodes!"
