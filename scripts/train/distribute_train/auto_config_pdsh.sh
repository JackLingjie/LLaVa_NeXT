#!/bin/bash

# 节点列表文件路径
NODES_FILE="scripts/train/config/nodes_2.txt"  # 节点列表文件

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

# 使用 pdsh 在所有节点上执行初始化操作
pdsh -R ssh -w ^$NODES_FILE bash -c "'
set -e  # 如果有任何命令失败，退出脚本

# 切换到临时目录
cd /tmp || { echo \"Failed to switch to /tmp directory\"; exit 1; }

# 克隆项目
if [ ! -d \"LLaVa_NeXT\" ]; then
    git clone https://github.com/JackLingjie/LLaVa_NeXT.git
fi

cd LLaVa_NeXT || { echo \"Failed to switch to LLaVa_NeXT directory\"; exit 1; }

# 检查并切换到 dev 分支
current_branch=\$(git rev-parse --abbrev-ref HEAD)
if [ \"\$current_branch\" != \"dev\" ]; then
    git checkout dev || git checkout -b dev origin/dev
fi

# 安装依赖
pip install -e \"[train]\"
pip install -U flash-attn==2.5.7 --no-build-isolation
'"

echo "Environment setup completed on all nodes!"
