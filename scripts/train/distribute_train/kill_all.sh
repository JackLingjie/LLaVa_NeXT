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

# 使用 pdsh 在所有节点上安装 lsof 并终止指定端口的进程
pdsh -R ssh -w ^$NODES_FILE bash -c "' 
set -e  # 如果有任何命令失败，退出脚本

# 安装 lsof
if ! command -v lsof &> /dev/null; then
    echo \"Installing lsof...\"
    sudo apt update && sudo apt install -y lsof
fi

# 终止运行在端口 12345 的进程
if sudo lsof -t -i :12345 &> /dev/null; then
    echo \"Killing processes on port 12345...\"
    sudo kill -9 \$(sudo lsof -t -i :12345)
else
    echo \"No processes running on port 12345.\"
fi
'"

echo "Commands executed successfully on all nodes!"
