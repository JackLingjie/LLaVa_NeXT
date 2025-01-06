#!/bin/bash  
  
# 配置节点数量  
NNODES=3  
  
# 遍历每个节点并通过SSH执行初始化操作  
for ((RANK=0; RANK<NNODES; RANK++)); do  
    NODE="node-$RANK"  
    echo "Setting up environment on $NODE"  
      
    ssh $NODE bash << 'EOF'
    set -e  # 如果有任何命令失败，退出脚本  
  
    # 切换到临时目录并克隆项目  
    cd /tmp  
    if [ ! -d "LLaVa_NeXT" ]; then  
        git clone https://github.com/JackLingjie/LLaVa_NeXT.git  
    fi  
  
    cd LLaVa_NeXT  

    # 检查当前分支并切换到 dev 分支  
    current_branch=$(git rev-parse --abbrev-ref HEAD)  
    if [ "$current_branch" != "dev" ]; then  
        git checkout dev || git checkout -b dev origin/dev  
    fi  
  
    # 安装依赖  
    pip install -e ".[train]"  
    pip install -U flash-attn==2.5.7 --no-build-isolation  
EOF

done  