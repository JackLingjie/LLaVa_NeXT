import torch
from transformers import (
    Qwen2VLForConditionalGeneration,  # Qwen2-VL
    AutoModelForCausalLM,                 # Qwen2 / Qwen2.5 coder
)

# 1) 加载 Qwen2-VL 模型
qwen2vl_model_path = "/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/vlm_checkpoints/Qwen2-VL-7B-Instruct"
qwen2vl = Qwen2VLForConditionalGeneration.from_pretrained(
    qwen2vl_model_path,
    torch_dtype="auto",  
    device_map="auto"
)

# 2) 加载 Qwen2.5 coder 模型 (或者 Qwen2-7B-Instruct 等)
qwen2coder_model_path = "/mnt/lingjiejiang/multimodal_code/checkpoints/llms/Qwen2.5-Coder-7B-Instruct"
qwen2coder = AutoModelForCausalLM.from_pretrained(
    qwen2coder_model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 取出 Qwen2VL 里 “LLM 部分”的 state_dict
qwen2vl_backbone_sd = qwen2vl.model.state_dict()

# 取出 Qwen2Coder (Qwen2.5) 里 “LLM 部分”的 state_dict
qwen2coder_sd = qwen2coder.model.state_dict()

# 遍历 coder 的每个参数，如果在 vl 里有同名参数且形状一致，就直接覆盖
for name, param in qwen2coder_sd.items():
    if name in qwen2vl_backbone_sd and qwen2vl_backbone_sd[name].shape == param.shape:
        # 用 coder 的参数覆盖 VL 的对应参数
        with torch.no_grad():
            qwen2vl_backbone_sd[name].copy_(param)

# 将更新后的 state_dict load 回去
qwen2vl.model.load_state_dict(qwen2vl_backbone_sd)

qwen2vl_lm_head_sd = qwen2vl.lm_head.state_dict()
qwen2coder_lm_head_sd = qwen2coder.lm_head.state_dict()
for name, param in qwen2coder_lm_head_sd.items():
    if name in qwen2vl_lm_head_sd and qwen2vl_lm_head_sd[name].shape == param.shape:
        with torch.no_grad():
            qwen2vl_lm_head_sd[name].copy_(param)
qwen2vl.lm_head.load_state_dict(qwen2vl_lm_head_sd)

qwen2vl.save_pretrained("/mnt/lingjiejiang/multimodal_code/checkpoints/vlms/Qwenvl2-coder2.5-7B-Instruct_merge")


import os, shutil
from transformers import AutoTokenizer

# 你已经在代码中加载了 Qwen2-VL
qwen2vl_model_path = "/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/vlm_checkpoints/Qwen2-VL-7B-Instruct"
# qwen2vl = Qwen2VLForConditionalGeneration.from_pretrained(...)
# /mnt/lingjiejiang/multimodal_code/checkpoints/llms/Qwen2.5-Coder-7B-Instruct
# 你的新模型保存目录
merged_path = "/mnt/lingjiejiang/multimodal_code/checkpoints/vlms/Qwenvl2-coder2.5-7B-Instruct_merge"

# 如果你也加载了原 Qwen2-VL 的 tokenizer:
vl_tokenizer = AutoTokenizer.from_pretrained(qwen2vl_model_path)
# 直接保存到新目录
vl_tokenizer.save_pretrained(merged_path)

# 某些 Qwen2-VL 版本还带有 "processor" 或 "chat_template.json"，也可类似保存或复制
# 如果原目录下有 chat_template.json，可以直接 shutil.copy
src_chat_template = os.path.join(qwen2vl_model_path, "chat_template.json")
dst_chat_template = os.path.join(merged_path, "chat_template.json")
if os.path.exists(src_chat_template):
    shutil.copy(src_chat_template, dst_chat_template)

# 如果有 preprocessor_config.json，也可以类似处理
src_preproc_config = os.path.join(qwen2vl_model_path, "preprocessor_config.json")
dst_preproc_config = os.path.join(merged_path, "preprocessor_config.json")
if os.path.exists(src_preproc_config):
    shutil.copy(src_preproc_config, dst_preproc_config)

print("All done. Check the merged_path folder to see if tokenizer & chat_template are there.")
