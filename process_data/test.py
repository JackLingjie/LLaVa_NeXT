import json  
from datasets import load_dataset  
from multiprocessing import cpu_count  
import os  
from PIL import Image  
  
# 加载数据集  
ds = load_dataset("/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-ReCap-118K")  
train_data = ds["train"]  