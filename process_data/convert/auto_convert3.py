import json  
from datasets import load_dataset  
from multiprocessing import cpu_count  
import os  
from PIL import Image  
from tqdm import tqdm  
  
def process_dataset(dirname, filename):  
    # 加载数据集  
    path = f"/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-OneVision-Data/{dirname}"  
    ds = load_dataset(path)  
    train_data = ds["train"]  
  
    IMAGE_PATH = f"{dirname}_images"  
    # 定义图像和JSON文件的输出路径  
    output_image_path = f"/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-Stage2-Si/{IMAGE_PATH}/"  
    output_json_file = f"/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-Stage2-Si/{filename}"  
  
    print(f"output_image_path: {output_image_path}")  
    print(f"output_json_file: {output_json_file}")  
  
    # 确保图像输出目录存在  
    os.makedirs(output_image_path, exist_ok=True)  
  
    def is_valid_entry(entry):  
        # 检查条目是否有效（即具有 id 和 image）  
        return entry.get('image') is not None and 'id' in entry  
  
    def process_entry(entry):  
        try:  
            img_id = entry['id']  
            image = entry['image']  
  
            # 如果图像是 RGBA 或 P 模式，则转换为 RGB  
            if image.mode in ['RGBA', 'P']:  
                image = image.convert('RGB')  
  
            # 确定图像文件名和路径  
            image_filename = f"{img_id}.jpg"  
            image_path = os.path.join(output_image_path, image_filename)  
  
            # 确保目录存在  
            os.makedirs(os.path.dirname(image_path), exist_ok=True)  
  
            # 将图像保存到指定位置  
            image.save(image_path)  
  
            # 修改会话以包含新的提示  
            conversations = entry['conversations']  
  
            # 创建新的数据条目  
            new_entry = {  
                "id": img_id,  
                "image_temp": f"{IMAGE_PATH}/{image_filename}",  
                "conversations": conversations  
            }  
  
            return new_entry  
        except Exception as e:  
            print(f"Error processing entry: {e}")  
            return None  
  
    # 过滤掉无效的条目  
    filtered_data = train_data.filter(is_valid_entry)  
  
    # 使用 map 和多进程处理数据，添加 tqdm 显示进度  
    processed_data = filtered_data.map(lambda entry: process_entry(entry), num_proc=24)  
  
    # 将处理后的数据转换为字典列表  
    processed_data_list = []  
    for entry in tqdm(processed_data, desc="Processing Entries"):  
        if entry is not None:  
            # 删除 image_temp 并将其替换为 image  
            entry['image'] = entry.pop('image_temp')  
            processed_data_list.append(entry)  
  
    # 将处理后的数据保存到 JSON 文件  
    with open(output_json_file, 'w', encoding='utf-8') as f:  
        json.dump(processed_data_list, f, ensure_ascii=False, indent=2)  
  
    print(f"Processed data has been saved to {output_json_file}")  
  
# 获取机器 ID  
machine_id = 3  
# 根据机器 ID 读取对应的 JSON 文件  
file_path = f'process_data/convert/exist_files_map_part_{machine_id}_supply.json'  
with open(file_path, 'r') as file:  
    dirname2filename = json.load(file)  
  
# 对于每个键值对，处理数据集  
for dirname, filename in tqdm(dirname2filename.items(), desc="Processing Datasets"):  
    process_dataset(dirname, filename)  