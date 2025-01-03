import json  
from datasets import load_dataset  
from multiprocessing import cpu_count  
import os  
from PIL import Image  
from tqdm import tqdm  
import uuid  

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
  
    def process_entry(entry):  
        try:  
            # Check if 'id' key exists, if not, generate a unique id  
            if 'id' not in entry:  
                img_id = str(uuid.uuid4())  # Generate a unique id  
            else:  
                img_id = entry['id']  
    
            image = entry['image']  
    
            # Convert image to RGB if it is in RGBA or P mode  
            if image.mode in ['RGBA', 'P']:  
                image = image.convert('RGB')  
    
            # Determine the image filename and path  
            image_filename = f"{img_id}.jpg"  
            image_path = os.path.join(output_image_path, image_filename)  
    
            # Ensure the directory exists  
            os.makedirs(os.path.dirname(image_path), exist_ok=True)  
    
            # Save the image to the specified location  
            image.save(image_path)  
    
            # Modify the conversations to include the new prompt  
            conversations = entry['conversations']  
            
            # Create the new data entry  
            new_entry = {  
                "id": img_id,  
                "image_temp": f"{IMAGE_PATH}/{image_filename}",  
                "conversations": conversations  
            }  
    
            return new_entry  
        except Exception as e:  
            print(f"Error processing entry: {e}")  
            return None    
  
    # 使用map和多进程处理数据，添加tqdm显示进度  
    # train_data = train_data.select(range(10))
    processed_data = train_data.map(lambda entry: process_entry(entry), num_proc=cpu_count())  
  
    # 将处理后的数据转换为字典列表  
    processed_data_list = []  
    for entry in tqdm(processed_data, desc="Processing Entries"):  
        if entry is not None:  
            # 删除 image_temp 并将其替换为 image  
            entry['image'] = entry.pop('image_temp')  
            processed_data_list.append(entry)  
  
    # 将处理后的数据保存到JSON文件  
    with open(output_json_file, 'w', encoding='utf-8') as f:  
        json.dump(processed_data_list, f, ensure_ascii=False, indent=2)  
  
    print(f"Processed data has been saved to {output_json_file}")  
  
# 获取机器 ID  
# machine_id = int(input("Enter Machine ID (1-5): "))  
machine_id = 5
# 根据机器 ID 读取对应的 JSON 文件  
file_path = f'process_data/convert/exist_files_map_part_{machine_id}.json'  
with open(file_path, 'r') as file:  
    dirname2filename = json.load(file)  
  
# 对于每个键值对，处理数据集  
for dirname, filename in tqdm(dirname2filename.items(), desc="Processing Datasets"):  
    process_dataset(dirname, filename)  