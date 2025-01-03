import json  
from datasets import load_dataset  
from multiprocessing import cpu_count  
import os  
from PIL import Image  
  
# 加载数据集  
ds = load_dataset("/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-ReCap-118K")  
train_data = ds["train"]  
  
# 定义图像和JSON文件的输出路径  
output_image_path = "/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-ReCap-118K/coco118k/"  
output_json_file = "/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-ReCap-118K/coco118k_stage1.5_finetune_w_prompt.json"  
  
# 确保图像输出目录存在  
os.makedirs(output_image_path, exist_ok=True)  
  
def process_entry(entry):  
    from PIL import Image  
      
    try:  
        # Extract the id and image  
        img_id = entry['id']  
        image = entry['image']  
          
        # Save the image to the specified location  
        image_filename = f"{img_id}.jpg"  
        image_path = os.path.join(output_image_path, image_filename)  
        image.save(image_path)  
          
        # Modify the conversations to include the new prompt  
        conversations = entry['conversations']  
        for conv in conversations:  
            if conv['from'] == 'human' and conv['value'] == '<image>':  
                conv['value'] += "\nPlease generate detailed descriptions of the given image."  
          
        # Create the new data entry  
        new_entry = {  
            "id": img_id,  
            "image_temp": f"coco118k/{image_filename}",  
            "conversations": conversations  
        }  
          
        return new_entry  
  
    except Exception as e:  
        print(f"Error processing entry with id {entry['id']}: {e}")  
        return None  
  
# 选择前10个条目进行测试  
train_data_subset = train_data  
  
# 使用map和多进程处理数据  
processed_data = train_data_subset.map(process_entry, num_proc=cpu_count())  
  
# 将处理后的数据转换为字典列表  
processed_data_list = []  
for entry in processed_data:  
    if entry is not None:  
        # 删除 image_temp 并将其替换为 image  
        entry['image'] = entry.pop('image_temp')  
        # 删除 data_source 键  
        if 'data_source' in entry:  
            del entry['data_source']  
        processed_data_list.append(entry)  
  
# 将处理后的数据保存到JSON文件  
with open(output_json_file, 'w', encoding='utf-8') as f:  
    json.dump(processed_data_list, f, ensure_ascii=False, indent=2)  
  
print(f"Processed data has been saved to {output_json_file}")  