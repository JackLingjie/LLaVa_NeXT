from copy import deepcopy
# 定义转换函数  
def convert_conversations_to_messages(data):  
    for item in data:  
        messages = []  
        for conversation in item['conversations']:  
            role = 'user' if conversation['from'] == 'human' else 'assistant'  
            # 确保 <image> 在消息的开头  
            content = conversation['value']  
            # if '<image>' in content:  
            #     content = '<image>' + content.replace('<image>', '').strip()  
            messages.append({  
                "content": content,  
                "role": role  
            })  
        # 用新格式替换旧的 conversations  
        item['messages'] = messages  
        item['images'] = [f"/mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-Stage2-Si/{item.pop('image')}"]  
        del item['conversations']  
    return data  
  
# 执行转换  
new_data = convert_conversations_to_messages(deepcopy(all_items_with_code))  
  