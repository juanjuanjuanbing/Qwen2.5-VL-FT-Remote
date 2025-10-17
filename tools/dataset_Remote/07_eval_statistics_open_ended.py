import json
import os
import glob
import argparse
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # 设置为评估模式

def get_bert_embedding(text):
    """使用BERT获取文本的嵌入向量"""
    try:
        if not text.strip():
            return None
            
        # 使用BERT tokenizer处理文本
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # 获取BERT嵌入
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 使用[CLS] token的嵌入作为整个句子的表示
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        return embedding
        
    except Exception as e:
        print(f"获取BERT嵌入时出错: {e}")
        return None

def calculate_bert_similarity(text1, text2):
    """使用BERT嵌入计算两个文本之间的余弦相似度"""
    try:
        # 清理文本
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = str(text).lower().replace('[', '').replace(']', '').replace("'", "").replace(".", "")
            return text
            
        text1_clean = clean_text(text1)
        text2_clean = clean_text(text2)
        
        # 如果任一文本为空，返回0相似度
        if not text1_clean.strip() or not text2_clean.strip():
            return 0.0
            
        # 获取BERT嵌入
        embedding1 = get_bert_embedding(text1_clean)
        embedding2 = get_bert_embedding(text2_clean)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
        
    except Exception as e:
        print(f"计算BERT相似度时出错: {e}")
        return 0.0

def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"无法获取图像尺寸: {image_path}, 错误: {e}")
        return None

def get_class_mappings():
    """获取所有类别映射"""
    # 对于open_ended任务，使用完整的类别映射
    return {
        'full_mapping': {
            'xview': {
                'Fixed-wing Aircraft': 0,
                'Small Aircraft': 1,
                'Cargo Plane': 2,
                'Helicopter': 3,
                'Passenger Vehicle': 4,
                'Small Car': 5,
                'Bus': 6,
                'Pickup Truck': 7,
                'Utility Truck': 8,
                'Truck': 9,
                'Cargo Truck': 10,
                'Truck w/Box': 11,
                'Truck Tractor': 12,
                'Trailer': 13,
                'Truck w/Flatbed': 14,
                'Truck w/Liquid': 15,
                'Crane Truck': 16,
                'Railway Vehicle': 17,
                'Passenger Car': 18,
                'Cargo Car': 19,
                'Flat Car': 20,
                'Tank car': 21,
                'Locomotive': 22,
                'Maritime Vessel': 23,
                'Motorboat': 24,
                'Sailboat': 25,
                'Tugboat': 26,
                'Barge': 27,
                'Fishing Vessel': 28,
                'Ferry': 29,
                'Yacht': 30,
                'Container Ship': 31,
                'Oil Tanker': 32,
                'Engineering Vehicle': 33,
                'Tower crane': 34,
                'Container Crane': 35,
                'Reach Stacker': 36,
                'Straddle Carrier': 37,
                'Mobile Crane': 38,
                'Dump Truck': 39,
                'Haul Truck': 40,
                'Scraper/Tractor': 41,
                'Front loader/Bulldozer': 42,
                'Excavator': 43,
                'Cement Mixer': 44,
                'Ground Grader': 45,
                'Hut/Tent': 46,
                'Shed': 47,
                'Building': 48,
                'Aircraft Hangar': 49,
                'Damaged Building': 50,
                'Facility': 51,
                'Construction Site': 52,
                'Vehicle Lot': 53,
                'Helipad': 54,
                'Storage Tank': 55,
                'Shipping container lot': 56,
                'Shipping Container': 57,
                'Pylon': 58,
                'Tower': 59
            },
            'visdrone': {
                'pedestrian': 0,
                'people': 1,
                'bicycle': 2,
                'car': 3,
                'van': 4,
                'truck': 5,
                'tricycle': 6,
                'awning-tricycle': 7,
                'bus': 8,
                'motor': 9
            }
        }
    }

def calculate_similarity_to_gt_class(generated_text, gt_object):
    """计算生成文本与真实类别(gt_object)之间的BERT相似度得分"""
    try:
        # 清理文本
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = str(text).lower().replace('[', '').replace(']', '').replace("'", "").replace(".", "")
            return text
            
        generated_text_clean = clean_text(generated_text)
        gt_object_clean = clean_text(gt_object)
        
        # 如果任一文本为空，返回0相似度
        if not generated_text_clean.strip() or not gt_object_clean.strip():
            return 0.0
            
        # 计算BERT相似度
        similarity = calculate_bert_similarity(generated_text_clean, gt_object_clean)
        return similarity
        
    except Exception as e:
        print(f"计算与真实类别相似度时出错: {e}")
        return 0.0

def get_true_class_id(gt_object, class_mapping):
    """从gt_object中提取真实类别ID"""
    try:
        # 清理gt_object文本
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = str(text).lower().replace('[', '').replace(']', '').replace("'", "").replace(".", "")
            return text
            
        gt_clean = clean_text(gt_object)
        
        # 尝试在类别映射中查找匹配的类别
        for class_name, class_id in class_mapping.items():
            if class_name.lower() in gt_clean or gt_clean in class_name.lower():
                return class_id
        
        # 如果没有找到匹配，返回默认类别ID 0
        return 0
        
    except Exception as e:
        print(f"获取真实类别ID时出错: {e}")
        return 0

def convert_json_to_coco_txt(json_file_path, output_folder):
    # 创建主输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有类别映射
    all_mappings = get_class_mappings()
    
    # 遍历所有JSON文件
    for json_file in glob.glob(os.path.join(json_file_path, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 判断数据来源
        image_path = data['image_path']
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
            
        # 获取图像尺寸
        img_size = get_image_size(image_path)
        if img_size is None:
            print(f"跳过文件 {json_file} 因为无法获取图像尺寸")
            continue
            
        img_width, img_height = img_size
        
        # 确定数据集类型
        dataset_type = 'xview' if 'xview' in image_path.lower() else 'visdrone'
        
        # 处理每种映射
        for mapping_name, mappings in all_mappings.items():
            # 创建子文件夹
            mapping_folder = os.path.join(output_folder, mapping_name)
            os.makedirs(mapping_folder, exist_ok=True)
            
            # 获取当前数据集类型的映射
            class_mapping = mappings[dataset_type]
            
            # 准备输出内容
            output_lines = []
            detections = data['detections']
            
            # 获取生成的对象描述和真实类别
            generated_object = data.get('object', '')
            gt_object = data.get('gt_object', '')
            
            # 计算相似度得分
            similarity_score = calculate_similarity_to_gt_class(generated_object, gt_object)
            
            # 从gt_object中提取真实类别ID
            true_class_id = get_true_class_id(gt_object, class_mapping)
            
            # 处理检测框
            if isinstance(detections, list) and len(detections) > 0:
                for bbox in detections:
                    if len(bbox) != 4:
                        print(f"警告: {json_file} 中的检测有无效的bbox: {bbox}")
                        continue
                        
                    x1, y1, x2, y2 = bbox
                    
                    # 计算中心点和宽高（归一化）
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # 确保坐标在0-1范围内
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    # 添加到输出，格式：类别 相似度 x_center y_center width height
                    output_lines.append(f"{true_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {similarity_score:.6f}")
            
            # 写入TXT文件
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_file = os.path.join(mapping_folder, f"{base_name}.txt")
            
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_lines))
            
            print(f"成功转换: {json_file} -> {output_file} (映射: {mapping_name}, 图像尺寸: {img_width}x{img_height}, 相似度: {similarity_score:.4f}, 真实类别ID: {true_class_id})")

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='将JSON标注转换为COCO格式的TXT文件')
    parser.add_argument('--file_name', type=str, required=True,
                        help='输入文件路径，如 ms-swift/output/export_v5_11968/open_ended')
    
    args = parser.parse_args()
    
    # 设置输入输出路径
    json_folder = f'./results/eval/labels/{args.file_name}/open_ended'
    output_folder = f'./results/eval/coco_labels/{args.file_name}/open_ended'
    
    # 执行转换
    convert_json_to_coco_txt(json_folder, output_folder)

if __name__ == '__main__':
    main()