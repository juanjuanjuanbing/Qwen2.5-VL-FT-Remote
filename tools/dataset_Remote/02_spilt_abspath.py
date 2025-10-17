import json
import random
import os
from collections import defaultdict
from pathlib import Path

'''
    v0.1.0 2025.06.12 @jialei
    将VLAD_Remote.json文件划分9：1的训练集和验证集
    保证验证集中不出现训练集的图片，避免数据泄漏
    将json文件中的image改为绝对路径（在训练时被LLaMA-Factory找到）
    (Important)在数据集不再被移动之后进行划分
    新增功能：同时保存测试集的图片路径到test_image_list.txt
'''

def convert_to_absolute_paths(data, base_dir):
    """Convert image paths in 'images' field to absolute paths"""
    for item in data:
        absolute_images = []
        for relative_path in item["images"]:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(base_dir), relative_path.replace("./", "")))
            absolute_images.append(abs_path)
        item["images"] = absolute_images
    return data

def split_dataset(input_json_path, train_ratio=0.9):
    """Split dataset into train and test sets, grouped by images"""
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_json_path)))
    data = convert_to_absolute_paths(data, base_dir)
    
    image_groups = defaultdict(list)
    for item in data:
        primary_image = item["images"][0]
        image_groups[primary_image].append(item)
    
    unique_images = list(image_groups.keys())
    random.shuffle(unique_images)
    
    split_idx = int(len(unique_images) * train_ratio)
    train_images = unique_images[:split_idx]
    test_images = unique_images[split_idx:]
    
    train_data = []
    test_data = []
    
    for img in train_images:
        train_data.extend(image_groups[img])
    
    for img in test_images:
        test_data.extend(image_groups[img])
    
    return train_data, test_data, test_images

def save_json(data, output_path):
    """Save data in compact JSON format with one sample per line"""
    with open(output_path, 'w') as f:
        f.write("[\n")
        for i, item in enumerate(data):
            json.dump(item, f, separators=(',', ':'))
            if i < len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n")

def save_test_image_list(test_images, output_dir):
    """Save test image paths to a text file"""
    output_path = Path(output_dir) / "test_image_list.txt"
    with open(output_path, 'w') as f:
        for img_path in test_images:
            f.write(f"{img_path}\n")

if __name__ == "__main__":
    input_json = "./datasets/VLAD_Remote/VLAD_Remote.json"
    output_dir = "./datasets/VLAD_Remote/"
    
    train_data, test_data, test_images = split_dataset(input_json)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(train_data, Path(output_dir) / "VLAD_Remote_train.json")
    save_json(test_data, Path(output_dir) / "VLAD_Remote_test.json")
    save_test_image_list(test_images, output_dir)
    
    print(f"Split complete!\nTrain samples: {len(train_data)}\nTest samples: {len(test_data)}")
    print(f"Unique train images: {len({item['images'][0] for item in train_data})}")
    print(f"Unique test images: {len({item['images'][0] for item in test_data})}")
    print(f"Test image paths saved to: {Path(output_dir) / 'test_image_list.txt'}")