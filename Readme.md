面向开放指令的多源遥感大模型

一、项目简介

本项目是一个基于Qwen2.5-VL模型的多模态视觉语言理解与生成系统，专注于遥感图像的分析和处理。项目通过微调训练和评估流程，实现对遥感图像中目标的检测、识别和描述生成。

主要功能

• 多模态模型微调：基于Qwen2.5-VL模型进行视觉语言联合训练

• 遥感图像分析：专门针对遥感图像数据进行优化处理

• 目标检测与识别：支持固定类别和开放类别的目标检测任务

• 开放描述生成：生成对遥感图像的自然语言描述

• 性能评估：提供完整的评估流程和指标计算（mAP、mF1等）

技术特点

• 基于先进的视觉语言大模型Qwen2.5-VL

• 使用ms-swift框架进行高效微调

• 支持LoRA和全参数微调两种模式

• 提供完整的评估流水线和可视化工具

• 兼容多种主流视觉语言模型（LLaVA、Falcon等）

应用场景

• 遥感图像智能解译

• 地理信息系统分析

• 环境监测与评估

• 城市规划与管理

二、环境部署

1. 文件迁移

1.1 本文件夹作为母文件夹

1.2 大文件合并

在FT/bigfiles.rar中存储着微调模型、微调结果和数据集三部分：
• 微调模型文件export_11xxx放到./Qwen2.5-VL-FT-Remote

• 微调结果results放到./Qwen2.5-VL-FT-Remote/results

• 数据集datasets（包含VLAD_R和VRSBench两部分）放到./Qwen2.5-VL-FT-Remote/datasets

2. 设置环境

2.1 新建虚拟环境

conda create -n qwen_vl python=3.10
conda activate qwen_vl


2.2 安装依赖

建议按照以下项目顺序安装，避免直接使用requirements.txt：
• Qwen部署框架：https://github.com/libing64/Qwen2.5-VL-Fine-Tuning

• Qwen微调框架：https://github.com/modelscope/ms-swift

注意：安装flash-attn包可能耗时较长

3. 前期准备

数据集的路径需要是绝对路径（ms-swift微调需求），分别运行：
python ./tools/dataset_Remote/02_spilt_abspath.py
python ./tools/dataset_VRSBench/02_abspath.py


4. 开始训练和评测

4.1 代码说明

./tools目录结构：
• bbox_detector：切换模型API并解析模型结果

• calculate_map_mf1：计算mAP和mf1分数

• generate_eval_instructions：生成评测集指令

• dataset_Remote：遥感数据集处理工具

  • 04：数据集统计

  • 05：固定和开放类别任务评测

  • 06：固定和开放类别评测统计

  • 07：开放任务评测统计

  • 08：目标框可视化

  • 09：统计结果可视化

完整流程：训练（约2天）→导出结果→vllm部署模型→05评测（约2H）→calculate_map_mf1→06→07→08→09

建议使用screen命令后台运行

4.2 常用指令

训练
swift web-ui


模型导出
# LoRA微调导出
swift export --adapters output/v0-20250801-164528/checkpoint-11968 --merge_lora true

# 全参微调导出
swift export --model output/v1-20250711-174411/checkpoint-978


模型部署
# 基础部署
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --gpu-memory-utilization 0.9 --max-model-len 8192 --limit-mm-per-prompt "image=2"

# 自定义端口部署
CUDA_VISIBLE_DEVICES=4 vllm serve ./ms-swift/output/export_v3_23936 --port 8001


# 评价
    # 1.微调实验
    python ./tools/dataset_Remote/05_eval.py 记得调bbox detector的模型api


    # 2.微调实验结果处理为coco格式
    # 微调
    # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name ms-swift/output/export_v5_11968 --task_type fixed

    # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name ms-swift/output/export_v5_11968 --task_type open

    # 处理open ended任务
    python ./tools/dataset_Remote/07_eval_statistics_open_ended.py --file_name ms-swift/output/export_v5_11968

    # 启用指令动态匹配机制处理open-ended任务
    1.检测
      首先修改instruction.py与eval_instruction.py中的模型api
      再修改eval_instruction.py中的图片路径（使用绝对路径）
      python ./tools/dataset_Remote/010_eval_instruction.py

    2.将结果转化为coco格式
      注意修改需要处理的json文件路径
      python ./tools/dataset_Remote/07_eval_statistics_open_ended.py 

    3.可视化
      注意修改图片路径，确保该图片启用与未启用的coco结果均已保存在对应路径下
      python ./tools/dataset_Remote/011_case_study.py
      
    # Qwen
    # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name Qwen/Qwen2.5-VL-7B-Instruct --task_type fixed

    # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name Qwen/Qwen2.5-VL-7B-Instruct --task_type open

    # 处理open ended任务
    python ./tools/dataset_Remote/07_eval_statistics_open_ended.py --file_name Qwen/Qwen2.5-VL-7B-Instruct


    # Falcon(不支持open ended)
    # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name Falcon --task_type fixed

    # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name Falcon --task_type open


    # llava
    # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name llava-hf/llava-v1.6-vicuna-7b-hf --task_type fixed

    # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name llava-hf/llava-v1.6-vicuna-7b-hf --task_type open
    
    # 处理open ended任务
    python ./tools/dataset_Remote/07_eval_statistics_open_ended.py --file_name llava-hf/llava-v1.6-vicuna-7b-hf


    # lae-dino
    # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name lae-dino --task_type fixed

    # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name lae-dino --task_type open
    
    # 处理open ended任务
    python ./tools/dataset_Remote/07_eval_statistics_open_ended.py --file_name lae-dino


    #3.计算IoU
    python ./tools/calculate_map_mf1.py 
    # 处理启用动态匹配机制的计算
    python ./tools/open_ended_result.py
