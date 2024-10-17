import argparse
import json
import os
import numpy as np
from transformers import AutoTokenizer
import torch

# 设置命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--base-model-path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="基础模型路径")
parser.add_argument("--jsonl-folder", type=str, default="/path/to/jsonl/folder",
                    help="包含jsonl文件的文件夹路径")
parser.add_argument("--jsonl-file-base", type=str, default="/path/to/jsonl/base/file.jsonl",
                    help="基准jsonl文件路径")
args = parser.parse_args()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

# 读取基准jsonl文件
jsonl_file_base = args.jsonl_file_base
data_base = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data_base.append(json_obj)

# 计算基准文件的总时间和总token
total_time_base = 0
total_token_base = 0
speeds_base = []
for datapoint in data_base:
    qid = datapoint["question_id"]
    answer = datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds_base.append(tokens / times)
    total_time_base += times
    total_token_base += tokens

# 遍历文件夹中的所有jsonl文件
folder_path = args.jsonl_folder
ratios=[]
all_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
for file_name in all_files:
    jsonl_file = os.path.join(folder_path, file_name)
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]['turns']
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens / times)

    # 计算当前文件的平均速度
    average_speed = np.array(speeds).mean()

    # 计算加速比
    average_speed_base = np.array(speeds_base).mean()
    ratio = average_speed / average_speed_base
    ratios.append(ratio)
    print(f"{file_name} ratio: {ratio}")
for r in ratios:
    print(r)
# print(ratios)