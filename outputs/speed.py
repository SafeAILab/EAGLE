import json
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/33B/")

jsonl_file = "/home/lyh/code/nlp/ess/Medusa/llm_judge/data/mt_bench/model_answer0/vicuna-33b-v1.3-0_baseline_fp16-greedy.jsonl"  # 用你的 JSONL 文件名替换这里

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        # 逐行解析 JSON 对象
        json_obj = json.loads(line)
        data.append(json_obj)

# 现在，'data' 列表中包含了 JSONL 文件中的所有数据
errorids_old=[85, 87, 114, 117, 140, 142, 147, 150]
errorids=[]
total_token=0
total_time=0
for datapoint in data:
    qid=datapoint["question_id"]
    # if qid in errorids_old:
    #     continue
    answer=datapoint["choices"][0]['turns']
    continue_flag=0
    for i in answer:
        if "ERROR"==i or "many, many, many" in i:
            errorids.append(qid)
            continue_flag=1
            break
    if continue_flag:
        continue
    tokens=0
    for i in answer:
        tokens+=(len(tokenizer(i).input_ids)-2)
    #tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    total_token+=tokens
    total_time+=times
print("total_token",total_token)
print("total_time",total_time)
print("speed",total_time/total_token)
print(errorids)
import transformers.generation.utils