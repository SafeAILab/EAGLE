import json
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl-file", type=str, default="/home/leihaodong/EAGLE3/mt_bench/lpf1",
                        help="1")
args = parser.parse_args()
json_files = args.jsonl_file
files = os.listdir(json_files)
folder_path = args.jsonl_file
json_files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]
# json_files=[
#     "1pj/EAGLE/mt_bench/greedy1/0_greedy_llama3_8b_chat-temperature-0.0.jsonl",
#     "1pj/EAGLE/mt_bench/greedy1/1_greedy_llama3_8b_chat-temperature-0.0.jsonl",
#     "1pj/EAGLE/mt_bench/greedy1/2_greedy_llama3_8b_chat-temperature-0.0.jsonl",
#     "1pj/EAGLE/mt_bench/greedy1/greedy-llama38b2_40-temperature-0.0.jsonl",
# ]
for jsonl_file in json_files:
    data=[]
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    # alphas=[0 for _ in range(5)]#初始化
    # alphas_num=[0 for _ in range(5)]
    alphas = 0
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        ids = sum(datapoint["choices"][0]['idxs'])
        # print([a / b for a, b in zip(tokens, ids)])
        alpha = sum([a / b for a, b in zip(datapoint["choices"][0]['new_tokens'], datapoint["choices"][0]['idxs'])])/len(datapoint["choices"][0]['new_tokens'])#计算平均树长
        alphas += alpha
        # alpha=datapoint["choices"][0]['alpha']
        # alpha_num = datapoint["choices"][0]['alpha_num']
        
        # for i in range(len(alpha)):
        #     alphas[i]+=alpha[i]#第i个位置的树高
        #     alphas_num[i] += alpha_num[i]#第i个位置的数量
    print(jsonl_file)
    print(alphas/len(data))
    # ar=np.array(alphas)/np.array(alphas_num)
    # print(np.round(ar, 2))