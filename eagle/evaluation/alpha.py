import json
import numpy as np

json_files=[
    "/home/lyh/code/nlp/EAGLE/data/mt_bench/model_answer/vicuna-7b-alpha-temperature-0.0.jsonl",
]


for jsonl_file in json_files:
    data=[]
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    alphas=[0 for _ in range(5)]
    alphas_num=[0 for _ in range(5)]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        ids = sum(datapoint["choices"][0]['idxs'])
        alpha=datapoint["choices"][0]['alpha']
        alpha_num = datapoint["choices"][0]['alpha_num']
        for i in range(len(alpha)):
            alphas[i]+=alpha[i]
            alphas_num[i] += alpha_num[i]


    ar=np.array(alphas)/np.array(alphas_num)
    print(np.round(ar, 2))