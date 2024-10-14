#默认温度是0
# python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct"
#需要跑个baseline

#greedy test
python speed.py --jsonl-file "/home/leihaodong/1pj/exp4eagle/outputs0/mt_bench/llama38b2_40-temperature-0.0.jsonl" --jsonl-file-base "/home/leihaodong/1pj/exp4eagle/outputs0/mt_bench/base-llama38b2_40-temperature-0.0.jsonl"
#entropy test
python eagle/evaluation/speed.py --jsonl-file "/home/leihaodong/1pj/exp4eagle/outputs0/mt_bench/entropy_llama38b2_40-temperature-0.0.jsonl" --jsonl-file-base "/home/leihaodong/1pj/exp4eagle/outputs0/mt_bench/base-llama38b2_40-temperature-0.0.jsonl"

python eagle/evaluation/speedfolder.py --jsonl-folder "/home/leihaodong/1pj/EAGLE/mt_bench/all1" --jsonl-file-base "/home/leihaodong/1pj/exp4eagle/outputs0/mt_bench/base-llama38b2_40-temperature-0.0.jsonl"


python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct"

python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "1_allacept_llama3_8b_chat" --expand-method "upp"

python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "2_allacept_llama3_8b_chat" --expand-method "upp"

python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "1_greedy_llama3_8b_chat" 

python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "2_greedy_llama3_8b_chat" 


nohup python -m eagle.evaluation.gen_ea_answer_lpf_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "llama3_8b_chat_lpf_d5" > entropyoutput1.log 2>&1 &

nohup python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "2_entropy_llama3_8b_chat" --expand-method "entropy" > entropyoutput2.log 2>&1 &

nohup python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "3_entropy_llama3_8b_chat" --expand-method "entropy" > entropyoutput3.log 2>&1 &

nohup python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "3_allacept_llama3_8b_chat" --expand-method "upp" > allaceptoutput1.log 2>&1 &

nohup python -m eagle.evaluation.gen_ea_answer_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "1_greedy_llama3_8b_chat" > greedyoutput1.log 2>&1 &

nohup python -m eagle.evaluation.gen_ea_answer_lpf_llama3chat  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --model-id "llama3_8b_chat_lpf_d5" > llama3_8b_chat_lpf_d5.log 2>&1 &

#baseline test
nohup python -m eagle.evaluation.gen_baseline_answer_llama3chat\
        --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  \
        --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" > log/baseline1.log 2>&1 &
#lpf test
#5
nohup python -m eagle.evaluation.gen_ea_answer_lpf_llama3chat\
  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  \
  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --lpfrog-model-path /data/lei/eagle3output/state_20 \
  --model-id "llama3_8b_chat_lpf_d5" > log/llama3_8b_chat_lpf_d5.log 2>&1 &
#8
nohup python -m eagle.evaluation.gen_ea_answer_lpf_llama3chat\
  --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B"  \
  --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --lpfrog-model-path /data/lei/eagle3output/state_20 \
  --depth 8 \
  --model-id "llama3_8b_chat_lpf_d8" > log/llama3_8b_chat_lpf_d8.log 2>&1 &
