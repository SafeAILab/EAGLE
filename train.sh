# python -m eagle.ge_data.allocation --outdir /home/leihaodong/eagle3traindata 
# nohup python ./eagle/ge_data/allocation.py --outdir /data/lei/eagle3traindata > getdata.log 2>&1 &
# nohup python ./eagle/ge_data/allocation.py --outdir /data/lei/eagle3traindata2 > getdata.log 2>&1 &
# set_gpu 2,3,1,0
# /data/lei/cachelei/huggingface/accelerate/default_config.yaml
nohup accelerate launch -m --mixed_precision=bf16 eagle.train.main_lpfrog\
    --tmpdir /data/lei/eagle3traindata2\
    --cpdir /home/leihaodong/eagle3output \
    --configpath /home/leihaodong/EAGLE3/eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
    --basepath /data/lei/localmodel/meta-llama/Meta-Llama-3-8B-Instruct > train_lpfrog_llama3.log 2>&1 &
# deepspeed main_deepspeed.py --deepspeed_config ds_config.json\
#     --configpath /home/leihaodong/EAGLE3/eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
#     --tmpdir /data/lei/eagle3traindata2\
#     --cpdir /home/leihaodong/eagle3output \
#     --basepath /data/lei/localmodel/meta-llama/Meta-Llama-3-8B-Instruct > train_lpfrog_llama3_deepspeed.log 2>&1 &
# Error loading data at index 10717: /data/lei/eagle3traindata/sharegpt_0_67999_mufp16/2/data_3250.ckpt