import argparse


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template


modelname="meta-llama/Llama-3.2-3B-Instruct"



def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files="training_data/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            # messages = [
            #     {"role": "system",
            #      "content": "You are a helpful assistant."},
            # ]
            # convroles=["user","assistant"]
            # roles = {"human": "user", "gpt": "assistant"}
            # source= examples['conversations'][i]
            # if roles[source[0]["from"]] != "user":
            #     # Skip the first one if it is not from human
            #     source = source[1:]
            # for j, sentence in enumerate(source):
            #     role = roles[sentence["from"]]
            #     assert role == convroles[j % 2], f"{i}"
            #     # if sentence["from"]=="gpt":
            #     #     sentence["value"]=" "+sentence["value"]
            #     messages.append(
            #         {"role": role, "content": sentence["value"]}
            #     )
            # conversation=tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=False,
            # )

            conv = get_conversation_template("llama-3-instruct")
            
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source = examples['conversations'][i]
            
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"]=="gpt":
                    sentence["value"]=" "+sentence["value"]
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            # if i==56:
            #     print(i)
            # if i==57:
            #     print(i)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)

            # sep = "<|im_end|>\n<|im_start|>assistant\n"
            sep = conv.sep + conv.roles[1] + " "
            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())
            print(total_len)

            turns = conversation.split(conv.sep2)


            # sep2="<|im_end|>\n<|im_start|>user\n"
            # turns = conversation.split(sep2)

            # turns[1]=turns[0]+sep2+turns[1]
            # turns=turns[1:]


            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids)

                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len+=2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0
            #     if i==0:
            #         loss_mask[0: cur_len + instruction_len-2] = 0
            #     else:
            #         loss_mask[cur_len-6: cur_len + instruction_len-2] = 0
            #     cur_len += turn_len
            #     cur_len += 5


            # loss_mask[cur_len:] = 0



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        #num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1

tokenizer = AutoTokenizer.from_pretrained(modelname,use_fast=False)
dataset = build_dataset_rank(tokenizer)

model = AutoModelForCausalLM.from_pretrained(modelname,  device_map="auto",torch_dtype=torch.float16)
model.eval()

@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    # probs = torch.softmax(outs_big.logits, dim=-1)
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id,data in enumerate(dataset):
    if id%100==0:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir,outdata)
