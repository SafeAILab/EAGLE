<img src="figs/logo.png" alt="EAGLE" width="220" align="left"><div align="center"><h1>&nbsp;EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty</h1></div>

<p align="center">
| <a href="https://arxiv.org/pdf/2401.15077.pdf"><b>Paper</b></a> | <a href="https://sites.google.com/view/
eagle-llm"><b>Blog</b></a> |
</p>


<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Version-v1.1.0-orange.svg" alt="Version">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/SafeAILab/EAGLE/issues">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/SafeAILab/EAGLE/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>


##

<p align="center">
  <img src="./figs/benchmark.png" alt="benchmark" width="790">
</p>

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is a new baseline for fast decoding of Large Language Models (LLMs) with provable performance maintenance. This approach involves extrapolating the second-top-layer contextual feature vectors of LLMs, enabling a significant boost in generation efficiency. EAGLE is building upon the following First Principle:

**The sequence of LLM feature vectors is compressible over time, making the prediction of subsequent feature vectors from previous ones easy.**

- EAGLE is:
	- achieving **2x** speedup on gpt-fast, one of the **fastest**-known open-sourced inferences.
	- **3x** faster than vanilla decoding (13B).
 	- **2x** faster than <a href="https://lmsys.org/blog/2023-11-21-lookahead-decoding/"><b>Lookahead</b></a> (13B).
 	- **1.6x** faster than <a href="https://sites.google.com/view/medusa-llm"><b>Medusa</b></a> (13B).
  	- provably maintaining the consistency with vanilla decoding in the distribution of generated texts.
  	- trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs. So even the GPU poor can afford it.
	- combinable with other parallelled techniques such as vLLM, DeepSpeed, Mamba, FlashAttention, quantization, and hardware optimization.

<p align="center">
  <img src="./figs/demosmall.gif" alt="demogif">
</p>

_Inference is conducted on RTX 3090 GPUs at fp16 precision using the Vicuna 33B model. For an enhanced viewing experience, the animation has been sped up fourfold._

## Update
**2023.12.8**: EAGLE v1.0 is released.

**2024.1.15**: We now support [batch size > 1](#batch-size--1) generation.

**2024.1.17**: We have integrated [gpt-fast](https://github.com/pytorch-labs/gpt-fast) into EAGLE, [further accelerating](https://github.com/SafeAILab/EAGLE/tree/eaglefast) the generation speed.

**2024.1.17**: We now support  [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

## Todo
- [x] Support non-greedy inference (provably maintaining text distribution).
- [x] Support bs > 1.
- [x] Support gpt-fast.
- [x] Support more LLMs such as Mixtral 8x7B.
- [ ] Support vLLM.


## Contents

- [Setup & Installation](#setup--installation)
- [EAGLE Weights](#eagle-weights)
- [Inference](#inference)
  - [With UI](#with-ui)
  - [With Code](#with-code)
  - [Batch size > 1](#batch-size--1)
- [Train](#train)
  - [Generate Train Data](#generate-train-data)
  - [Train the Auto-regression Head](#train-the-auto-regression-head)
  - [Inference on custom models](#inference-on-custom-models)
- [Evaluation](#evaluation)
- [With gpt-fast](#with-gpt-fast)
  - [Setup](#setup)
  - [Quantizing Weights](quantizing-weights)
  - [Modifying Path](modifying-path)

## Setup & Installation

```bash
pip install -r requirements.txt
```

## EAGLE Weights

| Base Model  | EAGLE on Hugging Face  | \# EAGLE Parameters | Base Model  | EAGLE on Hugging Face  | \# EAGLE Parameters |
|------|------|------|------|------|------|
| Vicuna-7B-v1.3 | [yuhuili/EAGLE-Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3) | 0.24B | LLaMA2-Chat 7B | [yuhuili/EAGLE-llama2-chat-7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B) | 0.24B |
| Vicuna-13B-v1.3 | [yuhuili/EAGLE-Vicuna-13B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-13B-v1.3) | 0.37B | LLaMA2-Chat 13B | [yuhuili/EAGLE-llama2-chat-13B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-13B) | 0.37B |
| Vicuna-33B-v1.3 | [yuhuili/EAGLE-Vicuna-33B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-33B-v1.3)| 0.56B | LLaMA2-Chat 70B| [yuhuili/EAGLE-llama2-chat-70B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-70B)| 0.99B |
| Mixtral-8x7B-Instruct-v0.1 | [yuhuili/EAGLE-mixtral-instruct-8x7B](https://huggingface.co/yuhuili/EAGLE-mixtral-instruct-8x7B)| 0.28B |


## Inference
The inference code we provide automatically allocates model weights (loading a model across multiple GPUs), allowing you to run models that exceed the memory of a single GPU.

### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m application.webui --ea-model-path [path of EAGLE weight]\ 
		--base-model-path [path of the original model]\
		--model-type [vicuna or llama-2-chat]
```
### With Code
You can use our provided "eagenerate" for speedup generation just like using 'generate' from Hugging Face. Here is an example.
```python
from model.ea_model import EaModel
from fastchat.model import get_conversation_template
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

your_message="Hello"

if use_llama_2_chat:
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

if use_vicuna:
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
```

**_Note: Vicuna and LLaMA2-Chat are both chat models. You need to use the correct chat template, otherwise it will cause abnormal output from the model and affect the performance of EAGLE._**

### Batch size > 1

Switch to the *bsne1* branch.

```bash
git checkout bsne1
```
Here is an example. Note that left padding is needed.
```python
from model.ea_model import EaModel
from fastchat.model import get_conversation_template

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
# left padding
model.eval()
model.tokenizer.padding_side = "left"
model.tokenizer.pad_token = model.tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

your_message="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
conv = get_conversation_template("llama-2-chat")
conv.system_message = sys_p
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt1 = conv.get_prompt()+" "

your_message="Hello"
conv = get_conversation_template("llama-2-chat")
conv.system_message = sys_p
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt2 = conv.get_prompt()+" "

input_s=model.tokenizer([prompt1,prompt2],return_tensors="pt",padding=True).to("cuda")
output_ids=model.eagenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15)
output=model.tokenizer.batch_decode(output_ids)
print(output)

# vanilla auto-regression
# output_ids, new_token, idx=model.naivegenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15,log=True)
```

## Train

### Generate Train Data
You can run the following command to generate the training data.
```bash
python -m ge_data.allocation --outdir [path of data]
```
### Train the Auto-regression Head
```bash
cd train
accelerate launch --mixed_precision=bf16 main.py --tmpdir [path of data]\
--cpdir [path of checkpoints]
```

### Inference on custom models

If the original LLM structure differs from LLaMA and Mixtral, you can utilize EAGLE in two ways.

#### 1. Using the generic modeling_eagle.py

This approach directly encapsulates the native Transformers LLM. Here is an example. **Note: transformers version should be higher than 4.36.**

```python
from modeling_eagle import EAGLE
from transformers import AutoModelForCausalLM,AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained(base_model_path)
model=AutoModelForCausalLM.from_pretrained("base_model_path",torch_dtype=torch.float16,device_map="auto",)
# for bs>1, the padding side should be right
if bs>1:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

text=prompt1
# text=[prompt1,prompt2]
inputs = tokenizer(text, return_tensors="pt",padding=True)

eagle=EAGLE(model,eagle_path)
outs=eagle.generate(**inputs, max_new_tokens=200,temperature=0.0)
output=tokenizer.decode(outs)
# output=tokenizer.batch_decode(outs)
```

#### 2. Modifying the code of the model

Copy the modeling_basemodelname.py from the Transformers library and proceed to make modifications to leverage the pre-allocated kv_cache for enhanced speed in the base model. You can refer to model/modeling_llama_kv.py for guidance, where places that require modifications are annotated with # [MODIFIED]. These modifications are minimal.


## Evaluation
You can test the speed of EAGLE on MT-bench using the following command.
```bash
python -m evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
```
If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m evaluation.gen_baseline_answer_vicuna\
		(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.

## With gpt-fast

GPT-Fast primarily accelerates generation through quantization and compilation, which we have integrated into EAGLE. Here is the result of an experiment conducted on MT-bench with a single RTX3090, using LLaMA2-chat 7B.

| Precision 	    | fp16      | int4      |
|-------------------|-----------|-----------|
| vanilla          | 24.5 tokens/s     | N/A     |
| gpt-fast          | 55.1 tokens/s      | 106.9 tokens/s     |
| EAGLE+gpt-fast    | 100.2 tokens/s    | 160.4 tokens/s    |



<p align="center">
  <img src="./figs/eaglefast.gif" alt="demogif">
</p>

_Inference is conducted on a single RTX3090 GPU at int4 precision using the LLaMA2-chat 7B model. No additional training required._

In EAGLE, using gpt-fast only requires three steps: setting up the environment, quantizing weights, and modifying the model path.

### Setup

Switch to the *eaglefast* branch.

```bash
git checkout eaglefast
```

Install the Preview (Nightly) version of PyTorch with CUDA 12.1, do not use "pip install torch" as it installs the Stable version, which lacks some of the new features used by gpt-fast. 

_This is a requirement for gpt-fast, whereas other branches of eagle can use the Stable version of PyTorch._

### Quantizing Weights

Convert Huggingface weights to the format required by gpt-fast.

```bash
python convert/convert_hf_checkpoint.py --checkpoint_dir path_of_base_model
python convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir path_of_eagle
```

Quantize weights.

```bash
python -m model.quantize_llama --checkpoint_path path_of_base_model/model.pth
python -m model.quantize_EAGLE --checkpoint_path path_of_eagle/model.pth
```

### Modifying Path

When specifying the model weights (including the base model and EAGLE), change "path" to "path/model_int4.g32.pth".

## 🌟 Our Contributors

A heartfelt thank you to all our contributors.

![Contributors](https://contrib.rocks/image?repo=SafeAILab/EAGLE)


## Reference
For technical details and full experimental results, please check [the paper](https://arxiv.org/pdf/2401.15077.pdf).
```
@article{li2024eagle, 
	author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
	title = {EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty}, 
	journal = {arXiv preprint arXiv:2401.15077},
	year = {2024}
}
```

## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [Medusa](https://github.com/FasterDecoding/Medusa), [FastChat](https://github.com/lm-sys/FastChat), and others. The logo is designed by GPT-4. We also appreciate many valuable discussions with Tianle Cai, Hao Zhang, Ziteng Sun, and others.
