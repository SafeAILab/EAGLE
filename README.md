<img src="logo.png" alt="EAGLE" width="130" align="left"><div align="center"><h1>&nbsp;Lossless Acceleration of LLM Decoding by Feature Extrapolation</h1></div>

<p align="center">
| <a href="https://sites.google.com/view/
medusa-llm"><b>Blog</b></a> |
</p>

## Introduction

## Contents

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Inference
The inference code we provide automatically allocates model weights (loading a model across multiple GPUs), allowing you to run models that exceed the memoryof a single GPU.

### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m application.webui --ea-model-path [path of EAGLE's weight]\ 
		--base-model-path [path of the original model]\
		--model-type [vicuna or llama-2-chat]
```
### With code
You can use our provided "eagenerate" for speedup generation just like using 'generate' from Hugging Face. Here is an example.
```python
from model.ea_model import EaModel
model = EaModel.from_pretrained(  
    base_model_path=base_model_path,  
    ea_model_path=eainfer_model_path,  
    torch_dtype=torch.float16,  
    low_cpu_mem_usage=True,  
    device_map="auto"  
)
prompt="Hello"
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
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
## evaluation
You can test the speed of EaInfer on MT-bench using the following command.
```bash
python -m evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of eainfer's weight]\ 
		 --base-model-path [path of the original model]\
```
If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m evaluation.gen_baseline_answer_vicuna\
		(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of eainfer's weight]\ 
		 --base-model-path [path of the original model]\
```

## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [Medusa](https://github.com/FasterDecoding/Medusa), [FastChat](https://github.com/lm-sys/FastChat), and others.
