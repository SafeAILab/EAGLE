from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
base_model_path = '/home/yzq/cuda_learning/EAGLE/eagle/Qwen/Qwen3-1.7B' # https://huggingface.co/Qwen/Qwen3-1.7B
EAGLE_model_path = '/home/yzq/cuda_learning/EAGLE/eagle/Qwen3-4B_eagle3' # https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
your_message="Hello"
conv = get_conversation_template("qwen")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=1024)
output=model.tokenizer.decode(output_ids[0])
print(output)