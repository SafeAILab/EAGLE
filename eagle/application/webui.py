import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import gradio as gr
import argparse
try:
    from ..model.ea_model import EaModel_lpf
except:
    from eagle.model.ea_model import EaModel_lpf
import torch
from fastchat.model import get_conversation_template
import re


def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]





def find_list_markers(text):

    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)


    return [(match.start(), match.end()) for match in matches]


def checkin(pointer,start,marker):
    for b,e in marker:
        if b<=pointer<e:
            return True
        if b<=start<e:
            return True
    return False

def highlight_text(text, text_list,color="black"):

    pointer = 0
    result = ""
    markers=find_list_markers(text)


    for sub_text in text_list:

        start = text.find(sub_text, pointer)
        if start==-1:
            continue
        end = start + len(sub_text)


        if checkin(pointer,start,markers):
            result += text[pointer:start]
        else:
            result += f"<span style='color: {color};'>{text[pointer:start]}</span>"

        result += sub_text

        pointer = end

    if pointer < len(text):
        result += f"<span style='color: {color};'>{text[pointer:]}</span>"

    return result


def warmup(model):
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if args.model_type == "llama-2-chat":
        prompt += " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    for output_ids in model.ea_generate(input_ids):
        ol=output_ids.shape[1]

def bot(history, temperature, top_p, use_EaInfer, highlight_EaInfer,session_state,):
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    assert args.model_type == "llama-2-chat" or "vicuna"
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"
    elif args.model_type == "llama-3-instruct":
        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

    for query, response in pure_history:
        if args.model_type == "llama-3-instruct":
            messages.append({
                "role": "user",
                "content": query
            })
            if response!=None:
                messages.append({
                    "role": "assistant",
                    "content": response
                })
        else:
            conv.append_message(conv.roles[0], query)
            if args.model_type == "llama-2-chat" and response:
                response = " " + response
            conv.append_message(conv.roles[1], response)

    if args.model_type == "llama-3-instruct":
        prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = conv.get_prompt()

    if args.model_type == "llama-2-chat":
        prompt += " "

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    naive_text = []
    cu_len = input_len
    totaltime=0
    start_time=time.time()
    total_ids=0
    if use_EaInfer:

        for output_ids in model.ea_generate(input_ids, temperature=temperature, top_p=top_p,
                                            max_new_tokens=args.max_new_token,is_llama3=args.model_type=="llama-3-instruct"):
            totaltime+=(time.time()-start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            if args.model_type == "llama-3-instruct":
                decode_ids = truncate_list(decode_ids, model.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )

            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))

            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_EaInfer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cu_len-input_len
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()


    else:
        for output_ids in model.naive_generate(input_ids, temperature=temperature, top_p=top_p,
                                            max_new_tokens=args.max_new_token,is_llama3=args.model_type=="llama-3-instruct"):
            totaltime += (time.time() - start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))
            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_EaInfer and use_EaInfer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            history[-1][1] = text
            pure_history[-1][1] = text
            new_tokens = cu_len - input_len
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()


def user(user_message, history,session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]],session_state


def regenerate(history,session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00",session_state
    history[-1][1] = None
    return history, None,"0.00 tokens/s","0.00",session_state


def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state




parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea-model-path",
    type=str,
    default="/home/lyh/weights/l38b/",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama3chat/8B/",
                    help="path of basemodel, huggingface project or local path")
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument("--model-type", type=str, default="llama-3-instruct",choices=["llama-2-chat","vicuna","mixtral","llama-3-instruct"])
parser.add_argument(
    "--total-token",
    type=int,
    default=60,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=512,
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

model = EaModel_lpf.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    total_token=args.total_token,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    device_map="auto",
)
model.eval()
warmup(model)

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": []})
    gr.Markdown('''## EAGLE-2 Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        compression_box = gr.Textbox(label="Compression Ratio", elem_id="speed", interactive=False, value="0.00")
    with gr.Row():
        with gr.Column():
            use_EaInfer = gr.Checkbox(label="Use EAGLE-2", value=True)
            highlight_EaInfer = gr.Checkbox(label="Highlight the tokens generated by EAGLE-2", value=True)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0.5)
        top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="top_p", value=0.9)
    note=gr.Markdown(show_label=False,interactive=False,value='''The Compression Ratio is defined as the number of generated tokens divided by the number of forward passes in the original LLM. If "Highlight the tokens generated by EAGLE-2" is checked, the tokens correctly guessed by EAGLE-2 
    will be displayed in orange. Note: Checking this option may cause special formatting rendering issues in a few cases, especially when generating code''')


    chatbot = gr.Chatbot(height=600,show_label=False)


    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")
    enter_event=msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, temperature, top_p, use_EaInfer, highlight_EaInfer,gs], [chatbot,speed_box,compression_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,compression_box,gs], queue=True)

    send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
        bot, [chatbot, temperature, top_p, use_EaInfer, highlight_EaInfer,gs], [chatbot,speed_box,compression_box,gs]
    )
    regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box,compression_box,gs],queue=True).then(
        bot, [chatbot, temperature, top_p, use_EaInfer, highlight_EaInfer,gs], [chatbot,speed_box,compression_box,gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,regenerate_event,enter_event])
demo.queue()
demo.launch(share=True)
