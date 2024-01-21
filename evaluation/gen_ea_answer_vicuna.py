"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from model.ea_model import EaModel
from model.kv_cache import initialize_past_key_values
from model.utils import *
from model.choices import *





def run_eval(
        bs,
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1


    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                bs,
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
            )
        )


def batchgenerate(
        questions,
        model,
        temperature,
        model_id,
        num_choice=3,
        save=False,
):
    tokenizer=model.tokenizer
    bs = len(questions)
    choices_l = [[] for _ in range(bs)]
    for i in range(num_choice):
        torch.manual_seed(i)

        conv_l=[]
        turns_l=[[] for _ in range(bs)]
        idxs_l=[[] for _ in range(bs)]
        new_tokens_l=[[] for _ in range(bs)]
        wall_time_l=[[] for _ in range(bs)]
        for b in range(bs):
            conv = get_conversation_template("vicuna")
            conv_l.append(conv)


        for j in range(len(questions[0]["turns"])):
            prompts=[]
            for b in range(bs):
                question=questions[b]
                conv = conv_l[b]


                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)

            input_s = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids_l, new_token_l, idx_l, times_l = model.eagenerate(input_s.input_ids, input_s.attention_mask, temperature=temperature,log=True)

            torch.cuda.synchronize()
            total_time = time.time() - start_time

            for b in range(bs):
                output_ids, new_token, idx=output_ids_l[b],new_token_l[b],idx_l[b]
                conv=conv_l[b]

                output_ids = output_ids[len(input_s.input_ids[0]):]


                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )


                conv.stop_str = "</s>"
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                turns_l[b].append(output)
                idxs_l[b].append(int(idx))
                new_tokens_l[b].append(int(new_token))
                wall_time_l[b].append(times_l[b])
                conv.messages[-1][-1] = output
        if save:
            for b in range(bs):
                choices_l[b].append({"index": i, "turns": turns_l[b], "idxs": idxs_l[b], "new_tokens": new_tokens_l[b], "wall_time": wall_time_l[b]})

    if save:
        for b in range(bs):
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": questions[b]["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices_l[b],
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")






@torch.inference_mode()
def get_model_answers(
        bs,
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id




    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    #warmup

    question_l=[]
    for question in questions:
        question_l.append(question)
        if len(question_l)==bs:
            batchgenerate(
                question_l,
                model,
                temperature,
                model_id,
            )
            break

    print('Warmup done')

    # questions=questions[6:]
    question_l = []
    for question in tqdm(questions):

        question_l.append(question)
        if len(question_l) == bs:
            batchgenerate(
                question_l,
                model,
                temperature,
                model_id,
                save=True,
                num_choice=num_choices,

            )
            question_l=[]




def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/root/eagle/7B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/root/7B",
                        help="1")
    parser.add_argument("--bs", type=int, default=2,
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-vicuna-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)+"-bs-"+str(args.bs)
    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.bs,
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.tree_choices,
    )

    reorg_answer_file(answer_file)
