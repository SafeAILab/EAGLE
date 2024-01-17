import copy
from torch import Tensor
import torch
import torch.nn as nn
from .utils import *
from transformers import AutoTokenizer
import os
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download
from .llama_fast0 import Transformer as llamafast
from .eagle_fast import EAGLE as eaglefast,find_multiple
from pathlib import Path
from .choices import mc_sim_7b_63
from .utils_c import generate_tree_buffers as generate_tree_buffersc


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        # 计算并打印经过的时间
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')

top_k=10
class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.base_model_name_or_path = base_model_name_or_path
        tokenpath=str(Path(self.base_model_name_or_path).parent)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenpath)
        # tokenizer_path=str(Path(self.base_model_name_or_path).parent/"tokenizer.model")
        # self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
        self.ea_layer = eaglefast.from_pretrained(ea_model_path)


        #device = base_model.output.weight.device

        #self.ea_layer.to(self.base_model.dtype).to(device)
        self.init_tree()
        #self.base_model.setup_caches(1,2000)
        #self.ea_layer.setup_caches(1, 2000)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):


        base_model = llamafast.from_pretrained(
            base_model_path
        )


        #configpath=str(Path(ea_model_path).parent/"config.json")
        model = cls(
            base_model,
            base_model_path,
            ea_model_path
        )


        return model



    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffersc(self.tree,self.ea_layer.fc.weight.device)


    def reset(self):
        self.tree_mask=None

    def draft_many(self,hidden_states: Tensor,idx: Tensor, input_pos: Tensor):
        logits, hidden=self.ea_layer(hidden_states,idx,input_pos)
        return logits,hidden


    def draft_one(self,hidden_states_one: Tensor,idx_one: Tensor, input_pos_one: Tensor):
        logits_one, hidden_one=self.ea_layer(hidden_states_one,idx_one,input_pos_one)
        return logits_one,hidden_one


    def repeat_hidden(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
        return torch.cat(new_hidden,dim=1)


    def sample(self,logits, logits_processor,k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs,probabilities






    def chaintopK_genrate(self,hidden_states, input_ids, logits_processor,sk=3):
        input_ids = input_ids[:, 1:]
        init_pos=input_ids.shape[1]-hidden_states.shape[1]
        input_ids = input_ids.to(hidden_states.device)
        input_ids = input_ids[:, -hidden_states.shape[1]:]
        ss_token, ss_prob, ss_op = [], [], []
        len_posi = init_pos+1
        input_pos=torch.arange(init_pos,init_pos+input_ids.shape[1],device=input_ids.device)


        input_ids=input_ids.clone()
        hidden_states=hidden_states.clone()


        #with Timer("draft many"):
        logits,hidden = self.draft_many(hidden_states, input_ids,input_pos)


        kv_len=len_posi

        tl=logits[:, -1]

        for i in range(sk-1):

            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(tl, logits_processor, k=1, )
            else:
                top=torch.topk(tl, 1, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                op = None

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

            topk_index = topk_index.view(-1)
            select_index = topk_index
            # len_sq=select_index.shape[0]
            input_ids = select_index[None, :]
            if i == 0:
                hidden_states = hidden[:, -1:].clone()
            else:
                hidden_states = hidden.clone()


            kv_len+=input_ids.shape[1]

            position_ids = len_posi + self.tree_buffer["position_ids"][i]

            #with Timer("draft one"):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                logits,hidden = self.draft_one(hidden_states, input_ids,position_ids)
            tl=logits[0]
            len_posi += 1


        if logits_processor is not None:
            topk_index,topk_prob,op=self.sample(logits[0],logits_processor,k=1)
        else:
            top = torch.topk(logits[0], 1, dim=-1)
            topk_index, topk_prob = top.indices, top.values
            op=None

        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)

        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op)



    def base_forward(self,idx: Tensor, input_pos: Tensor):
        logits,hidden=self.base_model(idx,input_pos)
        return logits,hidden


    def base_forward_one(self,idx_one: Tensor, input_pos_one: Tensor):
        logits,hidden=self.base_model(idx_one,input_pos_one)
        return logits,hidden








    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2000,
            tree_choices=mc_sim_7b_63,

    ):
        current_length_data = 0
        device = self.base_model.output.weight.device
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        # max_block_size = min(max_length, max_new_tokens + input_len+10)
        max_block_size = max_length
        max_block_size = find_multiple(max_block_size, 8)

        print(max_block_size)


        if hasattr(self, "max_block_size") and self.max_block_size == max_block_size:
            causal_mask = self.causal_mask


        else:


            causal_mask = torch.tril(
                torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
            self.causal_mask = causal_mask

            with torch.device(device):
                self.ea_layer.setup_caches(1, max_block_size)
                self.base_model.setup_caches(1, max_block_size)


            self.max_block_size = max_block_size



        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=device
            )

        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices


        tree_logits, logits, hidden_state, sample_token = chaininitialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], logits_processor, causal_mask
        )
        new_token = 0

        for idx in range(max_new_tokens):


            candidates=torch.cat((sample_token,tree_logits[0].view(1,-1)),dim=1)
            if logits_processor is not None:
                candidates_prob=torch.cat((torch.tensor([[1]],dtype=torch.long,device=tree_logits[1].device)
                                           ,tree_logits[1].view(1,-1)),dim=1)
            else:
                candidates_prob=None

            logits, hidden_state_new = chaintree_decoding(
                self,
                candidates,
                input_ids,
            )
            best_candidate, accept_length, sample_p = chainevaluate_posterior(
                logits, candidates, logits_processor, candidates_prob, tree_logits[2]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = chainupdate_inference_inputs(
                input_ids,
                candidates,
                accept_length,
                logits_processor,
                new_token,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1]+25 > max_length:
                break

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2000,
            tree_choices=mc_sim_7b_63,
            log=False
    ):
        current_length_data = 0
        device = self.base_model.output.weight.device
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        # max_block_size = min(max_length, max_new_tokens + input_len+10)
        max_block_size = max_length
        max_block_size = find_multiple(max_block_size, 8)

        print(max_block_size)
        # from IPython import embed
        # embed()

        if hasattr(self, "max_block_size") and self.max_block_size == max_block_size:
            causal_mask = self.causal_mask


        else:

            # print("!!!!!!!!!!!")
            causal_mask = torch.tril(
                torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
            self.causal_mask = causal_mask

            with torch.device(device):
                self.ea_layer.setup_caches(1, max_block_size)
                self.base_model.setup_caches(1, max_block_size)

            self.max_block_size = max_block_size



        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=device
            )

        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices


        tree_logits, logits, hidden_state, sample_token = chaininitialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], logits_processor, causal_mask
        )
        new_token = 0

        for idx in range(max_new_tokens):

            candidates=torch.cat((sample_token,tree_logits[0].view(1,-1)),dim=1)
            if logits_processor is not None:
                candidates_prob=torch.cat((torch.tensor([[1]],dtype=torch.long,device=tree_logits[1].device)
                                           ,tree_logits[1].view(1,-1)),dim=1)
            else:
                candidates_prob=None

            logits, hidden_state_new = chaintree_decoding(
                self,
                candidates,
                input_ids,
            )
            best_candidate, accept_length, sample_p = chainevaluate_posterior(
                logits, candidates, logits_processor, candidates_prob, tree_logits[2]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = chainupdate_inference_inputs(
                input_ids,
                candidates,
                accept_length,
                logits_processor,
                new_token,
                self,
                hidden_state_new,
                sample_p
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1]+25 > max_length:
                break
        if log:
            return input_ids, new_token, idx
        else:
            return input_ids


    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,
    ):
        current_length_data = 0
        device = self.base_model.output.weight.device
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        max_block_size = max_length
        max_block_size = find_multiple(max_block_size, 8)
        if hasattr(self, "max_block_size") and self.max_block_size == max_block_size:
            causal_mask = self.causal_mask


        else:
            causal_mask = torch.tril(
                torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
            self.causal_mask = causal_mask

            with torch.device(device):
                self.ea_layer.setup_caches(1, max_block_size)
                self.base_model.setup_caches(1, max_block_size)

            self.max_block_size = max_block_size

        # if hasattr(self, "causal_mask") and self.tree_choices == tree_choices:
        #     causal_mask = self.causal_mask
        # else:
        #     causal_mask = torch.tril(
        #         torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
        #     self.causal_mask=causal_mask

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=device
            )

        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        input_len = input_ids.shape[1]

        input_pos = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        mask = causal_mask[None, None, input_pos]



        logits, hidden_states = self.base_model(input_ids, input_pos)

        new_token = 0

        for idx in range(max_new_tokens):
            if logits_processor is not None:
                logits = logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = logits[:, -1:].argmax(dim=-1)
            input_pos=torch.arange(input_ids.shape[1], input_ids.shape[1]+1, device=input_ids.device)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                #with Timer("naive base"):
                logits, hidden_states = self.base_forward_one(input_id, input_pos)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1]+25 > max_length:
                break



    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2000,
            tree_choices=mc_sim_7b_63,
            log=False
    ):
        current_length_data = 0
        device = self.base_model.output.weight.device
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        max_block_size = max_length
        max_block_size = find_multiple(max_block_size, 8)
        if hasattr(self, "max_block_size") and self.max_block_size == max_block_size:
            causal_mask = self.causal_mask


        else:
            causal_mask = torch.tril(
                torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
            self.causal_mask = causal_mask

            with torch.device(device):
                self.ea_layer.setup_caches(1, max_block_size)
                self.base_model.setup_caches(1, max_block_size)


            self.max_block_size = max_block_size

        # if hasattr(self, "causal_mask") and self.tree_choices == tree_choices:
        #     causal_mask = self.causal_mask
        # else:
        #     causal_mask = torch.tril(
        #         torch.ones(max_block_size, max_block_size, dtype=torch.bool, device=device))
        #     self.causal_mask=causal_mask

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=device
            )
            # tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            #     self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        input_len = input_ids.shape[1]

        input_pos = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        mask = causal_mask[None, None, input_pos]



        logits, hidden_states = self.base_model(input_ids, input_pos)

        new_token = 0

        for idx in range(max_new_tokens):
            if logits_processor is not None:
                logits = logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = logits[:, -1:].argmax(dim=-1)
            input_pos=torch.arange(input_ids.shape[1], input_ids.shape[1]+1, device=input_ids.device)
            #with Timer("naive base"):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                logits, hidden_states = self.base_forward_one(input_id, input_pos)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1]+25 > max_length:
                break
        if log:
            return input_ids, new_token, idx
        else:
            return input_ids