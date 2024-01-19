import copy

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download




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
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        self.ea_layer = Model(config)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.device=device
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):

        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1],dim=-1)
                token = token[:,None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states

            ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor,attention_mask=attention_mask)
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token
            return ea_logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            attention_mask=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,
            log=False,

    ):

        bs = input_ids.shape[0]
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers

        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
            tree_buffers["tree_position_ids"]=tree_buffers["tree_position_ids"].to(self.base_model.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        tree_buffers["retrieve_indices_batch"]=tree_buffers["retrieve_indices"].expand(bs,-1,-1)

        bool_mask = attention_mask.bool()

        #out_inputids = [ids[mask].tolist() for ids, mask in zip(input_ids, bool_mask)]
        out_inputids = [ids.tolist() for ids, mask in zip(input_ids, bool_mask)]

        #out_inputids = [[]]*bs
        out_idx=[0]*bs
        out_newtokens=[0]*bs

        # if self.ea_layer.tree_buffer["bs"]!=bs:
        #     self.ea_layer.init_tree(bs=bs)

        # Initialize the past key and value states
        if hasattr(self, "past_key_values") and self.past_key_values[0][0].shape[0]==bs:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,bs=bs)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor,attention_mask=attention_mask
        )
        new_token = [0]*bs
        finish_flag=[False]*bs

        for idx in range(max_length):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
                attention_mask=attention_mask
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"],finish_flag
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token,attention_mask,newfinish_flag,new_outs = update_inference_inputs(
                input_ids,
                attention_mask,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices_batch"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p,
                finish_flag
            )
            min_uf_newtokens=max_length+10
            for batch in range(bs):
                if not finish_flag[batch]:
                    out_idx[batch]+=1
                    out_newtokens[batch]=new_token[batch]
                    out_inputids[batch].extend(new_outs[batch])
                    min_uf_newtokens=min(min_uf_newtokens,new_token[batch])
                # if finish_flag[batch]!=newfinish_flag[batch]:
                #     out_inputids[batch]=input_ids[batch].tolist()
            finish_flag=newfinish_flag

            if min(finish_flag):
                break
            if min_uf_newtokens > max_new_tokens:
                break
            if input_ids.shape[1]+10+len(tree_choices) > max_length:
                break

        if log:
            return out_inputids, out_newtokens, out_idx
        else:
            return out_inputids



    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            attention_mask=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,
            log=False,

    ):

        bs = input_ids.shape[0]
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers

        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        tree_buffers["retrieve_indices_batch"]=tree_buffers["retrieve_indices"].expand(bs,-1,-1)

        bool_mask = attention_mask.bool()

        #out_inputids = [ids[mask].tolist() for ids, mask in zip(input_ids, bool_mask)]
        out_inputids = [ids.tolist() for ids, mask in zip(input_ids, bool_mask)]
        #out_inputids = [[]]*bs
        out_idx=[0]*bs
        out_newtokens=[0]*bs

        # if self.ea_layer.tree_buffer["bs"]!=bs:
        #     self.ea_layer.init_tree(bs=bs)

        # Initialize the past key and value states
        if hasattr(self, "past_key_values") and self.past_key_values[0][0].shape[0]==bs:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,bs=bs)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)



        new_token = [0]*bs
        finish_flag=[False]*bs
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        zero_num=position_ids.shape[1]-position_ids.max(dim=-1).values-1
        zero_num=zero_num[:,None]
        len_posi = input_ids.shape[1]

        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True,attention_mask=attention_mask,position_ids=position_ids)

        for idx in range(max_length):

            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            input_ids = torch.cat([input_ids, input_id], dim=-1)
            position_ids=torch.ones(bs,1,dtype=torch.long,device=input_ids.device)*len_posi-zero_num
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(input_id, device=attention_mask.device, dtype=attention_mask.dtype)),
                dim=1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values,attention_mask=attention_mask,position_ids=position_ids)

            len_posi+=1

            newfinish_flag=(input_id==self.tokenizer.eos_token_id).squeeze().tolist()

            min_uf_newtokens = max_length + 10
            for batch in range(bs):
                if not finish_flag[batch]:
                    out_idx[batch]+=1
                    out_newtokens[batch]+=1
                    out_inputids[batch].append(input_id[batch].item())
                    min_uf_newtokens = min(min_uf_newtokens, new_token[batch])
                    finish_flag[batch]=newfinish_flag[batch]


            if min(finish_flag):
                break
            if min_uf_newtokens > max_new_tokens:
                break
            if input_ids.shape[1]+10+len(tree_choices) > max_length:
                break

        if log:
            return out_inputids, out_newtokens, out_idx
        else:
            return out_inputids

