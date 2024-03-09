import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_Mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download



class dmodel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d=d
    def forward(self, input_ids,position_ids, **kwargs):
        #bs,seq_len=input_ids.shape
        x=self.d[position_ids][None]
        return x

class EaModel(nn.Module):

    def __init__(
            self,
            based,
            draftd,
    ):

        super().__init__()
        self.base_model = dmodel(based)
        self.ea_layer = Model(draftd)
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
            based,
            draftd,
    ):


        model = cls(
            based,
            draftd,
        )


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
            outputs = None
            if output_orig:
                orig = self.base_model(input_ids, past_key_values=past_key_values, position_ids=position_ids,use_cache=True)
            hidden_states = None
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                # logits = logits_processor(None, logits)
                probabilities = logits
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states

            ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, None, logits_processor)
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
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,

    ):
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
                tree_choices, device="cpu"
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                "cpu")
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = None,None,None
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        #input_len = input_ids.shape[1]
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

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
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids


    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
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

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
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
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_steps):
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
