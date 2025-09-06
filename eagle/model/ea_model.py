import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
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
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
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
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    # @torch.no_grad()
    # def eagenerate(
    #         self,
    #         input_ids,
    #         temperature=0.0,
    #         top_p=0.0,
    #         top_k=0.0,
    #         max_new_tokens=512,
    #         max_length=2048,
    #         log=False,
    #         is_llama3=False,

    # ):
    #     if is_llama3:
    #         stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


    #     if temperature > 1e-5:
    #         logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
    #     else:
    #         logits_processor = None
    #     # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    #     # Avoid modifying the input_ids in-place

    #     padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    #     input_ids = input_ids.clone()
    #     self.ea_layer.reset_kv()

    #     # Initialize the past key and value states
    #     if hasattr(self, "past_key_values"):
    #         past_key_values = self.past_key_values
    #         past_key_values_data = self.past_key_values_data
    #         current_length_data = self.current_length_data
    #         # Reset the past key and value states
    #         current_length_data.zero_()
    #     else:
    #         (
    #             past_key_values,
    #             past_key_values_data,
    #             current_length_data,
    #         ) = initialize_past_key_values(self.base_model,max_length=max_length)
    #         self.past_key_values = past_key_values
    #         self.past_key_values_data = past_key_values_data
    #         self.current_length_data = current_length_data

    #     input_len = input_ids.shape[1]
    #     reset_tree_mode(self)
    #     # prefill
    #     draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, draft_log_scores = initialize_tree(
    #         input_ids, self, past_key_values, logits_processor
    #     )
    #     new_token = 0
    #     max_length = max_length - self.ea_layer.total_tokens - 10
    #     for idx in range(max_length):
    #         # with Timer("all"):
    #         self.base_model.model.tree_mask = tree_mask

    #         draft_tokens = draft_tokens.to(input_ids.device)
    #         # Target model forward, get logits
    #         logits, hidden_state_new, outputs = tree_decoding(
    #             self,
    #             draft_tokens,
    #             past_key_values,
    #             tree_position_ids,
    #             input_ids,
    #             retrieve_indices,
    #         )
    #         # retrieve_indices=tree_buffers["retrieve_indices"]
    #         # logits = logits[0, retrieve_indices]
    #         draft_tokens = torch.cat((draft_tokens, padding), dim=1)
    #         candidates = draft_tokens[0, retrieve_indices]
    #         # verification
    #         best_candidate, accept_length, sample_p = evaluate_posterior(
    #             logits, candidates, logits_processor
    #         )
    #         # print(accept_length)
    #         # Adjusting the input sequence, draft model forward
    #         input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
    #             input_ids,
    #             candidates,
    #             best_candidate,
    #             accept_length,
    #             retrieve_indices,
    #             logits_processor,
    #             new_token,
    #             past_key_values_data,
    #             current_length_data,
    #             self,
    #             hidden_state_new,
    #             sample_p
    #         )

    #         if is_llama3:
    #             if stop_token_id in input_ids[0, input_len:].tolist():
    #                 break

    #         if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
    #             break
    #         if new_token > max_new_tokens:
    #             break
    #         if input_ids.shape[1] > max_length:
    #             break
    #     if not log:
    #         return input_ids
    #     else:
    #         return input_ids, new_token, idx

    # NOTE: remove any @torch.no_grad() decorator above this function
    def eagenerate(
            self,
            input_ids=None,
            inputs_embeds=None,          # optional: continuous embeddings for prompt (for external optimization)
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            enable_grad=False,          # if True keep grad-enabled (for external backprop)
            return_step_scores=False,   # if True return per-step (draft_logp_first, verify_logp_first) lists
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
    
        # Input handling: require exactly one of input_ids or inputs_embeds
        assert (input_ids is None) ^ (inputs_embeds is None), "Provide exactly one of input_ids or inputs_embeds"
    
        device = None
        if input_ids is not None:
            device = input_ids.device
        else:
            device = inputs_embeds.device
    
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
    
        # copy input_ids if provided (do not detach or clone inputs_embeds)
        if input_ids is not None:
            input_ids = input_ids.clone()
    
        self.ea_layer.reset_kv()
    
        # Initialize past key-values
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (past_key_values, past_key_values_data, current_length_data) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
    
        if input_ids is not None:
            input_len = input_ids.shape[1]
        else:
            input_len = inputs_embeds.shape[1]
    
        reset_tree_mode(self)
    
        # PREFILL: initialize_tree must be updated to accept inputs_embeds (if provided)
        # and to return draft_log_scores (and any other draft score arrays)
        # We assume initialize_tree returns:
        # (draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
        #  logits, hidden_state, sample_token, draft_log_scores)
        # where draft_log_scores is shape [1, total_tokens] aligned with draft_tokens indices.
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, draft_log_scores = initialize_tree(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            model=self,
            past_key_values=past_key_values,
            logits_processor=logits_processor
        )
    
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
    
        # Prepare containers for external scoring
        step_draft_logps = []   # list of tensors (scalars) or floats (kept as tensors if enable_grad True)
        step_verify_logps = []
    
        # Use grad enabled depending on enable_grad
        torch.set_grad_enabled(enable_grad)
        try:
            for idx in range(max_length):
                self.base_model.model.tree_mask = tree_mask
    
                draft_tokens = draft_tokens.to(device)
    
                # Target model forward -> tree_decoding MUST accept inputs_embeds if provided
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    retrieve_indices=retrieve_indices,
                )
                # logits shape: [num_candidates, seq_len, vocab]
    
                # prepare candidates matrix
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]  # [num_candidates, seq_len]
    
                # --- compute verifier log-probs (log-space) ---
                log_probs = torch.log_softmax(logits, dim=-1)   # shape [num_candidates, seq_len, vocab]
    
                # first EA-generated token for candidate j is candidates[j, 1]
                first_ea_token_ids = candidates[:, 1].to(log_probs.device)    # shape [num_candidates]
    
                # verifier log-prob for first EA token per candidate: logits[:,0] predicts candidates[:,1]
                logp_verify_first_per_candidate = log_probs[torch.arange(candidates.shape[0], device=log_probs.device), 0, first_ea_token_ids]
                # shape [num_candidates]
    
                # --- compute draft first-token log-prob per candidate from draft_log_scores ---
                # we assume draft_log_scores is aligned with draft_tokens indices;
                # The first EA token index for candidate j is retrieve_indices[j, 1].
                # So:
                #   draft_first_logp_per_candidate = draft_log_scores[0, retrieve_indices[:,1]]
                # (make sure dtype/device align)
                draft_log_scores = draft_log_scores.to(device)
                first_indices = retrieve_indices[:, 1].to(device)   # shape [num_candidates]
                draft_first_logp_per_candidate = draft_log_scores[0, first_indices]  # shape [num_candidates]
    
                # Now call evaluate_posterior to pick best candidate and accept_length and sample_p
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                # make best index python int for indexing
                best_idx = int(best_candidate) if isinstance(best_candidate, torch.Tensor) else best_candidate
    
                # pick the two scalar log-probs for the selected candidate
                logp_verify_first = logp_verify_first_per_candidate[best_idx]        # tensor (requires_grad if upstream)
                logp_draft_first = draft_first_logp_per_candidate[best_idx]         # tensor (requires_grad if upstream)
    
                # store them (keep tensors — they will carry grad if enable_grad=True)
                if return_step_scores:
                    step_draft_logps.append(logp_draft_first)
                    step_verify_logps.append(logp_verify_first)
    
                # Update inference state as usual. update_inference_inputs may need inputs_embeds handling
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids=input_ids,
                    candidates=candidates,
                    best_candidate=best_candidate,
                    accept_length=accept_length,
                    retrieve_indices=retrieve_indices,
                    logits_processor=logits_processor,
                    new_token=new_token,
                    past_key_values_data_list=past_key_values_data,
                    current_length_data=current_length_data,
                    model=self,
                    hidden_state_new=hidden_state_new,
                    sample_p=sample_p
                )
    
                # stopping criteria
                if is_llama3:
                    if stop_token_id in input_ids[0, input_len:].tolist():
                        break
                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if new_token > max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        finally:
            # restore grad mode
            torch.set_grad_enabled(True)
    
        # Convert step lists to tensors if desired (stack)
        if return_step_scores:
            # stack into tensors (shape [num_steps])
            # if enable_grad=True these tensors will require grad if their elements do
            draft_scores_tensor = torch.stack(step_draft_logps) if len(step_draft_logps) > 0 else torch.empty(0, device=device)
            verify_scores_tensor = torch.stack(step_verify_logps) if len(step_verify_logps) > 0 else torch.empty(0, device=device)
    
        # Return values
        if not log:
            if return_step_scores:
                return input_ids, draft_scores_tensor, verify_scores_tensor
            else:
                return input_ids
        else:
            if return_step_scores:
                return input_ids, new_token, idx, draft_scores_tensor, verify_scores_tensor
            else:
                return input_ids, new_token, idx


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

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
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

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
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

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
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
