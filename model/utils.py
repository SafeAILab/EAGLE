import copy
import random

import torch
import time
# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

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

def prepare_logits_processor(
        temperature: float = 0.0, 
        repetition_penalty: float = 0.0, 
        top_p: float = 0.0, 
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature>1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    return tree_buffers



def chaininitialize_tree(input_ids, model, tree_attn_mask,  logits_processor,causal_mask):

    input_pos=torch.arange(0,input_ids.shape[1],device=input_ids.device)


    logits, hidden_states = model.base_model(input_ids, input_pos)

    if logits_processor is not None:
        logits = logits[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(logits[:, -1])
        token = token[None, None]


    model.ea_layer(hidden_states[:,:-1], input_ids[:,1:], input_pos[:-1])

    intokens=torch.cat((input_ids,token),dim=1)

    ea_logits  = model.chaintopK_genrate(hidden_states[:,-1:], intokens, logits_processor)



    return ea_logits,logits,hidden_states,token



def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits[0]

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = tree_logits[1]
        candidates_prob = torch.cat(
            [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
            dim=-1)

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)], dim=0)
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None
    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates





def tree_decoding(
        model,
        tree_candidates,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    len_pos=input_ids.shape[1]
    mask_pos=torch.arange(len_pos,len_pos+tree_candidates.shape[1],device=tree_candidates.device)
    tree_mask=model.tree_mask
    model_tree_mask=model.model_tree_mask(len_pos,mask_pos,tree_mask)

    # tree_candidates_base=tree_candidates.clone()
    # position_ids_base=position_ids.clone()
    # mask_pos_base=mask_pos.clone()
    # model_tree_mask_base=model_tree_mask.clone()
    #print(tree_candidates_base)

    #with Timer("sbase"):
    #print(mask_pos_base)
    tree_logits,hidden_state=model.base_forward(tree_candidates,position_ids,mask_pos,model_tree_mask)

    # tree_logits_out=tree_logits.clone()
    # hidden_state_out=hidden_state.clone()



    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state


def chaintree_decoding(
        model,
        candidates,
        input_ids,
):


    len_pos=input_ids.shape[1]
    mask_pos=torch.arange(len_pos,len_pos+candidates.shape[1],device=candidates.device)



    #with Timer("sbase"):
    #print(mask_pos_base)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        tree_logits,hidden_state=model.base_forward(candidates,mask_pos)

    # tree_logits_out=tree_logits.clone()
    # hidden_state_out=hidden_state.clone()



    logits = tree_logits
    return logits, hidden_state


def evaluate_posterior(
        logits, candidates, logits_processor, cart_candidates_prob, op, p_indices, tree_candidates, b_indices
):

    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        # breakflag=False
        for i in range(1, candidates.shape[1]):
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            if i != accept_length:
                # breakflag=True
                break
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            adjustflag = False
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    r = random.random()
                    x = candidates[j, i]
                    if x == 0:
                        continue
                    px = gtp[x]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        gtp = gtp / gtp.sum()
                        # has_nan = torch.isnan(gtp).any()
                        # if has_nan:
                        #     print(1)
                        adjustflag = True
        if adjustflag:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


def chainevaluate_posterior(
        logits, candidates, logits_processor, cart_candidates_prob, op,
):

    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        # breakflag=False
        for i in range(1, candidates.shape[1]):
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            if i != accept_length:
                # breakflag=True
                break
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            adjustflag = False
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    r = random.random()
                    x = candidates[j, i]
                    if x == 0:
                        continue
                    px = gtp[x]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][0].clone()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        gtp = gtp / gtp.sum()
                        # has_nan = torch.isnan(gtp).any()
                        # if has_nan:
                        #     print(1)
                        adjustflag = True
        if adjustflag:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        logits,
        tree_logits,
        new_token,
        past_key_values_data,
        current_length_data,
        model,
        hidden_state,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    # for past_key_values_data in past_key_values_data_list:
    tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # print(prev_input_len)
    # print(model.base_model.layers[0].attention.kv_cache.k_cache[0,0,prev_input_len])

    # Update the current length tensor (currently only support batch size is 1)
    #current_length_data=prev_input_len + tgt.shape[-2]

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    #hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    ea_input=torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    tree_logits = model.topK_genrate(accept_hidden_state_new,ea_input,logits_processor)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token



def chainupdate_inference_inputs(
        input_ids,
        candidates,
        accept_length,
        logits_processor,
        new_token,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence

    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[:, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    accept_hidden_state_new = hidden_state_new[:, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    #hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    ea_input=torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    tree_logits = model.chaintopK_genrate(accept_hidden_state_new,ea_input,logits_processor)
    #print(model.base_model.layers[0].attention.kv_cache.k_cache[0, 0, prev_input_len])


    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
