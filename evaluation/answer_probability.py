import torch, math, copy
import numpy as np
from transformers import DynamicCache

def past_key_values_expander(past_key_values, num_beams):
    '''
    helper function to expand past key values (obtained for e.g. on a constant prompt
    whereby past_key_values was obtained by passing a prefix text through the model). 
    - for decoder only models: shape of past key values is 
    (num_layers, attn layers, num_beams, ...). 

    '''
    pkv = []
    for layer in past_key_values:
        pkv_1 = []
        for attn_layer in layer:
            assert len(attn_layer) == 1 and attn_layer.dim() == 4, \
                (len(attn_layer), attn_layer.shape)
            pkv_1.append(attn_layer.repeat(num_beams,1,1,1))
        pkv.append(pkv_1)

    return pkv


def compute_continuation_llmaqalogprobs(pipeline_model, model_name, 
                               prefix: str, continuations: list):
    '''
    NOTE: this is intended for use with decoder-only models (non-chat models)
    credits: https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/10
    '''
    batch_size  = 2
    num_batches = math.ceil(len(continuations)/batch_size)
    device      = pipeline_model.device

    assert type(prefix) == str

    if model_name in ['gemma']:
        raise NotImplementedError # new past_key_values_expander needed for HybridCache
        # see https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.HybridCache
        # also https://huggingface.co/docs/transformers/v4.48.0/kv_cache#model-specific-cache-classes 
    else: 
        past_key_values = make_prompt_cache(pipeline_model, prefix)

    batch_log_prob  = []
    for bn in range(num_batches):

        batch = continuations[bn*batch_size:(1+bn)*batch_size]
        
        # NOTE: add_special_tokens = False to avoid adding bos token. 
        # llama chat template does not include bos_token before assistant reply.
        enc_conts   = pipeline_model.tokenizer(batch, padding = 'longest', padding_side = 'right', 
                                add_special_tokens = False, return_tensors = "pt").to(device)

        with torch.no_grad():
            pkv     = past_key_values_expander(copy.deepcopy(past_key_values), len(batch))
            pkv_obj = DynamicCache().from_legacy_cache(pkv)
            outputs = pipeline_model.model(past_key_values = pkv_obj, 
                                           input_ids = enc_conts['input_ids'], 
                                           use_cache = True, return_dict = True)
        del pkv

        log_probs = torch.log_softmax(outputs.logits, dim = -1).detach()

        # collect the probability of the generated token -- 
        # probability at index 0 corresponds to the token at index 1
        log_probs = log_probs[:, :-1, :]
        input_ids = enc_conts['input_ids'][:, 1:]
        gen_log_probs = torch.gather(log_probs, 2, input_ids[:, :, None]).squeeze(-1)
        
        for input_sentence, input_probs in zip(input_ids, gen_log_probs):
            sent_log_probs = []
            for token, prob in zip(input_sentence, input_probs):
                if token not in pipeline_model.tokenizer.all_special_ids:
                    sent_log_probs.append(prob.item())
            batch_log_prob.append(np.mean(sent_log_probs))
    
    return batch_log_prob


def make_prompt_cache(pipeline_model, prefix):

    enc_prefix = pipeline_model.tokenizer([prefix], return_tensors = "pt").to(pipeline_model.device)     
    
    with torch.no_grad():                                                              
        past_key_values  = pipeline_model.model(**enc_prefix, past_key_values = None,
                                            return_dict_in_generate = True).past_key_values
    
    return past_key_values


def make_one_prefix_llmqalogprobs(cfg, do_icl, model_name, tokenizer, 
                                  ctx, anc, qud, ans_cands, icl_messages):
    
    prefix_str          = cfg.prompts.answer_compat.prefix
    instructions_str    = cfg.prompts.answer_compat.instructions
    system_message      = cfg.prompts.answer_compat.system_message
    
    if do_icl: 
        # NOTE: if doing ICL, instructions added once (above 1st exemplar)
        prefix_str = prefix_str.replace('{{instructions}}', '')
    else:      
        prefix_str = prefix_str.replace('{{instructions}}', instructions_str)
    prefix_str = prefix_str.replace('{{context}}', f'{ctx}\n{anc}' if ctx else anc)
    prefix_str = prefix_str.replace('{{qud}}', qud)
    prefix_str = prefix_str.replace('{{ans_cands}}', ans_cands)
    
    if do_icl:
        messages = [{"role": "system", "content": system_message}]
        messages.extend(icl_messages)
        messages.append({"role": "user", "content": prefix_str})
    else:
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prefix_str},]
        
    # some models (e.g. gemma) don't have a system role in the chat template
    # add the system instructions to the first message
    if model_name in ['gemma']:
        messages = messages[1:]
        # messages[0]['content'] = f'{system_message}\n\n' + messages[0]['content']

    # NOTE: add_generation_prompt = True appends the bot, bos tokens (i.e. model 
    # generation will immediately provide the answer).
    # e.g. for llama, this appends "<|start_header_id|>assistant<|end_header_id|>" 
    # to the end of the last message (user)
    if tokenizer.chat_template is not None:
        prefix = tokenizer.apply_chat_template(messages, tokenize = False, 
                                            add_generation_prompt = True)
    else: 
        prefix = '\n'.join([m["content"] for m in messages])

    return prefix


def prepare_one_qud_instance_llmqalogprobs(cfg, holder_articles, qud_instance):
    add_index_nums  = cfg.answer_compat.add_index_nums
    ans_cands_form  = cfg.answer_compat.ans_cands_form

    article_contents = holder_articles[qud_instance.article_id]
    ctx              = ' '.join(qud_instance.extract_context(article_contents))
    anc              = qud_instance.extract_anchor(article_contents)
    continuations    = qud_instance.extract_answer_cands(article_contents, 
                                            ans_cands_form = ans_cands_form)
    if add_index_nums:
        continuations    = [f'[{str(i+1).zfill(2)}]\t{ans_cand}' \
                                for i, ans_cand in enumerate(continuations)]
        ans_cands        = '\n'.join(continuations)
    else:
        ans_cands        = '- ' + '\n- '.join(continuations)
    
    ans_idx          = qud_instance.answer_id

    return ctx, anc, ans_cands, continuations, ans_idx

def give_llmqalogprobs_icl(cfg, num_few_shot_examples = 2):
    ans_cands_form = cfg.answer_compat.ans_cands_form
    assert ans_cands_form in ['all', 'post_anc']
    
    icl_exemplars   = cfg.prompts.answer_compat.icl_exemplars
    instructions    = cfg.prompts.answer_compat.instructions
    add_index_nums  = cfg.answer_compat.add_index_nums
    # idxes = random.sample(sorted(icl_exemplars), num_few_shot_examples)
    # NOTE: idxes sorted by num ans_cands (907_18_19 has more than 30 ans cands,
    # leading to OOM issues)
    idxes = ['1445_3_4', '1443_12_13', '1445_12_13', '905_13_17', '907_18_19'][:num_few_shot_examples]

    icl_messages = []
    for __, idx in enumerate(idxes):
        holder      = icl_exemplars[idx]
        prefix_str  = cfg.prompts.answer_compat.prefix

        ctx = holder['ctx']
        ctx = '\n'.join(ctx)
        anc = holder['anc']
        qud = holder['qud']
        ans = holder['ans']
        ans_cands = holder['ans_cands_all' if ans_cands_form == 'all' else 'ans_cands']
        if add_index_nums:
            art_id, anc_id, ans_id = idx.split('_')
            ans = f'[{ans_id.zfill(2)}]\t{ans}'

            ans_cands = [f'[{str(i+1).zfill(2)}]\t{a}' for i, a in enumerate(ans_cands)]
            ans_cands = '\n'.join(ans_cands)
        else: 
            ans_cands = '- ' + '\n- '.join(ans_cands)
        

        if __ == 0: prefix_str = prefix_str.replace('{{instructions}}', instructions)
        else:       prefix_str = prefix_str.replace('{{instructions}}', '')
        prefix_str = prefix_str.replace('{{context}}', f'{ctx}\n{anc}' if ctx else anc)
        prefix_str = prefix_str.replace('{{qud}}', qud)
        prefix_str = prefix_str.replace('{{ans_cands}}', ans_cands)
        icl_messages.append({"role": "user",        "content": prefix_str},)
        icl_messages.append({"role": "assistant",   "content": ans},)

    return icl_messages

def compute_answer_compat(batch_log_prob, ans_idx, do_print = False):
    # NOTE: not reversing (i.e. worst -> best) 
    # since we set score below to be the higher the better, and normalised
    sorted_b_l_p    = sorted(batch_log_prob, reverse = False)
    sort_order      = {}
    for opos, sc in enumerate(batch_log_prob):
        npos = sorted_b_l_p.index(sc)
        # make 1-indexed (to be aligned with ans_idx)
        sort_order[npos]   = opos + 1
        sorted_b_l_p[npos] = None 
    
    ranked = [sort_order[npos] for npos in range(len(sorted_b_l_p))]
    score = ranked.index(ans_idx)/len(ranked)

    if do_print:
        # print('HERE 1', batch_log_prob)
        # print('HERE 2', sorted_b_l_p)
        print('HERE 2', ranked)
        print(f'HERE 3 ans_idx: {ans_idx} size: {len(ranked)} score: {score}')

    return score
