######################################
########## REWARD FUNCTIONS ##########
######################################
# format reward func from: 
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

import re, sys
sys.path.append('evaluation')
sys.path.append('data')
from data_utils import QUDInstance
from answer_probability import (prepare_one_qud_instance_llmqalogprobs, 
                                make_one_prefix_llmqalogprobs, 
                                compute_continuation_llmaqalogprobs, 
                                compute_answer_compat)
THINK_START, THINK_END  = '<think>', '</think>'
ANS_START, ANS_END      = '<answer>', '</answer>'


##### CONV VERSION #####
def strict_format_reward_func_conv(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    pattern = rf'^{THINK_START}\n.*?\n{THINK_END}\n{ANS_START}\n.*?\n{ANS_END}\n$'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func_conv(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    pattern = rf'{THINK_START}.*?{THINK_END}\s*{ANS_START}.*?{ANS_END}'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func_conv(completions, **kwargs) -> list[float]:
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]

##### TEXT VERSION #####
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    # NOTE: forces newline. but soft_format_reward_func relaxes this
    pattern = rf'^{THINK_START}\n.*?\n{THINK_END}\n{ANS_START}\n.*?\n{ANS_END}\n$'
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    pattern = rf'{THINK_START}.*?{THINK_END}\s*{ANS_START}.*?{ANS_END}'
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count(f'{THINK_START}\n') == 1:
        count += 0.125
    if text.count(f'\n{THINK_END}\n') == 1:
        count += 0.125
    if text.count(f'\n{ANS_START}\n') == 1:
        count += 0.125
        count -= len(text.split(f'\n{ANS_END}\n')[-1])*0.001
    if text.count(f'\n{ANS_END}') == 1:
        count += 0.125
        count -= (len(text.split(f'\n{ANS_END}')[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]

def extract_xml_answer_no_tags(text: str) -> str:
    answer = text.split(f'{ANS_START}')[-1]
    answer = answer.split(f'{ANS_END}')[0]
    return answer.strip()

def extract_xml_answer_strict_with_tags(text: str) -> str:
    answer = re.search(rf'{ANS_START}.+{ANS_END}', text)
    if answer:     return answer.group(0).strip()
    else:               return ''

def no_xml_in_answer(completions, reward_funcs_version, **kwargs):
    if      reward_funcs_version == 1:
        answers = [extract_xml_answer_no_tags(c) for c in completions]
        scores = [0.0 if re.search(r'<\w+>.+</\w+>', c) else 0.5 for c in answers] 
    elif   reward_funcs_version >= 2:
        answers = [extract_xml_answer_no_tags(c) for c in completions]
        scores = [0.0 if not c or re.search(r'<\w+>.+</\w+>', c) else 0.5 for c in answers] 
    else: raise NotImplementedError
    return scores

def extract_think_words_strict_no_tags(completion) -> str:
    think_words = re.search(rf'{THINK_START}(.*){THINK_END}', completion)
    if think_words:     return think_words.group(0).strip()
    else:               return ''

def extract_think_words_strict_with_tags(completion) -> str:
    think_words = re.search(rf'{THINK_START}.+{THINK_END}', completion)
    if think_words:     return think_words.group(0).strip()
    else:               return ''

def think_length_reward_func(completions, **kwargs) -> list[float]:
    scores = []
    for c in completions:
        think_words = extract_think_words_strict_no_tags(c)
        if think_words:
            len_think_words = len(think_words)

            if   250 <= len_think_words <= 350:scores.append(0.5)
            elif 200 <= len_think_words < 250: scores.append(0.25)
            elif 150 <= len_think_words < 200: scores.append(0.125)
            elif len_think_words < 150:        scores.append(0.0)
            elif 350 < len_think_words <= 400: scores.append(0.25)
            elif 400 < len_think_words <= 450: scores.append(0.125)
            elif len_think_words > 450:        scores.append(0.0)
            
        else: scores.append(0.0)
    
    return scores

def ensure_tighter_qud_gen(completions, reward_funcs_version = 1, **kwargs):
    scores = []
    if reward_funcs_version >= 3:
        answers = [extract_xml_answer_no_tags(c) for c in completions]
        for a in answers:
            if not a: 
                scores.append(0)
                continue
            s = 0.5
            for p in kwargs['tighter_qud_gen_phrases_reward']:
                if p.lower() in a.lower(): s -= 0.125
            scores.append(s)
    else: scores.append(0.0)
    
    return scores

def qud_length_reward(completions, reward_funcs_version = 1, **kwargs):
    scores  = []
    for c in completions:
        length = len(extract_xml_answer_no_tags(c).split()) 
        if   reward_funcs_version == 1:
            if   7 <= length <= 15: scores.append(0.5) # ~75% of DCQA train
            elif length <  4:       scores.append(0.0)
            elif length > 35:       scores.append(0.0)
            else:                   scores.append(0.125)
        elif reward_funcs_version == 2:
            if   7 <= length <= 15: scores.append(0.75) # ~75% of DCQA train
            else:                   scores.append(0.0)
        elif reward_funcs_version == 3:
            if   7 <= length <= 10: scores.append(0.75) # ~75% of DCQA train
            elif 5 <= length <   7: scores.append(0.5) 
            elif 10 < length <= 12: scores.append(0.5) 
            else:                   scores.append(0.0)
        else: 
            raise not NotImplementedError

    return scores

def criteria2_reward_func(worker_scores, qud_max_score, compare_pairs, 
                          non_worker_evaluator, reward_model_method = 'worker', 
                          qud_instance_id = None, **kwargs):
    if reward_model_method == 'rules': 
        scores  = [non_worker_evaluator.compute_comp(q, ans) for q,ans in compare_pairs]
        return scores
    
    elif reward_model_method == 'worker':
        # normalise to between 0 and 1 (same range as other rewards)
        ratio = qud_max_score
        return [s/ratio for s in worker_scores] 
    
    elif reward_model_method == 'llmqalogprobs':
        assert qud_instance_id is not None
        pipeline_model  = non_worker_evaluator
        cfg             = non_worker_evaluator.cfg
        holder_articles = non_worker_evaluator.holder_articles
        icl_messages    = non_worker_evaluator.icl_messages
        model_name      = pipeline_model.model_name
        do_icl          = pipeline_model.do_icl
        tokenizer       = pipeline_model.tokenizer

        article_id, anchor_id, answer_id = qud_instance_id.split('_')
        qud_instance = QUDInstance(article_id = int(article_id) \
                                   if type(article_id) == str and article_id.isdigit() else article_id, 
                                   anchor_id = int(anchor_id), 
                                    answer_id = int(answer_id), 
                                    qud_instance_id = qud_instance_id,
                                    qud_human = None, qud_candidates = None,
                                    do_tedq = 'talk' in qud_instance_id)
        ctx, anc, ans_cands, continuations, ans_idx = \
            prepare_one_qud_instance_llmqalogprobs(cfg, holder_articles, qud_instance)
        
        scores = []
        for qud, __ in compare_pairs:
            prefix = make_one_prefix_llmqalogprobs(cfg, do_icl, model_name, tokenizer, 
                                                    ctx, anc, qud, ans_cands, icl_messages)
            batch_log_prob = compute_continuation_llmaqalogprobs(pipeline_model, model_name, 
                                                        prefix, continuations)
            score = compute_answer_compat(batch_log_prob, ans_idx)
            scores.append(score)
        
        return scores

def criteria3_reward_func(worker_scores, qud_max_score, compare_pairs, 
                          non_worker_evaluator, reward_model_method = 'worker', 
                          **kwargs):
    if reward_model_method == 'rules': 
        scores  = [non_worker_evaluator.compute_givenness(q, ctx) for q,ctx in compare_pairs]
        return scores
    elif reward_model_method == 'worker':
        ratio = qud_max_score 
        return [s/ratio for s in worker_scores]
    else: raise NotImplementedError

def criteria4_reward_func(worker_scores, qud_max_score, compare_pairs, 
                          non_worker_evaluator, reward_model_method = 'worker', 
                          **kwargs):
    if reward_model_method == 'rules': 
        scores  = [non_worker_evaluator.compute_relevance(q, anc) for q,anc in compare_pairs]
        return scores
    elif reward_model_method == 'worker':
        ratio = qud_max_score 
        return [s/ratio for s in worker_scores]
    else: raise NotImplementedError

