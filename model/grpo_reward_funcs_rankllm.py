import re, itertools, math, os
import numpy as np
from pydantic_core import from_json
# HACK for runpod
if not os.path.exists('/root/nltk_data'):
    import nltk
    nltk.download('punkt')
from nltk import word_tokenize
from grpo_reward_funcs import (extract_xml_answer_strict_with_tags, 
                               extract_think_words_strict_with_tags,
                               THINK_START)

PATTERN_CAND        = re.compile('[\'"]candidate[\\\\\\\'"]+\s*:\s*[\\\\\\\'"]\[(\d+)\][\\\\\\\'"]')
PATTERN_SCORE       = re.compile('[\'"]score[\\\\\\\'"]+\s*:\s*(\d+)')
PATTERN_RATIONALE   = re.compile('[\'"]rationale[\\\\\\\'"]+\s*:\s*(.+)\s*,\s*[\\\\\\\'"]+score[\\\\\\\'"]+\s*:\s*.+')
PATTERN_RANK_SEQ    = re.compile('((?:\[\d+\](?:\s*>\s*\[\d+\])+)(?!.*\[\d+\]\s*>\s*\[\d+\]))', re.DOTALL)

### HELPERS ###
def json_extractor(completions):
    # NOTE: no re.DOTALL. we don't want '\n' in the JSONs
    jsons_list = [re.findall(r'{.+}', c) for c in completions]
    
    return jsons_list

def score_dicts_extractor(jsons_list):
    score_dicts_list = []
    for jl in jsons_list:
        sds = []
        for j in jl:
            can, sco, rat = get_process_one_cot_output(j) 
            # might not be possible to get a candidate iden
            # i.e. that JSON is not well-formed
            # but not setting "if can is None: continue"
            # we want to get signal on other keys of the JSON
            # i.e. "rationale" and "score"
            sds.append({'c': can, 's': sco, 'r': rat})
        
        score_dicts_list.append(sds)
    return score_dicts_list

def get_process_one_cot_output(line):
    try:    
        can = re.search(PATTERN_CAND,line).group(1)
        can = f"[{can.zfill(2)}]"
    except: can = None
    
    try:    sco = int(re.search(PATTERN_SCORE,line).group(1))
    except: sco = None
        
    try:    rat = re.search(PATTERN_RATIONALE,line).group(1)
    except: rat = None
    
    return can, sco, rat

def ranking_extractor(completions):
    # 1. extract answers
    ranks_list = []
    # for longest [\d+] > [\d+] sequence before end
    for c in completions:
        rank_seq = give_rank_seq(c)        
        ranks_list.append(rank_seq)
    return ranks_list

def give_rank_seq(line):
    rank_seq = re.search(PATTERN_RANK_SEQ, line)
    if rank_seq:  return rank_seq.group(1).strip()
    else:         return ''

def give_poss_rank_orders(unsorted_scores):
    most2least    = sorted(set(unsorted_scores), reverse = True)
    num_ties      = len(unsorted_scores) - len(most2least) + 1
    poss_rankings = [[] for i in range(math.factorial(num_ties))]
    positions     = list(range(len(unsorted_scores)))
    
    for ss in most2least:
        cut_pos = [__ for sss, __ in zip(unsorted_scores, positions) if sss == ss]
        perms = list(itertools.permutations(cut_pos))
        bsz = len(poss_rankings)//len(perms)
        for __, p in enumerate(perms):
            for pr in poss_rankings[__*bsz:(__+1*bsz)]: pr.extend(p)
    return poss_rankings

def get_0_indexed(pred_order):
    return [int(re.search(r'\d+', iden).group()) - 1 for iden in pred_order]

### REWARD FUNCS ###
def json_ratio(jsons_list, num_expected, **kwargs):
    '''
    gives the ratio of num json returned vs expected 
    '''
    ratios = [len(j)/num_expected for j in jsons_list]
    # keep to scale of 0.5 and less 
    # if ratio exceeds 1.0, set to 0. stop r-hacking
    return [r/2 if r <= 1.0 else 0.0 for r in ratios]

def rank_ratio(ranks_list, num_expected, **kwargs):
    '''
    gives the ratio of num identifiers in ranking returned vs expected 
    '''
    ratios = [len(re.findall(r'\[\d+\]', r))/num_expected for r in ranks_list]
    # similar to json_ratio
    return [r/2 if r <= 1.0 else 0.0 for r in ratios]

def check_answer_format_match_identifiers(ranks_list, num_expected, **kwargs):
    scores      = []
    # force identifiers to be zfill(2)
    exp_identifiers = set([f"[{str(i+1).zfill(2)}]" for i in range(num_expected)])
    elem_s1     = 0.125
    max_score   = elem_s1 * num_expected
    for rank_seq in ranks_list:
        ss = 0.0      
        if rank_seq: 
            pred_identifiers = re.findall('\[\d+\]', rank_seq)
            
            if exp_identifiers.issubset(set(pred_identifiers)): 
                ss += elem_s1 * len(exp_identifiers)

            # check how much additional/missing
            num_mismatch  = abs(num_expected - len(pred_identifiers))
            # deduct from base score for every additional/missing
            ss -= (elem_s1 * num_mismatch)

            # ensure no hacking (more than num_expected) 
            c1 = len(pred_identifiers) > len(set(pred_identifiers)) # repeated
            if c1: ss = 0.0

        s = min(max_score, ss)
        s = max(0.0, s)
        scores.append(s)

    return scores

def check_valid_json(jsons_list, num_expected, 
                     expected_keys = set(['candidate','rationale', 'score']),
                     **kwargs):
    '''
    checks whether a span that is supposed to be a json is json-parseable
    '''
    scores  = []
    elem_s1 = 0.25
    elem_s2 = 0.0625
    max_score = elem_s1 + elem_s2 * len(expected_keys)
    for jl in jsons_list:
        score = []
        for j in jl:
            ss = 0.0
            try:    parsed_json = from_json(j)
            except: parsed_json = {}
                
            if parsed_json: 
                ss += elem_s1 # just for being parseable
            
                # check that the expected keys are present and in the right number
                pred_keys = list(parsed_json.keys())
                # + 0.125 for every expected key
                ss += (elem_s2 * len(set(pred_keys).intersection(expected_keys)))
                # - 0.0625 for every extra key
                ss -= (elem_s2 * len(set(pred_keys).difference(expected_keys)))
            
            score.append(ss)
        s = min(max_score, np.mean(score) if score else 0.0)
        s = max(0.0, s)
        scores.append(s)

    return scores

def check_score_dicts_validity(score_dicts_list, num_expected, min_s = 1.0, max_s = 3.0, **kwargs):
    scores       = []
    per_elem_max = 0.125
    max_score    = num_expected * per_elem_max
    for sdl in score_dicts_list:
        ss = 0.0
        for sd in sdl:
            if sd['s'] is not None and min_s <= sd['s'] <= max_s: ss += per_elem_max
        # avoid reward hacking. e.g. via more JSON objects than candidates
        s = min(max_score, ss)
        s = max(0.0, s)
        scores.append(s)

    return scores

def check_score_dicts_rationales(score_dicts_list, num_expected, **kwargs):
    '''
    ensure that rationales are of meaningful length and not too long
    '''
    scores       = []
    per_elem_max = 0.125
    max_score    = num_expected * per_elem_max
    for sdl in score_dicts_list:
        ss = 0.0 
        # NOTE: could have no scores JSON (esp early steps)
        for sd in sdl:
            num_words = len(word_tokenize(sd['r']) if sd['r'] else '')
            if   75  <= num_words <= 100:            ss += per_elem_max
            elif 50  <= num_words <   75:            ss += per_elem_max/2
            elif 25  <= num_words <   50:            ss += per_elem_max/4
            elif 100 <  num_words <= 125:            ss += per_elem_max/2
            elif 125 <  num_words <= 150:            ss += per_elem_max/4
        s = min(max_score, ss)
        s = max(0.0, s)
        scores.append(s)

    return scores

def match_gpt4o_score(score_dicts_list, gold_docmap_list, num_expected, **kwargs): 
    '''
    1 for exact. -0.5 if score is more than 1 point from GPT4o score. 0.25 otherwise
    also adding a check that num_expected is met. to avoid reward hacking
    '''
    assert len(score_dicts_list) == len(gold_docmap_list)
    scores    = []
    elem_s1   = 0.25
    max_score = elem_s1 * num_expected
    for sdl, gdm in zip(score_dicts_list, gold_docmap_list):
        ss = 0.0
        # NOTE: could have no scores JSON (esp early steps)
        if len(sdl) != len(gdm) != num_expected: 
            scores.append(ss)
            continue
        # convert to docmap:
        pred_docmap = {sd['c']: {'score': sd['s']} for sd in sdl}
        for iden in gdm:
            if iden not in pred_docmap: continue
            # same score
            if pred_docmap[iden]['score'] == gdm[iden]['score']:        ss += elem_s1
            # score cannot be recovered
            elif pred_docmap[iden]['score'] is None:                    ss -= elem_s1
            # score difference of more than 1
            elif abs(pred_docmap[iden]['score']-gdm[iden]['score']) >1: ss -= elem_s1/2
        s = min(max_score, ss)
        s = max(0.0, s)
        scores.append(s)
    
    return scores

def score_rank_consistency(score_dicts_list, ranks_list, num_expected, **kwargs):
    '''
    the predicted ranking is consistent with the give scores.
    also adding a check that num_expected is met. to avoid reward hacking
    '''
    assert len(score_dicts_list) == len(ranks_list)
    scores = []
    for sdl, pred_order in zip(score_dicts_list, ranks_list):
        ss = 0.0
        pred_order  = re.findall(r'(\[\d+\])', pred_order)
        # NOTE: could have no scores JSON (esp early steps)
        if len(sdl) != len(pred_order) != num_expected: 
            scores.append(ss)
            continue
        pred_docmap = {sd['c']: sd['s'] for sd in sdl}
        if None in pred_docmap: 
            scores.append(ss)
            continue
        # sorted by iden, but not sorted by score
        unsorted_pred_scores = [pred_docmap[c] for c in sorted(pred_docmap)]
        if None in unsorted_pred_scores: 
            scores.append(ss)
            continue
        poss_rankings        = give_poss_rank_orders(unsorted_pred_scores)
        if get_0_indexed(pred_order) in poss_rankings: 
            ss = 1.0
        scores.append(ss)
    
    return scores

def answer_well_formed(completions, **kwargs):
    answers   = [extract_xml_answer_strict_with_tags(c) for c in completions]
    answers     = completions
    scores      = []
    elem_s1     = 0.125
    for ans in answers:
        ss = 0.0
        if ans:
            # a. only has exactly 1x [START] and 1x [STOP]
            if len(re.findall(r'\[START\]', ans)) == 1: 
                ss += elem_s1
            if len(re.findall(r'\[STOP\]', ans)) == 1:  
                ss += elem_s1
            # b. [START] must come before [STOP]
            if len(re.findall(r'\[START\].+\[STOP\]', ans)) == 1: 
                ss += elem_s1
            # c. the answer does not have anything other than [START] [STOP] and the ranking prediction
            if re.search(r'\[START\]\s*[\[\d+\] > ]+\[\d+\]\s*\[STOP\]', ans): 
                ss += elem_s1
        scores.append(ss)

    return scores

def thinking_well_formed(completions, **kwargs):
    thoughts  = [extract_think_words_strict_with_tags(c) for c in completions]
    thoughts    = completions
    scores      = []
    elem_s1     = 0.125
    for thought in thoughts:
        ss = 0.0
        if thought:
            # a. only has exactly 1x [S_COT] and 1x [E_COT]
            if len(re.findall(r'\[S_COT\]', thought)) == 1: 
                ss += elem_s1
            if len(re.findall(r'\[E_COT\]', thought)) == 1:  
                ss += elem_s1
            # b. [S_COT] must come before [E_COT]
            if len(re.findall(r'\[S_COT\].+\[E_COT\]', thought, re.DOTALL)) == 1: 
                ss += elem_s1
            # c. have some preliminary (task decomp) thinking before going into SCOTR
            task_decomp     = re.search(rf'{THINK_START}(.+)\[S_COT\]', thought, re.DOTALL)
            task_decomp     = task_decomp.group(1).strip() if task_decomp else ''
            c_task_decomp   = len(word_tokenize(task_decomp))
            if   40 <= c_task_decomp <= 60: ss += elem_s1
            elif 30 <= c_task_decomp <  40: ss += (elem_s1/2)
            elif 60 <  c_task_decomp <  75: ss += (elem_s1/2)
            else: pass 
        scores.append(ss)

    return scores