import random, tqdm, copy
from collections import defaultdict

def evaluate_qud_pick_stv(cfg, holder_panel_outputs, qud_criteria, keep_systems = None):
    from evaluation.stv_utils import compute_stv
    from utils_model import convert_defaultdict
    if keep_systems is not None: assert type(keep_systems) == set
        
    rankllm_choices = defaultdict(dict)
    for qud_instance_id, rankllm_sys_dict in holder_panel_outputs[qud_criteria].items():
        for rankllm_sys, proc_pred in rankllm_sys_dict.items():
            if keep_systems is not None and rankllm_sys not in keep_systems: continue
            rank_dict   = proc_pred['rerank_products']['predicted_order']
            # rank_dict is a dictionary where keys are 0 - n, 0 being best
            # and the values are the qud parsing sys name
            choices     = [rank_dict[i] for i in sorted(list(rank_dict))]
            rankllm_choices[rankllm_sys][qud_instance_id] = choices
    
    rankllm_choices = convert_defaultdict(rankllm_choices)
    
    stv_all_results = compute_stv(cfg, rankllm_choices)     
    stv_all_results = convert_defaultdict(stv_all_results)
    return stv_all_results

def evaluate_qud_rules_based(cfg, holder_articles, holder_quds):
    from evaluation.qud_rules_based import QUDRulesBasedEvaluator
    rules_based = QUDRulesBasedEvaluator()

    # 1. Rules-based 
    # obtain the rules-based score for each QUDCandidate for each criteria
    # obtain the rules-based ordering for the candidates in each QUDInstance
    holder_eval_rules_based = {}
    for qud_criteria in cfg.qud_criteria_list:
        print('🟧🟧\tWorking on Rules-based scoring for criteria:', qud_criteria)
        holder_eval_rules_based[qud_criteria] = {}

        for __, (qud_instance_id, qud_instance) in enumerate(tqdm.tqdm(holder_quds.items())):  

            gold_scores = [c.criteria_scores[qud_criteria] for c in qud_instance.qud_candidates]
            if cfg.exclude_same_scores and len(set(gold_scores)) == 1: continue


            article_contents = holder_articles[qud_instance.article_id]
            ctx = ' '.join(qud_instance.extract_context(article_contents))
            anc = qud_instance.extract_anchor(article_contents)
            ans = qud_instance.extract_answer(article_contents)
            
            qud_cands_scores    = {}
            for qud_cand in qud_instance.qud_candidates:
                qud = qud_cand.qud
                
                if   qud_criteria == 'criteria2':
                    score = rules_based.compute_comp(question = qud, answer = ans)
                
                elif qud_criteria == 'criteria3':
                    score = rules_based.compute_givenness(question = qud, context = ctx)
                
                elif qud_criteria == 'criteria4':
                    score = rules_based.compute_relevance(question = qud, anchor = anc)

                else: raise NotImplementedError

                qud_cands_scores[qud_cand.sysname] = score
            
            # convert scores to ordinal ranks
            # NOTE: important so that NDCG computation is directly comparable 
            ordinal_scores = make_score_ordinal(qud_cands_scores, break_tie = True)
            holder_eval_rules_based[qud_criteria][qud_instance_id] = {'ord': ordinal_scores, 
                                                                      'raw': qud_cands_scores}

    return holder_eval_rules_based

def evaluate_llmqalogprobs(cfg, holder_articles, holder_quds, model_name, model_size, device_map = None):
    from evaluation.answer_probability import (compute_continuation_llmaqalogprobs, make_one_prefix_llmqalogprobs,
                                               prepare_one_qud_instance_llmqalogprobs, compute_answer_compat,
                                               give_llmqalogprobs_icl)
    from utils_model import load_model
    import torch, gc

    #################
    # A. load model 
    qud_criteria    = 'criteria2'
    print('🟧🟧\tWorking on LLM QA log_probs scoring for criteria:', qud_criteria)

    pipeline_model = None
    gc.collect()
    torch.cuda.empty_cache()
    # NOTE: current set-up leads to graph graphs on torch.compile
    # https://pytorch.org/docs/2.6/torch.compiler_troubleshooting.html#graph-break
    # setting to no compile
    tokenizer, pipeline_model, device, cfg = \
        load_model(cfg, model_name, cfg.models[model_size][model_name], 
                model_size = model_size, do_compile = False, device_map = device_map)
    if tokenizer.pad_token is None: 
        pipeline_model.tokenizer.pad_token    = tokenizer.pad_token     = tokenizer.bos_token
        pipeline_model.tokenizer.pad_token_id = tokenizer.pad_token_id  = tokenizer.bos_token_id

    #################
    # B. re-usables
    do_icl          = cfg.answer_compat.do_icl    
    if do_icl: 
        icl_messages = give_llmqalogprobs_icl(cfg, 
                    num_few_shot_examples = cfg.answer_compat.num_few_shot_examples)
        
    
    #################
    # C. LLM QA
    holder_eval_llm_qa = {}
    holder_eval_llm_qa[qud_criteria] = {}
    for __, (qud_instance_id, qud_instance) in enumerate(tqdm.tqdm(holder_quds.items())):    

        gold_scores = [c.criteria_scores[qud_criteria] for c in qud_instance.qud_candidates]
        if cfg.exclude_same_scores and len(set(gold_scores)) == 1: continue

        ctx, anc, ans_cands, continuations, ans_idx = \
            prepare_one_qud_instance_llmqalogprobs(cfg, holder_articles, qud_instance)
        
        qud_cands_scores    = {}
        for qud_cand in qud_instance.qud_candidates:
            qud = qud_cand.qud
            
            prefix = make_one_prefix_llmqalogprobs(cfg, do_icl, model_name, tokenizer,
                                    ctx, anc, qud, ans_cands, icl_messages)

            batch_log_prob = compute_continuation_llmaqalogprobs(pipeline_model, model_name,
                                                        prefix, continuations)
            
            if cfg.exp_code == 'test':
                print('🟣'*20)
                print('HERE 1', prefix)
            do_print = True if cfg.exp_code == 'test' else False
            score = compute_answer_compat(batch_log_prob, ans_idx, do_print)
            qud_cands_scores[qud_cand.sysname] = score

            
                
                
        
        # convert scores to ordinal ranks
        # NOTE: important so that NDCG computation is directly comparable 
        ordinal_scores = make_score_ordinal(qud_cands_scores, break_tie = True)
        holder_eval_llm_qa[qud_criteria][qud_instance_id] = {'ord': ordinal_scores, 
                                                             'raw': qud_cands_scores}

        if cfg.exp_code == 'test':
            gold_scores_dict = \
                {c.sysname: c.criteria_scores[qud_criteria] for c in qud_instance.qud_candidates}
            print('HERE 4', ordinal_scores, '\t\tGOLD:', gold_scores_dict)
            print('🟣'*20)

    pipeline_model = None
    gc.collect()
    torch.cuda.empty_cache()
    return holder_eval_llm_qa

    
def make_score_ordinal(qud_cands_scores: dict, break_tie: bool = False):
    '''
    - reduces complete ties (if break_tie = True), 
        > so we don't ever have a case of all equivalently good
        > i.e. all candidate score the same
        > done by picking one of the ties and adding 1 to it
    - rescores to the following:
        > max score is set to num of candidates in qud_cands_scores
        > where ties are present, they share the same score  
        > e.g.          {'c': 1, 'b': 3, 'a': 3, 'd': 1}
        > becomes       {'c': 2, 'b': 4, 'a': 4, 'd': 2}
    '''
    q_c_scores = copy.deepcopy(qud_cands_scores)
    # for complete ties, randomly pick one
    if break_tie and len(set(q_c_scores.values())) == 1:
        random.seed(54506)
        pick = random.choice(list(q_c_scores))
        q_c_scores[pick] += 1
        assert len(set(q_c_scores.values())) > 1, q_c_scores

    max_score       = len(q_c_scores)
    sorted_scores   = sorted(q_c_scores.values(), reverse = True)

    ordinal_scores  = {sn: max_score-sorted_scores.index(sc) for sn, sc in q_c_scores.items()}

    return ordinal_scores

def evaluate_qudeval_gpt(cfg, holder_articles, holder_quds, model_name = 'gpt-4o', few_shot = False):
    from evaluation.qudeval.criteria2_answer_compatibility_gpt_response     import criteria2_do_one_line
    from evaluation.qudeval.criteria3_giveness_gpt_response                 import criteria3_do_one_line
    from evaluation.qudeval.criteria4_anchor_relevence_eval_by_threshold    import criteria4_do_one_line
    from evaluation.qudeval.few_shot_scoring                                import do_one_instance_few_shot

    few_shot_examples = None
    if few_shot:
        from evaluation.qudeval.few_shot_scoring import load_examples
        few_shot_examples = load_examples(cfg, holder_articles)

    from openai import OpenAI
    if model_name == 'gpt-4o':       
        base_url = None
        api_key  = cfg.token_dict['gpt']
    if model_name == 'deepseek':    
        base_url = "https://api.deepseek.com"
        api_key  = cfg.token_dict['deepseek']
    
    client = OpenAI(api_key = api_key, 
                    base_url = base_url,
                    max_retries = 1, 
                    timeout = 3 * 60)
    
    holder_eval_qudeval_gpt  = {}
    for qud_criteria in cfg.qud_criteria_list:
        print('🟧🟧\tWorking on QUDEval GPT scoring for criteria:', qud_criteria)
        holder_eval_qudeval_gpt[qud_criteria] = {}

        for __, (qud_instance_id, qud_instance) in enumerate(tqdm.tqdm(holder_quds.items())):  

            gold_scores = [c.criteria_scores[qud_criteria] for c in qud_instance.qud_candidates]
            if cfg.exclude_same_scores and len(set(gold_scores)) == 1: continue

            article_contents = holder_articles[qud_instance.article_id]
            article_text = [article_contents.contents[i].strip() for i in sorted(article_contents.contents)]
            ctx = ' '.join(qud_instance.extract_context(article_contents))
            anc = qud_instance.extract_anchor(article_contents)
            ans = qud_instance.extract_answer(article_contents)
            
            qud_cands_scores    = {}
            prompt_one_set      = {}
            cot_output_one_set  = {}
            for qud_cand in qud_instance.qud_candidates:
                qud = qud_cand.qud
                
                cot_output = None
                if   qud_criteria == 'criteria2':
                    if few_shot:
                        score, cot_output, prompt = do_one_instance_few_shot(cfg, few_shot_examples, ctx, anc, ans, qud, 
                                                                        qud_criteria, client, model_name)
                    else:
                        score, cot_output, prompt = criteria2_do_one_line(context_before_include_anchor = ctx,  # ✅
                                                  anchor_sentence = anc,                    # ✅
                                                  question_text = qud,                      # ✅
                                                  context = '\n'.join(article_text),        # ✅
                                                  client = client, model_name = model_name)
                
                elif qud_criteria == 'criteria3':
                    if few_shot:
                        score, cot_output, prompt = do_one_instance_few_shot(cfg, few_shot_examples, ctx, anc, ans, qud, 
                                                                        qud_criteria, client, model_name)
                    else:
                        score, cot_output, prompt = criteria3_do_one_line(context_before_include_anchor = ctx,  # ✅
                                                  question_text = qud,                      # ✅
                                                  answer_sentence= ans,                     # ✅
                                                  client = client, model_name = model_name)
                
                elif qud_criteria == 'criteria4':
                    if few_shot:
                        score, cot_output, prompt = do_one_instance_few_shot(cfg, few_shot_examples, ctx, anc, ans, qud, 
                                                                        qud_criteria, client, model_name)
                    else:
                        # using thresholds provided in their repo
                        # https://github.com/lingchensanwen/QUDeval/blob/main/code/criteria4_anchor_relevence/gpt-score-base/best_threshold_info.txt
                        score, cot_output, prompt = criteria4_do_one_line(question_text= qud,                   # ✅
                                                  anchor_sentence = anc,                    # ✅
                                                  client = client, model_name = model_name, # ✅
                                                  best_thresh1 = 0.0, best_thresh2 = 10.0)

                else: raise NotImplementedError

                assert len(score) == 1, score
                score = score[0]
                if not few_shot:
                    # reverse the scores to 3 best, 1 worst ... to align with out treatment
                    if   qud_criteria == 'criteria2':
                        if score == 3: score = 1
                        elif score == 1: score = 3
                        else: score = 0 
                    elif qud_criteria in ['criteria3', 'criteria4']:
                        score = 4-score
                        if score not in [1,2,3]: score = 0 
                qud_cands_scores[qud_cand.sysname]   = score 
                prompt_one_set[qud_cand.sysname]     = prompt
                cot_output_one_set[qud_cand.sysname] = cot_output 
            
            # convert scores to ordinal ranks
            # NOTE: important so that NDCG computation is directly comparable 
            ordinal_scores = make_score_ordinal(qud_cands_scores, break_tie = True)
            holder_eval_qudeval_gpt[qud_criteria][qud_instance_id] = {'ord': ordinal_scores, 
                                                                      'raw': qud_cands_scores,
                                                                      'prompt': prompt_one_set,
                                                                      'cot_output': cot_output_one_set}

    return holder_eval_qudeval_gpt

def evaluate_qudselect_classifiers(cfg, holder_articles, holder_quds):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if   torch.cuda.is_available(): device = torch.device('cuda')
    elif torch.cuda.is_available(): device = torch.device('mps')
    else: device = torch.device('cpu')
    
    def predict(text, model, tokenizer, max_seq_length = 512, device = device):
        inputs = tokenizer(text, return_tensors = "pt", 
                           padding = True, truncation = True, 
                           max_length = max_seq_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities   = torch.nn.functional.softmax(outputs.logits, dim = -1)
        predicted_class = torch.argmax(probabilities, dim = -1).item()
        
        # Map class numbers to meaningful labels
        return predicted_class, probabilities[0].tolist()

    holder_eval_qudselect_classifiers  = {}
    for qud_criteria in cfg.qud_criteria_list:
        crit_key = qud_criteria if qud_criteria != 'criteria3' else 'criteria3_withans'
        model_save_path = f'{cfg.dirpath}/results/qudselect_classifiers/{crit_key}'
        tokenizer   = AutoTokenizer.from_pretrained(model_save_path)
        model       = AutoModelForSequenceClassification.from_pretrained(model_save_path,
                                                        num_labels = 3, problem_type = 'single_label_classification')
        model.to(device)
        model.eval()


        print('🟧🟧\tWorking on QUDSelect classifier scoring for criteria:', qud_criteria)
        holder_eval_qudselect_classifiers[qud_criteria] = {}

        for __, (qud_instance_id, qud_instance) in enumerate(tqdm.tqdm(holder_quds.items())):  

            gold_scores = [c.criteria_scores[qud_criteria] for c in qud_instance.qud_candidates]
            # NOTE: HACK: cfg.do_model is for us to run post-GRPO evals
            if not cfg.do_model and cfg.exclude_same_scores and len(set(gold_scores)) == 1: 
                continue

            article_contents = holder_articles[qud_instance.article_id]
            ctx = ' '.join(qud_instance.extract_context(article_contents))
            anc = qud_instance.extract_anchor(article_contents)
            ans = qud_instance.extract_answer(article_contents)
            
            qud_cands_scores    = {}
            for qud_cand in qud_instance.qud_candidates:
                qud = qud_cand.qud
                
                if   qud_criteria == 'criteria2':
                    text = qud + tokenizer.sep_token + ans
                
                elif qud_criteria == 'criteria3':
                    if  crit_key == qud_criteria:
                        text = ctx + tokenizer.sep_token + qud
                    elif crit_key == 'criteria3_withans':
                        # we add anc to ctx, and also ans in withans
                        text = ctx + anc + tokenizer.sep_token + ans + tokenizer.sep_token + qud
                    else: raise NotImplementedError
                
                elif qud_criteria == 'criteria4':
                    text = qud + tokenizer.sep_token + anc

                else: raise NotImplementedError

                score, probs = predict(text, model, tokenizer, max_seq_length = 512)
                # NOTE: score + 1 because 0-indexed. 
                # NOTE: during training, we reversed the scores to 3 being the best.
                qud_cands_scores[qud_cand.sysname]   = score + 1
            
            # convert scores to ordinal ranks
            # NOTE: important so that NDCG computation is directly comparable 
            ordinal_scores = make_score_ordinal(qud_cands_scores, break_tie = True)
            holder_eval_qudselect_classifiers[qud_criteria][qud_instance_id] = {'ord': ordinal_scores, 
                                                                                'raw': qud_cands_scores,
                                                                                'probs': probs}

    return holder_eval_qudselect_classifiers

