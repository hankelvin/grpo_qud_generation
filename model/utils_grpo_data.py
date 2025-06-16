import sys, copy, json, re, os, math
from collections import Counter
from grpo_reward_funcs import THINK_START, THINK_END, ANS_START, ANS_END
SEED = 54506
COMPLETION_STRIP_REGEX = {
'llama': re.compile(re.escape('<|begin_of_text|>')+'.+Cutting Knowledge Date.+?'+re.escape('<|eot_id|>'), re.DOTALL),
'qwen':  re.compile(re.escape('''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'''))
}
COMPLETION_STRIP_GENPROMT = {
'llama': '<|start_header_id|>assistant<|end_header_id|>\n\n',
'qwen': '<|im_start|>assistant\n'
}

def load_grpo_rankllm_data(cfg, sft_tokenizer):
    sys.path.append(cfg.dirpath)
    sys.path.append('data')
    sys.path.append('tools')
    from dataloader import load_phase1, make_one_rankllm_request
    from data_utils import QUDInstance, QUDCandidate
    from rank_llm.src.rank_llm.data import Result
    from main_grpo_worker import WorkerNode

    holder_articles, holder_quds, num_cands2qud_idxes = load_phase1(cfg)

    with open('data/eval_qud_instance_ids.txt', encoding = 'utf-8') as f: 
        eval_qud_instance_ids = [i.strip() for i in f.readlines() if i.strip()]

    # collect rankllm Request objects. not possible to add them to datasets.Dataset
    eval_ranker_requests = {}
    
    # load data for testing
    # set up rankllm models (using openai client to avoid loading models)
    # leverage create_prompt to get the messages
    holder_dataset_eval  = [] # NOTE: not used
    exclude_same_scores  = True # important. align with main_phase1 to get 322x
    context_empty_symbol = cfg.prompts.rankllm.prefix.context_empty
    criteria2worker  = {}
    qud_criteria_cfg = {}
    for qud_criteria, model_info in cfg.model.reward.items():
        if qud_criteria not in ['criteria2', 'criteria3', 'criteria4']: continue
        eval_ranker_requests[qud_criteria] = {}
        # init a gpt4o-rankllm to utilise its create_prompt
        cfg_copy = copy.deepcopy(cfg) 
        cfg_copy.model.reward[qud_criteria].model = 'gpt4o'
        cfg_copy.role = qud_criteria
        cfg_copy.device_num = 0
        # we want to SFT a model that can work without few-shot exemplars needed
        for key, value in cfg.ranker_args.rm_criteria_settings[qud_criteria].items():
            cfg_copy.ranker_args[key] = value

        if cfg_copy.ranker_args.do_cot: 
            cfg_copy = set_cfg_prompts_for_post_train_rankllm(cfg_copy)
            cfg_copy.load_peft_ckpt_path = None

        criteria2worker[qud_criteria] = worker = \
            WorkerNode(cfg_copy, own_rank = None, master_rank = None, 
                        role = qud_criteria, launch_nccl = False)
        qud_criteria_cfg[qud_criteria] = cfg_copy
        
        for qud_instance_id, qud_instance in holder_quds.items():
            if qud_instance_id not in eval_qud_instance_ids: continue   
            prompt_msgs, request, gold_docmap = give_rankllm_prompt_request(make_one_rankllm_request, Result, 
                                                        holder_articles, qud_instance, qud_instance_id, qud_criteria, 
                                                        exclude_same_scores, context_empty_symbol, worker)
            if prompt_msgs is None: continue
            prompt = sft_tokenizer.apply_chat_template(prompt_msgs, tokenize = False, add_generation_prompt = True)
            request.prompt = prompt # for use in process_one_step_rankllm
            eval_ranker_requests[qud_criteria][qud_instance_id] = request
            line = {'qud_instance_id': qud_instance_id, 'prompt': prompt, 
                    'qud_criteria': qud_criteria, 'gold_docmap': gold_docmap, 
                    'completion': None}
            holder_dataset_eval.append(line)
    
    # load data for training
    holder_dataset_train = []
    exclude_same_scores = False
    placeholder = '^IDEN^'
    ctr_num, ctr_fail = 0,0
    if cfg.exclude_used_q_i_ids:
        q_i_ids_to_exclude = extract_qud_instance_ids_used(cfg, task = 'rankllm')
    else: q_i_ids_to_exclude = None
    for folder in cfg.ranker_args.reward_model_trg_data:
        with open(f'{cfg.dirpath}/results/main_grpo/{folder}/RANK0/train_outputs_final.json', encoding = 'utf-8') as f:
            holder = json.load(f)
            for step in holder:
                for qud_instance_id, line in holder[step].items():
                    ctr_num += 1
                    article_id, anchor_id, answer_id = qud_instance_id.split('_')
                    generated_quds      = line['generated_quds']
                    crit_scores_dict    = line['crit_scores_dict']
                    holder_sorted_cot_outputs = \
                        process_one_set_ranker_cot_prompt(line['ranker_cot_output'], line['ranker_prompts'], 
                                                            generated_quds, placeholder = placeholder)
                                        
                    qud_candidates = []
                    for qi, qud in enumerate(generated_quds):
                        # each crit_scores_dict value is a list of 1 list.
                        criteria_scores = {re.search('(criteria\d)', k).group(): v[0][qi]  for k,v in crit_scores_dict.items()}
                        rationales  = {crit: sorted_cot_outputs[qi] for crit, sorted_cot_outputs in holder_sorted_cot_outputs.items()}
                        qud_candidates.append(QUDCandidate(sysname = qi, qud = qud, 
                                                           rationales = rationales, 
                                                           criteria_scores = criteria_scores))
                    qud_instance = QUDInstance(article_id = int(article_id) \
                                               if type(article_id) == str and article_id.isdigit() else article_id, 
                                               anchor_id = int(anchor_id), answer_id = int(answer_id), 
                                               qud_instance_id = qud_instance_id, qud_human = None, 
                                               qud_candidates = qud_candidates, do_tedq = 'talk' in qud_instance_id)
                    
                    for qud_criteria in cfg.qud_criteria_list:
                        check_key = f'{qud_instance_id}_{qud_criteria}'
                        if cfg.exclude_used_q_i_ids and check_key in q_i_ids_to_exclude: continue
                        rankllm_model = cfg.model.sft.model
                        # a. get the prompt 
                        worker  = criteria2worker[qud_criteria]
                        try: 
                            prompt_msgs, request, gold_docmap = give_rankllm_prompt_request(make_one_rankllm_request, Result, 
                                                                    holder_articles, qud_instance, qud_instance_id, qud_criteria, 
                                                                    exclude_same_scores, context_empty_symbol, worker)
                            if prompt_msgs is None: continue 
                            prompt = sft_tokenizer.apply_chat_template(prompt_msgs, tokenize = False, 
                                                                       add_generation_prompt = True)
                        # there are cases where certain qud_instance don't have rationale for a particular criteria 
                        except KeyError: 
                            ctr_fail += 1
                            print(f'🚨🚨Request creation for {qud_instance_id} failed for qud_criteria {qud_criteria} ')
                            continue

                        assert len(sft_tokenizer.encode(prompt)) <= cfg.grpo_settings.max_prompt_length + 1200, \
                            (len(sft_tokenizer.encode(prompt)), cfg.grpo_settings.max_prompt_length + 1200)

                        # b. also get the completion (for SFT)
                        completion_text = give_rankllm_completion_sequence(cfg, gold_docmap, placeholder)
                        completion_msgs = [{'role': 'assistant', 'content': completion_text}]
                        completion = sft_tokenizer.apply_chat_template(completion_msgs, tokenize = False, 
                                                                       add_generation_prompt = False)
                        assert rankllm_model in COMPLETION_STRIP_REGEX, rankllm_model
                        # remove system messages portion for completion (assistant response)
                        completion = re.sub(COMPLETION_STRIP_REGEX[rankllm_model], '', completion)
                        # remove the generation prompt. we had to set add_generation_prompt = True above 
                        # because the prompts are also to be used at inference time. 
                        completion = completion.replace(COMPLETION_STRIP_GENPROMT[rankllm_model], '')
                        
                        line = {'qud_instance_id': qud_instance_id, 'prompt': prompt, 
                                'qud_criteria': qud_criteria, 'gold_docmap': gold_docmap, 
                                'completion': completion}                        
                        holder_dataset_train.append(line)

    print('TRAINING SPLIT: ratio of failed over available', round(ctr_fail/ctr_num, 4))
    return holder_dataset_train, holder_dataset_eval, holder_articles, holder_quds, \
        eval_ranker_requests, num_cands2qud_idxes, qud_criteria_cfg

def process_one_set_ranker_cot_prompt(ranker_cot_output, ranker_prompts, generated_quds, placeholder = '^IDEN^'):
    assert set(ranker_prompts.keys()) == set(ranker_cot_output.keys()) # keys are qud_criteria
    holder_sorted_cot_outputs = {}
    for qud_criteria, prompt in ranker_prompts.items():
        # recover iden and qud
        # take the last occurence of qud instance attempts
        __ = prompt.split('## QUD Instance Attempts:')[-1].split('\n\nRank')[0]
        # recover every instance line e.g. "[01] ...."
        __ = re.findall(r'(\[\d+\].+)', __)
        # split into iden:qud
        candidates = {}
        for c in __:
            try: 
                # NOTE: there are cases where qud candidate are empty strings
                iden, qud = re.search(r'(\[\d+\])\s*(.*)', c.strip()).groups()
                candidates[iden] = qud.strip()  
            except AttributeError: 
                print('🚨🚨 iden, qud extraction failed for this line', c)
                raise

        # ensure all generated quds present
        assert len(generated_quds) == len(candidates), (generated_quds, candidates)

        c_g_q = copy.deepcopy(generated_quds)
        order2iden = {}
        for iden, qud in candidates.items():
            try: 
                ord             = c_g_q.index(qud)
                order2iden[ord] = iden 
                # in case of exactly the same quds
                c_g_q[ord]      = None
            except: pass
            
            

        cot_output = ranker_cot_output[qud_criteria]
        assert len(cot_output) == 1, cot_output # cot_output is nested in a list
        cot_output = cot_output[0]
        if len(cot_output) == 1 and type(cot_output) is list:
            try:    cot_output = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cot_output[0])
            except: pass
        
        if len(cot_output) == len(candidates) == len(order2iden):
            iden2cot_output = {re.search(r'\[\d+\]', co).group(): co for co in cot_output}
            # replace the "candidate" key in 
            for iden in iden2cot_output:
                # assert iden2cot_output[iden].count(iden) == 1, iden2cot_output[iden]
                # NOTE: there are cases like this:
                # {"candidate": "[04]", "rationale": "the QUD is effectively the same as [03], asking about Honecker's successor. 
                # The ANS directly answers it by stating Egon Krenz succeeded him, making it a case of "direct and explicit", similar to [03]. 
                # Since [04] and [03] are equivalent in their QUD form and they are both "direct and explicit", they share the same ranking.", 
                # "score": 3}
                iden2cot_output[iden] = iden2cot_output[iden].replace(iden, placeholder)

            # this will be in the same order as generated_quds
            holder_sorted_cot_outputs[qud_criteria] = [iden2cot_output[order2iden[ord]] for ord in sorted(order2iden)]
        
        else: 
            print('🚨🚨 num elems in cot_output does not match num candidates', qud_criteria, cot_output, order2iden)
    
    return holder_sorted_cot_outputs

def load_grpo_qud_gen_data(cfg, sft_tokenizer, model_name):
    from utils_agent import (make_prefix_messages_qud_gen, make_one_qud_gen_turn, 
                             text_ctx_anc_ans_agent, PROMPT_CONTENT)
    print('🟧🟧\tPREPARING DATA...')
    sys.path.append('data')
    from dataloader import load_phase1, load_dcqa
    from data_utils import QUDInstance
    import pandas as pd
    reason_msg_version  = cfg.qud_gen.do_reasoning

    # preliminaries
    holder_articles, holder_quds, num_cands2qud_idxes = load_phase1(cfg)
    qud_gen_exemplar_messages = []
    if cfg.qud_gen.do_icl: raise ValueError

    prefix_messages = make_prefix_messages_qud_gen(cfg                  = cfg, 
                                                    model_name          = model_name,
                                                    openai_reasoning    = False, 
                                                    reason_msg_version  = reason_msg_version, 
                                                    exemplar_messages   = qud_gen_exemplar_messages)

    ##### TESTING DATA #####
    # a. load QUDEval data for testing
    holder_dataset_eval = []
    seen = set()
    for qud_instance_id, qud_instance in holder_quds.items():
        # NOTE: we departed from main_phase1.py here 
        # did not exclude all same scores
        # see if req is None: continue
        if qud_instance_id in seen: continue     
        line = one_line_test(text_ctx_anc_ans_agent, make_one_qud_gen_turn, cfg, sft_tokenizer, 
                  holder_articles, qud_instance, qud_instance_id, reason_msg_version, prefix_messages)
        holder_dataset_eval.append(line)
        seen.add(qud_instance_id)

    # b. include QSalience data 
    df_qsalience = pd.read_csv('data/qsalience/data/answerability/answerability.csv', encoding = 'utf-8')
    # keep only human portion of data, and where human_answerability is 2 and above
    df_qsalience = df_qsalience[(df_qsalience.model=='human') & (df_qsalience.human_answerability>=2)].copy()
    # df_qsalience = df_qsalience[(df_qsalience.dataset=='tedq')].copy()
    df_qsalience['ans_candidates'] = df_qsalience.apply(lambda x: give_qsalience_ans_sents(x), axis=1)

    qsal_ctr = 0 
    holder_dataset_eval_qsal = []
    for __, row in df_qsalience.iterrows():
        article_id  = row['article_id']
        anchor_id   = row['sentence_id']
        qud         = row['question']
        for answer_id in row['ans_candidates']:
            if article_id == answer_id: continue
            qud_instance_id = f'{article_id}_{anchor_id}_{answer_id}'
            standalone_instruct = True if qud_instance_id.startswith('talk') or qud_instance_id.startswith('1927') else False
            # if model_name == 'llama': standalone_instruct = False
            if cfg.exp_code != 'SA': standalone_instruct = False
            if qud_instance_id in seen: continue
            qud_instance = QUDInstance(article_id = int(article_id) \
                                       if type(article_id) == str and article_id.isdigit() else article_id, 
                                       anchor_id = int(anchor_id), answer_id = int(answer_id), 
                                       qud_instance_id = qud_instance_id, qud_human = qud, 
                                       qud_candidates = None, do_tedq = 'talk' in qud_instance_id)
            line = one_line_test(text_ctx_anc_ans_agent, make_one_qud_gen_turn, cfg, sft_tokenizer, 
                  holder_articles, qud_instance, qud_instance_id, reason_msg_version, prefix_messages,
                  standalone_instruct = standalone_instruct) # NOTE: standalone_instruct
            holder_dataset_eval_qsal.append(line)
            qsal_ctr += 1
    print('This number of instances from the QSalience data added:', qsal_ctr)

    # c. include TED-Q data 
    from dataloader import load_tedq_data
    tedq_ctr = 0 
    holder_articles_tedq, df_tedq_cut = load_tedq_data(cfg.dirpath)
    for __, row in df_tedq_cut.iterrows():
        qud_instance_id  = row['qud_instance_id']
        qud              = row['content']
        assert row['type'] == 'question'
        article_id, anchor_id, answer_id = qud_instance_id.split('_')
        standalone_instruct = True if qud_instance_id.startswith('talk') or qud_instance_id.startswith('1927') else False
        # if model_name == 'llama': standalone_instruct = False
        if cfg.exp_code != 'SA': standalone_instruct = False
        qud_instance = QUDInstance(article_id = article_id, anchor_id = int(anchor_id), 
                                    answer_id = int(answer_id), qud_instance_id = qud_instance_id,
                                    qud_human = qud, qud_candidates = None, do_tedq = 'talk' in qud_instance_id,)
        line = one_line_test(text_ctx_anc_ans_agent, make_one_qud_gen_turn, cfg, sft_tokenizer, 
                holder_articles_tedq, qud_instance, qud_instance_id, reason_msg_version, prefix_messages,
                standalone_instruct = standalone_instruct) # NOTE: standalone_instruct
        holder_dataset_eval_qsal.append(line)
        tedq_ctr += 1
    print('This number of instances from the TED-Q data added:', tedq_ctr)
    ########################
    
    # load data for training
    test_articles           = set(q_i_id.split('_')[0] for q_i_id in seen)
    holder_dcqa             = load_dcqa() 
    holder_dataset_train    = []
    q_i_ids_to_exclude = extract_qud_instance_ids_used(cfg, task = 'qud_gen')
    dropped = []
    for qud_instance_id, qud_human in holder_dcqa['train'].items():
        article_id, anchor_id, answer_id = qud_instance_id.split('_')
        # exclude all articles (and not just the (article, anc, ans) tuples) in QUDEval
        if qud_instance_id in q_i_ids_to_exclude: continue
        if article_id in test_articles: continue
        qud_instance = QUDInstance(article_id = int(article_id) \
                                   if type(article_id) == str and article_id.isdigit() else article_id, 
                                   anchor_id = int(anchor_id), answer_id = int(answer_id), 
                                   qud_instance_id = qud_instance_id, qud_human = qud_human, 
                                   qud_candidates = None, do_tedq = 'talk' in qud_instance_id)
        prompt_content, *__ = text_ctx_anc_ans_agent(holder_articles, qud_instance, 
                                                  cfg.prompts.qud_gen.prefix.context_empty)
        p_msgs = make_one_qud_gen_turn(cfg, qud_instance, prompt_content,
                                            reason_msg_version  = reason_msg_version, 
                                            prefix_messages     = prefix_messages, 
                                            exemplars           = False)
        prompt = sft_tokenizer.apply_chat_template(p_msgs, tokenize = False, add_generation_prompt = True)   
        if cfg.train_exclude_long_prompts:
            prompt_len = len(sft_tokenizer.encode(prompt))
            if prompt_len >= cfg.grpo_settings.max_prompt_length: 
                # print('🚨🚨 train_exclude_long_prompts TRUE... dropping', qud_instance_id, prompt_len)
                dropped.append(qud_instance_id)
                continue
            
        # also collect context and answer for use in initial rule-based eval
        __, ctx, anc, ans = text_ctx_anc_ans_agent(holder_articles, qud_instance, 
                                            cfg.prompts.rankllm.prefix.context_empty)
        line = {'qud_instance_id': qud_instance_id, 'prompt': prompt, 'completion': qud_human, 
                # NOTE: the following are used for rules-based
                # context is used for compute_givenness (criteria3). we take
                # the QUDEval definition of criteria3 (i.e. context + anchor)
                'context': ctx+anc, 'anchor': anc, 'answer': ans}
        holder_dataset_train.append(line)
    print('🚨🚨\tQUD INSTANCE IDS EXCLUDED:', len(q_i_ids_to_exclude))
    if len(dropped) >0: 
        print('🚨🚨\tQUD INSTANCE IDS DROPPED:', len(dropped))

    return holder_dataset_train, holder_dataset_eval, holder_dataset_eval_qsal, \
            holder_articles, holder_quds, num_cands2qud_idxes

def extract_qud_instance_ids_used(cfg, task = 'qud_gen'):

    def do_one_holder(holder, task):
        idxes = set()
        for gstep, gsholder in holder.items():
            for qud_instance_id in gsholder:
                if task in ['qud_gen']:
                    idxes.add(qud_instance_id)
                elif task in ['rankllm']: 
                    for q_i_id, line in gsholder.items():
                        assert len(set(line['qud_criterias'])) == 1, line['qud_criterias']
                        qud_criteria = line['qud_criterias'][0]
                        check_key    = f'{q_i_id}_{qud_criteria}'
                        idxes.add(check_key)
                else: raise NotImplementedError
        return idxes
        
    q_i_ids_to_exclude = set()

    if cfg.load_peft_ckpt_path is not None and cfg.exclude_used_q_i_ids:

        dp      = os.path.dirname(os.path.dirname(cfg.load_peft_ckpt_path))
        steps   = os.path.basename(cfg.load_peft_ckpt_path).split('-')[-1]
        fp      = os.path.join(dp, f'train_outputs_step{steps}.json')
        if not os.path.exists(fp): fp = os.path.join(dp, 'train_outputs_final.json')
        # in the case of SFT for RM we don't collect the training outputs, so can't 
        # check to remove.
        if os.path.exists(fp): 
            with open(fp, encoding = 'utf-8') as f: holder = json.load(f)    
            q_i_ids_to_exclude.update(do_one_holder(holder, task))
        else: assert 'SFT_rankllm' in fp, fp

    exc_fps = cfg.get('exclude_used_q_i_ids_files', '')
    if exc_fps:
        dp = cfg.dirpath + '/results/main_grpo'
        exc_fps = exc_fps.split(';')
        for fp in exc_fps:
            assert dp not in fp
            with open(f'{dp}/{fp}', encoding = 'utf-8') as f: holder = json.load(f)
            q_i_ids_to_exclude.update(do_one_holder(holder, task))            

    return q_i_ids_to_exclude

def give_rankllm_prompt_request(make_one_rankllm_request, Result, holder_articles, qud_instance, qud_instance_id, 
                                qud_criteria, exclude_same_scores, context_empty_symbol, worker):   
    
    # NOTE: we randomise the order of request.candidates inside make_one_rankllm_request()        
    request         = make_one_rankllm_request(SEED, holder_articles, qud_instance, qud_instance_id, 
                            qud_criteria, exclude_same_scores, context_empty_symbol, worker.rankllm_obj_pack,)
    
    if request is None: 
        prompt_messages, request, gold_docmap = None, None, None
    else:
        rerank_result   = Result(query      = copy.deepcopy(request.query),
                                candidates  = copy.deepcopy(request.candidates), 
                                ranking_exec_summary = [], )
        __, __, prompt_messages = worker.rankllm_model._agent.create_prompt(rerank_result, 
                            0,  len(rerank_result.candidates), 
                            qud_criteria = qud_criteria,)

        # we need this docmap during reward computation. for match_gpt4o_score
        gold_docmap      = {f"[{str(i+1).zfill(2)}]": # make 1-indexed
                                    {'sysname': c.docid,
                                    'score':   c.score,
                                    'qud':     c.doc['text'], 
                                    'rationale': c.rationale} \
                                for i,c in enumerate(request.candidates)}

    return prompt_messages, request, gold_docmap

def give_rankllm_completion_sequence(cfg, gold_docmap, placeholder):
    qud_min_score = cfg.ranker_args.qud_min_score
    qud_max_score = cfg.ranker_args.qud_max_score
    cot_start       = cfg.prompts.rankllm.post.cot_start
    cot_end         = cfg.prompts.rankllm.post.cot_end
    completion = f'{THINK_START} '
    # see _add_few_shot_examples in rank_listwise_os_llm.py of rankllm
    # completion += f'''First of all, I know that the higher the score the better, and the scores range from {qud_min_score} to {qud_max_score}. '''
    completion += f'''Let's think step-by-step and assess each candidate first before ranking them. I think the candidates should get these scores based on the following assessments and reasoning:\n{cot_start}\n'''
    scores = {}
    for iden in gold_docmap:
        scores[iden] = gold_docmap[iden]['score']
        rationale    = gold_docmap[iden]['rationale']
        rationale    = rationale.replace(placeholder, iden)
        rationale    = re.sub(r'\s{1,}', ' ', rationale) # remove trailing whitespace, newlines in JSON
        completion   += rationale + '\n'
    completion += f' {cot_end}' + f' {THINK_END}'
    
    # highest-to-lowest for scores
    sorted_scores = sorted(scores.values(), reverse = True)
    # smallest-to-lowest for idens. to follow prompt instructions
    # and return the smallest iden in cases of ties
    sorted_idens  = sorted(scores.keys(),   reverse = False)
    rank_sequence = []
    for s in sorted_scores:
        pool = [i for i in sorted_idens if scores[i] == s]
        pick = pool[0]
        rank_sequence.append(pick)
        sorted_idens.remove(pick)
    assert len(sorted_idens) == 0, sorted_idens
    assert len(set(rank_sequence)) == len(rank_sequence), rank_sequence
    response = ' > '.join(rank_sequence)

    # see _add_few_shot_examples in rank_listwise_os_llm.py of rankllm
    completion += f'\nBased on the above, I think the ranking should be as follows: {ANS_START} [START] {response} [STOP] {ANS_END}'

    return completion

def give_qsalience_ans_sents(line, min_agree = 2):
    '''
    helper func for QSalience data. Finds the set of candidate answer_ids that were
    selected by 2 or more of the annotators as a candidate answer to a question.
    '''
    ctr = Counter()
    for i, anno in ([0, 'Keziah reason'],
                    [1, 'Kathryn reason'],
                    [2, 'Karim reason'],):
        a = line[anno]
        if type(a) is float and math.isnan(a): pass
        else: 
            a = [aa.strip() for aa in a.split(',')]
            for aa in a: 
                try:    ctr.update([int(aa)])
                except: pass
    return [k for k, v in ctr.items() if v >= min_agree]

def one_line_test(text_ctx_anc_ans_agent, make_one_qud_gen_turn, cfg, sft_tokenizer, 
                  holder_articles, qud_instance, qud_instance_id, reason_msg_version, 
                  prefix_messages, standalone_instruct = False):
    prompt_content, ctx, anc, ans = text_ctx_anc_ans_agent(holder_articles, qud_instance, 
                                            cfg.prompts.qud_gen.prefix.context_empty)
    p_msgs = make_one_qud_gen_turn(cfg, qud_instance, prompt_content,
                                            reason_msg_version  = reason_msg_version, 
                                            prefix_messages     = prefix_messages, 
                                            exemplars           = False,
                                            standalone_instruct = standalone_instruct)
    # NOTE: add generation prompt here because prompt here is used for inference as well for SFT/GRPO RM
    prompt = sft_tokenizer.apply_chat_template(p_msgs, tokenize = False, add_generation_prompt = True)
    # also collect context and answer for use in initial rule-based eval
    line   = {'qud_instance_id': qud_instance_id, 'prompt': prompt, 'completion': None, 
                'context': ctx+anc, 'anchor': anc, 'answer': ans}
    return line

def set_cfg_prompts_for_post_train_rankllm(cfg):
    cfg.gen_args.rankllm.max_new_tokens = 256
    oline = cfg.prompts.rankllm.post.common_cot_replace['oline']
    nline = cfg.prompts.rankllm.post.common_cot_replace['nline']
    if cfg.ranker_args.cot_json:
        # NOTE: this was not in initial model run
        # this affects how the 'common' str is replaced inside _add_post_prompt in rank_listwise_os_llm.py
        add_str = 'It is very important that you give each score and rationale in a string that can be parsed into JSON. '
        nline =  add_str + nline
        cfg.prompts.rankllm.post.common_cot_replace['oline'] = \
            add_str + cfg.prompts.rankllm.post.common_cot_replace['oline']
        # also ensure that reasoning's common has add_str added
        # else replacement fails in _add_post_prompt in rank_listwise_os_llm.py
        cfg.prompts.rankllm.post.common_reasoning_replace['oline'] = \
            cfg.prompts.rankllm.post.common_reasoning_replace['oline'].replace(oline, nline)
    cfg.prompts.rankllm.post.common = cfg.prompts.rankllm.post.common.replace(oline, nline)
    return cfg