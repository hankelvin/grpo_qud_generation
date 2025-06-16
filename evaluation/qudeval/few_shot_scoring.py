import numpy as np
import re, json

def do_one_instance_few_shot(cfg, few_shot_examples, ctx, anc, ans, qud, qud_criteria, client, model_name):
    max_tokens  = 512
    key         = 'rankllm_single_score'
    text        = '## Context: {0} \n## Anchor Sentence: {1} \n## Answer Sentence: {2} '

    # 1. prepare system msg
    messages = []
    sys_message = cfg.prompts['rankllm']['system_message']
    messages.append({'role': 'developer', 'content': sys_message})

    qud_min_score, qud_max_score = cfg.ranker_args.qud_min_score, cfg.ranker_args.qud_max_score
    cot = think_start = think_end = ans_start = ans_end = ''
    c_reason = model_name in ['dsr1_llama', 'dsr1_qwen', 'o3mini']
    if c_reason:
        think_start, think_end  = '<think>', '</think>'     
        ans_start, ans_end      = '<answer>', '</answer>'
        think_str = f'''{think_start} ... ... {think_end} ... ... {{reasoning process truncated here}} ... ... '''
    
    # 2. add exemplars 
    input_context_cands_post = "{}\n\n## QUD Instance Attempt: "
    examples = few_shot_examples[qud_criteria]
    for fse_idx in range(cfg.ranker_args.num_few_shot_examples):
        ### CHANGE START ###
        np.random.seed(54506 + fse_idx)
        ex          = examples[fse_idx]
        perm_order  = np.random.permutation(len(ex['quds']))
        num_sys     = 1 # NOTE -- set to 1 here
        assert len(perm_order) >= num_sys
        perm_order  = perm_order[:num_sys]
        
        prompt = input_context_cands_post.format(ex['query'])

        if fse_idx == 0:
            # add prefix to front 
            prefix = _add_prefix_prompt(cfg, qud_criteria)
            prompt = f"{prefix}\n" + prompt

        response    = ''
        
        if cfg.ranker_args.do_cot:
            cot = f'''First of all, I know that the higher the score the better, and the scores range from {qud_min_score} to {qud_max_score}. '''
            if model_name in ['o3mini']:                        
                cot += '''Secondly, I know that it is very important that I follow the formatting instruction when returning the scoring results. The score and rationale for the candidate has to be a standalone JSON (i.e. it should be a top-level object, not nested inside another object). '''
            if cfg.ranker_args.add_task_decomp_cot:
                # NOTE: we can resuse task_decomposition from rankllm (not present under rankllm_single_score)
                task_decomp_str = cfg.prompts.rankllm.task_decomposition[qud_criteria]
                # remove header
                task_decomp_str = task_decomp_str.replace('## Guide to Scoring Scheme:\n', '').strip()
                # replace you/You with 'I'
                task_decomp_str = re.sub(r'you|You', 'I', task_decomp_str)
                cot += task_decomp_str

            if model_name in ['dsr1_llama', 'dsr1_qwen', 'o3mini']:
                # NOTE: chat template removes everything between think tokens
                cot += f'''{think_str} {ans_start} [S_COT] I think the candidate should get this score based on the following assessment and reasoning: '''
            else: 
                cot += f'''Let's think step-by-step and assess the candidate first before scoring it.\n[S_COT] I think the candidate should get this scores based on the following assessment and reasoning: '''
            
        for rank, pos in enumerate(perm_order):
            attempt     = ex['quds'][pos]
            prompt      += f"{attempt}\n"
            score       = int(ex['scores'][pos])
            rationale   = ex['rationales'][pos]
            rationale   = rationale[0].lower() + rationale[1:]
            
            response    += str(score)
            if cfg.ranker_args.do_cot:
                if cfg.ranker_args.cot_json:
                    cot += f'''\n{{"rationale": "{rationale}", "score": {score}}} . '''
                else:
                    cot += f'''Because: {rationale}... therefore, score of: {score}.'''
        
        prompt  += _add_post_prompt(cfg, c_reason)
        if cfg.ranker_args.do_cot:
            cot     += f' [E_COT] '
            response = cot + f'\nBased on the above, I think the score should be as follows: [START] {response} [STOP] {ans_end}'
        else: 
            if c_reason:
                response = f'{think_str} {ans_start} {response} {ans_end}'
            else: pass

        messages.append({'role': 'user', 'content': prompt})
        messages.append({'role': 'assistant', 'content': response})


    # 3. add instance        
    # ctx, anc, ans, qud  
    if not ctx: ctx = cfg.prompts[key].prefix.context_empty                   
    query = text.format(ctx, anc, ans)
    prompt = input_context_cands_post.format(query) + qud + '\n'
    messages.append({'role': 'user', 'content': prompt})
    
    # 4. dispatch to client 
    response = client.chat.completions.create(
                        model = model_name,
                        messages = messages,
                        temperature=0, #0
                        max_tokens = max_tokens)
    result = response.choices[0].message.content
    
    # 5. collect result
    cot_output = None
    if cfg.ranker_args.do_cot:
        try:
            cot_output = result.split('[START]')[0]
        except: 
            cot_output = result

        if cfg.ranker_args.cot_json:
            try: cot_output = re.findall(r'{.+}', cot_output)
            except: pass

        try: result = re.search(r'\[START\](.+)\[STOP\]', result).group(1).strip()
        except: pass 
    
    match = re.findall(r'[123]', result)
    if not match: score = 0.0
    else: score = float(match[-1])

    return [score], cot_output, prompt # note: score in [] to match the QUDEval GPT setup

def _add_prefix_prompt(cfg, qud_criteria = None):
    key = 'rankllm_single_score'
    terminology         = cfg['prompts']['rankllm']['prefix']['terminology']
    task_decomposition  = cfg['prompts']['rankllm'].task_decomposition[qud_criteria]
    prefix_prompt       = cfg['prompts']['rankllm']['prefix'][qud_criteria]
    
    prefix_dict     = cfg['prompts'][key]['prefix']
    common          = prefix_dict['common']
    common          = common.replace('{{terminology}}', terminology)
    prefix_prompt   = prefix_prompt.replace('''{{common}}''', common)
    
    task_decomp_str = ''
    if cfg.ranker_args.add_task_decomp_common:
        task_decomp_str = task_decomposition
    prefix_prompt = prefix_prompt.replace('{{task_decomposition}}', task_decomp_str)

    return prefix_prompt

def _add_post_prompt(cfg, c_reason):
    key = 'rankllm_single_score'
    post_dict   = cfg['prompts'][key]['post']
    common      = post_dict['common']
    if c_reason: 
        oline = post_dict['common_reasoning_replace']['oline']
        nline = post_dict['common_reasoning_replace']['nline']
        common = common.replace(oline, nline)
    
    return common

def load_examples(cfg, holder_articles, ):
    text = '## Context: {0} \n## Anchor Sentence: {1} \n## Answer Sentence: {2} '
    examples = {}
    with open(f'{cfg.dirpath}/data/exemplars/qudeval_mod_exemplars.json', encoding = 'utf-8') as f:
        qud_exemplars = json.load(f)
    
    rat_key = 'rationale_fine' if cfg.ranker_args.cot_fine else 'rationale'
    for qud_criteria, h1 in qud_exemplars.items():
        examples[qud_criteria] = []
        for q_i_id, h2 in h1.items():
            article_id, anchor_id, answer_id = q_i_id.split('_')
            article_id, anchor_id, answer_id = int(article_id), int(anchor_id), int(answer_id)
            article_contents = holder_articles[article_id]
            
            ctx = [article_contents.contents[i] for i in range(1, anchor_id)]
            if not ctx: ctx = cfg.prompts.rankllm_single_score.prefix.context_empty
            else:       ctx = ' '.join(ctx)
            anc = article_contents.contents[anchor_id]
            ans = article_contents.contents[answer_id]

            scores      = []
            quds        = []
            rationales  = []
            for c in h2.values():
                scores.append(c['new_score'])
                quds.append(c['altered_qud'])
                rationales.append(c[rat_key])
                       
            ex_dict = {'query':     text.format(ctx, anc, ans),
                        'quds':      quds,
                        'scores':    scores,
                        'rationales': rationales}
                
            examples[qud_criteria].append(ex_dict)
    
    return examples