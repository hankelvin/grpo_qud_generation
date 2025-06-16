import sys, re, json, torch, contextlib, copy
from pydantic_core import from_json
from utils_model import process_one_step_rankllm
from torch.nn.attention import SDPBackend, sdpa_kernel
PROMPT_CONTENT      = '## Context: {0} \n## Anchor Sentence: {1} \n## Answer Sentence: {2} '
RE_COT_REASON1      = re.compile(r'<think>(.*?)</think>', re.DOTALL)
RE_COT_REASON2      = re.compile(r'(.*?)<answer>', re.DOTALL)
RE_COT              = re.compile(r'\[S_COT\](.*?)\[E_COT\]', re.DOTALL)
RE_QUD_GEN_REASON1  = re.compile(r'<answer>(.*?)</answer>' , re.DOTALL)
RE_QUD_GEN_REASON2  = re.compile(r'(?:</think>)(.+\s*)', re.DOTALL)
RE_QUD_GEN          = re.compile(r'(?:\[START\])(.+\s*)(?:\[STOP\])', re.DOTALL)

##################################################
##### TOOLS ######################################

def generate_one_qud(cfg, prompt, pipeline_model,  
                     model_name, qud_gen_args, qud_gen_cfg):
    
    inputs = pipeline_model.tokenizer([prompt],
                    return_tensors = 'pt').to(pipeline_model.device)
    
    
    qud_gen_context = sdpa_kernel(SDPBackend.MATH) \
                if pipeline_model.model.attn_implementation == 'sdpa' \
                    else contextlib.nullcontext()

    with torch.no_grad() and qud_gen_context:
        output_ids = pipeline_model.model.generate(**inputs, **qud_gen_args, 
                                                generation_config = qud_gen_cfg)
    
    assert output_ids.shape[0] == 1, output_ids.shape
    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs_dec = pipeline_model.tokenizer.decode(output_ids, 
                                    skip_special_tokens = True)
    
    if cfg.qud_gen.do_cot:
        c_reason = True if model_name in ['dsr1_llama', 'dsr1_qwen', 'o3mini'] else False
        
        outputs, cot_output = qud_gen_cot_processor(outputs_dec, cfg.qud_gen.do_cot, 
                            cfg.qud_gen.cot_json, c_reason, model_name)
    else: 
        outputs     = outputs_dec
        cot_output  = None
    
    return outputs.strip(), cot_output


def rank_score_critique_one_qud(cfg, rankllm_model, window_size, 
                                request, qud_criteria):
    
    proc_predictions = process_one_step_rankllm(cfg, rankllm_model, 
                                window_size, request, qud_criteria)

    assert len(proc_predictions) == 1
    for q_i_id, proc_pred in proc_predictions.items():
        pass 

    return q_i_id, proc_pred


def give_qud_gen_gen_args(model_name, tokenizer):
    from transformers import StoppingCriteriaList, StoppingCriteria
    ####################################
    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_token_ids):
            self.stop_token_ids = stop_token_ids

        def __call__(self, input_ids, scores, **kwargs):
            return any([inp[-1].item() in self.stop_token_ids for inp in input_ids])
    ####################################
    gen_args = {}
    if model_name in ['llama']:
        terminators = [tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        gen_args = {'eos_token_id': terminators, 
                        # in llama, the pad_token_id is not set (rec: use the eos_token_id)
                        'pad_token_id': tokenizer.eos_token_id,}
    elif model_name in ['qwen']:
        # in mistral, the pad_token_id is not set (rec: use the eos_token_id)
        gen_args = {'pad_token_id': tokenizer.eos_token_id,}
    elif model_name in ['dsr1_llama', 'dsr1_qwen']:
        # there is a bug in the tokenizer for deepseek models '｜' instead of '|' used in the chat_template
        # this causes the special tokens to be broken up. 
        stop_tokens = [tokenizer.eos_token_id]
        if model_name in ['dsr1_llama']: 
            # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/generation_config.json
            stop_tokens.extend([128001, 128008, 128009])
        elif model_name in ['dsr1_qwen']:
            # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json
            stop_tokens.extend([151645, 151643])
        stopping_criteria = StoppingCriteriaList([StopOnToken(set(stop_tokens))])
        gen_args = {'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.eos_token_id,
                    'stopping_criteria': stopping_criteria}
        
    return gen_args

##################################################
##### PROMPT HELPERS #############################

def make_prefix_messages_qud_gen(cfg, model_name, openai_reasoning, 
                                 reason_msg_version, exemplar_messages):
    # 1. system_message
    # gemma: has no system message. 
    # openai: should be placed as developer message
    # deepseek models: system message folded into 1st user message (i.e. instructions)
    prefix_messages = []
    sys_msg_key = 'system_message_reasoning' if (openai_reasoning or reason_msg_version) else 'system_message'
    sys_message = cfg.prompts.qud_gen[sys_msg_key]

    # a. collect terminology and criteria
    term_criteria_str = ''
    if cfg.qud_gen.add_task_decomp_cot:
        term_criteria_str += cfg.prompts.rankllm.prefix.terminology
    if cfg.qud_gen.add_criteria_desc:
        if cfg.qud_gen.add_task_decomp_cot: 
            term_criteria_str += '\n\n\n###########\nNext, here are the set of Criteria you must follow to write a good QUD. It is very important that the QUD you write must score the maximum points on each of the Criteria. \n\n'
        else: 
            term_criteria_str += '\n\n\n###########\nFirstly, here are the set of Criteria you must follow to write a good QUD. It is very important that the QUD you write must score the maximum points on each of the Criteria. \n\n'
        # reuse the terminlogy used in rankllm
        for crit in cfg.qud_criteria_list:
            # reuse the criteria guide/scoring scheme used in rankllm
            crit = cfg.prompts.rankllm.prefix[crit].replace('{{common}}', '').replace('{{task_decomposition}}', '')
            term_criteria_str += crit.strip() +'\n\n'   
    # b. collect task decomposition instruction (if specified)
    if cfg.qud_gen.add_task_decomp_common: 
        term_criteria_str += f'\n\n\n{cfg.prompts.qud_gen.task_decomposition}'
    if cfg.qud_gen.add_term_desc_loc == 'system':
        sys_message += term_criteria_str
        term_criteria_str = ''

    # see https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B#usage-recommendations
    # but we add the sys prompt to the user message
    # certain models (e.g. gemma) require the 1st msg for input to apply_chat_template to be a user msg
    c_no_sys_msg = model_name in ['dsr1_llama', 'dsr1_qwen', 'gemma']
    if c_no_sys_msg: 
        prefix_messages.append({'role': 'user','content': f"{sys_message}\n\n"})
        assert prefix_messages[-1]['role'] == 'user', prefix_messages[-1]
        prefix_messages[-1]['content'] += term_criteria_str
    else: 
        if openai_reasoning:
            prefix_messages.append({'role': 'developer', 'content': sys_message})
        else:
            prefix_messages.append({'role': 'system', 'content': sys_message})
        
        prefix_messages.append({'role': 'user', 'content': term_criteria_str})

    # 2. add_few_shot_examples
    num_few_shot_examples = cfg.qud_gen.num_few_shot_examples
    for i, turn_msgs in enumerate(exemplar_messages[:num_few_shot_examples]):
        if i == 0:
            assert prefix_messages[-1]['role'] in ['user', 'developer', 'system']
            user, asst = turn_msgs
            prefix_messages[-1]['content'] += f"\n\n\n{user['content']}"
            prefix_messages.append(asst)
        else: 
            prefix_messages.extend(turn_msgs)

    return prefix_messages

def make_one_qud_gen_turn(cfg, qud_instance, prompt_content, reason_msg_version = False,
                          prefix_messages = None, exemplars = False, standalone_instruct = False): 
    '''
    params:
    - standalone_instruct (bool): whether to add instructions for the model to ensure the QUD generated
    is standalone (i.e. does not refer to parts of the prompt such as "the CTX mentions..."). Could be useful
    when applying tuned models to inputs of a different distribution.
    '''
    qud = qud_instance.qud_human    
    if reason_msg_version:
        prompt_content += f"\n\n{cfg.prompts.qud_gen.post.common_reason}\n"
    else:
        prompt_content += f"\n\n{cfg.prompts.qud_gen.post.common}\n"

    sa_instruct_prefix = sa_instruct_post = ''
    if standalone_instruct:
        sa_instruct_post = '''Write a QUD given the CTX, ANC and ANS above, which are from a speech. '''
        prompt_content = prompt_content.replace('Write a QUD given the CTX, ANC and ANS above. ', sa_instruct_post)


        sa_instruct_prefix = '''## Guide for QUDs on speeches:\nFor speeches, the QUD is to be written in the form of a question that could come to the mind of a person listening to the speech so far (i.e. past the sentences in the CTX, and right at the point where the ANC ends). The task is similar to that for normal QUDs (i.e. give a QUD that scores the maximum on all of the Criteria). The difference is that all, or some, of the CTX, ANC and ANS for a speech is in the first person. Nonetheless, the QUD must always stay in the second or third person; make sure it remains grammatical and fluent sounding. The following are very important additional requirements for QUDs on speeches:\n- the QUD has to be standalone from the instruction, which means that it must not refer to elements that are specific to the instructions (i.e. the QUD **must not** refer to the answer or contain terms such as: "CTX", "ANC", "ANS", "the context", "the anchor", "the answer sentence" etc, or their equivalents). This is because the QUD will be put to use later on with only the text corresponding to the CTX and ANC. Violating this requirement means the QUD will not be useful at all and so its score for all of the Criteria will drop to the lowest;\n- the QUD has to be short (ideally around seven words), precise and direct -- it **must not** hedge or include phrases such as "according to the speaker", "mentioned in the speech" or other generic references (for e.g. to the speaker, to the speech, or their equivalents) as these do not add to the information content of the question. The reason for this is because it will cost a lot more to process longer QUDs when they are used later on. If this requirement is violated, it means the QUD is much less useful and so its score for all of the Criteria will also drop to the lowest.\n'''

    if prefix_messages is not None:
        messages = copy.deepcopy(prefix_messages)
    else:
        messages = []

    if messages and messages[-1]['role'] == 'user':   
        messages[-2]['content'] += sa_instruct_prefix
        messages[-1]['content'] += prompt_content
    else:
        messages[-1]['content'] += sa_instruct_prefix
        messages.append({'role': 'user', 'content': prompt_content})
    
    
    if exemplars:
        messages.append({'role': 'assistant', 'content': qud})
    
    return messages


def text_ctx_anc_ans_agent(holder_articles, qud_instance, context_empty_symbol = '[EMPTY]'):
    article_id = qud_instance.article_id
    article_contents = holder_articles[article_id]
    
    ctx = qud_instance.extract_context(article_contents)
    # if Anchor is 1st sentence (i.e. nothing in context, use context_empty symbol)
    if ctx: ctx = ' '.join(ctx)
    else: ctx = context_empty_symbol
    
    anc = qud_instance.extract_anchor(article_contents)
    ans = qud_instance.extract_answer(article_contents)
    
    prompt_content = PROMPT_CONTENT.format(ctx, anc, ans)
    
    return prompt_content, ctx, anc, ans


def qud_gen_cot_processor(outputs, do_cot, cot_json, c_reason, model_name):
    cot_output = None
    if do_cot and not c_reason:
        # ~~~ 1. EXTRACT COT ~~~
        try:
            cot_output = extract_cot(outputs, cot_json, reason = False)
        except Exception as e: 
            if not model_name in ['qwen']: # qwen frequently fails to follow and generate cot
                print(f'COT EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

        # ~~~ 2. EXTRACT QUD GEN ~~~
        try: 
            outputs = extract_qud_gen(outputs, reason = False)
        except Exception as e: 
            print(f'QUD GEN EXTRACTION (FAILED) - \n🟥{e}\n', outputs)
    
    elif c_reason:
        # ~~~ 1. EXTRACT COT ~~~
        try:
            cot_output = extract_cot(outputs, cot_json, reason = True)
        except Exception as e: 
            print(f'COT EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

        # ~~~ 2. EXTRACT QUD GEN ~~~
        try: 
            outputs = extract_qud_gen(outputs, reason = True)
        except Exception as e:
            print(f'QUD GEN EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

    return outputs, cot_output

def extract_cot(outputs, cot_json, reason = False):
    if reason: 
        try: 
            cot_output = re.search(RE_COT, outputs).group(1)
            cot_output = cot_output.strip()
        except: 
            try: 
                cot_output = re.search(RE_COT_REASON1, outputs).group(1)
            except: 
                cot_output = re.search(RE_COT_REASON2, outputs).group(1)
        cot_output = cot_output.strip()
    else:
        if cot_json:
            cot_output = outputs
            cot_output = re.findall(r'{.+}', outputs)
            cot_output = [co.strip() for co in cot_output]
            for i, co in enumerate(cot_output): 
                # ensure that saved as escaped string (loadable with pydantic from_json)
                try: 
                    co = json.dumps(from_json(co))
                    cot_output[i] = co
                except: 
                    pass
        else:
            cot_output = re.search(RE_COT, outputs).group(1)
            cot_output = cot_output.strip()

    return cot_output

def extract_qud_gen(outputs, reason = False):
    try:
        # look for [START] [STOP] first
        outputs = re.search(RE_QUD_GEN, outputs).group(1)
    except:
        if reason:
            try: 
                outputs = re.search(RE_QUD_GEN_REASON1, outputs).group(1)
            except: 
                outputs = re.search(RE_QUD_GEN_REASON2, outputs).group(1)
        
    return outputs.strip()