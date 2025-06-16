import sys, torch, os, re
from collections import defaultdict
sys.path.append('/home/khan/agentic_qud')

def load_model(cfg, model_name, model_path, model_size = None, do_compile = True,
               force_load_in_nbit = None, device_map = None, skip_pipeline = False,
               optimum_bypass_bsz = None):
    from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                              pipeline)
    if cfg.use_optimum:
        from optimum.nvidia import AutoModelForCausalLM as AutoModelForCausalLM_opt
        from optimum.nvidia.pipelines import pipeline_opt

    
    if model_name in ['gpt4o', 'deepseek', 'o3mini']:
        device = tokenizer = None 
        from openai import OpenAI
        if model_name in ['gpt4o', 'o3mini']:       
            base_url = None
            api_key  = cfg.token_dict['gpt']
        if model_name in ['deepseek']:    
            base_url = "https://api.deepseek.com"
            api_key  = cfg.token_dict['deepseek']
        pipeline_model = OpenAI(api_key = api_key, 
                                base_url = base_url,
                                max_retries = 1, 
                                timeout = 3 * 60)
        return tokenizer, pipeline_model, device, cfg

    c_unsloth = 'unsloth' in model_path
    c_unsloth_4bit = c_unsloth and model_path.lower().endswith("-bnb-4bit")
    if model_size is None: model_size = cfg.model_size

    if device_map is None: 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        device = device_map
    if re.match(r'cuda:*\d*', device) and \
        not cfg.ranker_args.use_past_key_values and model_name not in ['gemma']:     
        attn_implementation = 'flash_attention_2'
    elif c_unsloth_4bit or cfg.ranker_args.use_past_key_values or model_name in ['gemma']:
        # see https://github.com/unslothai/unsloth-zoo/blob/c99dd1f88f6baf3e3dfc5e052674dbd5f149ca22/unsloth_zoo/vllm_utils.py#L1456
        attn_implementation = 'sdpa'
        do_compile = False
        # attn_implementation = 'eager'
    else:                   
        attn_implementation = None
    if cfg.get('flash_attn_override', None) is not None:
        if cfg.flash_attn_override == 'bypass':
            attn_implementation = None
        else: 
            attn_implementation = cfg.flash_attn_override
        
    # https://github.com/huggingface/transformers/issues/32848, very slow with FA
    # rerank tasks are effectively batch size 1 in rankllm setup
    if attn_implementation == 'flash_attention_2' and cfg.bsz > 1:
        # attn_implementation = None
        raise ValueError('🚨\t\tBatch size must be 1 for efficient use of Flash Attention 2.')
    print(f'\n👀 {model_path}... {"attn_implementation".upper()}:', attn_implementation)

    if torch.backends.mps.is_available(): device = 'mps'
    if device == 'mps': cache_loc = '../llm_models'
    else:               
        # cache_loc = f'/home/khan/synalp_me/llm_models/hub/{model_name}'
        cache_loc = cfg.hub_dirpath
    
    padding_side, load_in_4bit, load_in_8bit = give_model_loading_params(model_size)
    
    # cache_dir = f'{cache_loc}/{model_path}'
    cache_dir = f'{cache_loc}'
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    trust_remote_code = {}
    if model_name in ['phi']: 
        trust_remote_code = {'trust_remote_code': True}
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = padding_side, #use_fast = False,
                                    token = cfg.hf_token, **trust_remote_code)

    torch_dtype = torch.bfloat16
    c_gemma         = model_path in ['google/gemma-2-27b-it'] and (load_in_4bit or load_in_8bit)
    c_gemma_unsloth = model_path in ['unsloth/gemma-2-27b-it-bnb-4bit', 'unsloth/gemma-2-9b-it-bnb-4bit']
    if c_gemma or c_gemma_unsloth: 
        torch_dtype = torch.float32

    if   force_load_in_nbit == 4: load_in_4bit, load_in_8bit = True, False
    elif force_load_in_nbit == 8: load_in_4bit, load_in_8bit = False, True
    if   c_unsloth_4bit:          load_in_4bit, load_in_8bit = True, False  
        
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig        
        bnb_args = {'load_in_4bit': load_in_4bit, 'load_in_8bit': load_in_8bit}
        if model_size == 'large': 
            # https://huggingface.co/blog/4bit-transformers-bitsandbytes
            if model_name in ['llama', 'dsr1_llama']: 
                assert load_in_4bit, f'Large {model_name.upper()} model should be loaded in 4-bit quantisation'
            
            if load_in_4bit:
                bnb_args['bnb_4bit_quant_type']         = "nf4"
                bnb_args['bnb_4bit_use_double_quant']   = True
                bnb_args['bnb_4bit_compute_dtype']      = torch.bfloat16
                torch_dtype                             = bnb_args['bnb_4bit_compute_dtype']
        quantization_config = BitsAndBytesConfig(**bnb_args)

    else: quantization_config = {} if model_name == 'commandr_plus' else None
    # other set-able params for BitsAndBytesConfig: 
    # llm_int8_threshold, llm_int8_skip_modules, llm_int8_enable_fp32_cpu_offload
    # llm_int8_has_fp16_weight, bnb_4bit_compute_dtype, bnb_4bit_quant_type,
    # bnb_4bit_use_double_quant, bnb_4bit_quant_storage
    print(f'QUANTISATION SETTING: load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}')

    model_args  = trust_remote_code | {'cache_dir':             cache_dir, 
                                        'torch_dtype':          torch_dtype,
                                        'attn_implementation':  attn_implementation, 
                                        'token':                cfg.hf_token,
                                        'quantization_config':  quantization_config} 
    if device_map is not None: model_args['device_map'] = device_map
    if c_unsloth:
        model_args.pop('quantization_config')
        model_args.pop('torch_dtype')
        model_args['load_in_4bit'] = True

    if cfg.use_optimum:
        if optimum_bypass_bsz is not None:  max_batch_size = optimum_bypass_bsz
        else:                               max_batch_size = cfg.bsz

        model = AutoModelForCausalLM_opt.from_pretrained(model_path, 
                                                        use_fp8=True,
                                                        max_prompt_length = 1024,
                                                        max_output_length = 2048, 
                                                        max_batch_size = max_batch_size,
                                                        **model_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
    
    model.attn_implementation = attn_implementation
    
    ###### add pad token ##############################################
    add_tokens = []
    c1 = padding_side == 'left'
    c2 = cfg.load_ckpt_path is not None
    c3 = tokenizer.pad_token is None
    added_pad_token = False  # check if padding token is already added to tokenizer. If not, add it.
    if c1 and c2 and c3: 
        from transformers import AddedToken
        # for models with no padding token, we add a padding token to the tokenizer
        # see https://huggingface.co/docs/transformers/model_doc/llama3
        cfg.pad_token = "<|pad|>"
        add_tokens.append(AddedToken(cfg.pad_token, rstrip = False, lstrip = False, special = True))
        added_pad_token = True
    else: cfg.pad_token = tokenizer.pad_token
    if getattr(cfg, 'add_tokens', []):
        from transformers import AddedToken
        for token in cfg.add_tokens: 
            # NOTE: cannot be special tokens. for GRPO RM, we have reward funcs that specifically 
            # check for <think> <answer> tags. decoding is skip_special_tokens=True there. 
            # which is what we want because we don't want to keep the eos, eot tokens
            add_tokens.append(AddedToken(token, rstrip = False, lstrip = False, special = False))
    tokenizer.add_tokens(add_tokens)
    print('👀 TOKENIZER CREATED', model_name, 'add_tokens', add_tokens)
    if getattr(cfg, 'pad_token', None) is not None: 
        tokenizer.pad_token     = cfg.pad_token
        tokenizer.pad_token_id  = tokenizer.convert_tokens_to_ids(cfg.pad_token)

    # resize model, set model to self
    if len(tokenizer.get_vocab()) > model.config.vocab_size: 
        print('resizing embedding to:', len(tokenizer.get_vocab()))
        model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    if added_pad_token:
        emb_settings = {}
        for key in ['max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse']:
            emb_settings[key] = getattr(model.model.embed_tokens, key, None)

        # ensure that the Embedding returns zeros for added pad taken 
        emb = torch.nn.Embedding(*model.model.embed_tokens.weight.shape, 
                         padding_idx = tokenizer.pad_token_id,
                         dtype = model.model.embed_tokens.weight.dtype)
        for key, value in emb_settings.items():
            setattr(emb, key, value)
        model.model.embed_tokens.weight.data = emb.weight.data    
        model.model.config.pad_token_id = tokenizer.pad_token_id
    ###################################################################

    # NOTE: do not compile if doing sft with 4/8 bit model
    do_torch_compile = False
    if torch.cuda.is_available() and do_compile:
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] in (7,8,9):
            do_torch_compile = True
    if do_torch_compile:
        # https://huggingface.co/docs/transformers/main/perf_torch_compile
        print('🔥🔥\tdoing torch.compile')
        if cfg.ranker_args.use_past_key_values:
            model.forward = torch.compile(model.forward, 
                            mode = "reduce-overhead", fullgraph = True)
        else: 
            model = torch.compile(model)
        print('🔥🔥\ttorch.compile done.')

    # ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
    c_bnb = load_in_4bit or load_in_8bit
    if not c_bnb and device_map is None: 
        model.to(device)
        print(f'🔥🔥\tmodel shifted to {device}.')
    model.eval()

    if not skip_pipeline:
        pipeline_args = {'model': model, 'tokenizer': tokenizer,
                        'device': device if not c_bnb and device_map is None else None}
        if c_unsloth:
            pipeline_args.pop('device')
        if cfg.use_optimum:
            pipeline_model = pipeline_opt('text-generation', use_fp8 = True, **pipeline_args)
        else: 
            pipeline_model = pipeline('text-generation', **pipeline_args)
    else: pipeline_model = model
    # print('pipeline_model.device', pipeline_model.device)
    return tokenizer, pipeline_model, device, cfg

def load_model_rankllm(cfg, model_name, model_path, tokenizer = None, pipeline_model = None, 
                       device = None, qud_exemplars = None):
    from tools.rank_llm.src.rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
    from tools.rank_llm.src.rank_llm.rerank.reranker import Reranker
    print('🔮🔮\tLoading RankLLM Model:', cfg.ranker_args)

    c_reasoning = False
    c_vllm      = getattr(pipeline_model, 'is_vllm', False)
    if model_name in ['dsr1_llama', 'dsr1_qwen']:
        c_reasoning = True

    c1 = model_name in ['gpt4o', 'deepseek', 'o3mini']
    if  c1:
        constraints_dict = {}
        hf_model = pipeline_model
    
    else: 
        # NOTE: bypass so that prompts for DQG are loaded properly
        if tokenizer is None or pipeline_model is None or device is None:
            tokenizer, pipeline_model, device, cfg = load_model(cfg, model_name, model_path)
        cfg.ranker_args['device'] = str(device)
        
        hf_model = pipeline_model.model if not c_vllm else pipeline_model # NOTE: we pass the vllm LLM engine 

        ###### CONTROLLED GENERATION #####
        # A. constrained decoding controls (generate only tokens in ranking labels prediction sequence)
        # 1. settings to identify non-prediction sequence tokens to set logits to -inf
        # 2. settings to limit the number of tokens generated to prediction sequence length plus a little    
        cfg.ranker_args['rank_pred_seq'] = " > ".join([f"[{str(i+1).zfill(2)}]" for i in range(cfg['ranker_args']['window_size'])])
        cfg.ranker_args['rank_pred_seq_tokens']     = tokenizer.encode(cfg.ranker_args['rank_pred_seq'], add_special_tokens = False)
        cfg.ranker_args['rank_pred_seq_tokens_dec'] = [tokenizer.decode(t) for t in cfg.ranker_args['rank_pred_seq_tokens']]
        print(f'🔮🔮\tRank Prediction Sequence Tokens: {cfg.ranker_args["rank_pred_seq_tokens_dec"]}')

        # a. set max tokens
        for key in ['pad_token_id', 'eos_token_id']:
            cfg.gen_args[key] = getattr(pipeline_model.tokenizer, key)
        if model_name in ['phi']: 
            cfg.gen_args['min_new_tokens'] = 2 # NOTE: we have ranking instances of sizes (2, 3, 4)
            cfg.gen_args['pad_token_id'] = pipeline_model.tokenizer.pad_token_id,
            if getattr(tokenizer, 'vocab', None) is None: 
                tokenizer.vocab = {tokenizer.decode(i): i for t,i in tokenizer.get_vocab().items()}
        elif model_name in ['llama', 'dsr1_llama']:
            terminators = [pipeline_model.tokenizer.eos_token_id,
                        pipeline_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            # in llama, the pad_token_id is not set (rec: use the eos_token_id)
            cfg.gen_args['pad_token_id'] = terminators
            cfg.gen_args['eos_token_id'] = pipeline_model.tokenizer.eos_token_id
        
        # B. add Constrained Beam Search if specified (for ranking tasks)
        # constraints: list of PhrasalConstraint objects (each a sequence of rank labels, e.g. ['[01]', ..., '[20]')
        constraints_dict = {}
        if cfg.rerank_constrained_bs['do_constrained']: 
            from transformers import PhrasalConstraint
            rank_labels     = cfg.ranker_args['rank_pred_seq'].split(' > ')
            constraints     = [PhrasalConstraint(tokenizer(r_l, add_special_tokens=False).input_ids) for r_l in rank_labels]
            num_beams       = cfg.rerank_constrained_bs['num_beams']
            constraints_dict = {'constraints': constraints, 'num_beams': num_beams}

    ###### CONTROLLED GENERATION #####
    # cfg below: used at inference
    for key in ['step_size', 'top_k_candidates']: 
        cfg.ranker_args[key] = cfg.ranker_args['window_size']

    sys_msg_key = 'system_message_reasoning' if c_reasoning else 'system_message'
    rank_agent = RankListwiseOSLLM(model_path       = model_path, 
                                   model_name       = model_name,
                                   hf_model         = hf_model, 
                                   tokenizer        = tokenizer,
                                   cfg              = cfg, 
                                   constraints_dict = constraints_dict,
                                   system_message   = cfg.prompts['rankllm'][sys_msg_key],
                                   qud_exemplars    = qud_exemplars,
                                   **cfg.ranker_args)
    
    # pipeline_model is the reranker
    pipeline_model = Reranker(rank_agent)

    return pipeline_model, cfg, constraints_dict

def model_load_wrapper(cfg, llm_sys, model_path, model_size, 
                       rankllm_use_past_key_values = None, 
                       device_map = None, skip_pipeline = False, 
                       optimum_bypass_bsz = None):    
    # there is no need for CoT prompting for o1/o3 models
    # https://platform.openai.com/docs/guides/reasoning?lang=python&example=research#advice-on-prompting
    if rankllm_use_past_key_values is not None:
        if llm_sys in ['gemma', 'dsr1_llama', 'dsr1_qwen', 'gpt4o', 'deepseek', 'o3mini']:
            cfg.ranker_args.use_past_key_values = False
        else:
            cfg.ranker_args.use_past_key_values = rankllm_use_past_key_values
    
    # a. load LLM once  
    tokenizer, pipeline_model, device, cfg = \
        load_model(cfg, model_name = llm_sys, model_path = model_path, model_size = model_size,
                   force_load_in_nbit = True if model_path in cfg.models_force_4bit else None,
                   device_map = device_map, skip_pipeline = skip_pipeline, 
                   optimum_bypass_bsz = optimum_bypass_bsz)
    
    return tokenizer, pipeline_model, device, cfg

def give_model_loading_params(model_size):
    load_in_4bit = load_in_8bit = False
    padding_side = 'left'
    if model_size == 'large':
        load_in_4bit = True
        load_in_8bit = False

    return padding_side, load_in_4bit, load_in_8bit

def convert_defaultdict(dictionary):
    if isinstance(dictionary, defaultdict) or isinstance(dictionary, dict):
        dictionary = {k: convert_defaultdict(v) for k, v in dictionary.items()}
    
    return dictionary

def process_one_step_rankllm(cfg, pipeline_model, window_size, request, qud_criteria):
    ########################################
    # A. obtain ranking
    prompt = getattr(request, 'prompt', None)
    prompts = [prompt] if prompt else None
    predictions = pipeline_model.rerank_batch(requests = [request],
                        rank_end            = cfg.ranker_args['top_k_candidates'],
                        window_size         = window_size,
                        shuffle_candidates  = cfg.ranker_args['shuffle_candidates'],
                        logging             = cfg.ranker_args['print_prompts_responses'],
                        step                = cfg.ranker_args['step_size'],
                        qud_criteria        = qud_criteria,
                        prompts             = prompts)
    
    ########################################
    # B. extract ranking and info for eval
    proc_predictions = {}
    for prediction in predictions:
        # recover the gold support info:
        # 1. inside Request under 'candidates' attribute
        # 2. each Candidate object has a 'score' attribute
        perm_order      = prediction.perm_order
        gold_docmap     = {f"[{str(i+1).zfill(2)}]": # make 1-indexed
                                {'sysname': c.docid,
                                 'score':   c.score,
                                 'qud':     c.doc['text']} \
                            for i,c in enumerate(request.candidates)} 
        
        # recover the predicted ranking
        # NOTE: prediction is a Result object. It is initialised with a set of documents 
        # in .candidates ... then, as the sliding window moves for a given Request, 
        # the .candidates attribute is updated and sorted.
        predicted_order = {i: r.docid for i,r in enumerate(prediction.candidates)}

        proc_pred = \
            {'query'          : request.query.text,
            'qud_input':        {'context': request.query.context, 
                                'anchor': request.query.anchor, 
                                'answer': request.query.answer, },
            'rerank_products': {'prompt':           
                                [r.prompt   for r in prediction.ranking_exec_summary], 
                                # NOTE: r.response gives the label that is according 
                                # to the order presented to the model
                                'prompt_response':  
                                [r.response for r in prediction.ranking_exec_summary],
                                'gen_output': 
                                [r.gen_output for r in prediction.ranking_exec_summary],
                                'cot_output': 
                                [r.cot_output for r in prediction.ranking_exec_summary],
                                # "predicted_order" is the ranking prediction (docids, 
                                # after reordering docs from least to most)
                                'predicted_order':  predicted_order, 
                                'gold_docmap':      gold_docmap, 
                                'perm_order':       perm_order,},}   
        
        proc_predictions[request.query.qid] = proc_pred
    
    return proc_predictions

def set_peft_weights(cfg, model, adapter_name):
    '''NOTE: model here assumes a non-pipeline object and peft has already been applied '''
    import gc
    from peft.utils import load_peft_weights
    load_peft_ckpt_path = cfg['load_peft_ckpt_path']
    if not load_peft_ckpt_path.endswith('checkpoint-0'):
        # load the checkpoint peft weights
        peft_model_state_dict = load_peft_weights(load_peft_ckpt_path)
        # add adapter name prefix before .weight
        
        peft_model_state_dict = fix_peft_weights(peft_model_state_dict, adapter_name, model.do_compile)
        seen = set()
        peft_keys = set(peft_model_state_dict.keys())
        for mn, mw in model.named_parameters():
            if mn in peft_model_state_dict:
                # print('FOUND MATCH AND ASSIGNED', mn)
                # print(mw.data.mean(dim=0), peft_model_state_dict[mn].data.mean(dim =0))
                dtype = mw.data.dtype
                mw.data = peft_model_state_dict[mn].data.clone().to(dtype)
                seen.add(mn)
        # ensure all checkpoint param weights used
        assert seen == set(peft_model_state_dict.keys()), \
                (seen.difference(peft_keys), peft_keys.difference(seen))
        print('👀 PEFT WEIGHTS SUCCESSFULLY LOADED FROM:', load_peft_ckpt_path)
        model.set_adapter(adapter_name)   
        print('👀 ACTIVE ADAPTER', model.active_adapter)
        del peft_model_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    else: 
        print(f'👀 PEFT WEIGHTS set at {load_peft_ckpt_path}, i.e. ZEROSHOT...')
        model.unload()   
        model.delete_adapter(adapter_name)
        for mn, mw in model.named_parameters():
            assert '.lora' not in mn.lower(), mn

    return model

def fix_peft_weights(peft_model_state_dict, adapter_name, do_compile):
    peft_model_state_dict = {k.replace('.weight', f'.{adapter_name}.weight'): v \
                                        for k, v in peft_model_state_dict.items()}
    
    check_peft_weights_compiled = all('._orig_mod' in k for k,v in peft_model_state_dict.items())
    if do_compile and not check_peft_weights_compiled:
        peft_model_state_dict = {k.replace('.model.model', '.model._orig_mod.model'): v \
                                        for k, v in peft_model_state_dict.items()}
    elif not do_compile and check_peft_weights_compiled:
        peft_model_state_dict = {k.replace('.model._orig_mod.model', '.model.model'): v \
                                        for k, v in peft_model_state_dict.items()}
        
    return peft_model_state_dict

def make_peft_weight_vllm_loadable(load_peft_ckpt_path, fn = 'adapter_model.safetensors'):
    import shutil
    from safetensors.torch import save_file, safe_open
    
    src_path = load_peft_ckpt_path
    dst_path = load_peft_ckpt_path+'_vllm'
    if os.path.exists(dst_path): 
        print('Directory for VLLM-compatible peft weights exists...:', dst_path)
        # print('Directory for VLLM-compatible peft weights exists... deleting:', dst_path)
        # shutil.rmtree(dst_path)
    else: 
        print('Directory for VLLM-compatible peft weights does NOT exist...:', dst_path)
        shutil.copytree(src_path, dst_path) 
    
    tensors = {}
    open_fp = f'{load_peft_ckpt_path}/{fn}'
    with safe_open(open_fp, framework="pt", device=0) as f:
        for k in f.keys():
            # remove key added by torch.compile
            new_k = k.replace('_orig_mod.', '')
            tensors[new_k] = f.get_tensor(k)
    
    save_fp = f'{dst_path}/{fn}'
    save_file(tensors, save_fp)
    print('Directory for VLLM-compatible at:', dst_path)
    return dst_path

def give_vllm_from_peft_ckpt(load_peft_ckpt_path, model_path, temperature = 0, max_tokens = 1024, 
                             hf_token = None, device = 'cuda'):
    import json, os
    if hf_token: os.environ['HF_TOKEN'] = hf_token 

    from vllm.lora.request import LoRARequest
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    if not load_peft_ckpt_path.endswith('_vllm'): 
        load_peft_ckpt_path = make_peft_weight_vllm_loadable(load_peft_ckpt_path)
    with open(f'{load_peft_ckpt_path}/adapter_config.json', encoding = 'utf-8') as f:
        lora_rank = json.load(f)['r']

    # # https://github.com/vllm-project/vllm/issues/7939#issuecomment-2581553850
    dist_keys = [
                # "RANK",
                # "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_NAME",
                "OMP_NUM_THREADS",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHELASTIC_USE_AGENT_STORE",
                "TORCHELASTIC_MAX_RESTARTS",
                "TORCHELASTIC_RUN_ID",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "TORCHELASTIC_ERROR_FILE",]
    restore_dist_vals = {dist_key: os.environ[dist_key] for dist_key in dist_keys if dist_key in os.environ} 
    for dist_key in restore_dist_vals: del os.environ[dist_key]
    # https://vllm-dev.slack.com/archives/C07QP347J4D/p1741149503298339?thread_ts=1740962556.365409&cid=C07QP347J4D
    os.environ['RANK']          = '0'
    os.environ['LOCAL_RANK']    = '0'
    os.environ['WORLD_SIZE']    = '1'

    lora_int = 1
    lora_request = LoRARequest('default', lora_int, lora_path = load_peft_ckpt_path, base_model_name = model_path)
    #  https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
    vllm_model   = LLM(model = model_path, dtype = torch.bfloat16, enable_lora = True, max_lora_rank = lora_rank,
                       disable_async_output_proc = True, device = device,
                       #https://docs.vllm.ai/en/latest/getting_started/examples/torchrun_example.html
                       distributed_executor_backend = "external_launcher",)
    vllm_model.llm_engine.add_lora(lora_request)
    vllm_model.llm_engine.pin_lora(lora_int)
    vllm_model.token_prompt_class = TokensPrompt
    
    tokenizer = vllm_model.get_tokenizer()
    vllm_model.sampling_params = SamplingParams(temperature = temperature, 
                                                max_tokens  = max_tokens,
                                                stop = [tokenizer.eos_token])
    
    device = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model.device

    for dist_key in restore_dist_vals: os.environ[dist_key] = restore_dist_vals[dist_key]

    return tokenizer, vllm_model, device