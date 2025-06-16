import os, torch, re, sys, random, json, hydra, datetime, time
import functools, asyncio, yaml    
import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import login
SEED = 54506
torch.manual_seed(SEED)
torch._dynamo.config.capture_scalar_outputs = True

from utils_model import (load_model_rankllm, process_one_step_rankllm,
                         model_load_wrapper)
sys.path.append('data')
from dataloader import (load_articles_grpo, make_exemplar_objects, 
                        make_one_rankllm_request)
from data_utils import QUDCandidate, QUDInstance

@hydra.main(version_base = None, config_path = '../config', config_name = 'config_grpo.yaml')
def main(cfg: DictConfig):
    os.environ['HF_HOME'] = os.path.dirname(cfg.hub_dirpath)
    os.environ['HF_HUB_CACHE'] = cfg.hub_dirpath
    from utils_file import give_savepath_grpo
    sys.path.append(cfg.dirpath)

    token_path = 'api_tokens.json'
    if cfg.token_dict is None and os.path.exists(token_path):
        with open(token_path, encoding = 'utf-8') as f:
            cfg.token_dict = json.load(f)
        cfg.hf_token   = cfg.token_dict['hf']
        login(cfg.hf_token)
    else: cfg.hf_token = None
    
    if cfg.sft_or_grpo_rm:
        if cfg.load_peft_ckpt_path:
            model = re.search('(gemma|llama|phi|qwen)-', cfg.load_peft_ckpt_path).group(1)
            size  = re.search('(mini|small|unsloth_large)-', cfg.load_peft_ckpt_path).group(1)
            # assert cfg.use_vllm == True, cfg.use_vllm
            cfg.ranker_args.use_past_key_values  = False
            cfg.model.reward.use_past_key_values = False
            cfg.model.reward[cfg.role].model = model
            cfg.model.reward[cfg.role].size  = size
        cfg.ranker_args.reasoning_instruct  = True
        cfg.ranker_args.reasoning_rep_key   = 'nline_guided'
        for qud_criteria in cfg.qud_criteria_list:
            cfg.ranker_args.rm_criteria_settings[qud_criteria] = \
                {"do_cot":                  True,   # set COT to True (does not impact prompt. 
                                                    # but affects how we process the completion)
                                                    # i.e. we want to collect the cot_outputs
                 "cot_json":                True, 
                 "cot_fine":                True, 
                 "num_few_shot_examples":   0,
                 "add_task_decomp_common":  False, 
                 "add_task_decomp_cot":     False}

    qud_criteria    = cfg.role
    own_rank        = int(cfg.architecture.workers[qud_criteria].rank)
    master_rank     = int(cfg.architecture.master.rank)
    cfg.savepath    = os.path.join(give_savepath_grpo(cfg), f'RANK{own_rank}')
    print('🟩🟩\tSavepath created at:', cfg.savepath)
    
    print('🟧🟧\tInitialising WorkerNode...', cfg.architecture.master.addr)
    worker = WorkerNode(cfg, own_rank, master_rank)
    print('🟩🟩\tInitialised WorkerNode...')
    print('🥳🥳\tWorker successfully loaded... listening')
    
    ctr = 0
    outputs_holder = {}
    while True:
        command = worker.receive_command()
        # command are 'r' for rank, 'x' for exit
        if command == "rank":
            qud_cands_list = []
            start = time.time()
            qud_cands_list = worker.receive_qud_info()
            
            qud_cands_list  = qud_cands_list.split(cfg.qud_sep)
            qud_instance_id = qud_cands_list[0]
            qud_cands_list  = qud_cands_list[1:]
            
            scores, prompt, cot_output = worker.process_one_set(qud_instance_id, qud_cands_list)
            
            worker.send_to_master(cfg.score_sep.join([str(i) for i in scores]))
            ctr += 1
            print('qud_cands_list'.upper(), qud_cands_list)
            print('scores'.upper(),         scores)
            print(f'FINISHED: #{ctr}', time.time()- start)

            outputs_holder[qud_instance_id] = {'qud_cands_list': qud_cands_list,
                                               'rankllm_prompt': prompt,
                                               'scores': scores, 
                                               'cot_output': cot_output}
            
            if not os.path.exists(cfg.savepath): 
                os.makedirs(cfg.savepath)
            with open(os.path.join(cfg.savepath, 'ranker_outputs.json'),
                      encoding = 'utf-8', mode = 'w+') as f:
                json.dump(outputs_holder, f)

        elif command == "exit":
            break

    ## 3. Save cfg & model
    cfg_obj = OmegaConf.to_yaml(cfg)
    with open(f'{cfg.savepath}/config.yaml', encoding='utf-8', mode = 'w+') as f:
        yaml.dump(cfg_obj, f)
    
    dist.destroy_process_group()
    raise SystemExit
    
class WorkerNode:
    def __init__(self, cfg, own_rank, master_rank, role = None, launch_nccl = True):
        # role: to bypass and use WorkerNode for openai calling
        self.own_rank       = own_rank
        self.master_rank    = master_rank
        self.world_size     = cfg.world_size
        if cfg.device_num is not None and torch.cuda.is_available():
            self.device     = f"cuda:{cfg.device_num}"

        if launch_nccl:
            torch.cuda.set_device(self.device) 
            ##### NCCL settings #####
            os.environ['MASTER_ADDR']           = cfg.architecture.master.addr
            os.environ['MASTER_PORT']           = str(cfg.architecture.master.port)
            os.environ['NCCL_IB_DISABLE']       = '1' # NOTE: on runpod with NVL, this causes problems
            os.environ['NCCL_SHM_DISABLE']      = '1'
            os.environ['NCCL_P2P_DISABLE']      = '1'
            os.environ['NCCL_SOCKET_IFNAME']    = 'podnet1' # NOTE: runpod GN is not 'eth0' addresses
            os.environ['RANK']                  = str(self.own_rank)
            # os.environ['NCCL_DEBUG']        = 'INFO'
            # os.environ['NCCL_IB_DISABLE']   = '1'
            print('🟧🟧\tInitialising torch.distributed ...')
            init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            print('Initialising at:', init_method, f"rank : {own_rank} world size: {self.world_size}", )
            dist.init_process_group('nccl', 
                                    rank        = own_rank, 
                                    init_method = init_method, 
                                    world_size  = self.world_size,
                                    # device_id   = torch.device(self.device)
                                    timeout = datetime.timedelta(seconds = 60*60*4),
                                    )
            print(f'🟧🟧🟧🟧\tProcess group initialized successfully')
            print(f'🟧🟧🟧🟧🟧🟧\tWorld size: {dist.get_world_size()}')
            print(f'🟧🟧🟧🟧🟧🟧🟧🟧\tRank:    {dist.get_rank()}')
            print(f'🟩🟩🟩🟩🟩🟩🟩🟩\tInitialised torch.distributed ... global rank:', own_rank)
            #########################

        ## 0a. CoT generation settings
        sys.path.append(cfg.dirpath)
        from tools.rank_llm.src.rank_llm.data import Query, Candidate, Request
        self.rankllm_obj_pack = (Query, Candidate, Request)
        c_do_cot = cfg.ranker_args.num_few_shot_examples > 0 and cfg.ranker_args.do_cot
        if c_do_cot and not cfg.sft_or_grpo_rm: 
            cfg.gen_args.rankllm.max_new_tokens = 256
            oline = cfg.prompts.rankllm.post.common_cot_replace['oline']
            nline = cfg.prompts.rankllm.post.common_cot_replace['nline']
            if cfg.ranker_args.cot_json:
                nline = 'It is very important that you give each score and rationale in a string that can be parsed into JSON. ' + nline
            cfg.prompts.rankllm.post.common = cfg.prompts.rankllm.post.common.replace(oline, nline)
        elif cfg.sft_or_grpo_rm:
            from utils_grpo_data import set_cfg_prompts_for_post_train_rankllm
            cfg = set_cfg_prompts_for_post_train_rankllm(cfg)

        # set terminology in prompt
        cfg.prompts.rankllm.prefix.common = cfg.prompts.rankllm.prefix.common.replace('{{terminology}}', 
                                                                cfg.prompts.rankllm.prefix.terminology)
        # ensure common part of prefix has correct symbol for marking empty context.
        cfg.prompts.rankllm.prefix.common = cfg.prompts.rankllm.prefix.common.replace('{{context_empty}}', 
                                                                cfg.prompts.rankllm.prefix.context_empty)
        
        if cfg.ranker_args.do_cot: cfg.rerank_constrained_bs.do_constrained = False

        # load model 
        role = cfg.role if role is None else role
        model_name = cfg.model.reward[role].model
        model_size = cfg.model.reward[role].size
        model_path = cfg.model.models_list[model_size][model_name]
        
        if model_name in ['gpt4o', 'deepseek', 'o3mini']:
            cfg.ranker_args.use_past_key_values = False
        
        if cfg['use_vllm']:
            os.environ['LOCAL_RANK']            = str(self.own_rank)
            cfg.ranker_args.use_past_key_values = False
            from utils_model import give_vllm_from_peft_ckpt
            temperature     = cfg.grpo_settings.temperature
            max_seq_length  = cfg.grpo_settings.max_seq_length
            tokenizer, pipeline_model, device = give_vllm_from_peft_ckpt(cfg['load_peft_ckpt_path'], 
                                                      model_path, temperature, max_seq_length, cfg.hf_token,
                                                      device = self.device)
            pipeline_model.is_vllm = True
            print('Switched pipeline_model to vllm')
        else:
            if cfg['load_peft_ckpt_path'] is None:    
                use_past_key_values = cfg.ranker_args.use_past_key_values
                tokenizer, pipeline_model, device, cfg = \
                    model_load_wrapper(cfg, model_name, model_path, 
                                    model_size, use_past_key_values, self.device)
            else:    
                from utils_model import set_peft_weights
                from utils_grpo import load_llm_model
                from peft import get_peft_model
                from transformers import pipeline
                from main_grpo_master import PEFT_TARGET_MODULES
                if model_name in ['phi']: peft_key = model_name
                else:                     peft_key = 'else'
                cfg.grpo_settings.peft_target_modules = PEFT_TARGET_MODULES[peft_key]

                model, tokenizer, peft_config = \
                    load_llm_model(cfg = cfg, model_name = model_name,
                                    model_path = model_path, model_size = model_size,
                                    max_seq_length      = cfg.grpo_settings.max_seq_length, 
                                    lora_rank           = cfg.grpo_settings.lora_rank,
                                    gpu_memory_utilization = cfg.grpo_settings.gpu_memory_utilization, 
                                    target_modules      = cfg.grpo_settings.peft_target_modules,
                                    device_map          = self.device,
                                    dtype               = torch.bfloat16,
                                    enable_lora         = True if not cfg.use_own_loader else None,
                                    use_own_loader      = cfg.use_own_loader,
                                    optimum_bypass_bsz  = cfg.grpo_settings.num_cands,
                                    do_compile_overide  = True)
                device = model.device
                model = get_peft_model(model, peft_config)
                model.do_compile = True # this was set to True inside load_llm_model
                model = set_peft_weights(cfg, model, adapter_name = 'default')
                model.merge_and_unload() 
                # https://huggingface.co/bertin-project/bertin-alpaca-lora-7b/discussions/1
                # NOTE: model at this point is an OptimizedModule
                pipeline_model = pipeline('text-generation', 
                        model = model.base_model.model._orig_mod, tokenizer = tokenizer)
            pipeline_model.is_vllm = False

        self.qud_criteria = cfg.role if role is None else role
        assert self.qud_criteria in ['criteria2', 'criteria3', 'criteria4']
        self.holder_exemplars    = make_exemplar_objects(cfg)
        self.holder_articles     = load_articles_grpo(cfg)
        exemplar_requests   = {self.qud_criteria: {}}
        for num, (qud_instance_id, qud_instance) in \
            enumerate(self.holder_exemplars[self.qud_criteria].items()):
                rankllm_req = make_one_rankllm_request(SEED + num, self.holder_articles, 
                                    qud_instance, qud_instance_id, self.qud_criteria, 
                                    True, cfg.prompts.rankllm.prefix.context_empty, 
                                    rankllm_obj_pack = self.rankllm_obj_pack)
                exemplar_requests[self.qud_criteria][qud_instance_id] = rankllm_req

        qud_exemplars = list(exemplar_requests[self.qud_criteria].values())

        rm_criteria_settings = getattr(cfg.ranker_args, 'rm_criteria_settings', None)
        if rm_criteria_settings and cfg.do_grpo:
            settings_dict = rm_criteria_settings[self.qud_criteria]
            for k,v in settings_dict.items():
                if cfg.ranker_args[k] == v: continue
                print('🚨🚨CRITERIA SETTING OVERRIDE FOR GRPO', self.qud_criteria, 
                      f"{k}, from: {cfg.ranker_args[k]} to: {v}")
                cfg.ranker_args[k] = v
        
        rankllm_model, cfg, constraints_dict = load_model_rankllm(cfg = cfg, 
                            model_name = model_name, model_path = model_path, 
                            tokenizer = tokenizer, pipeline_model = pipeline_model,
                            device = device, qud_exemplars = qud_exemplars)
        
        self.window_size    = cfg.grpo_settings.num_cands
        self.rankllm_model  = rankllm_model
        self.device         = device
        self.cfg            = cfg    

    async def run_async_process_one_set(self, qud_instance_id, qud_list):
        # Use run_in_executor to run the synchronous method asynchronously
        loop = asyncio.get_event_loop()
        # Create a partial function with the instance method and parameter
        func = functools.partial(self.process_one_set, 
                       qud_instance_id = qud_instance_id, qud_list = qud_list)
        # Run it in the executor
        return await loop.run_in_executor(None, func)

    def process_one_set(self, qud_instance_id, qud_list):
        article_id, anchor_id, answer_id = qud_instance_id.split('_')
        article_id           = int(article_id) if type(article_id) == str and article_id.isdigit() else article_id
        anchor_id, answer_id = int(anchor_id), int(answer_id)

        qud_candidates = []
        for cand_id, qud in enumerate(qud_list):
            qud_cand = QUDCandidate(sysname         = cand_id, 
                                    qud             = qud, 
                                    criteria_scores = None)
            qud_candidates.append(qud_cand)
        qud_instance = QUDInstance(article_id = article_id, anchor_id = anchor_id, 
                                   answer_id = answer_id, qud_instance_id = qud_instance_id, 
                                   qud_human = None, qud_candidates = qud_candidates,
                                   do_tedq = 'talk' in qud_instance_id)

        rankllm_req = make_one_rankllm_request(seed = SEED + random.randint(0, 100), 
                                               holder_articles = self.holder_articles, 
                                                qud_instance= qud_instance, 
                                                qud_instance_id = qud_instance_id, 
                                                qud_criteria = self.qud_criteria, 
                                                exclude_same_scores = False, 
                                                context_empty_symbol = self.cfg.prompts.rankllm.prefix.context_empty,
                                                rankllm_obj_pack = self.rankllm_obj_pack)

        proc_predictions = process_one_step_rankllm(self.cfg, self.rankllm_model, 
                                    self.window_size, rankllm_req, self.qud_criteria)      
        
        assert len(proc_predictions) == 1, len(proc_predictions)
        rp          = proc_predictions[qud_instance_id]['rerank_products']
        prompt      = rp['prompt'][0]
        cot_output  = rp['cot_output']
        gold        = rp['gold_docmap']
        pred_ranks  = {int(k): v for k,v in rp['predicted_order'].items()}
        sys2iden    = {v['sysname']:k for k,v in gold.items() }
        iden2sys    = {v:k for k,v in sys2iden.items()}

        if self.cfg.test_print:
            print('~'*100)
            print('🤓 prompt >>',       rp['prompt'][0])
            print('🤓 cot_output >>',   cot_output)
            print('🤓 gold >>',         gold)
            print('🤓 iden2sys >>',     iden2sys)
            print('🤓 qud_list >>',     qud_list)
        
        assert len(cot_output) == 1, cot_output
        scores = [None for __ in range(len(qud_candidates))]

        # NOTE: self.cfg.reward_funcs_version == 1 check here for backward compat to initial model
        if self.cfg.reward_funcs_version == 1:       
            cos = cot_output[0]
            # HACK: hard code here. sft_or_grpo_rm training
            # leads to COT that has not extracted the JSONs
        else:     
            cos = re.findall(r'{.+}', cot_output[0]) 

        for co in cos:
            iden, score = get_process_one_cot_output(co, self.cfg.reward_funcs_version)
            if self.cfg.test_print: print('iden, score', iden, score, co)
            # could be unrecognisable iden e.g. '[5]' if max is '[4]'
            if iden is not None and iden in iden2sys:
                scores[iden2sys[iden]] = score

        # replace rank as score
        if None in scores:
            non_null = [s for s in scores if s is not None]
            if    not non_null:             min_score, max_score = 0, 1
            else:                           min_score, max_score = min(non_null), max(non_null)
            if    min_score == max_score:   min_score -= 1
            new_scores_pool = np.linspace(min_score, max_score, len(qud_candidates)).tolist()
            new_scores_pool.reverse() # make most to least
            new_scores      = [None for __ in range(len(qud_candidates))]
            
            # system name in pred_ranks is the order in which the QUDs arrive from master node
            # we want to return scores that follow the same order
            for rank, sys_num in pred_ranks.items():
                new_scores[sys_num] = new_scores_pool[rank]
            scores = new_scores
            print('🟥 NEW SCORES >>', new_scores)

        assert None not in scores, scores

        if self.cfg.test_print:
            # NOTE: scores are arranged by order of the completions sent
            # into worker for processing. 
            print('🤓 scores >>',       scores)
            print('~'*100, '\n\n')

        return (scores, prompt, cot_output)

    
    def send_to_master(self, data):
        """Send data back to master node."""
        # Send length first
        length_tensor = torch.tensor([len(data)], dtype=torch.long).to(self.device)#.cuda()
        dist.send(length_tensor, dst = self.master_rank)
        
        # Send actual data
        data_tensor = torch.tensor([ord(c) for c in data], dtype=torch.long).to(self.device)#.cuda()
        dist.send(data_tensor, dst = self.master_rank)
    
    def receive_command(self):
        """Receive command from master node."""
        length_tensor = torch.zeros(1, dtype = torch.long).to(self.device)#.cuda()
        dist.recv(length_tensor, src = self.master_rank)
        
        command_tensor = torch.zeros(length_tensor.item(), dtype = torch.long).to(self.device)#.cuda()
        dist.recv(command_tensor, src = self.master_rank)
        
        # Convert to string until first 0
        command = ''
        for i in command_tensor:
            if i == 0:
                break
            command += chr(i.item())
        return command
    
    def receive_qud_info(self):
        length_tensor = torch.zeros(1, dtype = torch.long).to(self.device)#.cuda()
        dist.recv(length_tensor, src = self.master_rank)

        data_tensor = torch.zeros(length_tensor.item(), dtype = torch.long).to(self.device)#.cuda()
        dist.recv(data_tensor, src = self.master_rank)
        
        # Convert to string until first 0
        command = ''
        for i in data_tensor:
            if i == 0:
                break
            command += chr(i.item())
        return command

pattern_cand1       = re.compile('[\'"]candidate[\\\'"]+\s*:\s*[\\\'"]\[(\d+)\][\\\'"]')
pattern_cand2       = re.compile('\[(\d+)\]')
pattern_score       = re.compile('[\'"]score[\\\'"]+\s*:\s*(\d+)')
pattern_rationale   = re.compile('[\'"]rationale[\\\'"]+\s*:\s*(.+)\s*,\s*[\\\'"]+score[\\\'"]+\s*:\s*.+')

PATTERN_CAND        = re.compile('[\'"]candidate[\\\\\\\'"]+\s*:\s*[\\\\\\\'"]\[(\d+)\][\\\\\\\'"]')
PATTERN_SCORE       = re.compile('[\'"]score[\\\\\\\'"]+\s*:\s*(\d+)')
PATTERN_RATIONALE   = re.compile('[\'"]rationale[\\\\\\\'"]+\s*:\s*(.+)\s*,\s*[\\\\\\\'"]+score[\\\\\\\'"]+\s*:\s*.+')
PATTERN_RANK_SEQ    = re.compile('((?:\[\d+\](?:\s*>\s*\[\d+\])+)(?!.*\[\d+\]\s*>\s*\[\d+\]))', re.DOTALL)

def get_process_one_cot_output(line, reward_funcs_version = 1, ):
    # get identifier
    c_rfv1 = reward_funcs_version == 1
    try:    
        try: can = re.search(pattern_cand1 if c_rfv1 else PATTERN_CAND, line).group(1)
        except: 
            try: can = re.search(pattern_cand2 if c_rfv1 else PATTERN_CAND, line).group(1)
            except: can = None
        can = f"[{can.zfill(2)}]"
    except: can = None
    
    # get score
    try:    sco = int(re.search(pattern_score if c_rfv1 else PATTERN_SCORE, line).group(1))
    except: sco = None

    # # get rationale
    # try:        rat = re.search(pattern_rationale1,line).group(1)
    # except: 
    #     try:    rat = re.search(pattern_rationale2,line).group(1)
    #     except: rat = None
    
    return can, sco #, rat

if __name__ == "__main__":
    main()