import os, sys, torch, hydra, json, datetime, yaml, copy
# os.environ['HF_HOME'] = '/home/khan/synalp_me/llm_models'
# os.environ['HF_HUB_CACHE'] = '/home/khan/synalp_me/llm_models/hub'
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import socket
SEED = 54506
torch.manual_seed(SEED)
torch._dynamo.config.capture_scalar_outputs = True
torch.compiler.reset()
PEFT_TARGET_MODULES = {'phi': ['qkv_proj', 'down_proj', 'gate_up_proj'], 
                        'else':["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]}

@hydra.main(version_base = None, config_path = '../config', config_name = 'config_grpo.yaml')
def main(cfg: DictConfig):
    from utils_file import give_savepath_grpo
    from huggingface_hub import login

    token_path = 'api_tokens.json'
    if cfg.token_dict is None and os.path.exists(token_path):
        with open(token_path, encoding = 'utf-8') as f:
            cfg.token_dict = json.load(f)
        cfg.hf_token   = cfg.token_dict['hf']
        login(cfg.hf_token)
    else: cfg.hf_token = None

    if cfg.grpo_settings.initial_rules_based_steps == 1e9: 
        cfg.world_size = None
        print('⛔️⛔️\tWorld size set to None', cfg.world_size)

    if cfg.sft_or_grpo_rm:
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
        
    own_rank = master_rank      = int(cfg.architecture.master.rank)
    cfg.savepath                = os.path.join(give_savepath_grpo(cfg), f'RANK{own_rank}')
    cfg.savepath_train_outputs  = os.path.join(cfg.savepath, 'train_outputs.json')
    cfg.savepath_test_outputs   = os.path.join(cfg.savepath, 'test_outputs.json')
    print('🟩🟩\tSavepath created at:', cfg.savepath)

    print('🟧🟧\tInitialising DistributedLLMManager...', cfg.architecture.master.addr)
    manager = DistributedLLMManager(cfg, master_rank)
    print('🟩🟩\tInitialised DistributedLLMManager...')
    
    print('🟧🟧\tSETTING UP MASTER NODE...')
    manager.init_master_node()
    print('🟩🟩\tMASTER NODE SET UP...')
    
    #########################################
    ## 1. DO training
    if not cfg.do_eval_only_bypass:
        print('🟧🟧\tStarting training (master)...')
        ## 1. Run training and save generated outputs across steps
        print("HERE 10", manager.post_trainer.model.device)
        manager.post_trainer.train()

        # NOTE: for SFT, we don't collect the generated outputs during training
        # did not mod the train step there.
        if manager.post_trainer.generated_outputs:
            fp = cfg.savepath_train_outputs.replace('.json', '_final.json')
            with open(fp, encoding = 'utf-8', mode = 'w+') as f:
                json.dump(manager.post_trainer.generated_outputs, f)
    #########################################
    
    #########################################
    ## 2. Run test and save generated outputs 
    # a. on eval_dataset for qud_gen
    # b. on eval_ranker_requests for rankllm
    # c. for qud_gen, also eval_dataset_qsal (QSalience data)
    print('🟧🟧\tStarting inference (master)...')
    holder_eval_qsal_outputs = None
    eval_bsz = 64
    if   cfg.grpo_settings.grpo_task == 'qud_gen':
        holder_eval_outputs = manager.post_trainer.inference_on_eval_dataset_qud_gen(manager.eval_dataset, bsz = eval_bsz)
        if manager.eval_dataset_qsal is not None:
            holder_eval_qsal_outputs = \
                manager.post_trainer.inference_on_eval_dataset_qud_gen(manager.eval_dataset_qsal, bsz = eval_bsz, tighter_gen = True)
    elif cfg.grpo_settings.grpo_task == 'rankllm':
        holder_eval_outputs = manager.post_trainer.inference_on_eval_dataset_rankllm(manager.eval_ranker_requests)
    
    if cfg.do_eval_only_bypass:
        import re
        step = re.search(r'\d+', os.path.basename(cfg.load_peft_ckpt_path)).group()
        task = cfg.grpo_settings.grpo_task
        # NOTE: 🚨🚨 hardcoded to exp launch values
        if task in ['qud_gen']:
            c1 = 'LPC' not in cfg.load_peft_ckpt_path and step == '2000'
            c2 = 'LPC'     in cfg.load_peft_ckpt_path and step == '2000'
            # NOTE: TODO: remove hard coded here
            if cfg.reward_funcs_version == 1:
                c2 = 'LPC'     in cfg.load_peft_ckpt_path and step == '500'
        elif task in ['rankllm']:
            c1 = 'LPCR-'   in cfg.load_peft_ckpt_path and step == '8000'
            c2 = 'SFT_'    in cfg.load_peft_ckpt_path and step == '6729'
        if c1 or c2: #HACK
            fp = cfg.savepath_test_outputs.replace('.json', '_final.json')
        else:
            fp = cfg.savepath_test_outputs.replace('.json', f'_step{step}.json')
    else: 
        fp = cfg.savepath_test_outputs.replace('.json', '_final.json')
    # for separating standalone_instructions set-up for QSAL/TEDQ
    exp_code_str = f'_{cfg.exp_code}.json' if cfg.exp_code is not None else '.json'
    fp = fp.replace('.json', exp_code_str)
    print('🤓🤓🤓🤓🤓SAVE PATH IS HERE', fp)
    with open(fp, encoding = 'utf-8', mode = 'w+') as f:
        json.dump(holder_eval_outputs, f)
    if holder_eval_qsal_outputs is not None:
        qsal_fp = fp.replace('.json', '_qsal.json')
        print('🤓🤓🤓🤓🤓QSAL SAVE PATH IS HERE', qsal_fp)
        with open(qsal_fp, encoding = 'utf-8', mode = 'w+') as f:
            json.dump(holder_eval_qsal_outputs, f)
    #########################################

    #########################################
    ## 3. Save model, clean up
    if not cfg.do_eval_only_bypass:
        ## 3. Save cfg & model
        cfg_obj = OmegaConf.to_yaml(cfg)
        with open(f'{cfg.savepath}/config.yaml', encoding='utf-8', mode = 'w+') as f:
            yaml.dump(cfg_obj, f)

        output_dir = os.path.join(cfg.savepath, 'model.ckpt')
        if cfg.use_own_loader:
            manager.post_trainer.accelerator.save_state(output_dir = output_dir, 
                                                        safe_serialization = False)
        
        else: 
            manager.sft_model.save_pretrained_merged(output_dir, manager.sft_tokenizer, 
                                                    save_method = "lora",)
        
        # terminate processes 
        if manager.world_size is not None and manager.world_size > 0:
            for worker_rank in range(1, cfg.world_size):
                manager.send_command_to_worker(worker_rank, 'exit', '')
                pass

            dist.destroy_process_group()
    #########################################

class DistributedLLMManager:
    def __init__(self, cfg, master_rank = 0):
        self.cfg            = cfg
        self.master_addr    = socket.gethostbyname(socket.gethostname()) \
                                if   cfg.architecture.master.addr is None \
                                else cfg.architecture.master.addr
        self.master_rank    = master_rank
        self.world_size     = cfg.world_size
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cfg.device_num is not None and torch.cuda.is_available():
            self.device     = f'cuda:{cfg.device_num}'
            torch.cuda.set_device(self.device) 
        
        if cfg.get('use_accelerate', False): 
            # issue with device number using accelerate and fsdp
            # https://github.com/huggingface/accelerate/issues/2963
            self.device = 'cpu'
        print('self.device', self.device)
        if self.world_size is not None and self.world_size > 0:
            ##### NCCL settings #####
            os.environ['MASTER_ADDR']           = cfg.architecture.master.addr
            os.environ['MASTER_PORT']           = str(cfg.architecture.master.port)
            os.environ['HOSTNAME']              = os.environ['MASTER_ADDR'] # NOTE: important for runpod Global Networking
            master_host_port                    = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            # os.environ['WORLD_SIZE']            = str(self.world_size)    # NOTE: do not set, starts up DDP in accelerate
            # os.environ['NCCL_DEBUG']            = 'INFO'
            # os.environ['NCCL_DEBUG_SUBSYS']     = 'ALL'
            os.environ['CUDA_LAUNCH_BLOCKING']  = '1'
            os.environ['NCCL_IB_DISABLE']       = '1' # NOTE: on runpod with NVL, this causes problems
            os.environ['NCCL_SHM_DISABLE']      = '1'
            os.environ['NCCL_P2P_DISABLE']      = '1'
            os.environ['NCCL_SOCKET_IFNAME']    = 'podnet1' # NOTE: runpod GN is not 'eth0' addresses
            os.environ['RANK']                  = str(self.master_rank)

            print('🟧🟧\tInitialising torch.distributed ...')
            init_method = f"tcp://{master_host_port}"
            print('Initialising at:', init_method, f"rank : {self.master_rank} world size: {self.world_size}", )
            dist.init_process_group('nccl', 
                                    rank        = self.master_rank, 
                                    init_method = init_method,
                                    world_size  = self.world_size,
                                    timeout = datetime.timedelta(seconds = 60*60*2),
                                    )
            print(f'🟧🟧🟧🟧\tProcess group initialized successfully')
            print(f'🟧🟧🟧🟧🟧🟧\tWorld size: {dist.get_world_size()}')
            print(f'🟧🟧🟧🟧🟧🟧🟧🟧\tRank:    {dist.get_rank()}')
            print(f'🟩🟩🟩🟩🟩🟩🟩🟩\tInitialised torch.distributed ... global rank:', self.master_rank)
        #########################
        
    def init_master_node(self):                
        from utils_grpo import load_llm_model
        from main_grpo_worker import WorkerNode

        ## 0a. CoT generation settings
        c_do_cot = self.cfg.qud_gen.do_icl and self.cfg.qud_gen.num_few_shot_examples > 0 and self.cfg.qud_gen.do_cot
        if c_do_cot: 
            self.cfg.gen_args.qud_gen.max_new_tokens = 256
            oline = self.cfg.prompts.qud_gen.post.common_cot_replace['oline']
            nline = self.cfg.prompts.qud_gen.post.common_cot_replace['nline']
            self.cfg.prompts.qud_gen.post.common = self.cfg.prompts.qud_gen.post.common.replace(oline, nline)
        
        ## 0b. set terminology in prompt
        self.cfg.prompts.qud_gen.prefix.common = self.cfg.prompts.qud_gen.prefix.common.replace('{{terminology}}', 
                                                                self.cfg.prompts.rankllm.prefix.terminology)

        # 1. load qud_gen model (to be sft)
        self.model_name = self.cfg.model.sft.model
        if self.model_name in ['phi']:  peft_key = self.model_name
        else:                           peft_key = 'else'
        self.cfg.grpo_settings.peft_target_modules = PEFT_TARGET_MODULES[peft_key]

        self.model_size = self.cfg.model.sft.size
        self.model_path = self.cfg.model.models_list[self.model_size][self.model_name]
        # NOTE: self.sft_model is a not a pipeline object (use_own_loader defaults to True in config)
        self.sft_model, self.sft_tokenizer, self.peft_config = \
            load_llm_model(cfg = self.cfg, model_name = self.model_name,
                            model_path = self.model_path, model_size = self.model_size,
                            max_seq_length      = self.cfg.grpo_settings.max_seq_length, 
                            lora_rank           = self.cfg.grpo_settings.lora_rank,
                            gpu_memory_utilization = self.cfg.grpo_settings.gpu_memory_utilization, 
                            target_modules      = self.cfg.grpo_settings.peft_target_modules,
                            device_map          = self.device,
                            dtype               = torch.bfloat16,
                            enable_lora         = True if not self.cfg.use_own_loader else None,
                            use_own_loader      = self.cfg.use_own_loader,
                            optimum_bypass_bsz  = self.cfg.grpo_settings.num_cands)
        print('PEFT CONFIG:', self.peft_config)
        print('🟩🟩\tLLM MODEL LOADED...')

        # 2. load data for training and inference (test)
        self.train_dataset, self.eval_dataset, self.eval_dataset_qsal, \
            self.eval_ranker_requests, num_cands2qud_idxes, qud_criteria_cfg = \
                self.give_datasets()
        
        # 3. set rewards 
        reward_funcs = self.give_reward_funcs()
        
        if   self.cfg.grpo_settings.grpo_task in ['qud_gen']:
            reward_weights = [1.0 if rf.__name__.startswith('criteria') else 0.5 for rf in reward_funcs]
        elif self.cfg.grpo_settings.grpo_task in ['rankllm']:
            reward_weights = [1.0 if rf.__name__ in \
                              ['check_score_dicts_validity', 'match_gpt4o_score', 'score_rank_consistency'] \
                                else 0.5 for rf in reward_funcs]
        
        # 4. instantiate trainer
        if self.cfg.do_grpo:
            from utils_grpo import give_grpo_trainer
            self.post_trainer = give_grpo_trainer(self.cfg, self.sft_model, self.sft_tokenizer, 
                                                train_dataset   = self.train_dataset, 
                                                eval_dataset    = self.eval_dataset,
                                                eval_dataset_qsal    = self.eval_dataset_qsal,
                                                eval_ranker_requests = self.eval_ranker_requests,
                                                num_cands2qud_idxes  = num_cands2qud_idxes,
                                                reward_funcs    = reward_funcs,
                                                reward_weights  = reward_weights,
                                                peft_config     = self.peft_config)
            self.post_trainer.post_init()
        else: 
            from sft_trainer_mods import give_sft_trainer
            self.post_trainer = give_sft_trainer(self.cfg, self.sft_model, self.sft_tokenizer, 
                                                train_dataset   = self.train_dataset, 
                                                eval_dataset    = self.eval_dataset, 
                                                eval_ranker_requests = self.eval_ranker_requests,
                                                num_cands2qud_idxes  = num_cands2qud_idxes,
                                                peft_config     = self.peft_config)
            self.post_trainer.post_init()
        self.post_trainer.qud_criteria_cfg = qud_criteria_cfg

        # 5. add rules-based evaluator / llmqalogprobs
        if self.cfg.grpo_settings.grpo_task in ['qud_gen']:
            sys.path.append(self.cfg.dirpath)
            from evaluation.qud_rules_based import QUDRulesBasedEvaluator
            c1 = self.cfg.grpo_settings.initial_rules_based_steps > 0
            c2 = self.cfg.grpo_settings.swop_rules_llmqalogprobs
            if c1:
                reward_model_device = 'cpu'
                if torch.cuda.device_count() > 1:
                    to_try = list(range(1, torch.cuda.device_count()))
                    while True:
                        reward_model_device = f'cuda:{to_try.pop(0)}'
                        if reward_model_device != self.device: break
                if self.cfg.rm_device_override is not None: 
                    reward_model_device = self.cfg.rm_device_override
                self.post_trainer.rules_based_evaluator = QUDRulesBasedEvaluator(device = reward_model_device)
                print('🟩🟩\tRULES-BASED EVALUATOR LOADED')
                if c2:
                    import gc
                    from utils_grpo import give_llmqalogprobs_model
                    self.post_trainer.rules_based_evaluator.classifier = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.post_trainer.llmqalogprobs_evaluator = \
                        give_llmqalogprobs_model(self.cfg, device_map = reward_model_device)
                    self.post_trainer.llmqalogprobs_evaluator.holder_articles = self.holder_articles
                    print('🟩🟩\tRULES-BASED EVALUATOR (C2) SWOPPED TO LLMQALOGPROBS')

            # 6. rankllm with remote worker LLM models / openai client
            # a. if using rankllm via nccl, collect mapping of worker node ranks
            self.post_trainer.criteria2rank_mapping = \
                {crit: self.cfg.architecture.workers[crit].rank  \
                for crit in self.cfg.architecture.workers}
            
            
            # b. if using openai clients for rankllm, individual clients (non-blocking)
            self.post_trainer.criteria2openai_worker = {}   
            for criteria, model_info in self.cfg.model.reward.items():
                if criteria not in ['criteria2', 'criteria3', 'criteria4']: continue
                if model_info['model'] not in ['gpt4o', 'deepseek', 'o3mini']: continue
                cfg_copy = copy.deepcopy(self.cfg)
                cfg_copy['load_peft_ckpt_path'] = None
                worker = WorkerNode(cfg_copy, own_rank = None, master_rank = None, 
                                    role = criteria, launch_nccl = False)
                self.post_trainer.criteria2openai_worker[criteria] = worker
 
    def give_datasets(self):
        import pandas as pd
        from datasets import Dataset
        from utils_grpo_data import load_grpo_qud_gen_data, load_grpo_rankllm_data

        eval_ranker_requests        = None
        holder_dataset_eval_qsal    = None
        qud_criteria_cfg            = None
        if   self.cfg.grpo_settings.grpo_task == 'qud_gen':
            holder_dataset_train, holder_dataset_eval, holder_dataset_eval_qsal, \
                self.holder_articles, self.holder_quds, num_cands2qud_idxes = \
                    load_grpo_qud_gen_data(self.cfg, self.sft_tokenizer, self.model_name)
        elif self.cfg.grpo_settings.grpo_task == 'rankllm':
            holder_dataset_train, holder_dataset_eval, \
                self.holder_articles, self.holder_quds, \
                    eval_ranker_requests, num_cands2qud_idxes, qud_criteria_cfg = \
                    load_grpo_rankllm_data(self.cfg, self.sft_tokenizer)
        else: 
            raise NotImplementedError

        df_dataset_train = pd.DataFrame(holder_dataset_train)
        df_dataset_train.sample(frac = 1.0, random_state = SEED)
        train_dataset = Dataset.from_pandas(df_dataset_train)
        print('🟩🟩\tPREPARED TRAIN DATA...', df_dataset_train.shape, df_dataset_train.columns)

        df_dataset_eval = pd.DataFrame(holder_dataset_eval)
        df_dataset_eval.sample(frac = 1.0, random_state = SEED)
        eval_dataset = Dataset.from_pandas(df_dataset_eval)
        print('🟩🟩\tPREPARED TEST DATA...', df_dataset_eval.shape, df_dataset_eval.columns)

        eval_dataset_qsal = None
        if holder_dataset_eval_qsal is not None:
            df_dataset_eval_qsal = pd.DataFrame(holder_dataset_eval_qsal)
            df_dataset_eval_qsal.sample(frac = 1.0, random_state = SEED)
            eval_dataset_qsal = Dataset.from_pandas(df_dataset_eval_qsal)
            print('🟩🟩\tPREPARED QSAL TEST DATA...', df_dataset_eval_qsal.shape, df_dataset_eval_qsal.columns)    
        
        return train_dataset, eval_dataset, eval_dataset_qsal, eval_ranker_requests, num_cands2qud_idxes, qud_criteria_cfg
    
    def give_reward_funcs(self):
        from grpo_reward_funcs import (xmlcount_reward_func, soft_format_reward_func, 
                                        strict_format_reward_func, criteria2_reward_func,
                                        criteria3_reward_func, criteria4_reward_func, 
                                        qud_length_reward, no_xml_in_answer, think_length_reward_func)
        reward_funcs = [xmlcount_reward_func, 
                        soft_format_reward_func, 
                        strict_format_reward_func, 
                        no_xml_in_answer,]
        if  self.cfg.grpo_settings.grpo_task == 'qud_gen':
            from grpo_reward_funcs import (criteria2_reward_func, criteria3_reward_func, 
                                        criteria4_reward_func, qud_length_reward, ensure_tighter_qud_gen)
            reward_funcs += [think_length_reward_func,
                            criteria2_reward_func,
                            criteria3_reward_func, 
                            criteria4_reward_func,
                            qud_length_reward, 
                            ensure_tighter_qud_gen]
        elif self.cfg.grpo_settings.grpo_task == 'rankllm':
            from grpo_reward_funcs_rankllm import (json_ratio, rank_ratio, 
                                                    check_answer_format_match_identifiers,
                                                    check_valid_json, check_score_dicts_validity,
                                                    check_score_dicts_rationales, match_gpt4o_score, 
                                                    score_rank_consistency, answer_well_formed, thinking_well_formed)
            reward_funcs += [json_ratio, 
                            rank_ratio, 
                            check_answer_format_match_identifiers,
                            check_valid_json,
                            check_score_dicts_validity,
                            check_score_dicts_rationales,
                            match_gpt4o_score, 
                            score_rank_consistency,
                            answer_well_formed, 
                            thinking_well_formed]
            
        return reward_funcs

if __name__ == '__main__':
    main()