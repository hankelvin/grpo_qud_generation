import re, sys, os, json
import omegaconf

######################################
########## MODEL LOADING     #########
######################################

def load_llm_model(cfg, model_name, model_path, model_size, 
                    max_seq_length, lora_rank,
                    gpu_memory_utilization, target_modules,
                    device_map, dtype, enable_lora, 
                    use_own_loader, optimum_bypass_bsz, do_compile_overide = False):
    import os
    os.environ['HF_HOME'] = os.path.dirname(cfg.hub_dirpath)
    os.environ['HF_HUB_CACHE'] = cfg.hub_dirpath
    if device_map not in [None, 'cpu']:
        device_map      = device_map if re.match(r'cuda:*\d*', device_map) else 'auto'
    from utils_model import load_model
    print('👀 device_map', device_map)
    peft_config = None
    if use_own_loader:
        from peft import get_peft_model, LoraConfig, TaskType
        from accelerate.utils import is_peft_model
        
        skip_pipeline = True
        force_load_in_nbit = cfg.model.sft.load_in_nbit 
        if      model_path in cfg.models_force_4bit: force_load_in_nbit = 4
        elif    model_path in cfg.models_force_8bit: force_load_in_nbit = 8
        do_compile = True
        if cfg.model.sft.load_in_nbit:           do_compile = False 
        if cfg.grpo_settings.num_iterations > 1: do_compile = False
        if do_compile_overide: do_compile = True
        # NOTE: model here is returned as a TextGenerationPipeline
        tokenizer, model, device, cfg = \
                    load_model(cfg, model_name      = model_name, 
                               model_path           = model_path, 
                               model_size           = model_size,
                                force_load_in_nbit = force_load_in_nbit,
                                device_map          = device_map, 
                                do_compile          = do_compile,
                                skip_pipeline       = skip_pipeline, 
                                optimum_bypass_bsz  = optimum_bypass_bsz)
        model.do_compile = do_compile
        if tokenizer.pad_token is None: # for pad to length
            tokenizer.pad_token       = tokenizer.eos_token
            tokenizer.pad_token_id    = tokenizer.eos_token_id
        
        # NOTE: peft set up delayed to GRPOTrainer initialisation
        print('🟧🟧\tCREATING PEFT CONFIG')
        task_type = TaskType.CAUSAL_LM    
        lora_alpha = cfg.grpo_settings.lora_alpha_ratio * lora_rank
        peft_config = LoraConfig(
            target_modules = list(target_modules), # else in omegaconf.ListConfig
            task_type      = task_type, 
            inference_mode = False,
            r              = lora_rank, 
            lora_alpha     = lora_alpha, 
            lora_dropout   = cfg.grpo_settings.lora_dropout,
            use_dora       = cfg.grpo_settings.use_dora,
            use_rslora     = True,
            # NOTE: pissa only works float32, float16, or bfloat16
            # init_lora_weights = 'pissa_niter_16', # https://arxiv.org/pdf/2404.02948
            )
        # NOTE: passing PEFTConfig into GRPOTrainer later. it will then 
        # set up the PEFT model inside its __init__
        # adapter_name = 'default'
        # model = get_peft_model(model, peft_config, adapter_name = adapter_name)
        # # see https://github.com/huggingface/peft/issues/1142
        # model.set_adapter(adapter_name)
        # print(f'🔮ADAPTER ({adapter_name}) ADDED TO MODEL')
        # model.print_trainable_parameters()       
        # print('🟧🟧\tPEFT MODEL LAYERS DONE', is_peft_model(model))

    else: 
        # NOTE: hitting a few issues with running via VLLM. best to use HF only for now
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL('GRPO', FastLanguageModel)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name      = model_path,
            max_seq_length  = max_seq_length,
            load_in_nbit    = False, 
            fast_inference  = False, 
            max_lora_rank   = lora_rank,
            gpu_memory_utilization = gpu_memory_utilization, 
            dtype           = dtype, 
            device_map      = device_map,
            fix_tokenizer   = False,
            trust_remote_code = True if model_name in ['phi'] else None,
            # https://github.com/unslothai/unsloth/issues/1268
            # quantization    = 'bitsandbytes' if 'bnb-4bit' in model_path else None, 
            # load_format     = 'bitsandbytes' if 'bnb-4bit' in model_path else 'auto', 
            enable_lora     = enable_lora
            )
    
        print('🟧🟧\tADDING PEFT MODEL LAYERS')
        model = FastLanguageModel.get_peft_model(model,
                                                r               = lora_rank, 
                                                target_modules  = target_modules,
                                                lora_alpha      = lora_rank,
                                                # Enable long context finetuning
                                                use_gradient_checkpointing = 'unsloth', 
                                                random_state    = 54506,
                                                )
        print('🟧🟧\tPEFT MODEL LAYERS DONE')

    return model, tokenizer, peft_config

def give_grpo_trainer(cfg, model, tokenizer, train_dataset, 
                    eval_dataset, eval_dataset_qsal,
                    eval_ranker_requests, num_cands2qud_idxes,
                    reward_funcs = [], reward_weights = None, 
                    peft_config = None): 
    from trl import GRPOConfig
    from grpo_trainer_mods import GRPOTrainerMod
    bf16 = fp16 = None
    if not cfg.use_own_loader: 
        from unsloth import is_bfloat16_supported
        bf16 = is_bfloat16_supported()
        fp16 = not is_bfloat16_supported()
    else: 
        bf16 = False if cfg.model.sft.model in ['gemma'] else True
        fp16 = False 
    
    ##### TRAINER ARGS #####    
    training_args = GRPOConfig(
        use_vllm                = True if not cfg.use_own_loader else False,
        beta                    = cfg.grpo_settings.beta, # hyperparam (coefficient for the KLDiv loss with ref_model)
        epsilon                 = cfg.grpo_settings.epsilon,
        reward_weights          = reward_weights,
        sync_ref_model          = cfg.grpo_settings.sync_ref_model,         
        ref_model_mixup_alpha   = cfg.grpo_settings.ref_model_mixup_alpha,  
        ref_model_sync_steps    = cfg.grpo_settings.ref_model_sync_steps,   
        learning_rate           = 5e-6,
        adam_beta1              = 0.9,
        adam_beta2              = 0.99,
        weight_decay            = 0.1,
        warmup_ratio            = cfg.warmup_ratio,
        lr_scheduler_type       = 'cosine',
        optim                   = 'paged_adamw_8bit',
        logging_steps           = 20,
        bf16                    = bf16,
        fp16                    = fp16,
        # per_device_train_batch_size should be a multple of num_generations 
        per_device_train_batch_size = cfg.grpo_settings.gen_bsz,  
        num_generations         = cfg.grpo_settings.num_cands, 
        num_iterations          = cfg.grpo_settings.num_iterations,
        gradient_accumulation_steps = cfg.grpo_settings.gradient_accumulation_steps,
        # gradient_checkpointing  = True if cfg.grpo_settings.gradient_accumulation_steps > 0 else False,
        # gradient_checkpointing_kwargs = {'use_reentrant': False},
        max_prompt_length       = cfg.grpo_settings.max_prompt_length,
        max_completion_length   = cfg.grpo_settings.max_seq_length,
        # torch_empty_cache_steps = 100,
        save_steps              = cfg.save_steps,
        save_total_limit        = 20,
        max_grad_norm           = 0.1,
        remove_unused_columns   = False, 
        report_to               = 'none', # Can use Weights & Biases
        log_level               = 'info',
        save_strategy           = 'steps',
        output_dir              = os.path.join(cfg.savepath, 'outputs'),
        ddp_find_unused_parameters = False,
        )
    
    # instead of modding the whole GRPOConfig, adding this here
    # https://github.com/huggingface/trl/pull/3135
    if cfg.reward_funcs_version == 1:
        # initial GRPO qud_gen (to obtain synthetic score+rank data) did not have scale_rewards
        setattr(training_args, 'scale_rewards', False)
        setattr(training_args, 'epsilon_high', None)
        # 12/04: added to trl, exclude truncated trajectories (for training stability)
        setattr(training_args, 'mask_truncated_completions', False)
    else: 
        setattr(training_args, 'scale_rewards', cfg.grpo_settings.scale_rewards)
        setattr(training_args, 'epsilon_high',  cfg.grpo_settings.epsilon_high)
        setattr(training_args, 'mask_truncated_completions', True)
    
    if cfg.grpo_settings.max_steps is None: 
        training_args.num_train_epochs  = 1, # Set to 1 for a full training run
    else: 
        training_args.max_steps         = cfg.grpo_settings.max_steps
    
    # force attach (GRPOConfig inherits TrainingArguments, which has generation_config)
    # see self.generation_config = GenerationConfig( line in grpo_trainer.py
    training_args.temperature           = cfg.grpo_settings.temperature
    training_args.num_return_sequences  = cfg.grpo_settings.num_cands
    training_args.ddp_backend           = None # causing a problem with multi-node
    # NOTE: not used in paper
    # see https://x.com/kalomaze/status/1926751357983154606
    # training_args.top_k                 = cfg.gen_args.top_k
    # training_args.top_p                 = cfg.gen_args.top_p
    
    ##### TRAINER ##### 
    trainer = GRPOTrainerMod(
        model               = model,
        processing_class    = tokenizer,
        reward_funcs        = reward_funcs,
        args                = training_args,
        train_dataset       = train_dataset,
        eval_dataset        = eval_dataset,
        peft_config         = peft_config, # peft_config only needed in v0.14dev
        ) 
    # ensure _prepare_inputs can access cfg info. 
    trainer.cfg                  = convert_to_json_compatible(cfg)
    trainer.eval_dataset_qsal    = eval_dataset_qsal
    trainer.training_args        = training_args
    trainer.eval_ranker_requests = eval_ranker_requests
    trainer.num_cands2qud_idxes  = num_cands2qud_idxes
    
    return trainer


def give_llmqalogprobs_model(cfg, device_map):
    from utils_model import load_model
    sys.path.append('evaluation')
    from answer_probability import give_llmqalogprobs_icl

    model_name = cfg.answer_compat.grpo_model_name
    model_size = cfg.answer_compat.grpo_model_size

    do_compile = False if '4bit' in model_name else True
    tokenizer, pipeline_model, device, cfg = \
        load_model(cfg, model_name, cfg.model['models_list'][model_size][model_name], 
                model_size = model_size, do_compile = do_compile, device_map = device_map)
    if tokenizer.pad_token is None: 
        pipeline_model.tokenizer.pad_token    = tokenizer.pad_token     = tokenizer.bos_token
        pipeline_model.tokenizer.pad_token_id = tokenizer.pad_token_id  = tokenizer.bos_token_id

    #################
    # B. re-usables
    pipeline_model.cfg          = cfg
    pipeline_model.model_name   = model_name
    pipeline_model.model_size   = model_size
    pipeline_model.do_icl       = pipeline_model.cfg.answer_compat.do_icl
    pipeline_model.icl_messages = None
    if pipeline_model.do_icl:
        fp = f'{cfg.dirpath}/data/exemplars/answer_compat_llm_qa.json'
        with open(fp, encoding = 'utf-8') as f:
            pipeline_model.cfg.prompts.answer_compat.icl_exemplars = json.load(f)

        pipeline_model.icl_messages = give_llmqalogprobs_icl(pipeline_model.cfg, 
                    num_few_shot_examples = pipeline_model.cfg.answer_compat.num_few_shot_examples)
    
    return pipeline_model


def convert_to_json_compatible(cfg):

    if isinstance(cfg, (omegaconf.ListConfig, list)):
        return [convert_to_json_compatible(item) for item in cfg]
    
    elif isinstance(cfg, (omegaconf.DictConfig, dict)):
        return {key: convert_to_json_compatible(value) for key, value in cfg.items()}
    
    else:
        return cfg
