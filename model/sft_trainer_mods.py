import os, tqdm, torch
from omegaconf import DictConfig
from collections import defaultdict
from functools import partial
from trl import SFTTrainer, SFTConfig


class SFTTrainerMod(SFTTrainer):

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.generated_outputs = {}
        from grpo_trainer_mods import GRPOTrainer
        self.inference_on_eval_dataset_rankllm = partial(GRPOTrainer.inference_on_eval_dataset_rankllm, self)

    def post_init(self):
        adapter_name = 'default'
        if self.cfg['load_peft_ckpt_path'] is not None:    
            from utils_model import set_peft_weights
            self.model  = set_peft_weights(self.cfg, self.model, adapter_name)
        else: self.model.set_adapter(adapter_name)  
        

    def inference_on_eval_dataset_rankllm(self, eval_ranker_requests):
        from trl.models.utils import unwrap_model_for_generation
        from accelerate.utils import is_peft_model
        from utils_model import load_model_rankllm, process_one_step_rankllm
        from transformers import pipeline
        # NOTE: similar method in GRPOTrainer
        if self.cfg['grpo_settings']['grpo_task'] in ['qud_gen']: raise NotImplementedError
        rankllm_sys             = self.cfg['model']['sft']['model']
        model_size              = self.cfg['model']['sft']['size']
        model_path              = self.cfg['model']['models_list'][model_size][rankllm_sys]
        
        qud_exemplars           = {}
        holder_eval_outputs     = {}
        if is_peft_model(self.model):   model = self.model.merge_and_unload()
        else:                           model = self.model
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            for qud_criteria in self.cfg['qud_criteria_list']:
                holder_eval_outputs[qud_criteria] = defaultdict(dict)
                for num_cands in sorted(self.num_cands2qud_idxes, reverse = True):
                    print('\n\n')
                    print(f'🟧🟧\tWorking on {qud_criteria} and num_cands: {num_cands}')
                    qud_idxes       = self.num_cands2qud_idxes[num_cands]
                    requests_dict   = {q_i_id: rankllm_req for q_i_id, rankllm_req \
                                        in eval_ranker_requests[qud_criteria].items() \
                                            if q_i_id in qud_idxes}
                    
                    # self.cfg['ranker_args']['use_past_key_values'] = True
                    pipeline_model = pipeline('text-generation', model = unwrapped_model, tokenizer = self.tokenizer)
                    rankllm_model, cfg, constraints_dict = load_model_rankllm(cfg = DictConfig(self.qud_criteria_cfg[qud_criteria]), 
                                                        model_name = rankllm_sys, model_path = model_path, 
                                                        tokenizer = self.tokenizer, pipeline_model = pipeline_model,
                                                        device = self.model.device, qud_exemplars = qud_exemplars)
                    print('🟩🟩\tMODEL LOADED', rankllm_sys, model_path, f'for num_cands: {num_cands}')
                    
                    # ensure window size set
                    window_size = cfg.ranker_args['window_size'] = num_cands

                    # B. start running through QUDInstances
                    for __, (qud_instance_id, rankllm_req) in enumerate(tqdm.tqdm(requests_dict.items())):
                        proc_predictions = process_one_step_rankllm(cfg, rankllm_model, 
                                                    window_size, rankllm_req, qud_criteria)

                        assert len(proc_predictions) == 1
                        for q_i_id, proc_pred in proc_predictions.items():
                            
                            if __ == 0: print(f'{"~"*50}\nPROCESSED:', proc_pred, f'\n{"~"*50}')
                            
                            holder_eval_outputs[qud_criteria][q_i_id][rankllm_sys] = proc_pred
        
        return holder_eval_outputs
    
def collate_fn(batch_samples, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts     = [x['prompt']      for x in batch_samples]
    completions = [x['completion']  for x in batch_samples]
    inputs      = [p+c for p,c in zip(prompts, completions)]

    # Tokenize the sequences
    batch_tokens = tokenizer(inputs, padding = 'longest', return_tensors = 'pt')
    # see https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
    c1 = tokenizer.eos_token_id is not None
    c2 = tokenizer.eos_token_id == tokenizer.pad_token_id \
        and not torch.all(batch_tokens["input_ids"][:,-1] == tokenizer.eos_token_id)
    c3 = tokenizer.eos_token_id != tokenizer.pad_token_id \
        and tokenizer.eos_token_id not in batch_tokens["input_ids"][:,-1]
    if c1 and (c2 or c3):  
        bsz = batch_tokens["input_ids"].size(0)
        add_eos         = torch.LongTensor([[tokenizer.eos_token_id] for i in range(bsz)])
        add_eos_mask    = torch.LongTensor([[1] for i in range(bsz)])
        batch_tokens["input_ids"]       = torch.cat([batch_tokens["input_ids"], add_eos], dim = -1)
        batch_tokens["attention_mask"]  = torch.cat([batch_tokens["attention_mask"], add_eos_mask], dim = -1)

    batch_tokens['labels'] = batch_tokens['input_ids'].clone()
    batch_tokens['labels'][batch_tokens['labels'] == tokenizer.pad_token] = -100

    # for i, (prompt, completion) in enumerate(zip(prompts, completions)):
    #     # Tokenize just the prompt to find its length
    #     prompt_length = len(tokenizer.encode(prompt))
        
    #     # Set labels for prompt tokens to -100
    #     batch_tokens["labels"][i][:prompt_length] = torch.LongTensor([-100] * prompt_length)

    return batch_tokens

def give_sft_trainer(cfg, model, tokenizer, train_dataset, eval_dataset, 
                     eval_ranker_requests, num_cands2qud_idxes, peft_config):
    bf16 = False if cfg.model.sft.model in ['gemma'] else True
    fp16 = False 

    training_args = SFTConfig(
        learning_rate           = 1e-4,
        adam_beta1              = 0.9,
        adam_beta2              = 0.99,
        weight_decay            = 0.1,
        warmup_ratio            = cfg.warmup_ratio,
        lr_scheduler_type       = 'linear',
        optim                   = 'paged_adamw_8bit',
        logging_steps           = 20,
        bf16                    = bf16,
        fp16                    = fp16,
        gradient_accumulation_steps = cfg.grpo_settings.gradient_accumulation_steps,
        max_length              = cfg.grpo_settings.max_prompt_length + 1000, # HACK: to keep config_grpo.yaml clean
        # torch_empty_cache_steps = 100,
        per_device_train_batch_size = cfg.grpo_settings.gen_bsz,
        num_train_epochs        = 3,
        # max_steps               = 1,
        packing                 = False,
        save_steps              = cfg.save_steps,
        save_total_limit        = 20,
        max_grad_norm           = 0.1,
        remove_unused_columns   = False, 
        report_to               = 'none', # Can use Weights & Biases
        log_level               = 'info',
        save_strategy           = 'steps',
        output_dir              = os.path.join(cfg.savepath, 'outputs'),
        ddp_find_unused_parameters = False,
        dataset_kwargs          = {"skip_prepare_dataset": True}, # use collate_fn
    )

    trainer = SFTTrainerMod(
        cfg                 = cfg,
        model               = model,
        processing_class    = tokenizer,
        args                = training_args,
        train_dataset       = train_dataset,
        # eval_dataset        = eval_dataset, 
        peft_config         = peft_config, # peft_config only needed in v0.14dev
        data_collator       = lambda batch_samples: collate_fn(batch_samples, tokenizer = tokenizer)
        ) 
    trainer.training_args           = training_args
    trainer.eval_ranker_requests    = eval_ranker_requests
    trainer.num_cands2qud_idxes     = num_cands2qud_idxes

    return trainer