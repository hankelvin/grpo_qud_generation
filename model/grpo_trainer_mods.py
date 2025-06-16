import torch, time, inspect, asyncio, tqdm, gc, json, math
from collections import defaultdict
from omegaconf import DictConfig
from typing import Any, Union, Dict, Mapping
from trl import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad, selective_log_softmax
from trl.extras.profiling import profiling_decorator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model
import torch.distributed as dist
from peft.utils import prepare_model_for_kbit_training
from grpo_reward_funcs import (extract_xml_answer_no_tags, 
                               extract_think_words_strict_with_tags)
from grpo_reward_funcs_rankllm import (json_extractor, score_dicts_extractor, 
                                       ranking_extractor)

######################################
########## MODS TO TRL GRPOTRainer ###
######################################
class GRPOTrainerMod(GRPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = kwargs.get('args', {})
        self.epsilon_high               = getattr(args, 'epsilon_high', None)
        self.mask_truncated_completions = getattr(args, 'mask_truncated_completions', False)

    ### CHANGE START ###
    def post_init(self):
        print('👀 PEFT MODEL STATUS -->', is_peft_model(self.model))
        adapter_name = 'default'
        if self.cfg['load_peft_ckpt_path'] is not None:    
            from utils_model import set_peft_weights
            self.model  = set_peft_weights(self.cfg, self.model, adapter_name)
        else: self.model.set_adapter(adapter_name)         

        mname = self.cfg['model']['sft']['model']
        msize = self.cfg['model']['sft']['size']
        mpath = self.cfg['model']['models_list'][msize][mname]
        c1 = '4bit' in mpath   
        if c1:
            # https://github.com/huggingface/peft/issues/1745
            self.model = prepare_model_for_kbit_training(self.model)
        else: 
            pass
            # # https://github.com/huggingface/trl/commit/9b3c5bf64fd88526481ec32de85539e2bbdda92b
            # if self.args.gradient_checkpointing:
            #     self.model = self._enable_gradient_checkpointing(self.model, self.args)
        
        self.model.print_trainable_parameters()   
        self.model.is_model_parallel = False
        self.generated_outputs          = defaultdict(dict)
        self.holder_eval_outputs_status = set()
        if getattr(self, 'temperature', None) is None: 
            self.temperature = self.training_args.temperature

        # https://github.com/huggingface/trl/commit/88cec7e13fc112809768c0fa0a4dfa9b17372952
        if mname in ['gemma']: cache_implementation = 'hybrid'
        else:                  cache_implementation = 'static' 
        self.generation_config.cache_implementation = cache_implementation
        grpo_task               = self.cfg['grpo_settings']['grpo_task']
        forced_bos_token_id     = self.processing_class.vocab.get(self.cfg['gen_args']['forced_bos_token'][grpo_task], None)
        if forced_bos_token_id is not None and self.cfg['reward_funcs_version'] > 1:
            # NOTE: initial GRPO qud_gen model did not set forced_bos_token_id
            self.generation_config.forced_bos_token_id = forced_bos_token_id

        # 03/25: for the DAPO upper epsilon
        self.epsilon_low  = self.epsilon
        self.epsilon_high = self.epsilon_high if self.epsilon_high is not None else self.epsilon
        self.bad_words_ids = [self.processing_class(phrase, add_special_tokens = False)['input_ids'] \
                              for phrase in self.cfg['gen_args']['tighter_qud_gen_phrases']]

    def inference_on_eval_dataset_qud_gen(self, eval_dataset, bsz = 8, tighter_gen = False):
        from accelerate.utils import is_peft_model
        if self.args.use_vllm: raise NotImplementedError
        assert len(eval_dataset) > 0, len(eval_dataset)
        num_batches = max(1, math.ceil(len(eval_dataset)/bsz))
        self.model.eval()
        
        holder_eval_outputs = {}
        if is_peft_model(self.model):   model = self.model.merge_and_unload()
        else:                           model = self.model
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            for __, bn in enumerate(tqdm.tqdm(range(num_batches))):
                ### CHANGE START ###
                cut_inputs          = eval_dataset[bn * bsz : (bn+1) * bsz] # slice of Dataset object is a dict
                qud_instance_idxes  = cut_inputs['qud_instance_id']
                prompts_text        = [maybe_apply_chat_template({'prompt': example}, self.processing_class)['prompt'] \
                                                                        for example in cut_inputs['prompt']]
                ### CHANGE END ###
                prompt_inputs = self.processing_class(
                    prompts_text, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False
                )
                prompt_inputs = self._prepare_inputs_Trainer(prompt_inputs)
                prompt_ids, prompt_mask = prompt_inputs['input_ids'], prompt_inputs['attention_mask']

                if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
                    prompt_ids  = prompt_ids[:, -self.max_prompt_length :]
                    prompt_mask = prompt_mask[:, -self.max_prompt_length :]
                
                # Regular generation path
                # bad_words_ids = self.bad_words_ids if tighter_gen else None
                bad_words_ids = None

                with torch.no_grad():
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, 
                        ### CHANGE START ###
                        forced_bos_token_id = getattr(self.generation_config, 'forced_bos_token_id', None),
                        bad_words_ids       = bad_words_ids,
                        ### CHANGE START ###
                        generation_config=self.generation_config
                    )
            
                # Compute prompt length and extract completion ids
                prompt_length   = prompt_ids.size(1)
                prompt_ids      = prompt_completion_ids[:, :prompt_length]
                completion_ids  = prompt_completion_ids[:, prompt_length:]
                completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                assert len(prompts_text) == len(completions_text) == len(qud_instance_idxes), \
                    (len(prompts_text), len(completions_text), len(qud_instance_idxes))
                
                for pt, ct, q_i_id in zip(prompts_text, completions_text, qud_instance_idxes):
                    holder_eval_outputs[q_i_id] = {'prompt': pt, 'generated': ct}
        
        return holder_eval_outputs
    
    def inference_on_eval_dataset_rankllm(self, eval_ranker_requests):
        from trl.models.utils import unwrap_model_for_generation
        from accelerate.utils import is_peft_model
        from utils_model import load_model_rankllm, process_one_step_rankllm
        from transformers import pipeline
        from accelerate.utils import is_peft_model
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

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        ### CHANGE START ###
        qud_instance_idxes  = [x['qud_instance_id'] for x in inputs]
        rules_based_pack    = {'contexts':       [x.get('context', None) for x in inputs],
                                'anchors':       [x.get('anchor',  None)  for x in inputs],
                                'answers':       [x.get('answer',  None)  for x in inputs],
                                'qud_criterias': [x.get('qud_criteria', None)  for x in inputs],
                                'gold_docmaps':  [x.get('gold_docmap', None)  for x in inputs]}     
        ### CHANGE END ###
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                ### CHANGE START ###
                inputs = self._generate_and_score_completions(inputs, qud_instance_idxes, rules_based_pack)
                ### CHANGE END ### 
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            ### CHANGE START ###
            inputs = self._generate_and_score_completions(inputs, qud_instance_idxes, rules_based_pack)
            ### CHANGE END ### 
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]],
        ### CHANGE START ###
        qud_instance_idxes: list[str], 
        rules_based_pack: dict, 
        ### CHANGE END ###
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        ### CHANGE START ###
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = self._prepare_inputs_Trainer(prompt_inputs)
        ### CHANGE END ###
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        ### CHANGE START ###
        # max_prompt_length longer than length of prompt_ids leading to empty tensors
        if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
            ### CHANGE END ###
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:                
                prompt_completion_ids = unwrapped_model.generate(
                prompt_ids, attention_mask=prompt_mask, 
                ### CHANGE START ###
                forced_bos_token_id = self.generation_config.forced_bos_token_id,
                ### CHANGE START ###
                generation_config=self.generation_config
            )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        ### CHANGE START ###
        # 12/04: added to trl, exclude truncated trajectories (for training stability)
        # https://github.com/huggingface/trl/pull/3248
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()
        ### CHANGE END ###

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        # set prompts_text again (in case truncated by max_prompt_length)
        prompts_text        = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text    = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        if self.state.global_step in range(5) or self.state.global_step % 100 == 0:
            print('Global step: #', self.state.global_step)
            print('Prompts\t',      prompts_text)
            print('Completions\t',  completions_text)

        ### CHANGE START ###
        rewards_per_func        = torch.zeros(len(completions_text), \
                                              len(self.reward_funcs), device = device)
        qud_instance_idxes_set  = set(qud_instance_idxes)
        bsz                     = int(len(completions)/len(qud_instance_idxes_set))
        assert len(completions) % len(qud_instance_idxes_set) == 0
        assert bsz == self.cfg['grpo_settings']['num_cands']
        
        c_qud_gen = self.cfg['grpo_settings']['grpo_task'] in ['qud_gen']
        c_rankllm = self.cfg['grpo_settings']['grpo_task'] in ['rankllm']
        crit_scores_dict = {k: [] for k in self.cfg['criteria_reward_funcs']}
        for bn, qud_instance_id in enumerate(qud_instance_idxes_set):
            start                   = time.time()

            cut_completions_text    = completions_text[bn*bsz: (bn+1)*bsz]
            cut_completions_ids     = completion_ids[bn*bsz: (bn+1)*bsz]
            assert cut_completions_text, cut_completions_text
            cut_think_words         = [extract_think_words_strict_with_tags(c) for c in cut_completions_text]
            # recover only the <answer> </answer> part
            cut_outputs = None
            if  c_qud_gen:
                # NOTE: cut_quds is only used in the qud_gen criteria reward funcs 
                # (to get the quds for the ranking). it is not used for the GRPO RM reward funcs
                cut_outputs    = [extract_xml_answer_no_tags(c).replace('\n','').replace('\t','').strip() \
                                        for c in cut_completions_text]
                # model could generate nothing
                mark_zero   = [1 if c else 0 for c in cut_outputs]
                # also might not be a question
                mark_zero   = [mz*0 if not c.endswith('?') else mz*1 for mz, c in zip(mark_zero, cut_outputs)]
            elif c_rankllm:
                cut_outputs = [extract_xml_answer_no_tags(c).replace('\n','').replace('\t','').strip() \
                                        for c in cut_completions_text]
                # model could generate nothing
                mark_zero   = [1 if c else 0 for c in cut_outputs]

            # set overly long trajectories to 0. from DAPO, shaping overly long trajectories
            # not done for initial model (i.e. version 1)
            if self.cfg['reward_funcs_version'] > 1:
                mark_zero   = [mz*0 if len(cid) > self.cfg['grpo_settings']['max_seq_length'] else mz*1 \
                                        for mz, cid in zip(mark_zero, cut_completions_ids)]


            if self.cfg['test_print']: 
                print('qud_instance_id =\t',    qud_instance_id)
                print('PROMPTs =\t',            prompts_text[bn*bsz: (bn+1)*bsz])
                print('COMPLETIONS =\t',        completions_text[bn*bsz: (bn+1)*bsz])
                print('QUD GENS/RANKING =\t',   cut_outputs)

            # rules-based / llmqalogprobs (for criteria2 only) / remote rankllm or openai client
            c_rbs           = self.cfg['grpo_settings']['initial_rules_based_steps']
            do_rules_based  = c_rbs is not None and self.state.global_step < c_rbs
            if c_rbs and self.state.global_step >= c_rbs:
                self.rules_based_evaluator.classifier = None # free up cuda
            
            if not do_rules_based and c_qud_gen:
                data_to_send            = [qud_instance_id] + cut_outputs
                data_to_send_joined     = f"{self.cfg['qud_sep']}".join(data_to_send)
                reward_model_method     = 'worker'
                (s_2, rprompt_2, s_2_co), (s_3, rprompt_3, s_3_co), (s_4, rprompt_4, s_4_co) = \
                    asyncio.run(self.score_qud_criteria(data_to_send_joined))
                
                crit_scores_dict['criteria2_reward_func'].append(s_2)
                crit_scores_dict['criteria3_reward_func'].append(s_3)
                crit_scores_dict['criteria4_reward_func'].append(s_4)
            
                if self.cfg['test_print']: 
                    print('SCORES =\t',                  s_2, s_3, s_4)
                    print('TIME TAKEN FOR 1 BATCH =',    time.time()-start)
            else: 
                s_2 = s_2_co = s_3 = s_3_co = s_4 = s_4_co = None
                rprompt_2 = rprompt_3 = rprompt_4 = None
                reward_model_method = 'rules'
                cut_contexts        = rules_based_pack['contexts'][bn*bsz: (bn+1)*bsz]
                cut_anchors         = rules_based_pack['anchors'][bn*bsz: (bn+1)*bsz]
                cut_answers         = rules_based_pack['answers'][bn*bsz: (bn+1)*bsz]
                cut_qud_criterias   = rules_based_pack['qud_criterias'][bn*bsz: (bn+1)*bsz]
                cut_gold_docmaps    = rules_based_pack['gold_docmaps'][bn*bsz: (bn+1)*bsz]
                assert len(cut_contexts) == len(cut_anchors) == len(cut_answers) == len(cut_outputs), \
                        (len(cut_contexts), len(cut_anchors), len(cut_answers), len(cut_outputs))
        
            # # NOTE: line below moved to top
            # rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)    
            ### CHANGE END ###

            ### CHANGE START ###
            zipped = zip(self.reward_funcs, self.reward_processing_classes)
            ## /////////////////////////////////////////////////////////////////// ##
            ## prep the necessary kwargs for c. all other reward funcs 
            # Repeat all input columns (but 'prompt' and 'completion') to match the number of generations
            keys = [key for key in inputs[0] if key not in ['prompt', 'completion']]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_kwargs['reward_funcs_version'] = self.cfg['reward_funcs_version']
            reward_kwargs['tighter_qud_gen_phrases_reward'] = self.cfg['gen_args']['tighter_qud_gen_phrases_reward']
            cut_prompts     = prompts[bn*bsz: (bn+1)*bsz]
            cut_completions = completions[bn*bsz: (bn+1)*bsz]
            if c_rankllm:
                reward_kwargs['gold_docmap_list']   = cut_gold_docmaps
                reward_kwargs['num_expected']       = self.cfg['grpo_settings']['num_cands']
                reward_kwargs['jsons_list']         = json_extractor(cut_completions)
                reward_kwargs['score_dicts_list']   = score_dicts_extractor(reward_kwargs['jsons_list'])
                reward_kwargs['ranks_list']         = ranking_extractor(cut_completions)
            ## /////////////////////////////////////////////////////////////////// ##
            
            ## /////////////////////////////////////////////////////////////////// ##
            for i, (reward_func, reward_processing_class) in enumerate(zipped):
                r_p_c = reward_processing_class
                ## a. model-based RM 
                if isinstance(reward_func, torch.nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        ### CHANGE START ###
                        messages        = [{'messages': p + c} for p, c in \
                                           zip(cut_prompts, cut_completions)]
                        # messages = [{'messages': p + c} for p, c in zip(prompts, completions)]
                        ### CHANGE END ###
                        texts = [apply_chat_template(x, r_p_c)['text'] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = r_p_c(texts, return_tensors = 'pt', padding = True, 
                                            padding_side = 'right', add_special_tokens = False)
                    ### CHANGE START ###
                    # reward_inputs = super()._prepare_inputs(reward_inputs)
                    reward_inputs = self._prepare_inputs_Trainer(reward_inputs)
                    ### CHANGE END ###
                    with torch.inference_mode():
                        ### CHANGE START ###
                        # rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                        batch_reward = reward_func(**reward_inputs, **reward_kwargs).logits[:, 0]
                        ### CHANGE END ###
                ### CHANGE START ###
                ## b. QUD Gen criteria-specific reward funcs
                elif reward_func.__name__ in self.cfg['criteria_reward_funcs']:
                    if do_rules_based:
                        
                        if   reward_func.__name__ == 'criteria2_reward_func':
                            compare_pairs = zip(cut_outputs, cut_answers)
                        
                        elif reward_func.__name__ == 'criteria3_reward_func':
                            compare_pairs = zip(cut_outputs, cut_contexts)
                        
                        elif reward_func.__name__ == 'criteria4_reward_func':
                            compare_pairs = zip(cut_outputs, cut_anchors)
                        
                        c1 = reward_func.__name__ == 'criteria2_reward_func'
                        c2 = self.cfg['grpo_settings']['swop_rules_llmqalogprobs']
                        if c1 and c2:
                            rules_based_evaluator = self.llmqalogprobs_evaluator
                            worker_scores         = None
                            reward_model_method   = 'llmqalogprobs'
                        else:
                            rules_based_evaluator = self.rules_based_evaluator
                            worker_scores         = None
                            reward_model_method   = 'rules' # ensure properly set for c3 and c4
                    
                    else:
                        rules_based_evaluator = None
                        compare_pairs         = None
                        worker_scores         = crit_scores_dict[reward_func.__name__][bn]
                        reward_model_method   = 'worker'
                    
                    qud_max_score = self.cfg['ranker_args']['qud_max_score']
                    output_reward_func = reward_func(worker_scores          = worker_scores,
                                                    qud_max_score           = qud_max_score,
                                                    compare_pairs           = compare_pairs,
                                                    reward_model_method     = reward_model_method,
                                                    qud_instance_id         = qud_instance_id,
                                                    non_worker_evaluator    = rules_based_evaluator)
                    ### CHANGE START ###
                    # if the generation was empty, set to zero score
                    output_reward_func  = [mz*r for mz, r in zip(mark_zero, output_reward_func)]
                    batch_reward        = torch.tensor(output_reward_func, dtype = torch.float32, device = device)
                ### CHANGE END ###
                ## c. all other reward funcs
                else:
                    # output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    output_reward_func  = reward_func(completions = cut_completions, **reward_kwargs)
                    if c_rankllm:
                        # NOTE: not using mark_zero (not as clear-cut as qud_gen about which reward funcs to apply mark_zero on)
                        # output_reward_func = [mz*r for mz, r in zip(mark_zero, output_reward_func)]
                        pass
                    batch_reward        = torch.tensor(output_reward_func, dtype = torch.float32, device=device)
                    ### CHANGE END ###

                rewards_per_func[bn*bsz: (bn+1)*bsz, i] = batch_reward    
                ## /////////////////////////////////////////////////////////////////// ##

            if self.accelerator.num_processes > 1: raise NotImplementedError
            train_step_outputs = {'think_tokens':     cut_think_words,
                                  'rewards_per_func': {r.__name__: rewards_per_func[:,i].detach().tolist() \
                                                for i, r in enumerate(self.reward_funcs)},}
            if  c_qud_gen:
                 train_step_outputs['generated_quds']       = cut_outputs
                 train_step_outputs['crit_scores_dict']     = crit_scores_dict
                 train_step_outputs['ranker_prompts']       = {'criteria2': rprompt_2, 'criteria3': rprompt_3, 
                                                               'criteria4': rprompt_4}
                 train_step_outputs['ranker_cot_output']    = {'criteria2': s_2_co, 'criteria3': s_3_co, 
                                                               'criteria4': s_4_co}
            elif c_rankllm:
                train_step_outputs['prompts']               = prompts_text[bn*bsz: (bn+1)*bsz]
                train_step_outputs['generated_rankings']    = cut_outputs
                train_step_outputs['qud_criterias']         = cut_qud_criterias
                train_step_outputs['gold_docmap_list']      = reward_kwargs['gold_docmap_list']
                train_step_outputs['jsons_list']            = reward_kwargs['jsons_list']
                train_step_outputs['ranks_list']            = reward_kwargs['ranks_list']
            else: raise NotImplementedError

            self.generated_outputs[self.state.global_step][qud_instance_id] = train_step_outputs
            
            if self.is_in_train and self.state.global_step % self.cfg['save_steps'] == 0:
                fp = self.cfg['savepath_train_outputs'].replace('.json', f'_step{self.state.global_step}.json')
                with open(fp, encoding= 'utf-8', mode = 'w+') as f:
                    json.dump(self.generated_outputs, f)
                print(f'🔮🔮 Train outputs at {self.state.global_step} saved to: ', fp)

                # necessary check: global step only increments per gradient_accumulation_steps
                c1 = self.state.global_step not in self.holder_eval_outputs_status
                c2 = self.cfg['eval_every_save_step']
                if c1 and c2 and c_qud_gen: 
                    holder_eval_outputs = self.inference_on_eval_dataset_qud_gen(self.eval_dataset, bsz = 32)
                    
                    fp = self.cfg['savepath_test_outputs'].replace('.json', f'_step{self.state.global_step}.json')
                    with open(fp, encoding = 'utf-8', mode = 'w+') as f:
                        json.dump(holder_eval_outputs, f)
                    self.holder_eval_outputs_status.add(self.state.global_step)
                    print(f'🔮🔮 Eval at {self.state.global_step} saved to: ', fp)

                    if getattr(self.eval_data_qsal, None):
                        holder_eval_qsal_outputs = self.inference_on_eval_dataset_qud_gen(self.eval_dataset_qsal, bsz = 32)
                        
                        fp = self.cfg['savepath_test_outputs'].replace('.json', f'_step{self.state.global_step}.json')
                        fp = fp.replace('.json', '_qsal.json')
                        with open(fp, encoding = 'utf-8', mode = 'w+') as f:
                            json.dump(holder_eval_qsal_outputs, f)
                        print(f'🔮🔮 Eval for QSAL data at {self.state.global_step} saved to: ', fp)
                    
                    self.model.train()
            ### CHANGE END ###

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        ### CHANGE START ###
        # Dr. GRPO fix for remove per-question (i.e. QUD/rank instance here) bias from dividing over std
        # https://github.com/huggingface/trl/pull/3135
        # advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards: # change to config in give_grpo_trainer()
            advantages = advantages / (std_grouped_rewards + 1e-4)
        
        ### CHANGE END ###

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            ### CHANGE START ### add torch. to nn.Module
            if isinstance(reward_func, torch.nn.Module):  # Module instead of PretrainedModel for compat with compiled models
            ### CHANGE END ###
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            ### CHANGE START ###
            # https://github.com/huggingface/trl/pull/3145
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
            # self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            ### CHANGE END ###

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            # if wandb.run is not None and self.accelerator.is_main_process:
            #     wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    ### NOTE: 
    # 03/25: this was added recently in TRL https://github.com/huggingface/trl/pull/3029
    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
    
    # ## NOTE:
    # 03/25: recently fixed in TRL https://github.com/huggingface/trl/pull/3145
    # - adding the upper epsilon as per DAPO
    # - is_clipped logic
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        ### CHANGE START ###
        # 12/04: added to trl, exclude truncated trajectories (for training stability)
        # https://github.com/huggingface/trl/pull/3248
        # loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        ### CHANGE END ###

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            ### CHANGE START ###
            # https://github.com/huggingface/trl/pull/3248
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())
            ### CHANGE END ###


        is_clipped = (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        ### CHANGE START ###
        # 12/04: added to trl, exclude truncated trajectories (for training stability)
        # https://github.com/huggingface/trl/pull/3248
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).nanmean().item())
        ### CHANGE END ###
        return loss

    def _prepare_input_Trainer(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs_Trainer(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
    
    async def send_command_to_worker_async(self, worker_rank, command, data):
        '''Send command to worker node asynchronously'''
        # Send length first
        length_tensor = torch.tensor([len(command)], dtype = torch.long).to(self.accelerator.device)
        req1 = dist.isend(length_tensor, dst=worker_rank)
        
        # Send command
        command_tensor = torch.tensor([ord(c) for c in command], dtype = torch.long).to(self.accelerator.device)
        req2 = dist.isend(command_tensor, dst=worker_rank)
        if self.cfg['test_print']: print('sent command to worker rank:', worker_rank)
        
        # Wait for both sends to complete
        await asyncio.get_event_loop().run_in_executor(None, req1.wait)
        await asyncio.get_event_loop().run_in_executor(None, req2.wait)
        
        print('sent to worker rank:', worker_rank)
        
        # Send data if provided
        if data:
            # Send length first
            length_tensor = torch.tensor([len(data)], dtype=torch.long).to(self.accelerator.device)
            req3 = dist.isend(length_tensor, dst=worker_rank)
            
            data_tensor = torch.tensor([ord(c) for c in data], dtype = torch.long).to(self.accelerator.device)
            req4 = dist.isend(data_tensor, dst=worker_rank)
            if self.cfg['test_print']: print('sent data to worker rank:', worker_rank)
            
            # Wait for both data sends to complete
            await asyncio.get_event_loop().run_in_executor(None, req3.wait)
            await asyncio.get_event_loop().run_in_executor(None, req4.wait)

    async def receive_from_worker_async(self, worker_rank):
        '''Receive results from worker node asynchronously'''
        # Receive response length first
        length_tensor = torch.zeros(1, dtype = torch.long).to(self.accelerator.device)
        req1 = dist.irecv(length_tensor, src=worker_rank)
        
        # Wait for length receive to complete
        try:
            await asyncio.get_event_loop().run_in_executor(None, req1.wait)
        except RuntimeError:
            return None
            
        # Receive actual response
        response_tensor = torch.zeros(length_tensor.item(), dtype = torch.long).to(self.accelerator.device)
        req2 = dist.irecv(response_tensor, src=worker_rank)
        
        # Wait for response receive to complete
        try:
            await asyncio.get_event_loop().run_in_executor(None, req2.wait)
        except RuntimeError:
            return None
        
        return ''.join([chr(i) for i in response_tensor.tolist()])
    
    async def score_qud_criteria(self, data_joined):

        (s_2, rprompt_2, s_2_co), (s_3, rprompt_3, s_3_co), (s_4, rprompt_4, s_4_co) = await asyncio.gather(
                    self.score_criteria(command = 'rank', criteria = 'criteria2', data = data_joined),
                    self.score_criteria(command = 'rank', criteria = 'criteria3', data = data_joined),
                    self.score_criteria(command = 'rank', criteria = 'criteria4', data = data_joined),)

        return (s_2, rprompt_2, s_2_co), (s_3, rprompt_3, s_3_co), (s_4, rprompt_4, s_4_co)

    async def score_criteria(self, criteria = 'criteria2', command = 'rank', data = None):
        rankllm_prompt = cot_output = None
        
        if criteria in self.criteria2openai_worker: 
            if self.cfg['test_print']: print(f'Starting work on criteria {criteria} with openai client worker')
            
            worker = self.criteria2openai_worker[criteria]
            
            __ = data.split(self.cfg['qud_sep'])
            qud_instance_id = __[0]
            qud_cands_list  = __[1:]
            
            (scores, rankllm_prompt, cot_output) = await worker.run_async_process_one_set(qud_instance_id, qud_cands_list)

            return (scores, rankllm_prompt, cot_output)

        else: 
            worker_rank = self.criteria2rank_mapping[criteria]
            if self.cfg['test_print']: print(f'Starting work on criteria {criteria} with worker {worker_rank}')
            
            await self.send_command_to_worker_async(worker_rank, command = command, data = data)
            if self.cfg['test_print']: print(f'Sent command to worker {worker_rank} for criteria {criteria}')
            
            while True:
                scores = await self.receive_from_worker_async(worker_rank)
                
                if not inspect.iscoroutine(scores) and scores is not None: # ensure its results (not coroutine obj) before proceeding
                    if self.cfg['test_print']:  print(f'Received response from worker {worker_rank} for criteria {criteria}')
                    # Now scores is the actual string, not a coroutine
                    scores = [float(i) for i in scores.split(self.cfg['score_sep'])]
                    return (scores, rankllm_prompt, cot_output)
                
                await asyncio.sleep(0.1)

### CHANGE START ###
# https://github.com/huggingface/trl/pull/3145
# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)
### CHANGE END ###