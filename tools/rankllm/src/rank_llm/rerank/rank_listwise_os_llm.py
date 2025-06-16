import json, re, torch, random, copy
from typing import Optional, Tuple
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from transformers.generation import GenerationConfig
from torch.nn.attention import SDPBackend, sdpa_kernel

### CHANGE START ###
import sys
sys.path.append('/home/khan/agentic_qud')
from tools.rank_llm.src.rank_llm.rerank.rankllm import PromptMode, RankLLM
from tools.rank_llm.src.rank_llm.data import Result
import numpy as np
from pydantic_core import from_json
from transformers import StoppingCriteria, StoppingCriteriaList
import contextlib 
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any([inp[-1].item() in self.stop_token_ids for inp in input_ids])

RE_COT_REASON1  = re.compile(r'<think>(.*?)</think>', re.DOTALL)
RE_COT_REASON2  = re.compile(r'(.*?)<answer>', re.DOTALL)
RE_COT          = re.compile(r'\[S_COT\](.*?)\[E_COT\]', re.DOTALL)
RE_RANK_REASON1 = re.compile(r'<answer>(.*?)</answer>' , re.DOTALL)
RE_RANK_REASON2 = re.compile(r'(?:</think>)(.+\s*)', re.DOTALL)
# capture cases such as: '''So the ranking would be [02] > [03] > [01].
# </answer> [02] > [03] > [01] </answer>'''
RE_RANK_REASON3 = re.compile(r'(?:\[\d+\] > )*\[\d+\](?!.*(?:\[\d+\] > )+\[\d+\])', re.DOTALL)
RE_RANK         = re.compile(r'(?:\[START\])(.+\s*)(?:\[STOP\])', re.DOTALL)

PKV_STRIP_REGEX = {'llama': re.compile(re.escape('<|begin_of_text|>')+'.+Cutting Knowledge Date.+?'+re.escape('<|eot_id|>'), re.DOTALL),
                #    'gemma': '', # TODO
                #    'phi': '',   # TODO
                   'qwen': re.compile(re.escape('''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'''))}
### CHANGE END ###




class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        ### CHANGE START ###
        model_path: str,
        ### CHANGE END ###
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        ### CHANGE START ###
        model_name: str = None,
        bypass_fsc_load: bool = False,
        hf_model = None,
        tokenizer = None,
        rerank_task_name = None,
        
        cfg = {},
        constraints_dict = None,
        qud_min_score = 1.0, 
        qud_max_score = 3.0,
        qud_exemplars = None,
        do_cot = False,
        cot_json = False, 
        **kwargs,
        ### CHANGE END ###
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(
            ### CHANGE START ###
            model_path, # model
            ### CHANGE END ###
            context_size, prompt_mode, num_few_shot_examples)
        ### CHANGE START ###
        self.bypass_fsc_load        = bypass_fsc_load
        self.model_path             = model_path
        self.model_name             = model_name
        self.rerank_task_name       = rerank_task_name
        self.c_qud                  = self.rerank_task_name == 'qud'
        self.input_context_cands_post = "{}\n\n## QUD Instance Attempts: \n"
        self.use_past_key_values    = cfg.ranker_args.use_past_key_values
        self.past_key_values        = None # will be set later (if use_past_key_values)
        self.past_key_values_len    = None # will be set later (if use_past_key_values)
        self.cfg                    = cfg
        self.constraints            = None
        self.num_beams              = None
        self.do_cot                 = do_cot
        self.cot_json               = cot_json
        self.qud_min_score          = qud_min_score
        self.qud_max_score          = qud_max_score
        for k,v in constraints_dict.items():
            setattr(self, k, v) # self.constraints, self.num_beams

        ### CHANGE END ###
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        ### CHANGE START ###
        self.use_openai_client = False
        self.openai_reasoning  = False
        if bypass_fsc_load:
            c1 = any(n in self.model_path for n in ['gpt-4o', 'deepseek-chat'])
            c2 = self.model_path.startswith('o3-mini')
            
            if c1 or c2:
                self.use_openai_client = True
            else: 
                assert hf_model is not None and tokenizer is not None
            
            if c2: 
                self.openai_reasoning = True
            
            self._llm, self._tokenizer = hf_model, tokenizer
            if self._tokenizer is not None and self._tokenizer.pad_token is None: # for pad to length
                self._tokenizer.pad_token       = self._tokenizer.eos_token
                self._tokenizer.pad_token_id    = self._tokenizer.eos_token_id
        
        else: self._llm, self._tokenizer = load_model(model_path, device=device, num_gpus=num_gpus)

        if getattr(self._llm, 'config', None) is not None:
            # no config for openai client
            self.gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        else: self.gen_cfg = None # could be openai client
        
        if self.use_past_key_values:
            assert self.num_beams in [1, None], self.num_beams

        if self.use_openai_client: self.cfg.ranker_args.use_past_key_values = False

        self.c1_check = any(n in self.model_path for n in ['GritLM/GritLM-7B'])
        self.c2_check = any(n in self.model_path for n in ['mistralai/Mistral-7B-Instruct-v0.3'])
        self.c_reason = self.model_name in ['dsr1_llama', 'dsr1_qwen', 'o3mini']
        if getattr(self.cfg.ranker_args, 'reasoning_instruct', False): 
            self.c_reason = True # for train/use of SFT/GRPO RM

        ### CHANGE END ###
        self._variable_passages = variable_passages
        self._window_size       = window_size
        self._system_message    = system_message
        self._output_token_estimate = None

        ### CHANGE START ###
        self.cot_gen_max_length_factor = 0
        if self.c_reason: 
            # allow model to <think> ... <\think>
            self.cot_gen_max_length_factor = 512
        ### CHANGE END ###
        if num_few_shot_examples > 0:
            ### CHANGE START ###
            if self.rerank_task_name in ['qud']:
                if self.cfg.ranker_args.do_cot:
                    self.cot_gen_max_length_factor += 256
                    if self.cfg.ranker_args.cot_fine: 
                        # min 2, else stops gen before rank result
                        self.cot_gen_max_length_factor *= 1.25 
                    if self.cfg.ranker_args.add_task_decomp_cot:
                        self.cot_gen_max_length_factor += 128
                assert qud_exemplars is not None
                self._examples = []
                for ex in qud_exemplars:
                    scores      = [c.score          for c in ex.candidates]
                    quds        = [c.doc['text']    for c in ex.candidates]
                    rationales  = [c.rationale      for c in ex.candidates]

                    # ensure exemplars have same number of candidates as the window size set
                    # ensure that the picked candidates have varying scores
                    random.seed(54506)
                    while True:
                        pick_pos = random.sample(list(range(len(scores))), self._window_size)
                        if len(set([scores[pos] for pos in pick_pos])) > 1: break
                    
                    ex_dict = {'query':     ex.query.text,
                               'quds':      [quds[pos]          for pos in pick_pos],
                               'scores':    [scores[pos]        for pos in pick_pos],
                               'rationales':[rationales[pos]    for pos in pick_pos],}
                    
                    self._examples.append(ex_dict)
            else:
            ### CHANGE END ###
                with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                    self._examples = list(json_file)[1:-1]

        self.max_new_tokens = self.cfg.gen_args.rankllm.max_new_tokens
        if (self._num_few_shot_examples > 0 and self.do_cot) or self.c_reason:
            # i.e. include for no few-shot & CoT, but deepseek R1 models
            self.max_new_tokens += int(self._window_size * self.cot_gen_max_length_factor) 
            # HACK 
            if self.model_path in ['google/gemma-2-27b-it']:
                self.max_new_tokens = min(768, self.max_new_tokens)
            
            print('👀 RANKLLM max_new_tokens set at:', self.max_new_tokens)
        

    def run_openai_client(self, messages: list, current_window_size: Optional[int] = None,):
        if current_window_size is None:
            current_window_size = self._window_size
        
        max_new_tokens = self.max_new_tokens
        if self._num_few_shot_examples > 0 and (self.do_cot or self.openai_reasoning):
            if self.openai_reasoning: 
                max_new_tokens = max(10000, max_new_tokens)
            
        run     = True
        tries   = 2
        rando   = [i for i in range(current_window_size)]
        random.shuffle(rando)
        outputs = ' > '.join([f'[{i+1}]' for i in rando])
        cot_output = None
        while run == True and tries > 0:
            try:
                input_pack = {'model': self.model_path, 'messages': messages, 
                    'max_completion_tokens': max_new_tokens, 'stream': False,
                    'temperature': 0}
                
                if self.openai_reasoning:
                    input_pack['reasoning_effort'] = self.cfg.models['openai_reasoning']['effort']

                response = self._llm.chat.completions.create(**input_pack)
                outputs = response.choices[0].message.content
                outputs, cot_output = self.cot_processor(outputs)
                run = False

            except Exception as e:
                print('FAILED (tries):', e, messages)
                tries -= 1
        return outputs, None, cot_output
        
        
    def run_llm(
        self, prompt: str, 
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        if current_window_size is None:
            current_window_size = self._window_size
        
        ### CHANGE START ###
        if getattr(self._model, 'is_vllm', False):
            raw_outputs, out_token_count = self.run_llm_vllm_branch(prompt)
        else: 
            raw_outputs, out_token_count = self.run_llm_hf_branch(prompt, current_window_size)
        
        outputs, cot_output = self.cot_processor(raw_outputs)
        
        return outputs, out_token_count, cot_output, raw_outputs
        ### CHANGE END ###

    ### CHANGE START ###
    def run_llm_vllm_branch(self, prompt):
        tokenizer        = self._model.get_tokenizer()
        prompt_token_ids = [self._model.token_prompt_class(prompt_token_ids = tokenizer.encode(prompt))]

        outputs         = self._model.generate(prompts = prompt_token_ids, 
                                       sampling_params = self._model.sampling_params, use_tqdm = False)
        assert len(outputs) == (outputs[0].outputs) == 1, (len(outputs), (outputs[0].outputs))
        outputs         = outputs[0].outputs[0].text
        out_token_count = len(outputs[0].outputs[0].token_ids)

        return outputs, out_token_count

    def run_llm_hf_branch(self, prompt, current_window_size):
        if self.model_name in ['dsr1_llama', 'dsr1_qwen']:
            # there is a bug in the chat_template where the special tokens use '|' instead of '｜'
            # this leads to special tokens being tokenized into many subwords
            prompt = prompt.replace('|', '｜')

        if self.use_past_key_values:
            c_llama = 'Cutting Knowledge Date' in prompt
            c_qwen  = 'You are Qwen,' in prompt
            if c_llama or c_qwen: prompt = self.pkv_suffix_strip_excess(prompt)
            inputs = None # will prepare below
            
        else:
            inputs = self._tokenizer([prompt], return_tensors = "pt").to(self._device)
        gen_cfg                = self.gen_cfg
        gen_cfg.max_new_tokens = self.max_new_tokens
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        if (self._num_few_shot_examples > 0 and self.do_cot) or self.c_reason:
            pass
        else: 
            gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False

        ### CHANGE START ### for Llama models (eos_token_id needs to be set to terminators)
        gen_args = {}
        if self.bypass_fsc_load:
            if self.model_name in ['llama']:
                terminators = [self._tokenizer.eos_token_id,
                            self._tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                gen_args = {'eos_token_id': terminators, 
                            # in llama, the pad_token_id is not set (rec: use the eos_token_id)
                            'pad_token_id': self._tokenizer.eos_token_id,}
            elif self.c1_check:
                gen_args = {'pad_token_id': self._tokenizer.pad_token_id,}
            elif (self.c2_check or self.model_name in ['qwen']) \
                and not (self.model_name in ['qwen'] and not self.c_reason):
                # in mistral, the pad_token_id is not set (rec: use the eos_token_id)
                # 21/03/25: "and not (self.model_name in ['qwen'] and not self.c_reason)"" for backward compat
                # there is pad_token for qwen. SFT RM used qwen pad_token
                gen_args = {'pad_token_id': self._tokenizer.eos_token_id,}
            elif self.model_name in ['dsr1_llama', 'dsr1_qwen']:
                # there is a bug in the tokenizer for deepseek models '｜' instead of '|' used in the chat_template
                # this causes the special tokens to be broken up. 
                stop_tokens = [self._tokenizer.eos_token_id]
                if self.model_name in ['dsr1_llama']: 
                    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/generation_config.json
                    stop_tokens.extend([128001, 128008, 128009])
                elif self.model_name in ['dsr1_qwen']:
                    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json
                    stop_tokens.extend([151645, 151643])
                stopping_criteria = StoppingCriteriaList([StopOnToken(set(stop_tokens))])
                gen_args = {'eos_token_id': self._tokenizer.eos_token_id,
                            'pad_token_id': self._tokenizer.eos_token_id,
                            'stopping_criteria': stopping_criteria}

        if self.use_past_key_values:
            if self.past_key_values is None:
                raise ValueError('Past key values not set.')
            
            inputs = self._tokenizer([self.pkv_prompt + prompt], padding = 'longest', 
                                    padding_side = 'right', pad_to_multiple_of = 64, 
                                    return_tensors = "pt").to(self._device)
            
            # inputs['input_ids']             = torch.cat([self.pkv_inputs['input_ids'], inputs['input_ids']], dim=-1)
            # inputs['attention_mask']        = torch.cat([self.pkv_inputs['attention_mask'], inputs['attention_mask']], dim=-1)
            # https://github.com/huggingface/transformers/issues/32896#issuecomment-2298486425
            # 24/01/25: problem with cache_position with generate()
            # https://github.com/huggingface/transformers/issues/35707
            past_key_values                 = copy.deepcopy(self.past_key_values)
            gen_args['use_cache']           = True
            gen_args['cache_position'] = torch.arange(self.past_key_values_len, 
                                                    inputs['input_ids'].shape[1], device = self._device)

        gen_args['return_dict_in_generate'] = True # esp for pkv, should be False. else issues with mask creation
        if self.constraints is not None:  gen_args['constraints']   = self.constraints
        if self.num_beams is not None:    gen_args['num_beams']     = self.num_beams
        if self.c_reason and '<think>' in self._tokenizer.vocab:  
            # NOTE: '<think>' tokens in dsr1 models. 
            # but not present if say we sft/grpo qwen
            gen_cfg.forced_bos_token_id = \
                gen_args['forced_bos_token_id'] = \
                    self._tokenizer.vocab['<think>']

        ### CHANGE START ###
        # a. use generate to obtain the ranker response
        # context = sdpa_kernel(SDPBackend.MATH) \
        #     if getattr(self._llm, 'attn_implementation') == 'sdpa' \
        #         else contextlib.nullcontext()
        with torch.no_grad():# and context:
            if self.use_past_key_values:
                output_holder = self._llm.generate(input_ids = inputs['input_ids'],
                                                    attention_mask = inputs['attention_mask'],
                                                    past_key_values = past_key_values,
                                                    **gen_args, generation_config = gen_cfg)
            
            else:
                output_holder = self._llm.generate(**inputs, **gen_args, generation_config = gen_cfg)
        
        if gen_args['return_dict_in_generate']: 
            output_ids = output_holder['sequences']
        else: output_ids = output_holder
        ### CHANGE END ###

        # # ~~~~~~~~~~~~~~~~~~~~~~~~ #
        # print('🟧'*100)
        # print('HERE 1.1', prompt)
        # # print('🟦'*100)
        # # print('HERE 1.2', outputs)
        # print('🟩'*100)
        # print('HERE 1.3', self._tokenizer.decode(output_ids[0], skip_special_tokens=False, spaces_between_special_tokens=False))
        # print('🟥'*100)
        # raise
        # # ~~~~~~~~~~~~~~~~~~~~~~~~ #

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        del output_holder
        out_token_count = output_ids.size(0)

        return outputs, out_token_count
    ### CHANGE END ###        

    def pkv_suffix_strip_excess(self, pkv_suffix):
        
        if self.model_name in PKV_STRIP_REGEX:
            pkv_suffix = re.sub(PKV_STRIP_REGEX[self.model_name], '', pkv_suffix)

        return pkv_suffix

    def cot_processor(self, outputs):
        cot_output = None
        if self.c_reason: cot_output = outputs

        if self.do_cot and not self.c_reason:
            # ~~~ 1. EXTRACT COT ~~~
            try:
                cot_output = self.extract_cot(outputs, reason = False)
            except Exception as e: 
                if not self.model_name in ['qwen']: # qwen frequently fails to follow and generate cot
                    print(f'COT EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

            # ~~~ 2. EXTRACT RANK ~~~
            try: 
                outputs = self.extract_rank(outputs, reason = False)
            except Exception as e: 
                print(f'RANKING EXTRACTION (FAILED) - \n🟥{e}\n', outputs)
        
        elif self.c_reason:
            # ~~~ 1. EXTRACT COT ~~~
            try:
                cot_output = self.extract_cot(outputs, reason = True)
            except Exception as e: 
                print(f'COT EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

            # ~~~ 2. EXTRACT RANK ~~~
            try: 
                outputs = self.extract_rank(outputs, reason = True)
            except Exception as e:
                print(f'RANKING EXTRACTION (FAILED) - \n🟥{e}\n', outputs)

        return outputs, cot_output

    def extract_cot(self, outputs, reason = False):
        # done this way (even though we have cot_json + reason cases)
        # so that we try to extract all reasoning traces and not just the json score
        if reason: 
            try: 
                cot_output = re.search(RE_COT, outputs).group(1)
                cot_output.strip()
            except: 
                try: 
                    cot_output = re.search(RE_COT_REASON1, outputs).group(1)
                except: 
                    cot_output = re.search(RE_COT_REASON2, outputs).group(1)
            cot_output.strip()

        else:
            if self.cot_json:
                cot_output = outputs
                # note: dotall works for openai's formatted json with newlines, but 
                # messes up llama etc's flat format
                cot_output = re.findall(r'{.+}', outputs) #, re.DOTALL)
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
                cot_output.strip()

        return cot_output

    def extract_rank(self, outputs, reason = False):
        try:
            if self.model_name in ['o3mini']:
                outputs     = re.search(RE_RANK_REASON3, outputs).group()
            else:
                # look for [START] [STOP] first
                outputs     = re.search(RE_RANK, outputs).group(1)
        except:
            if reason:
                try: 
                    outputs     = re.search(RE_RANK_REASON1, outputs).group(1)
                except: 
                    try: 
                        outputs     = re.search(RE_RANK_REASON2, outputs).group(1)
                    except: 
                        # NOTE: no group
                        outputs     = re.search(RE_RANK_REASON3, outputs).group()
                
        return outputs.strip()

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            ### CHANGE START ### zfill(2)
            seq     = " > ".join([f"[{str(i+1).zfill(2)}]" for i in range(current_window_size)])
            seq_tok = self._tokenizer.encode(seq) # to be safe, round-around check instead of skip_special_tokens=True
            if self._tokenizer.bos_token_id is not None:
                if seq_tok[0] == self._tokenizer.bos_token_id: seq_tok = seq_tok[1:]
            if self._tokenizer.eos_token_id is not None:
                if seq_tok[-1] == self._tokenizer.eos_token_id: seq_tok = seq_tok[:-1]
            # _output_token_estimate = (
            #     len(
            #         self._tokenizer.encode(seq)
            #     )
            #     - 1
            # )
            _output_token_estimate = len(seq_tok)
            ### CHANGE END ### 
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def _add_prefix_prompt(self, query: str, num: int,
                           ### CHANGE START ###
                           qud_criteria: str = None,
                           ### CHANGE END ###
                           ) -> str:
        ### CHANGE START ###
        if self.c_qud:
            prefix_dict     = self.cfg['prompts']['rankllm']['prefix']
            common          = prefix_dict['common']
            common          = common.replace('{{terminology}}', prefix_dict['terminology'])
            prefix_prompt   = prefix_dict[qud_criteria].replace('''{{common}}''', common).replace('''{{num_cands}}''', str(num))
            
            task_decomp_str = ''
            if self.cfg.ranker_args.add_task_decomp_common:
                task_decomp_str = self.cfg.prompts.rankllm.task_decomposition[qud_criteria]
            prefix_prompt = prefix_prompt.replace('{{task_decomposition}}', task_decomp_str)
        else: 
            prefix_prompt = f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

        return prefix_prompt
        ### CHANGE END ###

    
    def _add_post_prompt(self, query: str, num: int) -> str:
        ### CHANGE START ### example_ordering and self.rerank_task
        example_ordering = "[02] > [01]" if self._variable_passages else "[04] > [02]"
        if self.c_qud:
            post_dict   = self.cfg['prompts']['rankllm']['post']
            common      = post_dict['common']
            if self.c_reason: 
                rep_key = getattr(self.cfg.ranker_args, 'reasoning_rep_key', 'nline')
                oline   = post_dict['common_reasoning_replace']['oline']
                nline   = post_dict['common_reasoning_replace'][rep_key]
                common  = common.replace(oline, nline)
                # for SFT/GRPO RM training
                common  = common.replace('{{qud_min_score}}', str(self.qud_min_score))
                common  = common.replace('{{qud_max_score}}', str(self.qud_max_score))
                for key in ['cot_start', 'cot_end']:
                    cot_tag = post_dict.get(key, None)
                    if cot_tag is not None: common  = common.replace('{{' + key + '}}', cot_tag)                
            post_prompt = common.replace('''{{num_cands}}''', str(num))
            post_prompt = post_prompt.replace('''{{example_ordering}}''', example_ordering)
            return post_prompt
        
        else: 
            return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."
        ### CHANGE END ###

    def _add_few_shot_examples(self, conv, rank_start = None, rank_end = None,
                               ### CHANGE START ###
                               qud_criteria: str = None,
                               ### CHANGE END ###
                               ):
        assert len(self._examples) >= self._num_few_shot_examples

        cot = think_start = think_end = ans_start = ans_end = ''
        if self.c_reason:
            think_start, think_end  = '<think>', '</think>'     
            ans_start, ans_end      = '<answer>', '</answer>'
            think_str = f'''{think_start} ... ... {think_end} ... ... {{reasoning process truncated here}} ... ... '''

        for fse_idx in range(self._num_few_shot_examples):
            ### CHANGE START ###
            if self.rerank_task_name in ['qud']:
                np.random.seed(54506 + fse_idx)
                ex          = self._examples[fse_idx]
                perm_order  = np.random.permutation(len(ex['quds']))
                num_sys     = rank_end - rank_start
                assert len(perm_order) >= num_sys
                perm_order  = perm_order[:num_sys]
                
                prompt = self.input_context_cands_post.format(ex['query'])

                if fse_idx == 0:
                    # add prefix to front 
                    prefix = self._add_prefix_prompt(None, num_sys, qud_criteria = qud_criteria)
                    prompt = f"{prefix}\n" + prompt

                order       = np.argsort([ex['scores'][pos] for pos in perm_order])[::-1]
                response    = ' > '.join([f'[{i+1}]' for i in order]) + '\n'
                
                if self.do_cot:
                    cot = f'''First of all, I know that the higher the score the better, and the scores range from {self.qud_min_score} to {self.qud_max_score}. '''
                    if self.model_name in ['o3mini']:                        
                        cot += '''Secondly, I know that it is very important that I follow the formatting instruction when returning the rank results, as well as the scores and rationales. The score and rationale for each candidate has to be a standalone JSON (i.e. each should be a top-level object, not nested inside another object). '''
                    if self.cfg.ranker_args.add_task_decomp_cot:
                        task_decomp_str = self.cfg.prompts.rankllm.task_decomposition[qud_criteria]
                        # remove header
                        task_decomp_str = task_decomp_str.replace('## Guide to Scoring Scheme:\n', '').strip()
                        # replace you/You with 'I'
                        task_decomp_str = re.sub(r'you|You', 'I', task_decomp_str)
                        cot += task_decomp_str

                    if self.model_name in ['dsr1_llama', 'dsr1_qwen', 'o3mini']:
                        # NOTE: chat template removes everything between think tokens
                        cot += f'''{think_str} {ans_start} [S_COT] I think the candidates should get these scores based on the following assessments and reasoning: '''
                    else: 
                        cot += '''Let's think step-by-step and assess each candidate first before ranking them.\n[S_COT] I think the candidates should get these scores based on the following assessments and reasoning: '''
                    
                for rank, pos in enumerate(perm_order):
                    attempt     = ex['quds'][pos]
                    identifier  = str(rank+1).zfill(2)
                    prompt      += f"[{identifier}] {self._replace_number(attempt)}\n"
                    score       = int(ex['scores'][pos])
                    rationale   = self._replace_number(ex['rationales'][pos])
                    rationale   = rationale[0].lower() + rationale[1:]
                    if self.do_cot:
                        if self.cot_json:
                            cot += f'''\n[{identifier}]\t{{"candidate": "[{identifier}]", "rationale": "{rationale}", "score": {score}}} . '''
                        else:
                            cot += f'''\n[{identifier}]\tBecause: {rationale}... therefore, score of: {score}.'''
                
                prompt  += self._add_post_prompt(None, len(perm_order))
                if self.do_cot:
                    cot     += f' [E_COT] '
                    response = cot + f'\nBased on the above, I think the ranking should be as follows: [START] {response} [STOP] {ans_end}'
                else: 
                    if self.c_reason:
                        response = f'{think_str} {ans_start} {response} {ans_end}'
                    else: pass

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], response)
            
            else: 
            ### CHANGE END ###
                ex = random.choice(self._examples)
                obj = json.loads(ex)
                prompt = obj["conversations"][0]["value"]
                response = obj["conversations"][1]["value"]
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int,
        ### CHANGE START ###
        qud_criteria: str = None,
        ### CHANGE END ###
    ) -> Tuple[str, int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        ### CHANGE START ###
        # change max_length setting (DCQA articles could be longer than 300 tokens)
        # max_length = 300 * (20 / (rank_end - rank_start))
        max_length = self._context_size
        ### CHANGE END
        while True:
            ### CHANGE START ###
            if self.bypass_fsc_load: 
                # force use of rankLLM tempalate, else fastchat will retrieve a very unsuitable default
                conv = get_conversation_template('castorini/rank_zephyr_7b_v1_full')
            else: conv = get_conversation_template(self._model)
            ### CHANGE END ###
            if self._system_message:
                conv.set_system_message(self._system_message)
            ### CHANGE START ### 
            if self._num_few_shot_examples:
                conv = self._add_few_shot_examples(conv, rank_start = rank_start, rank_end = rank_end,
                                                   qud_criteria = qud_criteria)
                input_context = ""
            else:
                prefix = self._add_prefix_prompt(query, num, qud_criteria = qud_criteria)
                input_context = f"{prefix}\n"
            ### CHANGE END ###

            rank = 0
            ### CHANGE START ### zfill(2) & input_context_cands_post
            if self.rerank_task_name in ['qud']:
                input_context_cands_post = self.input_context_cands_post.format(query)
            else: input_context_cands_post = ''
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = self.covert_doc_to_prompt_content(cand.doc, max_length)
                input_context_cands_post += f"[{str(rank).zfill(2)}] {self._replace_number(content)}\n"
            
            input_context_cands_post += self._add_post_prompt(query, num)
            ### CHANGE END ###

            # append the entire prompt (i.e. with the doc cands and post prompt)
            # conv.append_message(conv.roles[0], input_context + input_context_cands_post)
            ### CHANGE START ### using past key values
            # assume num is fixed (k_candidates is fixed, and window is set.)
            pkv_msg_idx = None
            if self.use_past_key_values: 
                pkv_msg_idx = len(conv.messages)+1
            else: 
                # append the entire prompt (i.e. with the doc cands and post prompt)
                conv.append_message(conv.roles[0], input_context + input_context_cands_post)
                # NOTE: below not necessary... we set add_generation_prompt = True
                # conv.append_message(conv.roles[1], None)
            ### CHANGE END ###
            
            ### CHANGE START ###
            if self.bypass_fsc_load: 
                ##### SYSTEM PROMPT
                messages = []
                if self.rerank_task_name in ['qud']:
                    sys_message = self._system_message
                else: 
                    sys_message = 'You are an intelligent assistant that can rank the relevance of a set of passages to a given query.'
                    
                # some models have no system prompts... in these cases, we add sys prompt to first message
                # same for deepseek-r1 models... 
                # see https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B#usage-recommendations
                # but we add the sys prompt to the user message
                c_no_sys_msg = (self.model_name in ['dsr1_llama', 'dsr1_qwen']) or \
                                'mistralai' in self.model_path or 'gemma' in self.model_path
                if c_no_sys_msg: 
                    conv.messages[0][1] = f"{sys_message}\n\n" + conv.messages[0][1]
                else: 
                    if self.openai_reasoning:
                        messages.append({'role': 'developer', 'content': sys_message})
                    else:
                        messages.append({'role': 'system', 'content': sys_message})

                for m in conv.messages:
                    role = m[0].lower().replace('|', '').replace('<', '').replace('>', '')
                    if  role  in ['human', 'user']: message = {'role' : 'user',}
                    elif role in ['assistant']:     message = {'role': 'assistant',}
                    elif role in ['system', 'developer']:        
                        if self.openai_reasoning:
                            # https://platform.openai.com/docs/guides/reasoning?lang=python&example=research#advice-on-prompting
                            message = {'role': 'developer',}
                        else: 
                            message = {'role': 'system',}
                    else: raise NotImplementedError(f'🚨\t\tUnknown role: {role}')
                    
                    # apply_chat_template will add the gen prompt (add_generation_prompt = True)
                    if m[1] is None: continue
                    
                    message['content'] = m[1]
                    messages.append(message)

                ### CHANGE START ### for using past key values
                if self.use_past_key_values and not self.use_openai_client:
                    # separate the pkv portion, and the rest. 
                    # NOTE: 
                    # 1. at this point all of input_context_cands_post is left out of either 'pkv_msgs' or 'messages'
                    # 2. once 'messages' is converted to prompt below, we will prepend input_context_cands_post to it
                    # 3. very important to set add_generation_prompt False
                    pkv_msgs, messages = messages[:pkv_msg_idx], messages[pkv_msg_idx:]
                    assert len(messages) == 0, f'🚨\t\t"messages" should be empty after split, \n{pkv_msgs}\n{messages}' 
                    # pkv_msgs.append({'role': 'assistant', 'content': 'I understand the instructions.'})
                    # certain models (e.g. gemma) require the 1st msg for input to apply_chat_template to be a user msg
                    messages = [{'role': 'user', 'content': input_context_cands_post}]
                    
                    if self.past_key_values is None:
                        # see https://huggingface.co/docs/transformers/v4.48.0/kv_cache#iterative-generation-with-cache
                        self.pkv_prompt             = self._tokenizer.apply_chat_template(pkv_msgs, tokenize = False, 
                                                                    add_generation_prompt = False)
                        self.pkv_inputs             = self._tokenizer([self.pkv_prompt], 
                                                                    return_tensors = 'pt').to(self._llm.device)
                        self.past_key_values_len    = self.pkv_inputs['input_ids'].shape[1]                        
                        self._llm.model.eval()
                        with torch.no_grad():
                            # https://huggingface.co/docs/transformers/main/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
                            # prompt_cache = StaticCache(config = self._llm.model.config, max_batch_size = 2, 
                            #                         max_cache_len = self.past_key_values_len + self.max_new_tokens + 128, 
                            #                         device = self._llm.device, dtype = self._llm.dtype)
                            # __ = self._llm.model(**self.pkv_inputs, past_key_values = prompt_cache)#.past_key_values
                            # self.past_key_values = prompt_cache.to_legacy_cache()

                            self.past_key_values = self._llm.model(**self.pkv_inputs, past_key_values = None,
                                                                   return_dict_in_generate = True).past_key_values
                            
                            
                        print('🎉'*100)
                        print('Past key values set:\n', self.pkv_prompt, '\n')
                        print('🎉'*100)
                        
                        if self.num_beams is not None:
                            raise NotImplementedError
                            self.past_key_values = past_key_values_expander(self.past_key_values, self.num_beams)
                            print('\t\t' +'🎉'*10, f'{self.num_beams} beams specified... past key values expanded.\n'+'🎉'*10)
                
                if not self.use_openai_client:
                    prompt = self._tokenizer.apply_chat_template(messages, tokenize = False, 
                                                                  add_generation_prompt = True)  
                else: prompt = message
                ### CHANGE END ### 

            else: prompt = conv.get_prompt()

            ### CHANGE START ###     
            if not self.use_openai_client:
                prompt = fix_text(prompt)
                num_tokens = self.get_num_tokens(prompt)
                if num_tokens <= self.max_tokens() - self.num_output_tokens(
                    rank_end - rank_start
                ):
                    break
                else:
                    max_length -= max(
                        1,
                        (
                            num_tokens
                            - self.max_tokens()
                            + self.num_output_tokens(rank_end - rank_start)
                        )
                        // ((rank_end - rank_start) * 4),
                    )
            
            else: break
            
            ### CHANGE END ###
        
        # print('^'*100)
        # print(prompt) 
        # print('^'*100, '\n')
        # raise ValueError
        return prompt, self.get_num_tokens(prompt), messages

    def get_num_tokens(self, prompt: str) -> int:
        ### CHANGE START ###
        if type(prompt) != str: return None
        ### CHANGE END ###
        else: return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0


def past_key_values_expander(past_key_values, num_beams):
    '''
    helper function to expand past key values (obtained for e.g. on a constant prompt
    whereby past_key_values was obtained by passing a prefix text through the model). 
    Done so that past_key_values can be used for multiple beams search in generate()
    - for decoder only models: shape of past key values is 
    (num_layers, attn layers, num_beams, ...). 

    '''
    pkv = []
    for layer in past_key_values:
        pkv_1 = []
        for attn_layer in layer:
            assert len(attn_layer) == 1 and attn_layer.dim() == 4, \
                (len(attn_layer), attn_layer.shape)
            pkv_1.append(attn_layer.repeat(num_beams,1,1,1))
        pkv.append(pkv_1)

    return pkv