import hydra, tqdm, json, sys, os, yaml
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict, Counter
import itertools
ROUND = 4
SEED  = 54506

@hydra.main(version_base = None, config_path = "../config", config_name = "config_phase1.yaml")
def main_phase1(cfg: DictConfig):
    os.environ['HF_HOME'] = os.path.dirname(cfg.hub_dirpath)
    import gc, torch
    from utils_model import (model_load_wrapper, load_model_rankllm, 
                             convert_defaultdict, process_one_step_rankllm)
    from utils_file import (give_savepath, load_panel_outputs, save_panel_outputs,
                            give_panel_outputs_fp)
    sys.path.append(cfg.dirpath)
    from data.dataloader import load_phase1, make_one_rankllm_request, make_exemplar_objects
    from tools.rank_llm.src.rank_llm.data import Query, Candidate, Request

    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(54506)
    use_past_key_values = cfg.ranker_args.use_past_key_values

    ########################################
    ## 0a. CoT generation settings
    c_do_cot = cfg.ranker_args.num_few_shot_examples > 0 and cfg.ranker_args.do_cot
    if c_do_cot: 
        cfg.gen_args.rankllm.max_new_tokens = 256
        oline = cfg.prompts.rankllm.post.common_cot_replace['oline']
        nline = cfg.prompts.rankllm.post.common_cot_replace['nline']
        if cfg.ranker_args.cot_json:
            add_str = 'It is very important that you give each score and rationale in a string that can be parsed into JSON. '
            nline =  add_str + nline
            cfg.prompts.rankllm.post.common_cot_replace['oline'] = \
                    add_str + cfg.prompts.rankllm.post.common_cot_replace['oline']
            # also ensure that reasoning's common has add_str added. 
            # else replacement fails in _add_post_prompt in rank_listwise_os_llm.py
            cfg.prompts.rankllm.post.common_reasoning_replace['oline'] = \
                    cfg.prompts.rankllm.post.common_reasoning_replace['oline'].replace(oline, nline)
        cfg.prompts.rankllm.post.common = cfg.prompts.rankllm.post.common.replace(oline, nline)

    # set terminology in prompt
    cfg.prompts.rankllm.prefix.common = cfg.prompts.rankllm.prefix.common.replace('{{terminology}}', 
                                                            cfg.prompts.rankllm.prefix.terminology)
    # ensure common part of prefix has correct symbol for marking empty context.
    cfg.prompts.rankllm.prefix.common = cfg.prompts.rankllm.prefix.common.replace('{{context_empty}}', 
                                                            cfg.prompts.rankllm.prefix.context_empty)
    
    if cfg.ranker_args.do_cot: cfg.rerank_constrained_bs.do_constrained = False
    
    if type(cfg.insert_new_gens) == str: 
        print(cfg.insert_new_gens)
        cfg.insert_new_gens = eval(cfg.insert_new_gens)
    dp_ng = cfg.dirpath + '/results/main_grpo'
    cfg.insert_new_gens_mapping = {
        'v1': {
        'RB2K': f'{dp_ng}/grpo_settings-ms-2000-irbs-1000000000.0-b-0.04-rw-N-srm-F-rmma-0.9-rmss-64-qgb-4-nc-4-gas-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0_models-llama-mini-full-rulesbasedQUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON/RANK0/test_outputs_final.json',
        'LPC0': f'{dp_ng}/grpo_settings-ms-500-si-N-irbs-0-b-0.04-e-0.2-rw-N-srm-F-rmma-0.9-rmss-64-qgb-4-nc-4-gas-4-ni-1-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0_models-llama-mini-C2-gp-C3-gp-C4-gpQUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON_LPC-0/RANK0/test_outputs_final.json',
        'LPC1': f'{dp_ng}/grpo_settings-ms-500-si-N-irbs-0-b-0.04-e-0.2-rw-N-srm-F-rmma-0.9-rmss-64-qgb-4-nc-4-gas-4-ni-1-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0_models-llama-mini-C2-gp-C3-gp-C4-gpQUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON_LPC-1/RANK0/test_outputs_final.json',
        'LPC2': f'{dp_ng}/grpo_settings-ms-500-si-N-irbs-0-b-0.04-e-0.2-rw-N-srm-F-rmma-0.9-rmss-64-qgb-4-nc-4-gas-4-ni-1-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0_models-llama-mini-C2-gp-C3-gp-C4-gpQUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON_LPC-2/RANK0/test_outputs_final.json',
        'QS3B': f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_3B.json', 
        'QS8B': f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_8B.json', },
        
        'v2': {
        'QRB2K':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-1000000000.0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-2-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0-sr-F-qwen-mini-full-rulesbased_RFv2QUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON/RANK0/test_outputs_final.json',
        'QRB2Kqs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-1000000000.0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-2-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0-sr-F-qwen-mini-full-rulesbased_RFv2QUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON/RANK0/test_outputs_final_qsal.json',
        'QZS':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step0.json',
        'QZSqs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step0_qsal.json',
        'QLPC0':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step1000.json',
        'QLPC0qs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step1000_qsal.json',
        'QLPC1':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step1500.json',
        'QLPC1qs':    f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_step1500_qsal.json',
        'QLPC2':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_final.json',
        'QLPC2qs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-qwen-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-1/RANK0/test_outputs_final_qsal.json',
        
        'LRB2K':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-1000000000.0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-2-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0-sr-F-llama-mini-full-rulesbased_RFv2QUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON/RANK0/test_outputs_final.json',
        'LRB2Kqs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-1000000000.0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-2-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1200-gmu-0.6-t-1.0-sr-F-llama-mini-full-rulesbased_RFv2QUD_GEN-RANK-COMTDECOMP_COT-3-FINE-JSON/RANK0/test_outputs_final_qsal.json',
        'LZS':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step0.json',
        'LZSqs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step0_qsal.json',
        'LLPC0':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step1000.json',
        'LLPC0qs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step1000_qsal.json',
        'LLPC1':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step1500.json',
        'LLPC1qs':    f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_step1500_qsal.json',
        'LLPC2':        f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_final.json',
        'LLPC2qs':      f'{dp_ng}/grpo-ms-2000-si-N-irbs-0-b-0.04-e-0.2-eh-0.28-rw-N-srm-F-rmma-0.9-rmss-64-gb-4-nc-4-gas-4-ni-4-srl-T-ptm-N-ud-F-lr-32-lar-0.5-ld-0.05-msl-512-mpl-1400-gmu-0.6-t-1.0-sr-F-llama-mini-C2-qw-mi-C3-qw-mi-C4-qw-mi_RFv3QG-RK-CD_COT-3-FI-JS_LPC-2/RANK0/test_outputs_final_qsal.json',
        
        'LQS3B':         f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_llama_3B.json', 
        'LQS3Bqs':       f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_llama_3B.json', 
        'LQS8B':         f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_llama_8B.json', 
        'LQS8Bqs':       f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_llama_8B.json', 
        
        'QQS3B':         f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_qwen_3B.json', 
        'QQS3Bqs':       f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_qwen_3B.json', 
        'QQS7B':         f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_qwen_7B.json', 
        'QQS7Bqs':       f'{cfg.dirpath}/tools/qudselect/qud_parser_joint/data/processed/final_quds_QUDEVAL_RANK_qwen_7B.json', }
    }
    
    ########################################
    ## 0b. create save folder, hf token
    cfg.savepath = give_savepath(cfg, phase = 1)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    print('✅ DIRECTORY CREATED AT:', cfg.savepath)

    token_path = 'api_tokens.json'
    if cfg.token_dict is None and os.path.exists(token_path):
        with open(token_path, encoding = 'utf-8') as f:
            cfg.token_dict = json.load(f)
        cfg.hf_token   = cfg.token_dict['hf']
    else: cfg.hf_token = None

    
    ########################################
    # 1. load data 
    holder_articles, holder_quds, num_cands2qud_idxes = load_phase1(cfg)
    holder_exemplars = make_exemplar_objects(cfg)
    print('🟩🟩\tDATA LOADED')
    
    ########################################
    # 2. prepare inputs and results holder
    # holder for "rankllm_choices"
    holder_panel_outputs    = {}
    ranker_requests         = {}
    exemplar_requests       = {}
    for qud_criteria in cfg.qud_criteria_list:
        holder_panel_outputs[qud_criteria] = defaultdict(dict) 
        # prepare Request objects
        ranker_requests[qud_criteria]   = {}
        exemplar_requests[qud_criteria] = {} 
        for hname, holder in (('quds', holder_quds), 
                              ('exemplars', holder_exemplars[qud_criteria])):
            for num, (qud_instance_id, qud_instance) in enumerate(holder.items()):
                req = make_one_rankllm_request(SEED + num, holder_articles, 
                                    qud_instance, qud_instance_id, qud_criteria, 
                                    cfg.exclude_same_scores, 
                                    cfg.prompts.rankllm.prefix.context_empty,
                                    rankllm_obj_pack = (Query, Candidate, Request))
                
                if req is None: continue
                
                if hname == 'quds':
                    ranker_requests[qud_criteria][qud_instance_id] = req
                elif hname == 'exemplars':
                    exemplar_requests[qud_criteria][qud_instance_id] = req
                else: raise NotImplementedError

        print(f'🟦🟦\tNUM requests per num_cand... {qud_criteria} ', 
                Counter(len(r.candidates) for r in ranker_requests[qud_criteria].values()))     
    
    ########################################
    # 3. begin panel-of-LLM for 3x criteria
    models_to_use = dict(cfg.models[cfg.model_size])
    if cfg.ranker_args.phase1_use_closed_models:
        models_to_use.update(dict(cfg.models['closed_models']))
    for rankllm_sys, model_path in models_to_use.items():
        if cfg.do_model is not None and rankllm_sys != cfg.do_model: 
            print('👀 cfg.do_model specified... skipping', rankllm_sys)
            continue
        c_panel_save = False
        panel_outputs_fp = give_panel_outputs_fp(cfg, rankllm_sys)

        
        print('👀 panel_outputs_fp:', panel_outputs_fp)
        # if panel outputs for rankllm_sys already present, reuse
        if os.path.exists(panel_outputs_fp) and not cfg.reinitialise:
            c_load, holder_panel_outputs = \
                load_panel_outputs(cfg, panel_outputs_fp, holder_panel_outputs, rankllm_sys)
            if c_load is True: continue

        # NOTE (11/02): ignoring o3 doc and proceeding with CoT-like for o3
        # there is no need for CoT prompting for o1/o3 models
        # https://platform.openai.com/docs/guides/reasoning?lang=python&example=research#advice-on-prompting
        # if rankllm_sys in ['o3mini'] and cfg.ranker_args['do_cot']: continue
        
        # a. load LLM once  
        tokenizer = rankllm_model = pipeline_model = None
        gc.collect()
        torch.cuda.empty_cache()  
        tokenizer, pipeline_model, device, cfg = \
            model_load_wrapper(cfg, rankllm_sys, model_path, 
                               cfg.model_size, use_past_key_values)
        
        # b. run through criteria
        for qud_criteria in cfg.qud_criteria_list:
            qud_exemplars = list(exemplar_requests[qud_criteria].values())
            
            # process by num_cands 
            for num_cands in sorted(num_cands2qud_idxes, reverse = True):
                qud_idxes       = num_cands2qud_idxes[num_cands]
                requests_dict   = {q_i_id: rankllm_req for q_i_id, rankllm_req \
                                    in ranker_requests[qud_criteria].items() \
                                        if q_i_id in qud_idxes}
                
                print('\n\n')
                print(f'🟧🟧\tWorking on {qud_criteria} and num_cands: {num_cands}')
                
                # ensure window size set
                window_size = cfg.ranker_args['window_size'] = num_cands
                # A. load num_cands-specific rankllm 
                rankllm_model, cfg, constraints_dict = load_model_rankllm(cfg = cfg, 
                                                        model_name = rankllm_sys, model_path = model_path, 
                                                        tokenizer = tokenizer, pipeline_model = pipeline_model,
                                                        device = device, qud_exemplars = qud_exemplars)
                print('🟩🟩\tMODEL LOADED', rankllm_sys, model_path, f'for num_cands: {num_cands}')

                # B. start running through QUDInstances
                for __, (qud_instance_id, rankllm_req) in enumerate(tqdm.tqdm(requests_dict.items())):
                    c1 = qud_instance_id in holder_panel_outputs[qud_criteria]
                    if c1: continue
                    else:  c_panel_save = True
                    proc_predictions = process_one_step_rankllm(cfg, rankllm_model, 
                                                window_size, rankllm_req, qud_criteria)

                    assert len(proc_predictions) == 1
                    for q_i_id, proc_pred in proc_predictions.items():
                        
                        if __ == 0: print(f'{"~"*50}\nPROCESSED:', proc_pred, f'\n{"~"*50}')
                        
                        holder_panel_outputs[qud_criteria][q_i_id][rankllm_sys] = proc_pred
        
        # save ranksys_llm's panel outputs
        if c_panel_save: save_panel_outputs(cfg, panel_outputs_fp, holder_panel_outputs, rankllm_sys)
    
    tokenizer = rankllm_model = pipeline_model = None
    gc.collect()
    torch.cuda.empty_cache()  

    holder_articles         = convert_defaultdict(holder_articles)
    holder_panel_outputs    = convert_defaultdict(holder_panel_outputs)

    ########################################
    # 4. run evaluation
    holder_eval_rules_based, holder_eval_qudeval_gpt, holder_eval_qudeval_gpt_few_shot, \
        holder_eval_qudselect_classifiers, holder_eval_stv, holder_eval_llm_qa, \
            holder_ranx_results, holder_t_r_a_results = \
        evaluation_phase1(cfg, holder_articles, holder_quds, 
                          holder_panel_outputs, num_cands2qud_idxes, 
                          rankllm_systems = list(models_to_use.keys()))
    
    ########################################
    # 5. save out to file
    data_objs       = [cfg, holder_panel_outputs, holder_eval_qudeval_gpt, holder_eval_qudeval_gpt_few_shot, 
                       holder_eval_qudselect_classifiers, holder_eval_rules_based, holder_eval_stv, 
                       # holder_eval_llm_qa, 
                       holder_ranx_results,holder_t_r_a_results]
    data_objs_names = ['cfg', 'holder_panel_outputs', 'holder_eval_qudeval_gpt', 'holder_eval_qudeval_gpt_few_shot',
                       'holder_eval_qudselect_classifiers', 'holder_eval_rules_based', 'holder_eval_stv', 
                       # 'holder_eval_llm_qa',  - we save holder_eval_llm_qa inside evaluation_phase1()
                       'holder_ranx_results','holder_t_r_a_results']
    for obj, obj_name in zip(data_objs, data_objs_names):
        if obj_name == 'cfg': 
            obj = OmegaConf.to_yaml(obj)
            savepath = f'{cfg.savepath}/{obj_name}.yaml'
            with open(savepath, mode = 'w+', encoding = 'utf-8') as f:
                yaml.dump(obj, f)
        else: 
            savepath = f'{cfg.savepath}/{obj_name}.json'
            with open(savepath, mode = 'w+', encoding = 'utf-8') as f:
                    json.dump(obj, f)
    print('🥳🥳\tALL DONE!')

            
def evaluation_phase1(cfg, holder_articles, holder_quds, 
                      holder_panel_outputs, num_cands2qud_idxes, rankllm_systems):
    print(f'EVAL -- {"rankllm_systems".upper()}:', rankllm_systems)
    import numpy as np
    sys.path.append('evaluation')
    from phase1_metrics import (top_ranked_accuracy, do_one_run_ranx)
    from utils_model import convert_defaultdict
    from utils_evaluate import (evaluate_qud_rules_based, evaluate_qudeval_gpt,
                                evaluate_qud_pick_stv, evaluate_llmqalogprobs, 
                                evaluate_qudselect_classifiers, make_score_ordinal)
    
    # ########################################
    # 1a. Rules-based metrics
    if not cfg.do_model:
        holder_eval_rules_based = evaluate_qud_rules_based(cfg, holder_articles, holder_quds)

    ########################################
    # 1b. LLM QA log_prob computation for answer compatability
    if not cfg.do_model and cfg.answer_compat.do_icl:    
        fp = f'{cfg.dirpath}/data/exemplars/answer_compat_llm_qa.json'
        with open(fp, encoding = 'utf-8') as f:
            cfg.prompts.answer_compat.icl_exemplars = json.load(f)

        savepath = f'{cfg.savepath}/holder_eval_llm_qa.json'
        if not os.path.exists(savepath):
            holder_eval_llm_qa = {}
            for model_size in cfg.answer_compat.model_sizes:
                for model_name in cfg.answer_compat.model_names:
                    holder_eval_llm_qa[f'{model_size}-{model_name}'] = \
                        evaluate_llmqalogprobs(cfg, holder_articles, holder_quds,
                                                            model_name, model_size)
                    print('LLM QA log_probs done for:', model_name, model_size)
            
            with open(savepath, mode = 'w+', encoding = 'utf-8') as f:
                json.dump(holder_eval_llm_qa, f)
        else: 
            with open(savepath, encoding = 'utf-8') as f:
                holder_eval_llm_qa = json.load(f)
    else: 
        holder_eval_llm_qa      = {}

    ########################################
    # 1c. QUDEval's GPT scoring
    fp = f"{cfg.dirpath}/results/main_phase1/holder_eval_qudeval_gpt.json"
    if cfg.exp_code == 'test': fp = fp.replace('.json', '_test.json')
    if not os.path.exists(fp):
        holder_eval_qudeval_gpt = \
            evaluate_qudeval_gpt(cfg, holder_articles, holder_quds, model_name = 'gpt-4o')
        with open(fp, encoding = 'utf-8', mode = 'w+') as f:
            json.dump(holder_eval_qudeval_gpt,f)
    else: 
        with open(fp, encoding = 'utf-8') as f:
            holder_eval_qudeval_gpt = json.load(f)

    # 1d. GPT scoring with few-shot (aligned with rankllm-style instructions etc if specified)
    # purpose is to provide a like for like comparison about single-QUD scoring vs multiple-QUD ranking+scoring
    cot_str = ''
    c_do_cot = cfg.ranker_args.num_few_shot_examples > 0 and cfg.ranker_args.do_cot
    if c_do_cot: 
        cot_str += f'_COT-{cfg.ranker_args.num_few_shot_examples}'
        if cfg.ranker_args.cot_fine:  cot_str += '-FINE'
        if cfg.ranker_args.cot_json:  cot_str += '-JSON'
    elif not c_do_cot and  cfg.ranker_args.num_few_shot_examples > 0:
        cot_str += f'_ICL-{cfg.ranker_args.num_few_shot_examples}'
    fp = f"{cfg.dirpath}/results/main_phase1/holder_eval_qudeval_gpt_few_shot{cot_str}.json"
    if cfg.exp_code == 'test': fp = fp.replace('.json', '_test.json')
    if not os.path.exists(fp):
        holder_eval_qudeval_gpt_few_shot = \
            evaluate_qudeval_gpt(cfg, holder_articles, holder_quds, model_name = 'gpt-4o', few_shot = True)
        with open(fp, encoding = 'utf-8', mode = 'w+') as f:
            json.dump(holder_eval_qudeval_gpt_few_shot,f)
    else: 
        with open(fp, encoding = 'utf-8') as f:
            holder_eval_qudeval_gpt_few_shot = json.load(f)

    # 1e. the classifiers proposed by QUDSelect
    if cfg.insert_new_gens and cfg.do_model: 
        from utils_file import give_panel_outputs_fp
        dp = os.path.dirname(give_panel_outputs_fp(cfg, cfg.do_model))
        fp = f"{dp}/holder_eval_qudselect_classifiers.json"
    else:                   
        fp = f"{cfg.dirpath}/results/main_phase1/holder_eval_qudselect_classifiers.json"
    if cfg.exp_code == 'test': fp = fp.replace('.json', '_test.json')
    if not os.path.exists(fp):
        holder_eval_qudselect_classifiers = \
            evaluate_qudselect_classifiers(cfg, holder_articles, holder_quds)
        with open(fp, encoding = 'utf-8', mode = 'w+') as f:
            json.dump(holder_eval_qudselect_classifiers,f)
    else: 
        with open(fp, encoding = 'utf-8') as f:
            holder_eval_qudselect_classifiers = json.load(f)

    # NOTE: moved from utils_file.py. to allow us to do QUDSelect classifiers too
    if cfg.do_model: raise SystemExit

    ########################################
    # 2. compute NDCG for each of Rules-based and STV
    exclude_rankllm_sys = []#['o3mini'] 
    
    holder_ranx_results     = {}
    holder_eval_stv         = {}
    holder_t_r_a_results    = {}
    # rankllm_systems         = list(cfg.models[cfg.model_size])
    print(f'{"rankllm_systems".upper()}:', rankllm_systems)
    if cfg.ranker_args['do_cot']: 
        rankllm_systems = [s for s in rankllm_systems if s not in exclude_rankllm_sys]
    rankllm_combis          = []
    min_num_sys             = min(2, len(rankllm_systems))
    for num_sys in range(min_num_sys, len(rankllm_systems)+1):
        rankllm_combis += list(itertools.combinations(rankllm_systems, r = num_sys))
    # also do per-system NDCG
    rankllm_combis += [[sys] for sys in rankllm_systems]

    for combi in rankllm_combis:
        combi           = sorted(combi)
        combi_name      = '-'.join(combi)
        print('WORKING ON STV FOR: ', combi_name)
        # 1c. STV picking
        holder_eval_stv[combi_name] = {}
        for qud_criteria in cfg.qud_criteria_list:
            holder_eval_stv[combi_name][qud_criteria] = \
                evaluate_qud_pick_stv(cfg, 
                    holder_panel_outputs, qud_criteria, keep_systems = set(combi))
        
        ########################################
        ranx_results    = defaultdict(dict)
        t_r_a_results   = defaultdict(dict)
        for qud_criteria in cfg.qud_criteria_list:
            
            qrels_dict      = {}
            include_q_i_ids = set()

            for q_i_id, qud_instance in holder_quds.items():
                line = {c.sysname : \
                        c.criteria_scores[qud_criteria] \
                        for c in qud_instance.qud_candidates}
                line = make_score_ordinal(line, break_tie = False)
                
                # only include rankable instances where NOT all QUD candidates have the same gold score
                # (it is not meaningful to compute NDCG with those cases)
                # NOTE: in data/dataloader.py, cfg.exclude_same_scores would have removed 
                # such cases from being set as Requests to be sent for rankllm inference.
                if len(set(line.values())) != 1: include_q_i_ids.add(q_i_id)
                qrels_dict[q_i_id] = line

            run_preds_rules     = holder_eval_rules_based[qud_criteria]
            run_preds_qeval     = holder_eval_qudeval_gpt[qud_criteria]
            run_preds_qeval_fs  = holder_eval_qudeval_gpt_few_shot[qud_criteria]
            run_preds_qsclass   = holder_eval_qudselect_classifiers[qud_criteria]
            run_preds_stv       = {q_i_id: {'ord': __['stv_scoring'], 'raw': __['stv_scoring']} for q_i_id, __ \
                                in holder_eval_stv[combi_name][qud_criteria].items()}
            run_preds_runoff    = {q_i_id: {'ord': {x: i+1 for i,x in enumerate(__['runoff_winners'])},
                                            'raw': {x: i+1 for i,x in enumerate(__['runoff_winners'])},} \
                                   for q_i_id, __ in holder_eval_stv[combi_name][qud_criteria].items()}
            run_preds_llm_qa    = {}
            if qud_criteria in ['criteria2']:
                run_preds_llm_qa = {f"llm_qa-{k}": holder_eval_llm_qa[k][qud_criteria] for k in holder_eval_llm_qa}
            
            ###########################################################
            print('\n\n' + '🟧'*50)
            print(f'qud_criteria ({combi_name.upper()}):', qud_criteria.upper())

            run_preds = {'rules':   run_preds_rules, 
                         'qeval':   run_preds_qeval,
                         'qeval_fs':run_preds_qeval_fs,
                         'qsclass': run_preds_qsclass,       
                         'stv':     run_preds_stv, 
                         'runoff':  run_preds_runoff,
                         **run_preds_llm_qa}
            
            # global NDCG
            metrics     = ['ndcg@1'] + [f'ndcg@{i}' for i in sorted(num_cands2qud_idxes)] + ['recall@1']
            report_obj  = do_one_run_ranx(qrels_dict, metrics, run_preds, 
                                          include_q_i_ids, round = ROUND)
            
            ranx_results[qud_criteria]['global'] = \
                report_obj.to_dataframe().to_dict()
            print(f'\nRANX REPORT (GLOBAL):', report_obj)

            outputs_t_r_a = top_ranked_accuracy(qrels_dict, run_preds, include_q_i_ids)
            print(f'\nTOP RANKED MATCH REPORT (GLOBAL):', 
                  {k: np.mean(v) for k,v in outputs_t_r_a.items()})
            t_r_a_results[qud_criteria]['global'] = outputs_t_r_a

            ###########################################################
            # NDCG at num_cands level
            for num_cands in sorted(num_cands2qud_idxes):
                print('qud_criteria:', qud_criteria.upper())
                cut_q_i_idxes = num_cands2qud_idxes[num_cands]
                cut_qrels_dict          = {q_i_id: __ for q_i_id, __ \
                                        in qrels_dict.items() if q_i_id in cut_q_i_idxes}

                cut_run_preds_rules     = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_rules.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_qeval     = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_qeval.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_qeval_fs  = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_qeval_fs.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_qsclass   = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_qsclass.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_stv       = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_stv.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_runoff    = {q_i_id: __ for q_i_id, __ \
                                        in run_preds_runoff.items() if q_i_id in cut_q_i_idxes}
                cut_run_preds_llm_qa    = {}
                if qud_criteria in ['criteria2']:
                    cut_run_preds_llm_qa = {k: {q_i_id: __ for q_i_id, __ \
                                                in v.items() if q_i_id in cut_q_i_idxes} \
                                                    for k,v in run_preds_llm_qa.items()}
                
                cut_run_preds = {'rules':   cut_run_preds_rules,
                                 'qeval':   cut_run_preds_qeval,  
                                 'qeval_fs':cut_run_preds_qeval_fs,  
                                 'qsclass': cut_run_preds_qsclass,  
                                 'stv':     cut_run_preds_stv,
                                 'runoff':  cut_run_preds_runoff,
                                 **cut_run_preds_llm_qa}
                
                print('\n' + '🟦'*50)
                metrics     = [f'ndcg@{i+1}' for i in range(num_cands)] + ['recall@1']  
                # NOTE: be sure to use include_q_i_ids (not cut_q_i_idxes)
                # in do_one_run_ranx and top_ranked_accuracy. 
                # include_q_i_ids only keeps rank instances that are not having all the same human scores.
                report_obj  = do_one_run_ranx(cut_qrels_dict, metrics, cut_run_preds, include_q_i_ids, 
                                              round = ROUND)
                
                ranx_results[qud_criteria][f'num_cands_{num_cands}'] = \
                    report_obj.to_dataframe().to_dict()
                print(f'\nRANX REPORT (NUM CANDS {num_cands}):', str(report_obj))

                outputs_t_r_a = top_ranked_accuracy(cut_qrels_dict, cut_run_preds, include_q_i_ids)
                print(f'\n MATCH GOLD REPORT (NUM CANDS {num_cands}):', 
                      {k: np.mean(v) for k,v in outputs_t_r_a.items()})
                t_r_a_results[qud_criteria][f'num_cands_{num_cands}'] = outputs_t_r_a

        holder_ranx_results[combi_name]     = convert_defaultdict(ranx_results)
        holder_t_r_a_results[combi_name]    = convert_defaultdict(t_r_a_results)

    return holder_eval_rules_based, holder_eval_qudeval_gpt, holder_eval_qudeval_gpt_few_shot, \
            holder_eval_qudselect_classifiers, holder_eval_stv, holder_eval_llm_qa, \
                holder_ranx_results, holder_t_r_a_results


if __name__ == "__main__":
    ## 1. collect panel-of-LLMs rankings and run evaluation
    main_phase1()
    
