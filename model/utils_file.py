import re, json, os, sys, copy

def give_savepath(cfg, phase = 1):
    
    if phase == 1: 
        c_do_cot = cfg.ranker_args.num_few_shot_examples > 0 and cfg.ranker_args.do_cot
        savepath = f'{cfg.dirpath}/results/main_phase1/RANK'
        if not cfg.ranker_args.do_cot:
            cfg.ranker_args.add_task_decomp_cot = False
        if cfg.ranker_args.add_task_decomp_cot:
            savepath += '-TASK_DECOMP'
            cfg.ranker_args.add_task_decomp_common = False        
        if cfg.ranker_args.add_task_decomp_common:
            savepath += '-COMMON_TASK_DECOMP'
        if c_do_cot: 
            savepath += f'_COT-{cfg.ranker_args.num_few_shot_examples}'
            if cfg.ranker_args.cot_fine: 
                savepath += '-FINE'
            if cfg.ranker_args.cot_json: 
                savepath += '-JSON'
        elif not c_do_cot and  cfg.ranker_args.num_few_shot_examples > 0:
            savepath += f'_ICL-{cfg.ranker_args.num_few_shot_examples}'

        c_insert_new_gens = cfg.get('insert_new_gens', [])
        if len(c_insert_new_gens) > 0:
            savepath += '_ING'
            for new_sys_name in c_insert_new_gens:
                ing_map = cfg.insert_new_gens_mapping[cfg.insert_new_gens_version]
                assert new_sys_name.replace('qsSA', 'qs') in ing_map, new_sys_name
                savepath += f'-{new_sys_name}'
        
        sz_str = cfg.model_size
        mn_str = '-'.join(sorted(cfg.models[sz_str]))
        if cfg.ranker_args.phase1_use_closed_models:
            mn_str = '-'.join(sorted(cfg.models['closed_models']))
        savepath += f"_MODELS-{sz_str}-{mn_str}"
        
        mn_str = '-'.join(sorted(cfg.answer_compat.model_names))
        sz_str = '-'.join(sorted(cfg.answer_compat.model_sizes))
        savepath += f"_LLMQA-{sz_str}-{mn_str}"
        
        if cfg.exp_code is not None: 
            savepath += f'_{cfg.exp_code}'
        if cfg.ranker_args.use_past_key_values: 
            savepath += '_PKV'

    elif phase == 2:
        
        savepath = f'{cfg.dirpath}/results/main_phase2/'
        for ti, task_setting in enumerate(['qud_gen', 'ranker_args']):
            if   ti > 0:                        savepath += '-'
            if   task_setting == 'qud_gen':     savepath += 'QUD_GEN'
            elif task_setting == 'ranker_args': savepath += 'RANK'
            else: raise NotImplementedError

            if not getattr(cfg, task_setting).do_cot:
                getattr(cfg, task_setting).add_task_decomp_cot = False     
            if getattr(cfg, task_setting).add_task_decomp_cot:
                savepath += '-TASK_DECOMP'
                getattr(cfg, task_setting).add_task_decomp_common = False        
            if getattr(cfg, task_setting).add_task_decomp_common:
                savepath += '-COMMON_TASK_DECOMP'
            c_do_cot = getattr(cfg, task_setting).num_few_shot_examples > 0 and getattr(cfg, task_setting).do_cot
            if c_do_cot: 
                savepath += f'_COT-{getattr(cfg, task_setting).num_few_shot_examples}'
                if getattr(cfg, task_setting).cot_fine: 
                    savepath += '-FINE'
                if getattr(cfg, task_setting).cot_json: 
                    savepath += '-JSON'
            elif not c_do_cot and  getattr(cfg, task_setting).num_few_shot_examples > 0:
                savepath += f'_ICL-{getattr(cfg, task_setting).num_few_shot_examples}'
            sz_str = cfg.model_size
            if task_setting == 'qud_gen':
                mns = getattr(cfg, task_setting).phase2_model_names
            else:
                mns = getattr(cfg, task_setting).phase2_model_names.values()
                mns = sorted(set([m2 for m1 in mns for m2 in list(m1)]))
                cfg.ranker_args.phase_model_names_set = mns
            mn_str = '-'.join(mns)
            savepath += f"_MODELS-{sz_str}-{mn_str}"
        
        if cfg.exp_code is not None: 
            savepath += f'_{cfg.exp_code}' 
    
    else: raise NotADirectoryError

    return savepath


def give_panel_outputs_fp(cfg, rankllm_sys):
    sp = cfg.savepath.split('_MODELS')[0] + f'_{cfg.exp_code}'
    sp = os.path.join(os.path.dirname(sp), 'panel_outputs', os.path.basename(sp))
    panel_outputs_fp = f'{sp}/panel_outputs_{rankllm_sys}_{cfg.model_size}.json'
    
    return panel_outputs_fp


def save_panel_outputs(cfg, panel_outputs_fp, holder_panel_outputs, rankllm_sys):
    holder = {}
    for qud_criteria in cfg.qud_criteria_list:
        holder[qud_criteria] = {}
        for q_i_id in holder_panel_outputs[qud_criteria]:
            holder[qud_criteria][q_i_id] = \
                holder_panel_outputs[qud_criteria][q_i_id][rankllm_sys]
    
    dp = os.path.dirname(panel_outputs_fp)
    if not os.path.exists(dp): os.makedirs(dp)
    
    with open(panel_outputs_fp, encoding = 'utf-8', mode = 'w+') as f:
        json.dump(holder, f) 
    print(f'🟧🟧\tPANEL OUTPUTS FOR {rankllm_sys} SAVED TO : {panel_outputs_fp}')
    # if cfg.do_model: raise SystemExit


def give_savepath_grpo(cfg):
    savepath = os.path.join(cfg.dirpath, 'results/main_grpo')

    # 1. grpo settings
    if cfg.do_grpo:
        if cfg.reward_funcs_version == 1: grpo_settings = '/grpo_settings'
        else:                             grpo_settings = '/grpo'
        for k, v in cfg.grpo_settings.items():
            if cfg.reward_funcs_version == 1 and k in ['epsilon_high']: continue
            if k in ['grpo_task'] and v == 'qud_gen': continue
            k = ''.join([kk[0] for kk in k.split('_')])
            if   str(v) == 'True':  v = 'T'
            elif str(v) == 'False': v = 'F'
            elif str(v) == 'None':  v = 'N'
            grpo_settings += f'-{k}-{v}' 
        
        savepath += grpo_settings
    else: 
        savepath += f'/SFT_{cfg.grpo_settings.grpo_task}'
        

    # 2. models, sft, reward, criteria
    model_settings = ''
    if cfg.reward_funcs_version > 1:
        sft = cfg.model.sft
        model_settings += f'-{sft.model}-{sft.size}'
    if cfg.grpo_settings.initial_rules_based_steps == 1e9:
        model_settings += '-full-rulesbased'
    else:
        reward = cfg.model.reward
        for q_c in reward:
            if reward[q_c] in [True, False, None]: continue
            q_q = q_c[0] + q_c[-1]
            if reward[q_c].model in ['gpt4o', 'deepseek', 'o3mini']:
                mname = ''.join([m[:2] for m in reward[q_c].model.split('_')])
                model_settings += f'-{q_q.upper()}-{mname}'
            else:
                mname = ''.join([m[:2] for m in reward[q_c].model.split('_')])
                sname = ''.join([m[:2] for m in reward[q_c].size.split('_')])
                model_settings += f'-{q_q.upper()}-{mname}-{sname}'

    savepath += model_settings
    if cfg.reward_funcs_version > 1: savepath += f'_RFv{cfg.reward_funcs_version}'
    
    # 3. ranker_args
    for ti, task_setting in enumerate(['qud_gen', 'ranker_args']):
        if   ti > 0:                        savepath += '-'
        if   task_setting == 'qud_gen':     savepath += 'QUD_GEN'
        elif task_setting == 'ranker_args': savepath += 'RANK'
        else: raise NotImplementedError

        if not getattr(cfg, task_setting).do_cot:
            getattr(cfg, task_setting).add_task_decomp_cot = False     
        if getattr(cfg, task_setting).add_task_decomp_cot:
            savepath += '-TDECOMP'
            getattr(cfg, task_setting).add_task_decomp_common = False        
        if getattr(cfg, task_setting).add_task_decomp_common:
            savepath += '-COMTDECOMP'
        c_do_cot = getattr(cfg, task_setting).num_few_shot_examples > 0 and getattr(cfg, task_setting).do_cot
        if c_do_cot: 
            savepath += f'_COT-{getattr(cfg, task_setting).num_few_shot_examples}'
            if getattr(cfg, task_setting).cot_fine: 
                savepath += '-FINE'
            if getattr(cfg, task_setting).cot_json: 
                savepath += '-JSON'
        elif not c_do_cot and  getattr(cfg, task_setting).num_few_shot_examples > 0:
            savepath += f'_ICL-{getattr(cfg, task_setting).num_few_shot_examples}'
    
    if cfg.load_peft_ckpt_path is not None and not cfg.do_eval_only_bypass: 
        task_str = '' if cfg.grpo_settings.grpo_task == 'qud_gen' else f'_{cfg.grpo_settings.grpo_task}'
        fp = os.path.join(os.path.dirname(savepath), f'checkpoints_mapping{task_str}.json')
        
        if not os.path.exists(fp):
            mapping = {}
            code = len(mapping)
        else:
            with open(fp) as f: mapping = json.load(f)
            code = mapping.get(cfg.load_peft_ckpt_path, len(mapping))

        if code not in mapping.values():
            mapping.update({cfg.load_peft_ckpt_path: code})
        
        with open(fp, mode = 'w+') as f: json.dump(mapping, f)

        task_str = '' if cfg.grpo_settings.grpo_task == 'qud_gen' else f'{cfg.grpo_settings.grpo_task[0].upper()}'
        savepath += f'_LPC{task_str}-{code}'

    if cfg.exp_code is not None: 
        savepath += f'_{cfg.exp_code}' 

    if cfg.shorten_filename:
        savepath = savepath.replace('QUD_GEN', 'QG')
        savepath = savepath.replace('RANK', 'RK')
        savepath = savepath.replace('COMTDECOMP_COT', 'CD_COT')
        savepath = savepath.replace('FINE', 'FI')
        savepath = savepath.replace('JSON', 'JS')

    if cfg.do_eval_only_bypass:
        savepath = os.path.dirname(cfg.load_peft_ckpt_path)
        savepath = os.path.dirname(savepath)
        savepath = os.path.dirname(savepath)            
        c1 = not re.search(r'RANK\d+', savepath)
        c2 = 'grpo_settings_' in savepath or 'main_grpo/grpo' in savepath
        c3 = 'SFT_' in savepath
        c4 = 'grpo-gt-rankllm-' in savepath
        assert c1 and (c2 or c3 or c4), savepath
    return savepath



def load_panel_outputs(cfg, panel_outputs_fp, holder_panel_outputs, rankllm_sys):
    if not os.path.exists(panel_outputs_fp): 
        return False, holder_panel_outputs

    with open(panel_outputs_fp, encoding = 'utf-8') as f:
        holder = json.load(f) 
    
    for qud_criteria in cfg.qud_criteria_list:
        for q_i_id in holder[qud_criteria]:
            holder_panel_outputs[qud_criteria][q_i_id][rankllm_sys] = \
                holder[qud_criteria][q_i_id]
    
    print(f'🟧🟧\tPANEL OUTPUTS FOR {rankllm_sys} LOADED FROM : {panel_outputs_fp}')

    return True, holder_panel_outputs


def load_qud_gen_fp_outputs(qud_gen_dp, hnames  = ['holder_qud_candidates', 
                                                   'ranker_requests', 
                                                   'exemplar_requests']):
    sys.path.append('/home/khan/agentic_qud')
    from tools.rank_llm.src.rank_llm.data import Query, Candidate, Request
    
    checks  = []
    holders = {}
    for hn in hnames:
        fp = f'{qud_gen_dp}/{hn}.json'
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                holders[hn] = json.load(f)

            # convert dict to rankllm Request object
            if hn in ['ranker_requests', 'exemplar_requests']:
                for qud_criteria in holders[hn]:
                    for q_i_id, req_dict in holders[hn][qud_criteria].items():
                        r = convert_dict_request(Query, Candidate, Request, req_dict)
                        holders[hn][qud_criteria][q_i_id] = r
            print(f'🟧🟧\QUD GEN OUTPUTS {hn} FOR LOADED FROM : {fp}')
        else: 
            holders[hn] = None
            checks.append(False)
    
    return all(checks), holders


def save_qud_gen_fp_outputs(qud_gen_dp, holders):

    if not os.path.exists(qud_gen_dp): os.makedirs(qud_gen_dp)

    for hn, obj in holders.items():
        c_obj = copy.deepcopy(obj)
        fp = f'{qud_gen_dp}/{hn}.json'

        if hn in ['ranker_requests', 'exemplar_requests']:
            for q_c, req_dict in c_obj.items():
                for req_id in req_dict: 
                    c_obj[q_c][req_id] = c_obj[q_c][req_id].asdict()

        with open(fp, encoding='utf-8', mode = 'w+') as f:
            json.dump(c_obj, f)

        print(f'🟧🟧\QUD OUTPUTS ({hn}) saved to: {fp}')


def convert_dict_request(Query, Candidate, Request, req_dict):
    q_info = req_dict['query']
    c_info = req_dict['candidates']

    r = Request(Query(**q_info))

    for c in c_info:
        candidate   = Candidate(**c)
        r.candidates.append(candidate)

    return r