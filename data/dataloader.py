import sys, random, json, glob, re, os
QUD_FAIL = 'QUD GENERATION FAILURE (NOT EXTRACTABLE FROM OUTPUT).'
SEED = 54506

def load_phase1(cfg):
    sys.path.append(cfg.dirpath)
    # sys.path.append('tools')
    # from rank_llm.src.rank_llm.data import Query, Candidate, Request
    import pandas as pd
    from collections import Counter, defaultdict
    from data.data_utils import ArticleContents, QUDInstance, QUDCandidate
    # NOTE: qudeval scores are 1: best, 3: worst (i.e. lower better)
    # for NDCG (higher better), we reverse this (i.e. 1->3, 2->2, 3->1)
    MAX_SCORE = 3+1
    
    # 1. create collection of articles 
    art_fps = sorted(glob.glob(f'{cfg.dirpath}/data/qudeval/article2/*.txt'))
    holder_articles = {}
    for fp in art_fps:
        
        with open(fp, encoding='utf-8') as f:
            contents = f.readlines()
        contents = [re.search(r'(\d+)(?:\s+)(.+)', l).groups() for l in contents]
        contents = {int(k): v.strip() for (k,v) in contents}
        
        article_id  = int(os.path.basename(fp).replace('.txt', ''))
        art_obj     = ArticleContents(article_id, contents)
        holder_articles[article_id] = art_obj

    # a. add Ted-Q (QSALIENCE) article
    #  we will keep the Ted-Q article id 1927 (the DCQA articles go up to 1500 only)
    with open(f'{cfg.dirpath}/data/qsalience/data/tedq_1927.txt', encoding = 'utf-8') as f:
        contents = f.readlines()
    contents = [re.search(r'(\d+)(?:\s+)(.+)', l).groups() for l in contents]
    contents = {int(k): v.strip() for (k,v) in contents}
    # NOTE: the TEDQ data has a speaker name and title at the start of each discourse. 
    # the DCQA articles do not have this and all the models (qud_gen and ranking reward model) would not have seen data 
    # of this form. keeping it means that it will be shown in the context (if the anchor sentence id is > 1)
    # one can argue that we do not expect a QUD to be triggered by the speaker name and title.
    title_str = 'Chris McKnett: The investment logic for sustainability'
    assert title_str in contents[1], contents[1]
    contents[1] = contents[1].replace(title_str, '').strip()
    article_id  = 1927
    art_obj     = ArticleContents(article_id, contents)
    holder_articles[article_id] = art_obj

    # b. add TED-Q articles
    # NOTE: directly call in utils_data_grpo.py
    # holder_articles_tedq, df_tedq_cut = load_tedq_data(cfg.dirpath)

    # 2. load qudeval dataset 
    holder_quds = {}
    df = pd.read_csv(f'{cfg.dirpath}/data/qudeval/data-collection.csv')
    
    # a. exclude human-written QUDs (we want to use DCQA for test later)
    # also remove instances annotated 'skipped' 
    df['questions'] = df['questions'].fillna('') #, inplace = True)
    c0 = df.system    != 'human'
    c2 = df.criteria2 != 'skipped'
    c3 = df.criteria3 != 'skipped'
    c4 = df.criteria4 != 'skipped'
    df = df[c0 & c2 & c3 & c4].copy()
    
    # b. to get unique (anc, ans)
    df['eid_anc_ans'] = df.apply(lambda x: str(x['essay_id']) \
                                   + '_' + str(x['anchor_id']) \
                                   + '_' + str(x['answer_id']), axis = 1)
    
    # c. keep only (anc, ans) where there are at least 2 system generations
    criteria_keys   = ['criteria1', 'criteria2', 'criteria3', 'criteria4']
    ctr             = Counter(df.eid_anc_ans)
    to_keep         = {k:v for k,v in ctr.items() if v >= 2}
    df_keep         = df[df['eid_anc_ans'].isin(to_keep)].copy()

    # d. (optional) slot new systems (not in QUDEval) as new candidates. 
    # for eval using main_phase1.py
    c_insert_new_gens   = cfg.get('insert_new_gens', [])
    ing_version         = cfg.get('insert_new_gens_version', 'v1')
    holder_new_quds = {}
    if c_insert_new_gens: 
        sys.path.append('model')
        from model.grpo_reward_funcs import ANS_START, ANS_END

        fp_eval_q_i_ids = f'{cfg.dirpath}/data/eval_qud_instance_ids.txt'
        with open(fp_eval_q_i_ids, encoding = 'utf-8') as f:
            eval_q_i_ids = set(i.strip() for i in f.readlines())
        
        c_qsal = set([k.endswith('qs') for k in [kk.replace('qsSA', 'qs') for kk in c_insert_new_gens]])
        assert len(c_qsal) == 1, c_insert_new_gens
        c_qsal = True in c_qsal
        if c_qsal:
            fp_eval_q_i_ids = f'{cfg.dirpath}/data/eval_qud_instance_ids_qsal.txt'
            with open(fp_eval_q_i_ids, encoding = 'utf-8') as f:
                eval_q_i_ids = set(i.strip() for i in f.readlines())
            # create new "dummy" df_keep to iterate over below
            new_df_lines = []
            for qud_instance_id in eval_q_i_ids:
                for sname in c_insert_new_gens:
                    newline = {'eid_anc_ans': qud_instance_id, 
                                'system': sname, 'questions': None}
                    for crit in criteria_keys: newline[crit] = -99
                    new_df_lines.append(newline)
            df_keep = pd.DataFrame(new_df_lines)
            # load tedq data
            holder_articles_tedq, df_tedq_cut = load_tedq_data(cfg.dirpath)
            for article_id, art_obj in holder_articles_tedq.items():
                holder_articles[article_id] = art_obj

    for new_sys_name in c_insert_new_gens:
        print('NEW SYS NAME:', new_sys_name)
        if new_sys_name.endswith('qsSA'):
            fp = cfg.insert_new_gens_mapping[ing_version].get(new_sys_name.replace('qsSA', 'qs'), 'NONE')
            fp = fp.replace('_qsal.json', '_SA_qsal.json')
        else: 
            fp = cfg.insert_new_gens_mapping[ing_version].get(new_sys_name, 'NONE')
        assert os.path.exists(fp), fp
        holder_new_quds[new_sys_name] = {}

        with open(fp, encoding = 'utf-8') as f: 
            __hh = json.load(f)
        c_qudselect = new_sys_name.startswith('QQS') or new_sys_name.startswith('LQS')
        if c_qudselect:
            __hh = {x['qud_instance_id']: x['question'] for x in __hh \
                    if x['qud_instance_id'] in eval_q_i_ids}
        
        for qud_instance_id, gen in __hh.items():
            if qud_instance_id not in eval_q_i_ids: continue
            ### ATTEMPTS TO EXTRACT WELL-FORMED QUD ###
            # A. for QUDSelect models
            qud = gen['generated'].strip() if ing_version != 'v1' and not c_qudselect else gen.strip()
            qud = extract_qud(qud, c_qudselect, ANS_START, ANS_END, QUD_FAIL)
            holder_new_quds[new_sys_name][qud_instance_id] = qud

    if len(holder_new_quds) not in [0, 1, 4]: raise NotImplementedError

    # e. populating QUDInstances
    for qud_instance_id in df_keep.eid_anc_ans.unique():
        if cfg.exp_code == 'test' and len(holder_quds) >= 10: continue

        qud_instance_grp = df_keep[df_keep.eid_anc_ans == qud_instance_id]
        
        article_id, anchor_id, answer_id = qud_instance_id.split('_')
        article_id           = int(article_id) if type(article_id) == str and article_id.isdigit() else article_id
        anchor_id, answer_id = int(anchor_id), int(answer_id)

        qud_instance_candidates = []
        for i, cand_row in qud_instance_grp.iterrows():
            qud_cand = QUDCandidate(sysname      = cand_row['system'], 
                                    qud          = cand_row['questions'], 
                                    # NOTE: qudeval scores are between 1 and 3
                                    # with 3 the worst, and 1 the best
                                    # to avoid confusion, we reverse this
                                    criteria_scores  = {k: MAX_SCORE-float(cand_row[k]) for k in criteria_keys})
            qud_instance_candidates.append(qud_cand)

        ##########################################################################################
        if c_insert_new_gens:
            if not all(qud_instance_id in v for v in holder_new_quds.values()): 
                print('🟣🟣\tINSIDE insert_new_gens. skipping qud_instance_id not in eval_q_i_ids', qud_instance_id)
                continue
            max_cands = 4
            num_new   = len(holder_new_quds)
            assert num_new <= max_cands, (max_cands, num_new)
            # NOTE: we validate rankllm up to 4cands only. so we maintain max as 4cands
            # drop existing if exceeds max_cands
            excess = len(qud_instance_candidates) + num_new - max_cands
            if excess > 0:
                k = abs(len(qud_instance_candidates) - excess)
                if k == 0:
                    qud_instance_candidates = []
                else:
                    random.seed(SEED+i)
                    qud_instance_candidates = random.sample(qud_instance_candidates, k = k)
            
            for new_sys_name in holder_new_quds:
                qud_cand = QUDCandidate(sysname      = new_sys_name, 
                                        qud          = holder_new_quds[new_sys_name][qud_instance_id], 
                                        criteria_scores  = {k: None for k in criteria_keys})
                qud_instance_candidates.append(qud_cand)
        ##########################################################################################

        qud_instance = QUDInstance(article_id           = article_id, 
                                    anchor_id           = anchor_id,
                                    answer_id           = answer_id,
                                    qud_instance_id     = qud_instance_id,
                                    qud_human           = None,
                                    qud_candidates      = qud_instance_candidates,
                                    do_tedq             = 'talk' in qud_instance_id)
        
        holder_quds[qud_instance_id] = qud_instance
    
    num_cands2qud_idxes = defaultdict(set)
    for qud_instance_id, qud_instance in holder_quds.items():
        num_cands2qud_idxes[len(qud_instance.qud_candidates)].add(qud_instance_id)
    print('🟦🟦\tNUM requests per num_cand (GLOBAL)', {k: len(v) for k,v in num_cands2qud_idxes.items()})                          
    return holder_articles, holder_quds, num_cands2qud_idxes

def text_ctx_anc_ans(holder_articles, qud_instance, context_empty_symbol = '[EMPTY]'):
    article_id = qud_instance.article_id
    article_contents = holder_articles[article_id]
    
    ctx = qud_instance.extract_context(article_contents)
    # if Anchor is 1st sentence (i.e. nothing in context, use context_empty symbol)
    if ctx: ctx = ' '.join(ctx)
    else: ctx = context_empty_symbol
    
    anc = qud_instance.extract_anchor(article_contents)
    ans = qud_instance.extract_answer(article_contents)
    
    prompt_text = '## Context: {0} \n## Anchor Sentence: {1} \n## Answer Sentence: {2} '
    prompt_text = prompt_text.format(ctx, anc, ans)
    
    return prompt_text, ctx, anc, ans

def make_phase1_data_requests(cfg, holder_articles, holder_quds, 
                              rankllm_obj_pack, qud_criteria = 'criteria2'):
    ranker_requests = {}
    for num, (qud_instance_id, qud_instance) in enumerate(holder_quds.items()):
        seed = SEED + num

        ranker_requests[qud_instance_id] = \
            make_one_rankllm_request(seed, holder_articles, qud_instance, 
                                     qud_instance_id, qud_criteria, 
                                     cfg.exclude_same_scores, 
                                    cfg.prompts.rankllm.prefix.context_empty,
                                    rankllm_obj_pack)

    return ranker_requests

def make_one_rankllm_request(seed, holder_articles, qud_instance, qud_instance_id, 
                     qud_criteria, exclude_same_scores, context_empty_symbol, rankllm_obj_pack):
    
    (Query, Candidate, Request) = rankllm_obj_pack
    
    text, ctx, anc, ans = \
    text_ctx_anc_ans(holder_articles, qud_instance, 
                        context_empty_symbol = context_empty_symbol)

    r = Request(Query(text = text, qid = qud_instance_id,
                    # NOTE: below not used
                    context = ctx, 
                    anchor = anc,
                    answer = ans))

    for qud_cand in qud_instance.qud_candidates:
        score       = qud_cand.give_criteria_score(qud_criteria)
        rationale   = qud_cand.give_criteria_rationale(qud_criteria)
        candidate   = Candidate(docid = qud_cand.sysname, 
                                score = score, 
                                doc = {'text': qud_cand.qud},
                                rationale = rationale,)
        r.candidates.append(candidate)

    # randomize order (NOTE: we set shuffle_candidates = False in config yaml)
    # NOTE: keep same perm order regardless of which rankllm model used
    random.seed(seed)
    random.shuffle(r.candidates)

    # collect perm_order information
    gold_scores          = [c.score for c in r.candidates]

    # NOTE: exclude ranking instances where all candidates have the same human score
    c1 = exclude_same_scores
    c2 = len(set(gold_scores)) == 1
    c3 = None not in gold_scores # these are cases with insert_new_gens
    if c1 and c2 and c3: return None
    if all([isinstance(s, float) or isinstance(s, int) for s in gold_scores]):
        sorted_scores   = sorted(gold_scores, reverse = True)
        sort_order      = {}
        for opos, sc in enumerate(gold_scores):
            npos = sorted_scores.index(sc)
            sort_order[npos]    = opos
            sorted_scores[npos] = None 
        r.perm_order = [sort_order[npos] for npos in range(len(sorted_scores))]
    else: r.perm_order = None

    return r 

def make_exemplar_objects(cfg): 
    sys.path.append(cfg.dirpath)
    from data.data_utils import QUDInstance, QUDCandidate
    MAX_SCORE = 4

    fp = f'{cfg.dirpath}/data/exemplars/qudeval_mod_exemplars.json'
    with open(fp, encoding='utf-8') as f:
        exemplars = json.load(f)

    holder_exemplars    = {}
    criteria_keys       = list(exemplars)
    exps                = exemplars[criteria_keys[0]]
    rationale_key       = 'rationale_fine' if cfg.ranker_args.cot_fine else 'rationale'
    for qud_criteria, exps in exemplars.items():
        holder_exemplars[qud_criteria] = {}

        for qud_instance_id, cand_dict in exps.items():
            article_id, anchor_id, answer_id = qud_instance_id.split('_')
            article_id           = int(article_id) if type(article_id) == str and article_id.isdigit() else article_id
            anchor_id, answer_id = int(anchor_id), int(answer_id)

            qud_instance_candidates = []
            for sysname, cand_row in cand_dict.items():
                criteria_scores  = {}
                rationales       = {}
                for k in criteria_keys:
                    criteria_scores[k]  = MAX_SCORE-float(exemplars[k][qud_instance_id][sysname]['new_score'])
                    rationales[k]       = exemplars[k][qud_instance_id][sysname][rationale_key]

                qud_cand = QUDCandidate(sysname      = sysname, 
                                        qud          = cand_row['altered_qud'], 
                                        # NOTE: qudeval scores are between 1 and 3
                                        # with 3 the worst, and 1 the best
                                        # to avoid confusion, we reverse this
                                        rationales      = rationales,
                                        criteria_scores = criteria_scores)
                qud_instance_candidates.append(qud_cand)        

            exp_instance = QUDInstance(article_id           = article_id, 
                                        anchor_id           = anchor_id,
                                        answer_id           = answer_id,
                                        qud_instance_id     = qud_instance_id,
                                        qud_human           = None,
                                        qud_candidates      = qud_instance_candidates,
                                        do_tedq             = 'talk' in qud_instance_id)
            
            holder_exemplars[qud_criteria][qud_instance_id] = exp_instance
    
    return holder_exemplars

def load_phase2(cfg):
    return load_phase1(cfg)


def load_articles_grpo(cfg):
    holder_articles, holder_quds, num_cands2qud_idxes = load_phase1(cfg)
    return holder_articles

     
def load_dcqa():
    dp      = 'data/dcqa'
    fp_map  = {'test':           'test.json',
               'validation':     'val.json',
               'train':          'train.json'}
    holder_dcqa = {}
    for split, fp in fp_map.items():
        holder_dcqa[split] = {}
        with open(f'{dp}/{fp}', encoding = 'utf-8') as f:
            data = json.load(f)

        for art_list in data:
            assert len(art_list['Article']) == 1
            art_dict = art_list['Article'][0]
            article_id = art_dict['ArticleID']
            for q_info in art_dict['qas']:
                anchor_id = q_info['AnchorSentenceID']
                answer_id = q_info['AnswerSentenceID']
                qud = q_info['Question']
                qud_instance_id = f'{article_id}_{anchor_id}_{answer_id}'

                holder_dcqa[split][qud_instance_id] = qud  

    return holder_dcqa

def load_tedq_data(dirpath, cutoff = 20):
    import pandas as pd
    import glob, spacy
    from data_utils import ArticleContents 
    # nlp = spacy.load('en_core_web_lg')
    fps = glob.glob(f'{dirpath}/data/ted_mdb/English/raw/01/*.txt')
    
    holder_articles_tedq = {}
    bad_char = '\ufeff ' # talk_2009_en.txt has a weird char (encoding artifact) at the start
    holder_art_idx_tedq = {}
    df_tqelicit = pd.read_csv(f'{dirpath}/data/tedq/TED-Q_elicitation.csv')
    df_tqelicit.chunk_start = df_tqelicit.chunk_start.astype(int)
    df_tqelicit.chunk_end   = df_tqelicit.chunk_end.astype(int)
    gb = df_tqelicit.groupby('source')
    for fp in fps:
        article_id = os.path.basename(fp)
        source_key = f'Ted-MDB-Annotations/English/raw/01/{article_id}'
        if source_key not in gb.groups: continue # talk_2150_en_intra.txt is not in ted-q
        group           = gb.get_group(source_key)
        s_starts, s_ends = sorted(set(group.chunk_start.tolist())), sorted(set(group.chunk_end.tolist()))
        # verify that every subsequent start after 1st can be found in the ends (i.e. unlikely to have missing boundaries)
        assert set(s_starts[1:]).difference(s_ends[:]) == set(), (s_starts, s_ends)
        sent_boundaries = sorted(set(group.chunk_start.tolist() + group.chunk_end.tolist()))
        article_id = article_id.replace('.txt', '').replace('_', '-')
        with open(fp, encoding = 'utf-8') as f: text = f.read()

        # remove speaker and title (replace with whitespace to maintain span indexing)
        talk_id_str = re.search(r'(\ntalkid:\s*\d+\n\n.+:.+\n\n)', text).group()
        text        = text.replace(talk_id_str, ' '*len(talk_id_str))
        # talk_2009_en.txt has a weird char (encoding artifact) at the start
        if text.startswith(bad_char): text = text.replace(bad_char, ' '*len(bad_char)) 
        
        sents   = []
        indices = []
        # for s in nlp(text).sents:
        #     sents.append(s.text.strip())
        #     indices.append((s.start_char, s.end_char))
        # NOTE: use the sentence boundaries in the TED-Q dataset
        # NOTE: these articles have first chunks that fall on the article title or part of it (e.g. 'ustainability' for 1927)
        c_skip_first = '1927' in article_id or '1978' in article_id or '2009' in article_id or '2150' in article_id
        for si in range(len(sent_boundaries)-1):
            if c_skip_first and si == 0: continue
            start_char = sent_boundaries[si]
            end_char = sent_boundaries[si+1]
            sentence = text[start_char:end_char]
            sents.append(sentence.strip())
            indices.append((start_char, end_char))
        contents                        = {i+1: s for i,s in enumerate(sents)}
        art_obj                         = ArticleContents(article_id, contents, fulltext = text)
        holder_articles_tedq[article_id]= art_obj
        holder_art_idx_tedq[article_id] = {s: i+1 for i,s in enumerate(indices)}

    # b ii) keep only the questions, and ending with ? and not of "unknown" type
    df_tedq_cut = df_tqelicit[(df_tqelicit.type == 'question')  & (df_tqelicit.answered>=5.0) & \
                              (df_tqelicit.wh_type !='unknown') & (df_tqelicit.content.str.endswith('?'))].copy()
    
    for i, row in df_tedq_cut.iterrows():
        article_id = os.path.basename(row['source'])
        article_id = article_id.replace('.txt', '').replace('_', '-')

        # c i) get anchor
        anchor_id = None
        anc_start, anc_end              = int(row['highlight_start']), int(row['highlight_end'])
        anc_chunk_start, anc_chunk_end  = int(row['chunk_start']), int(row['chunk_end'])
        for (ss, se), sid in holder_art_idx_tedq[article_id].items():
            # if anc_start >= ss and anc_end <= se and holder_articles_tedq[article_id].contents.get(sid, ''):
            #     anchor_id = sid
            #     break
            if anc_chunk_start == ss: 
                assert anc_start >= anc_chunk_start, (ss, se, anc_start, anc_end, anc_chunk_start, anc_chunk_end)
                assert anc_start >= ss and anc_end >= ss, (ss, se, anc_start, anc_end, anc_chunk_start, anc_chunk_end)
                anchor_id = sid
                break
                
        # c ii) get answer
        ans_row = df_tqelicit[df_tqelicit.id == row['best_answer']]
        assert len(ans_row) == 1    
        ans_row = ans_row.reset_index().loc[0].to_dict()
        ans_start, ans_end = int(ans_row['highlight_start']), int(ans_row['highlight_end'])

        answer_id = None
        for (ss, se), sid in holder_art_idx_tedq[article_id].items():
            if ans_start >= ss and ans_end <= se and holder_articles_tedq[article_id].contents.get(sid, ''):
                answer_id = sid
                break
        
        if anchor_id is None or answer_id is None: continue
        if anchor_id > cutoff or answer_id > cutoff: continue # follow DCQA distribution and use first 20 sentences only
        if anchor_id == answer_id: continue # should not have same anchor and answer
        qud_instance_id = f"{article_id}_{anchor_id}_{answer_id}"
        df_tedq_cut.loc[i, 'qud_instance_id'] = qud_instance_id

    df_tedq_cut = df_tedq_cut[df_tedq_cut['qud_instance_id'].notna()]
    df_tedq_cut = df_tedq_cut.drop_duplicates('qud_instance_id').copy()

    return holder_articles_tedq, df_tedq_cut

def extract_qud(qud, c_qudselect, ans_start, ans_end, qud_fail):
    '''
    
    '''
    if c_qudselect: 
        # NOTE: many QUDSelect model outputs contain continuations of generated context etc
        # an indication of this is '###' in the output.
        if '###' in qud:
            # 1. if the generation contains a segment "Question:", use that to get the QUD
            try:            qud = re.search(r'(?:Question:)(.+\?)',     qud, re.DOTALL).group(1).strip()    
            except: 
                # 2. if that fails and the generation contains "### Instruction", get the sequence before it
                try:        qud = re.search(r'(.+)(?:### Instruction)', qud, re.DOTALL).group(1).strip()
                except:     qud = qud_fail
            # 3. finally, if the extraction does not contain a "?", this is likely context-only 
            if '?' not in qud:  qud = qud_fail
    # B. for our models and related (i.e. zeroshot, RB2K and GRPO models)
    else:
        
        # 1. try to extract assuming strict format adherence
        # NOTE: ensure greedy with ?, in case of multiple <answer> </answer> pairs
        try:        qud = re.search(rf'{ans_start}(.+?){ans_end}', qud, re.DOTALL).group(1).strip()
        except: 
            # 2. try to extract with sort format adherence
            try:    qud = re.search('</think>\s*(.+)', qud, re.DOTALL).group(1).strip()
            except: qud = qud.strip()
            # 3. one more check to ensure not degenerate text (e.g. for zeroshot or RB2K)
            # NOTE: some variants give poorly formatted outputs. these are usually longer than 200 chars
            if len(qud) > 200:
                try:
                    # get up to the last "?"
                    qud = re.search(r'(.+\?).*?', qud, re.DOTALL).group(1)
                    # get capitalised word up to "?" which is now at sequence end (without DOTALL)
                    qud = re.search(r'([A-Z].+\?)$', qud).group(1)
                except: qud = qud_fail
                if '?' not in qud:  qud = qud_fail
    return qud

if __name__ == "__main__": 
    pass