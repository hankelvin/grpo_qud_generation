from ranx import Qrels, Run, compare as ranx_compare

def top_ranked_accuracy(qrels_dict, run_preds, include_q_i_ids = set()):

    filt_qrels_dict = {q_i_id: sc for q_i_id, sc in qrels_dict.items() \
                                if q_i_id in include_q_i_ids}
    
    sc_key          = 'ord' # key for the score to use. we use the ord scores for top ranked
    runs            = {name: {q_i_id: line[sc_key] for q_i_id, line in run_p.items() \
                                if  q_i_id in include_q_i_ids} \
                                    for name, run_p in run_preds.items() \
                                        if run_p is not None}
            
    outputs_t_r_a = {}
    for name, run_dict in runs.items():
        outputs_t_r_a[name] = []
        for q_i_id, line in run_dict.items():
            gold_line = filt_qrels_dict[q_i_id]
            max_gold = max(gold_line.values())  # score
            top_gold = set(k for k,v in gold_line.items() if v == max_gold)

            max_line = max(line.values())       # score
            top_line = set(k for k,v in line.items() if v == max_line)

            check = top_gold.intersection(top_line) 
            if len(check)>0:    outputs_t_r_a[name].append(1)
            else:               outputs_t_r_a[name].append(0)
    
    check = [len(__) for name, __ in outputs_t_r_a.items()]
    assert len(set(check)) == 1, check
    return outputs_t_r_a


def do_one_run_ranx(qrels_dict, metrics, run_preds = {}, 
                    include_q_i_ids = set(), exclude_run_names = set(['runoff']),
                    round = 4):
    
    filt_qrels_dict = {q_i_id: line for q_i_id, line in qrels_dict.items() \
                                if q_i_id in include_q_i_ids}
    qrels           = Qrels(filt_qrels_dict)
    
    sc_key          = 'raw' # key for the score to use. we use the raw scores for NDCG
    runs            = [Run({q_i_id: line[sc_key] for q_i_id, line in run_p.items() \
                                    if q_i_id in include_q_i_ids},  name = name) \
                                        for name, run_p in run_preds.items() \
                                            if run_p is not None and name not in exclude_run_names]

    report_obj  = ranx_compare(qrels = qrels, runs = runs, 
                            metrics = metrics, 
                            rounding_digits = round, 
                            random_seed = 54506, 
                            max_p =  0.05,
                            make_comparable = True)

    return report_obj