import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)
# HOME = '/home/khan'
HOME = '/workspace'

def process_question_data(data):
    # construct question generation data based on predicted anchors 
    prompt_format_a = "\nSentence [{answer_id}] is anchored by sentence [{anchor_id}],"

    processed_data = []
    anchor_num = []
    seen_q_i_ids = set()
    for i in pred_data:
        ### CHANGE START ###
        # load qud_instance_ids to predict 
        with open(f'{HOME}/agentic_qud/data/eval_qud_instance_ids_noexcludes.txt', encoding = 'utf-8') as f:
            qiid_lines = f.readlines()
        
        holder = {}
        for qud_instance_id in qiid_lines:
            art, anc, ans = qud_instance_id.split('_')
            art, anc, ans = int(art), int(anc), int(ans)
            if art not in holder: holder[art] = set()
            holder[art].add((anc, ans,))
        
        ### CHANGE START ###
        article_id, anchor_id, answer_id = i['qud_instance_id'].split('_')
        article_id = int(article_id) if article_id.isdigit() else article_id
        if article_id not in holder and 'talk' not in str(article_id):
            print('NOT IN SUBSET', article_id)
            continue
        # was not aded in eval_qud_instance_ids_noexcludes
        if 'talk' in str(article_id):
            if article_id not in holder: holder[article_id] = set()
            holder[article_id].add((anchor_id, answer_id,))
        ### CHANGE END ###
        
        for idx, (anchor_id, answer_id) in enumerate(holder[article_id]):
            qud_instance_id =  f'{article_id}_{anchor_id}_{answer_id}'
            if qud_instance_id in seen_q_i_ids: continue
            processed_data.append({
                'dataset': 'DCQA-single-joint-question-val',
                'id': i['id']+'_'+str(idx),
                'prompt': i['prompt'] + prompt_format_a.format(answer_id = answer_id, anchor_id = anchor_id),
                'reference': i['reference'],
                ### CHANGE START ###
                'qud_instance_id': qud_instance_id, 
                ### CHANGE END ###
                'meta': {'answer_id': answer_id, 'pred_anchor_id': anchor_id}
            })
            seen_q_i_ids.add(qud_instance_id)
        ### CHANGE END ###

        # anchors = []
        # for j in i['output']:
            # splits = re.split('Sentence \[(\d+)\] is anchored by sentence \[(\d+)\],', j)
            #     # if len(splits) > 2:
            #     #     answer_id = splits[1]
            #     #     pred = splits[2]
            #     #     if not pred in anchors:
            #     #         anchors.append(pred)
            # anchor_num.append(len(anchors))
            # for idxj, j in enumerate(anchors):
            #     if int(answer_id) > int(j) and int(j) > 0:
            #         processed_data.append({
            #             'dataset': 'DCQA-single-joint-question-val',
            #             'id': i['id']+'_'+str(idxj),
            #             'prompt': i['prompt'] + prompt_format_a.format(answer_id = answer_id, anchor_id = j),
            #             'reference': i['reference'],
            #             'meta': {'answer_id': answer_id, 'pred_anchor_id': j}
            #         })

    print(len(pred_data), len(processed_data))
    # print(np.mean(anchor_num))
    
    return processed_data
            
pred_data = []
### CHANGE START ### change name to the file created at data_generation.py
with open('data/processed/single_joint_val.jsonl', 'r') as f:
    for line in f:
        pred_data.append(json.loads(line))
print('NUMBER OF PRED LINES', len(pred_data))

processed_question_data = process_question_data(pred_data)
with open('data/processed/single_joint_question_val.jsonl', 'w') as f:
    for i in processed_question_data:
        f.write(json.dumps(i)+'\n')