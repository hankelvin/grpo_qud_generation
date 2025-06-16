import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)

def main(args):
    # HOME = '/home/khan'
    HOME = '/workspace'

    articles={}
    ### CHANGE START ###
    directory=f'{HOME}/agentic_qud/data/dcqa/article2/'
    ### CHANGE END ###
    for filename in sorted(os.listdir(directory)):
        article = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(article):
            each_article=[]
            file=open(article,'r')
            for line in file:
                each_article.append(" ".join(line.strip().split(" ")[1:]))
            articles[filename[:4]] = each_article[:20]

    from data_generation import load_tedq_data, load_qsalience_data
    directory=f'{HOME}/agentic_qud'     
    holder_articles_tedq, df_tedq_cut = load_tedq_data(directory)
    holder_articles_qsal, holder_dataset_eval_qsal = load_qsalience_data(directory)
    for qud_instance_id, art_obj in holder_articles_tedq.items():
        articles[qud_instance_id.split('_')[0]] = [art_obj.contents[i] for i in sorted(art_obj.contents)]
    for qud_instance_id, art_obj in holder_articles_qsal.items():
        if type(qud_instance_id) == int: continue
        articles[qud_instance_id.split('_')[0]] = [art_obj.contents[i] for i in sorted(art_obj.contents)]
            
    data = []
    ### CHANGE START ###
    MODEL_NAME = args.model_name
    MODEL_SIZE = args.model_size #'8B'
    with open(f'data/processed/single_joint_question_val_outputs_{MODEL_NAME}_{MODEL_SIZE}.jsonl', 'r') as f:
        ### CHANGE END ###
        for line in f:
            data.append(json.loads(line))
            
    reformat_pred = []
    used_answer = dict()
    for i in data:
        article_id = i['id'][:4] if 'talk' not in i['id'] else i['id'].split('_')[0]
        ### CHANGE START ###
        qud_instance_id = i['qud_instance_id']
        ### CHANGE END ###
        answer_id = str(i['meta']['answer_id']).zfill(2)
        anchor_id = str(i['meta']['pred_anchor_id']).zfill(2)
        
        #### CHANGE START ### change article_id key to qud_instance_id
        if not qud_instance_id in used_answer:
            sentence_num = len(articles[article_id])
            used_answer[qud_instance_id] = []
            for j in range(sentence_num-1):
                used_answer[qud_instance_id].append([])
        if int(answer_id) > len(used_answer[qud_instance_id])+1:
            continue
        ### CHANGE END ###
        for j in i['output']:
            
            try:
                splits = re.split('answering the question of "(.*?)".', j) 
                question = splits[1]
            ### CHANGE START ###
            except:
                try:
                    splits = re.split('answering the question "(.*?)".', j)
                    question = splits[1]
                except:
                    question = j.replace(' answering the question of', '').replace('"', '').strip()
                    print('QG failed', qud_instance_id, j)
            ### CHANGE END ###
            used_answer[qud_instance_id][int(answer_id)-2].append({
                'anchor_id': anchor_id,
                'question': question,
                ### CHANGE START ###
                'qud_instance_id': qud_instance_id, 
            })
            
    ### CHANGE START ###        
    for qud_instance_id in used_answer:
        for i in range(len(used_answer[qud_instance_id])):
            reformat_pred.append({
                'qud_instance_id': qud_instance_id,
                'answer_id': str(i+2).zfill(2),
                'candidates': used_answer[qud_instance_id][i]
            })  
    ### CHANGE END ###

    print(len(reformat_pred))
    ### CHANGE START ###
    sorted_list = sorted(reformat_pred, key=lambda x: (x['qud_instance_id'], x['answer_id']))
    ### CHANGE END ###

    ### CHANGE START ###
    with open(f'data/processed/reformat_single_joint_question_val_outputs_{MODEL_NAME}_{MODEL_SIZE}.json', 'w') as f:
        ### CHANGE END ###
        f.write(json.dumps(sorted_list, indent=2)) 

if __name__ == '__main__':
    import argparse        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',             type = str, default = 'llama')
    parser.add_argument('--model_size',             type = str, default = '3B')
    args = parser.parse_args()
    main(args)

