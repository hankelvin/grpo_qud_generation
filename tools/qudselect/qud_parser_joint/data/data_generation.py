import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)

articles={}
### CHANGE START ###
# directory='../../dcqa/article2/'
HOME = '/home/khan'
HOME = '/workspace'
PROMPT_FORMAT = "### Instruction:\n{instruction}\n\n### Input:\nContext: {context}\nAnswer sentence: {answer_sentence}\n\n### Response:"
INSTRUCTION   = "Given the answer sentence, reason through the context to find the most likely sentence where a question can be generated."

def give_articles():
    directory     = f'{HOME}/agentic_qud/data/dcqa/article2/'

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
    return articles
        
def process_data(data, split, skip_seen = False):
    # transform the train/val data into instruction format
    articles = give_articles()

    processed_data = []
    seen = set()
    for k in data:
        article_id = k['Article'][0]['ArticleID'].zfill(4)
        current_article = articles[article_id]

        context = ""
        # add sentence id before each sentence
        for sentence in range(len(current_article)):
            context = context+' XT'+str(sentence+1).zfill(2)+' '+ current_article[sentence]  
        context = context[1:]

        used_answer_id = set()
        for idx, qas in enumerate(k['Article'][0]['qas']):
            answer_sentence_id = qas['AnswerSentenceID']
            if answer_sentence_id in used_answer_id or answer_sentence_id > 20:
                continue
            used_answer_id.add(answer_sentence_id)

            cur_context = context[:context.find('XT'+str(answer_sentence_id).zfill(2))].strip()
            answer_sentence = 'XT'+str(answer_sentence_id).zfill(2)+' '+ current_article[answer_sentence_id - 1]

            question_id = article_id + '_' + str(idx)

            anchor_sentence_id = qas['AnchorSentenceID']
            anchor_text = 'XT'+str(anchor_sentence_id).zfill(2)
            anchor_start = cur_context.find(anchor_text)

            question = qas['Question']
            answer = f'Sentence [{answer_sentence_id}] is anchored by sentence [{anchor_sentence_id}], answering the question of "{question}".'
            qud_instance_id = f'{article_id}_{anchor_sentence_id}_{answer_sentence_id}'
            if anchor_start >= 0:
                if skip_seen and qud_instance_id in seen: continue
                seen.add(qud_instance_id)
                if split == 'train':
                    processed_data.append({
                        'dataset': 'DCQA-single-joint-train',
                        'id': question_id,
                        ### CHANGE START ###
                        'article_id': article_id, 
                        'qud_instance_id': qud_instance_id,
                        ### CHANGE END ###
                        'messages': [{"role": "user", "content": PROMPT_FORMAT.format(instruction = INSTRUCTION, context = cur_context, answer_sentence = answer_sentence)}, {"role": "assistant", "content": answer}]
                    })
                else:
                    processed_data.append({
                        'context' : cur_context,
                        'answer' : answer_sentence,
                        'question' : question,
                        ### CHANGE START ###
                        'id': question_id,
                        'article_id': article_id, 
                        'qud_instance_id': qud_instance_id,
                        'prompt': PROMPT_FORMAT.format(instruction = INSTRUCTION, context = cur_context, answer_sentence = answer_sentence),
                        'reference': answer
                        ### CHANGE END ###
                    })
            
            # if anchor_start >= 0:
            #     if split == 'train':
            #         processed_data.append({
            #             'dataset': 'DCQA-single-joint-train',
            #             'id': question_id,
            #             'messages': [{"role": "user", "content": PROMPT_FORMAT.format(instruction = INSTRUCTION, context = cur_context, answer_sentence = answer_sentence)}, {"role": "assistant", "content": answer}]
            #         })
            #     else:
            #         processed_data.append({
            #             'dataset': 'DCQA-single-joint-val',
            #             'id': question_id,
            #             'prompt': PROMPT_FORMAT.format(instruction = INSTRUCTION, context = cur_context, answer_sentence = answer_sentence),
            #             'reference': answer
            #         })

    print(len(data), len(processed_data))
    return processed_data
    

### CHANGE START ###
class ArticleContents:
    def __init__(self, article_id: str, contents: dict, fulltext: str = None):
        '''
        fulltext (str): the utf-8 string containing the original unsegmented text. 
        to be used for segmenting with chunk indices (e.g. for TED-Q)
        '''
        self.article_id         = article_id
        self.contents           = contents
        self.fulltext           = fulltext

def give_qsalience_ans_sents(line, min_agree = 2):
    '''
    helper func for QSalience data. Finds the set of candidate answer_ids that were
    selected by 2 or more of the annotators as a candidate answer to a question.
    '''
    from collections import Counter
    import math
    ctr = Counter()
    for i, anno in ([0, 'Keziah reason'],
                    [1, 'Kathryn reason'],
                    [2, 'Karim reason'],):
        a = line[anno]
        if type(a) is float and math.isnan(a): pass
        else: 
            a = [aa.strip() for aa in a.split(',')]
            for aa in a: 
                try:    ctr.update([int(aa)])
                except: pass
    return [k for k, v in ctr.items() if v >= min_agree]

def load_qsalience_data(directory):
    import pandas as pd
    import glob
    qsal_ctr = 0 
    df_qsalience = pd.read_csv(f'{directory}/data/qsalience/data/answerability/answerability.csv', encoding = 'utf-8')
    # keep only human portion of data, and where human_answerability is 2 and above
    df_qsalience = df_qsalience[(df_qsalience.model=='human') & (df_qsalience.human_answerability>=2)].copy()
    # df_qsalience = df_qsalience[(df_qsalience.dataset=='tedq')].copy()
    df_qsalience['ans_candidates'] = df_qsalience.apply(lambda x: give_qsalience_ans_sents(x), axis=1)

    holder_articles_qsal = {}
    with open(f'{directory}/data/qsalience/data/tedq_1927.txt', encoding = 'utf-8') as f:
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
    holder_articles_qsal[article_id] = art_obj
    holder_articles_qsal[str(article_id)] = art_obj

    art_fps = sorted(glob.glob(f'{directory}/data/qudeval/article2/*.txt'))
    for fp in art_fps:
        with open(fp, encoding='utf-8') as f:
            contents = f.readlines()
        contents = [re.search(r'(\d+)(?:\s+)(.+)', l).groups() for l in contents]
        contents = {int(k): v.strip() for (k,v) in contents}
        
        article_id  = int(os.path.basename(fp).replace('.txt', ''))
        art_obj     = ArticleContents(article_id, contents)
        holder_articles_qsal[article_id] = art_obj
        holder_articles_qsal[str(article_id)] = art_obj

    holder_context = {}
    for article_id, art_obj in holder_articles_qsal.items():
        art_cont = art_obj.contents
        context = ''
        for sid in sorted(art_cont):
            # NOTE: sid already 1-indexed
            context = context+' XT'+str(sid).zfill(2)+' '+ art_cont[sid]  
        holder_context[article_id] = context

    holder_dataset_eval_qsal = []
    for __, row in df_qsalience.iterrows():
        article_id  = row['article_id']
        anchor_id   = row['sentence_id']
        qud         = row['question']
        for answer_id in row['ans_candidates']:
            if anchor_id == answer_id: continue
            qud_instance_id = f'{article_id}_{anchor_id}_{answer_id}'
            line = add_one_line_qsalience_tedq(qud_instance_id, qud, holder_articles_qsal, holder_context)
            holder_dataset_eval_qsal.append(line)
            qsal_ctr += 1
    print('This number of instances from the QSalience data added:', qsal_ctr)
    return holder_articles_qsal, holder_dataset_eval_qsal

def load_tedq_data(dirpath, cutoff = 20):
    import pandas as pd
    import glob, spacy
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

def add_qsalience_tedq_data(directory):
    directory=f'{HOME}/agentic_qud'                
    processed_data = []
    holder_articles_tedq, df_tedq_cut = load_tedq_data(directory)
    holder_context = {}
    for article_id, art_obj in holder_articles_tedq.items():
        art_cont = art_obj.contents
        context = ''
        for sid in sorted(art_cont):
            # NOTE: sid already 1-indexed
            context = context+' XT'+str(sid).zfill(2)+' '+ art_cont[sid]  
        holder_context[article_id] = context
    
    import spacy
    nlp = spacy.load('en_core_web_lg')
    for i, row in df_tedq_cut.iterrows():
        qud_instance_id  = row['qud_instance_id']
        qud              = row['content']
        assert row['type'] == 'question', row
        line = add_one_line_qsalience_tedq(qud_instance_id, qud, holder_articles_tedq, holder_context, 
                                           do_tedq = True)
        processed_data.append(line)
    holder_articles_qsal, holder_dataset_eval_qsal = load_qsalience_data(directory)
    processed_data += holder_dataset_eval_qsal
    return processed_data

def add_one_line_qsalience_tedq(qud_instance_id, question, holder_articles, holder_context, 
                                do_tedq = False):
    article_id, anchor_sentence_id, answer_sentence_id = qud_instance_id.split('_')
    anchor_sentence_id, answer_sentence_id = int(anchor_sentence_id), int(answer_sentence_id)
    

    current_article     = holder_articles[article_id]
    context             = holder_context[article_id]
    cur_context         = context[:context.find('XT'+str(answer_sentence_id).zfill(2))].strip()
    anchor_text = 'XT'+str(anchor_sentence_id).zfill(2)
    anchor_start = cur_context.find(anchor_text)
    assert anchor_start >= 0, (qud_instance_id, anchor_start, current_article.contents.keys(), anchor_text, cur_context)

    # NOTE: current_article is a dict here (and we made it 1-indexed earlier). we can use answer_sentence_id 
    # to directly retrieve the answer sentence
    answer_sentence = 'XT'+str(answer_sentence_id).zfill(2)+' '+ current_article.contents[answer_sentence_id]

    question_id = qud_instance_id
    answer = f'Sentence [{answer_sentence_id}] is anchored by sentence [{anchor_sentence_id}], answering the question of "{question}".'

    line = {'context' : cur_context,
            'answer' : answer_sentence,
            'question' : question,
            ### CHANGE START ###
            'id': question_id,
            'article_id': article_id, 
            'qud_instance_id': qud_instance_id,
            'prompt': PROMPT_FORMAT.format(instruction = INSTRUCTION, context = cur_context, answer_sentence = answer_sentence),
            'reference': answer
            ### CHANGE END ###
            }
    return line
### CHANGE END ###

def main(directory):
    ### CHANGE START         
    train_data = json.load(open(f'{directory}/data/dcqa/train.json', 'r'))
    processed_train_data = process_data(train_data, 'train')
    with open('data/processed/single_joint_train.jsonl', 'w') as f:
        for i in processed_train_data:
            f.write(json.dumps(i)+'\n')

    # directory='../../dcqa/article2/'  
    val_data = json.load(open(f'{directory}/data/dcqa/val.json', 'r'))
    val_data += json.load(open(f'{directory}/data/dcqa/test.json', 'r'))
    processed_val_data = process_data(val_data, 'val', skip_seen = True)
    print('BEFORE', len(processed_val_data))
    processed_val_data += add_qsalience_tedq_data(directory)
    print('AFTER', len(processed_val_data))
    # with open('data/processed/gpt4.jsonl', 'w+') as f:
    with open('data/processed/single_joint_val.jsonl', 'w+') as f:
        for i in processed_val_data:
            f.write(json.dumps(i)+'\n')
    ### CHANGE END ###  
            
    # val_data = json.load(open(f'{directory}/dcqa/test.json', 'r'))
    # processed_val_data = process_data(val_data, 'val')
    # with open('data/processed/single_joint_anchor_test.jsonl', 'w') as f:
    #     for i in processed_val_data:
    #         f.write(json.dumps(i)+'\n')

if __name__ == '__main__':
    import argparse        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size',             type = str, default = '3B')
    args = parser.parse_args()
    main(directory = f'{HOME}/agentic_qud')