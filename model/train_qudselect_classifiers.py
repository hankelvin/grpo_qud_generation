import pandas as pd, re, random, torch, numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments, Trainer, EvalPrediction)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import torch
from torch.utils.data import Dataset
SEED = 54506
torch.manual_seed(SEED)
np.random.seed(SEED)


# load data from QUDSelect repo
# create dataloader and collate_fn
# load models (roBERTa for comp. and relv.; longformer for give.)
# create savepath 
DIRPATH = '/workspace/agentic_qud/'
SETTINGS_DICT = \
{
'criteria2': {'model': 'FacebookAI/roberta-large', 
              'data_dir': 'tools/qudselect/automatic_evaluators/data/oversample_qudeval/comp/'}, 
'criteria3': {'model': 'allenai/longformer-base-4096', 
              'data_dir': 'tools/qudselect/automatic_evaluators/data/oversample_qudeval/givn/'},
'criteria4': {'model': 'FacebookAI/roberta-large', 
              'data_dir': 'tools/qudselect/automatic_evaluators/data/oversample_qudeval/relv/'},
}
BATCH_SIZE      = 32
NUM_LABELS      = 3
LEARNING_RATE   = 1e-5
NUM_EPOCHS      = 10
MAX_SEQ_LENGTH  = 512
def main():    
    for criteria in ['criteria2', 'criteria3', 'criteria3_withans', 'criteria4']:
        crit_key = criteria.replace('_withans', '')
        training_args = TrainingArguments(
                        output_dir          = f'{DIRPATH}/results/qudselect_classifiers/{criteria}',
                        num_train_epochs    = NUM_EPOCHS,
                        learning_rate       = LEARNING_RATE, 
                        per_device_train_batch_size = BATCH_SIZE,
                        per_device_eval_batch_size  = BATCH_SIZE,
                        logging_dir         = f'{DIRPATH}/results/qudselect_classifiers/{criteria}/logs',
                        logging_steps       = 10,
                        evaluation_strategy = "epoch",
                        save_strategy       = "no",
                        load_best_model_at_end = False,
                        report_to           = 'none')
        mname       = SETTINGS_DICT[crit_key]['model']
        model       = AutoModelForSequenceClassification.from_pretrained(mname,
                                                        num_labels = NUM_LABELS,
                                                        problem_type = 'single_label_classification')
        tokenizer   = AutoTokenizer.from_pretrained(mname)
        
        data_holder = {'train': None, 'dev': None, 'test': None}
        if   criteria == 'criteria2': 
            names = ['question', 'answer', 'label']
            for split in data_holder:
                data   = pd.read_csv(SETTINGS_DICT[crit_key]['data_dir'] + f'{split}.tsv', names = names, sep = '\t')
                texts  = data.apply(lambda x: x.question + tokenizer.sep_token + x.answer, axis = 1).tolist()
                # reverse the order so that 3 is best, -1 for 0-indexed 
                labels = data['label'].apply(lambda x: abs(x-4)-1).tolist() 
                print(np.unique(labels))
                ds_obj = TextClassificationDataset(texts, labels, tokenizer, MAX_SEQ_LENGTH)
                data_holder[split] = ds_obj

        elif criteria.startswith('criteria3'): 
            names = ['context', 'question', 'label']
            for split in data_holder:
                data   = pd.read_csv(SETTINGS_DICT[crit_key]['data_dir'] + f'{split}.tsv', names = names, sep = '\t')
                
                if criteria == 'criteria3_withans':                    
                    sample_size = len(data)
                    df_qudeval  = pd.read_csv(f'{DIRPATH}data/qudeval/data-collection.csv')
                    # 1. filter QUDEval for only the rows with (i) the same questions as the Classifier data csv
                    # and (ii) which have not been skipped 
                    qud_set     = data.question.tolist()
                    c1          = df_qudeval.questions.isin(qud_set)
                    c2          = df_qudeval.criteria3 != 'skipped'
                    cut_df_qudeval = df_qudeval[c1 & c2].copy()

                    # 2. load articles
                    holder_articles = {}
                    for article_id in cut_df_qudeval.essay_id.unique():
                        fp = f'{DIRPATH}data/dcqa/article2/{str(article_id).zfill(4)}.txt'
                        with open(fp, encoding = 'utf-8') as f: lines = f.readlines()
                        art = {}
                        for l in lines:
                            lid, line = re.search(r'(\d+) (.+)', l.strip()).groups()
                            art[int(lid)] = line
                        holder_articles[article_id] = art
                    
                    # 3. construct new dataframe with (C, q, ans) and score
                    new_data = []
                    for __, row in cut_df_qudeval.iterrows():
                        nline = {}
                        art                     = holder_articles[row['essay_id']]
                        nline['question']    = row['questions']
                        nline['label']       = int(row['criteria3'])
                        nline['context']     = ' '.join([art[i] for i in range(1,row['anchor_id']+1)])
                        nline['answer']      = art[row['answer_id']]
                        print('CTX', nline['context'])
                        print('ANS', nline['answer'])
                        print('QST', nline['question'])
                        print('~'*50, '\n\n')
                        new_data.append(nline)
                    new_data = random.choices(new_data, k = sample_size)
                    data = pd.DataFrame(new_data)
                    # add answer too
                    texts  = data.apply(lambda x: x.context + tokenizer.sep_token \
                                                + x.answer + tokenizer.sep_token + x.question, axis = 1).tolist()
                else:
                    data['context'] = data['context'].fillna('') # some context is empty, i.e. anchor is 1st sentence
                    texts  = data.apply(lambda x: x.context + tokenizer.sep_token + x.question, axis = 1).tolist()
                
                # reverse the order so that 3 is best, -1 for 0-indexed 
                labels = data['label'].apply(lambda x: abs(x-4)-1).tolist()
                print(np.unique(labels))
                ds_obj = TextClassificationDataset(texts, labels, tokenizer, MAX_SEQ_LENGTH)
                data_holder[split] = ds_obj 

        elif criteria == 'criteria4': 
            names = ['anchor', 'question', 'label']
            for split in data_holder:
                data   = pd.read_csv(SETTINGS_DICT[crit_key]['data_dir'] + f'{split}.tsv', names = names, sep = '\t')
                texts  = data.apply(lambda x: x.question + tokenizer.sep_token + x.anchor, axis = 1).tolist()
                # reverse the order so that 3 is best, -1 for 0-indexed 
                labels = data['label'].apply(lambda x: abs(x-4) - 1).tolist()  
                print(np.unique(labels))
                ds_obj = TextClassificationDataset(texts, labels, tokenizer, MAX_SEQ_LENGTH)
                data_holder[split] = ds_obj 

        trainer = Trainer(model = model, args = training_args, train_dataset = data_holder['train'],
            eval_dataset = data_holder['test'], compute_metrics = compute_metrics,)
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"{criteria}... Evaluation results: {eval_results}")

        model_save_path = f'{DIRPATH}/results/qudselect_classifiers/{criteria}'
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"{criteria}... MODEL SAVED TO: {model_save_path}")


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation = True, padding = 'max_length', 
                                 max_length = max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define compute_metrics function for the Trainer
def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    macrof1 = f1_score(y_true = labels, y_pred = preds, average = 'macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'macrof1': macrof1,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':
    main()



