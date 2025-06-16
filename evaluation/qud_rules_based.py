'''
This script implements a QUDRulesBasedEvaluator class that holds all the necessary 
tools and functions needed to compute the rules-based criteria scorers used 
in QUDSelect (Suvarna et al 2024). 

Credits to (Suvarna et al 2024) for the implementations for compute_givenness, compute_comp and 
compute_relevance, which can be found at: 
https://github.com/asuvarna31/qudselect/blob/main/selective_decoding/rule_based_approaches.py

'''
import torch

class QUDRulesBasedEvaluator:
    def __init__(self, device = None):
        import spacy, torch
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
        
        model = 'en_core_web_sm'
        if not spacy.util.is_package(model): spacy.cli.download(model)
        self.nlp            = spacy.load(model)
        if device is None: 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_name  = "facebook/bart-large-mnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device = device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, device = device)
        
        pipeline_args = {'model': model, 'tokenizer': tokenizer,'device': device}
        self.classifier     = pipeline("zero-shot-classification", **pipeline_args)
        
        print('self.classifier device', self.classifier.device)
        self.label_mapping  = ['valid', 'neutral', 'invalid']

    def compute_givenness(self, question, context):
        
        doc_context = self.nlp(context)
        ### CHANGE START ###
        # place in set for faster search
        context_lemmas_list = set(token.lemma_ for token in doc_context if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'})
        ### CHANGE END ###

        doc_question = self.nlp(question)
        question_lemmas_list = [token.lemma_ for token in doc_question]
        ### CHANGE START ###
        if len(question_lemmas_list) == 0:
            score = 0
        else: 
            new_lemmas=0
            for lemma in question_lemmas_list:
                if lemma not in context_lemmas_list:
                    new_lemmas+=1
                        
            score = new_lemmas/len(question_lemmas_list)
        ### CHANGE END ###
        return score

    def compute_comp(self, question, answer):
        ### CHANGE START ###
        with torch.no_grad():
            ### CHANGE END ###
            probs = self.classifier(question+answer, self.label_mapping)
        return float(probs['scores'][0])

    def compute_relevance(self, question, anchor):
        
        doc = self.nlp(question)
        # print(question)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        if len(noun_phrases) != 0:
            max_noun_phrase = max(noun_phrases, key=len)
            doc_np = self.nlp(max_noun_phrase)
            question_lemmas_list = [token.lemma_ for token in doc_np]
        else:
            question_lemmas_list = []

        ### CHANGE START ###
        if len(question_lemmas_list) == 0:
            score = 0
        else: 
            #anchor sentence
            doc_anchor = self.nlp(anchor)
            ### CHANGE START ###
            # place in set for faster search
            anchor_lemmas_list = set(token.lemma_ for token in doc_anchor if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'})
            ### CHANGE END ###
            
            new_lemmas=0
            for lemma in question_lemmas_list:
                if lemma in anchor_lemmas_list:
                    new_lemmas+=1
            
            score = new_lemmas/len(question_lemmas_list)
            ### CHANGE END ###
        
        return score