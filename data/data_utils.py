class QUDInstance:
    def __init__(self, article_id: int, 
                 anchor_id: int, answer_id: int, 
                 qud_instance_id: str,
                 qud_human: str, qud_candidates: dict,
                 do_tedq: bool = False):
        self.article_id         = article_id
        self.anchor_id          = anchor_id
        self.answer_id          = answer_id
        self.qud_instance_id    = qud_instance_id
        self.qud_human          = qud_human
        self.qud_candidates     = qud_candidates
        self.rank_predictions   = None
        self.do_tedq            = do_tedq

    def extract_context(self, article_contents):
        # article_contents is 1-indexed
        return [article_contents.contents[i] for i in range(1, self.anchor_id)]

    def extract_anchor(self, article_contents):
        if self.do_tedq: 
            # anchor for tedq is 2-sentence. we set anchor_id to be the first sentence
            return ' '.join([article_contents.contents[self.anchor_id], 
                             article_contents.contents[self.anchor_id+1]])
        else:
            return article_contents.contents[self.anchor_id]

    def extract_answer(self, article_contents):
        return article_contents.contents[self.answer_id]
    
    def extract_answer_cands(self, article_contents, ans_cands_form = 'all'):
        assert ans_cands_form in ['all', 'post_anc']
        if ans_cands_form == 'post_anc': 
            return [sent for i, sent in article_contents.contents.items() if i > self.anchor_id]
        elif ans_cands_form == 'all':
            return [sent for i, sent in article_contents.contents.items()]

class QUDCandidate:
    def __init__(self, sysname: str, qud: str, criteria_scores: dict = None, 
                 rationales: dict = None):
        self.sysname            = sysname
        self.qud                = qud
        self.rationales         = rationales
        self.criteria_scores    = criteria_scores

    def give_criteria_score(self, criteria):
        if self.criteria_scores is None: return None
        else: return self.criteria_scores[criteria]
    
    def give_criteria_rationale(self, criteria):
        if self.rationales is None: return None
        else: return self.rationales[criteria]

class ArticleContents:
    def __init__(self, article_id: str, contents: dict, fulltext: str = None):
        '''
        fulltext (str): the utf-8 string containing the original unsegmented text. 
        to be used for segmenting with chunk indices (e.g. for TED-Q)
        '''
        self.article_id         = article_id
        self.contents           = contents
        self.fulltext           = fulltext
