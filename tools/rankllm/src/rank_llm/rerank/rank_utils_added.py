import torch
from transformers import LogitsProcessor    
from collections import defaultdict
import math, numpy as np

def past_key_values_expander(past_key_values, num_beams):
    '''
    helper function to expand past key values (obtained for e.g. on a constant prompt
    whereby past_key_values was obtained by passing through the model). 
    Done so that past_key_values can be used for multiple beams search in generate()
    - for decoder only models: shape of past key values is 
    (num_layers, attn layers, num_beams, ...). We expand the last dimension

    '''
    pkv = []
    for l1 in past_key_values:
        pkv_1 = []
        for l2 in l1:
            pkv_2 = []
            assert len(l2) == 1
            for i in range(num_beams): pkv_2.append(l2[0])
            pkv_1.append(torch.stack(pkv_2, dim = 0))
        pkv.append(pkv_1)

    return pkv

########## LogitsProcessor ##########
# credits: Karel D'Oosterlinck
# https://colab.research.google.com/drive/1ezT24sogpVyr2HJLOvXHzjv61JZJ1gMT?usp=sharing#scrollTo=KwJ4OLx6CRqc
# https://towardsdatascience.com/the-power-of-constrained-language-models-cf63b65a035d

class RankLabelTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, contraints_pred_seq, pred_seq_num_pos, pred_seq_num_idxes, 
                 prefix_length, pred_seq_transform_model = None, tokenizer = None):
        super().__init__()
        # NOTE: pred_seq_num_pos is 0-indexed
        self.contraints_pred_seq    = contraints_pred_seq
        self.pred_seq_num_pos       = pred_seq_num_pos
        self.dedup_pred_seq_num_idxes     = pred_seq_num_idxes
        self.prefix_length          = prefix_length
        self.pred_seq_transform_model = pred_seq_transform_model
        # self.predicted_labels is a dict, each key is the beam index (only 0 key if not beam search)
        # the value is a list holding the predicted labels for each position in the sequence so far
        self.predicted_labels       = defaultdict(list)
        # similar to predicted_labels, but holds the past scores for each beam
        self.past_scores            = defaultdict(list)
        self.tokenizer              = tokenizer

        
    def __call__(self, input_ids, scores):
        # input_ids is of the shape (beam_size, sequence_length till now)
        assert len(input_ids.shape) == 2, input_ids.shape
        assert len(set(ii.shape[-1] for ii in input_ids)) == 1, set(ii.shape[-1] for ii in input_ids)
        curr_newgen_pos     = input_ids[0].shape[-1] - self.prefix_length
        curr_constraints    = self.contraints_pred_seq[curr_newgen_pos]

        tokens_to_set = {'positive_inf': [], 'handamard_product': [], 'positive_one': []}
        for beam_index, beam_scores in enumerate(scores):
            __holder  = {'positive_inf': [], 'handamard_product': [], 'positive_one': []}
            
            for key in ['allow', 'force']:
                indices = curr_constraints[key]
                if key == 'allow':
                    
                    if self.pred_seq_transform_model is None: 
                        action = 'positive_one'
                        __holder[action].extend(indices)
                    else:
                        action = 'handamard_product'
                        # take the already collected scores (extract self.dedup_pred_seq_num_idxes)
                        # stack and feed into self.pred_seq_transform_model
                        # get the weights, then expand into position in the mask, then apply to scores
                        # mask should be negative infinity every where else except the allowed tokens
                        past_scores = torch.stack(self.past_scores[beam_index], dim = 0)
                        past_scores = past_scores[self.dedup_pred_seq_num_idxes]

                        transform_weights = self.pred_seq_transform_model(past_scores)

                        product_mask = \
                            create_num_tokens_transform_mask(scores.shape[-1], 
                                len(self.pred_seq_num_pos), self.pred_seq_num_pos, 
                                transform_weights)
                        
                        __holder[action].extend([product_mask])
                
                elif key == 'force' and curr_constraints[key] is not None:
                    action = 'positive_inf'
                    __holder[action].extend(indices)
            
            # collect info at beam level for all keys
            for hk in __holder: tokens_to_set[hk].append(__holder[hk])
            
        # execute the modifications across beam 
        for action, tokens_to_set in tokens_to_set.items():
            if action != 'handamard_product': product_mask = None 
            else:
                assert len(tokens_to_set) == 1, len(tokens_to_set)
                product_mask = tokens_to_set[0]
                # there might not be any mask to apply, skip if so
                if not product_mask: continue
            
            scores = self.set_tokens_scores(scores, tokens_to_set, 
                                              action = action, 
                                              product_mask = product_mask)

        # collect scores for the current timestep 
        if curr_constraints['is_rank_token']:
            # collect current timestep scores and add the new predicted tokens to the set of predicted tokens
            for beam_index, beam_scores in enumerate(scores):
                assert len(beam_scores.shape) == 1, beam_scores.shape
                # a. collect the score at this timestep (that is a rank token position)
                self.past_scores[beam_index] = beam_scores[self.dedup_pred_seq_num_idxes]

                # b. add the new predicted token to the set of predicted tokens
                pred_index = beam_scores.argmax(dim = -1).item()
                self.predicted_labels[beam_index].append(pred_index)

                # print('PREDICTED', pred_index, 
                #       self.tokenizer.decode([pred_index]),
                #       self.predicted_labels[beam_index])

        return scores
  
    def set_tokens_scores(self, scores, tokens_to_set, action = 'positive_inf', product_mask = None):
        """
        Modifies the scores in place by setting the tokens_to_set token positions to `-inf`. 
        tokens_to_set is expected to be a
        list of list of tokens to exclude from generation in the format [[batch index, vocabulary position],...
        if positive = True, then tokens_to_set token position (has to be sole token) are set to  +inf

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            tokens_to_set: list of list of tokens to modify scores for, is of length (batch_size)
        """
        if action in ['positive_inf', 'positive_one']: 
            # create neg mask to zero out everywhere else 
            neg_mask = torch.zeros_like(scores, dtype = scores.dtype, device = scores.device)

            # collect positions to set to positive infinity
            inf_mask_list = []
            for batch_idx, batch_tts in enumerate(tokens_to_set):
                for token in batch_tts:
                    inf_mask_list.append([batch_idx, token])
            if not inf_mask_list: return scores

            inf_mask = torch.LongTensor(inf_mask_list)
            indices  = torch.ones(len(inf_mask))          

            # NOTE: torch.sparse.LongTensor deprecated
            # inf_mask = torch.sparse.LongTensor(inf_mask.t(), 
            #             indices, scores.size()).to(scores.device).to_dense().bool()
            inf_mask = torch.sparse_coo_tensor(inf_mask.t(), indices, scores.shape, 
                    dtype=torch.long, device=scores.device).to_dense().bool()

            # set +inf positions into neg_mask
            if action == 'positive_inf':   fill_val = float("inf") 
            elif action == 'positive_one': fill_val = 1.0
            else: raise NotImplementedError
            neg_mask = neg_mask.masked_fill(inf_mask, fill_val)
            
            # apply the mask
            scores *= neg_mask
            # the eot_id token in llama was nan, and argmax was returning nan as max.
            # this was forcing the model to generate the eot_id token at the first step
            scores = scores.nan_to_num(nan=-float('inf'))
            
        elif action == 'handamard_product':
            raise NotImplementedError
            # extract num_tokens, softmax this set of scores
            # multiply the mask with the scores
            # and then set it back to scores 
            scores = scores * product_mask
        
        return scores

def create_num_tokens_transform_mask(vocab_size, pred_seq_len, 
                                     pred_seq_num_pos, transform_weights):
    mask = torch.zeros(pred_seq_len, vocab_size, dtype = torch.float32)
    assert transform_weights.shape == (len(pred_seq_len), len(pred_seq_num_pos),), \
    f"transform_weights {transform_weights.shape} should \
        have shape (pred_seq_len, pred_seq_num_pos) {len(pred_seq_len), len(pred_seq_num_pos)}"
        
    for seq_pos in range(pred_seq_len):
        for i, pos in enumerate(pred_seq_num_pos):
            mask[seq_pos, pos] = transform_weights[seq_pos, i]

    return mask

# see # https://discuss.pytorch.org/t/quantization-of-a-single-tensor/189474
def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.cpu().data.numpy()
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    min_val = X.min().astype(dtype=np.float32)
    max_val = X.max().astype(dtype=np.float32)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    scale = (np.float64(max_val) - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = np.float64(0.1)
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max
    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return [scale.astype(np.float32), int(nudged_zero_point)]

def quantize_tensor(X):
    # quantisation does not work with bfloat16
    if X.dtype == torch.bfloat16: X = X.to(torch.float32)
    # find neg and pos inf and set to min, max -/+ 1 of the tensor
    X = manage_infs(X)
    scale, zero_point = _calculate_dynamic_qparams(X, torch.quint8)
    return torch.quantize_per_tensor(X, scale, zero_point, torch.quint8) 

def manage_infs(X):
    # certain models (e.g. LLAMA) have inf values in the tensor (eos/bos)
    # replace with min/max -/+ 1 of the tensor
    X[X == -float('inf')] = X[(X != -float('inf')) & (~X.isnan()) ].min() - 1
    X[X ==  float('inf')] = X[(X !=  float('inf')) & (~X.isnan()) ].max() + 1
    X[X.isnan()] = X[(X != -float('inf')) & (~X.isnan()) ].min() - 1
    return X