import json
from tqdm import tqdm

def main(args):
    ### CHANGE START ###
    MODEL_NAME = args.model_name
    MODEL_SIZE = args.model_size # '8B'
    # HOME = '/home/khan'
    HOME = '/workspace'
    dp = f'{HOME}/agentic_qud/tools/qudselect/qud_parser_joint/data/processed/'
    ### CHANGE END ###
    with open(dp + f'scored_single_joint_val_{MODEL_NAME}_{MODEL_SIZE}.json', 'r') as f:
        val_data = json.load(f)
        

    new_list = []
    for i in tqdm(val_data):
        new = {}
        ### CHANGE START ###
        # new['article_id'] = i['article_id']
        q_i_id = i['qud_instance_id']
        ### CHANGE END ###
        new['answer'] = i['answer_id']
        if len(i['candidates']):
            maxscore = max(i['candidates'], key=lambda x:x['score']) 
            new['score'] = maxscore['score']
            new['anchor_id'] = maxscore['anchor_id']
            new['question'] = maxscore['question']
            ### CHANGE START ###
            new['qud_instance_id'] = maxscore['qud_instance_id']
            ### CHANGE END ###
            new_list.append(new)
                
    with open(f'{dp}/final_quds_QUDEVAL_RANK_{MODEL_NAME}_{MODEL_SIZE}.json', 'w') as f:
        f.write(json.dumps(new_list, indent=2)) 


if __name__ == '__main__':
    import argparse        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',             type = str, default = 'llama')
    parser.add_argument('--model_size',             type = str, default = '3B')
    args = parser.parse_args()
    main(args)
        