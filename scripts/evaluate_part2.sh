conda deactivate
conda activate unsloth
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["QZS","QRB2K","QLPC2","QQS7B"]' insert_new_gens_version='v2' 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["QZS","QRB2K","QLPC2","QQS3B"]' insert_new_gens_version='v2' 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["QZSqs","QRB2Kqs","QLPC2qs","QQS3Bqs"]' insert_new_gens_version='v2' 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["LZS","LRB2K","LLPC2","LQS8B"]' insert_new_gens_version='v2' 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["LZS","LRB2K","LLPC2","LQS3B"]' insert_new_gens_version='v2' 

python  model/main_phase1.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=True model_size='small' do_model='gpt4o' insert_new_gens='["LZSqs","LRB2Kqs","LLPC2qs","LQS3Bqs"]' insert_new_gens_version='v2' 
