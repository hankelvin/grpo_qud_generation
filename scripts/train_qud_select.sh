conda deactivate
conda activate unsloth

cd tools/qudselect/qud_parser_joint
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000
./scripts/finetune_single_joint_lora_with_accelerate.sh 8B 3.1 llama
./scripts/finetune_single_joint_lora_with_accelerate.sh 3B 3.2 llama
./scripts/finetune_single_joint_lora_with_accelerate.sh 7B '' qwen
./scripts/finetune_single_joint_lora_with_accelerate.sh 3B '' qwen


python data/data_generation.py
python data/prepare_question_pred_data.py
bash scripts/eval_single_joint_question.sh 3B llama
bash scripts/eval_single_joint_question.sh 8B llama
python data/reformat_output.py --model_size 3B llama
python data/reformat_output.py --model_size 8B llama
bash scripts/eval_single_joint_question.sh 3B qwen
bash scripts/eval_single_joint_question.sh 7B qwen
python data/reformat_output.py --model_size 3B qwen
python data/reformat_output.py --model_size 7B qwen

cd ..
python selective_decoding/rule_based_approaches.py --model_size 3B llama
python selective_decoding/rule_based_approaches.py --model_size 8B llama
python selective_decoding/get_final_quds.py --model_size 3B llama
python selective_decoding/get_final_quds.py --model_size 8B llama
python selective_decoding/rule_based_approaches.py --model_size 3B qwen
python selective_decoding/rule_based_approaches.py --model_size 7B qwen
python selective_decoding/get_final_quds.py --model_size 3B qwen
python selective_decoding/get_final_quds.py --model_size 7B qwen
