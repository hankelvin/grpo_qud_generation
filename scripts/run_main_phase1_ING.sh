conda deactivate
conda activate unsloth # agentic_qud

cd ~
cd agentic_qud
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python  model/main_phase1.py ranker_args.do_cot=$1 ranker_args.cot_json=$2 ranker_args.cot_fine=$3 ranker_args.add_task_decomp_cot=$4 model_size=$5 do_model=$6 insert_new_gens=$7

