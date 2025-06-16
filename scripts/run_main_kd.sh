conda deactivate
conda activate unsloth

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 torchrun model/main_grpo_master.py architecture.master.addr=$1 role='master' device_num=0 load_peft_ckpt_path=$2 grpo_settings.max_steps=$3 grpo_settings.initial_rules_based_steps=$4 grpo_settings.num_iterations=$5 train_exclude_long_prompts=$6 exclude_used_q_i_ids_files=$7

