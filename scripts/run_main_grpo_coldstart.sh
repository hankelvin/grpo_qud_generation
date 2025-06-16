conda deactivate
conda activate unsloth

MODEL=$1
HOSTNAME=$2
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 torchrun model/main_grpo_master.py do_grpo=True architecture.master.addr=$HOSTNAME role='master' device_num=0 grpo_settings.grpo_task="qud_gen" grpo_settings.initial_rules_based_steps=1e9 grpo_settings.max_steps=500 grpo_settings.epsilon_high=0.28 grpo_settings.sync_ref_model=False  grpo_settings.num_iterations=2 grpo_settings.gradient_accumulation_steps=4 grpo_settings.num_cands=4  grpo_settings.temperature=1.0 grpo_settings.lora_rank=32 grpo_settings.max_prompt_length=1200 grpo_settings.max_seq_length=512 grpo_settings.epsilon_high=0.28 save_steps=500 warmup_ratio=0.1 reward_funcs_version=2 exclude_used_q_i_ids=False flash_attn_override='bypass' eval_every_save_step=False model.sft.model=$MODEL