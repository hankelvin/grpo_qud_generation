conda deactivate
conda activate unsloth

MODEL=$1
HOSTNAME=$2
MMASTER_PORT=$3
QUDPEFTCKPT=$4
CUDA_VISIBLE_DEVICES=$5 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 python model/main_grpo_master.py do_grpo=True architecture.master.addr=${MMASTER_ADDR} architecture.master.port=${MMASTER_PORT} device_num=0 world_size=4 local_rank=0 role='master' grpo_settings.grpo_task="qud_gen" grpo_settings.initial_rules_based_steps=0 grpo_settings.max_steps=2000 grpo_settings.sync_ref_model=False grpo_settings.num_iterations=4 grpo_settings.gradient_accumulation_steps=4 grpo_settings.num_cands=4  grpo_settings.temperature=1.0 grpo_settings.lora_rank=32 grpo_settings.max_prompt_length=1400 grpo_settings.max_seq_length=512 grpo_settings.epsilon_high=0.28 save_steps=500 warmup_ratio=0.1 load_peft_ckpt_path=$QUDPEFTCKPT reward_funcs_version=3 model.reward="{use_past_key_values: False, criteria2: {model: qwen, size:mini}, criteria3: {model: qwen, size:mini}, criteria4: {model: qwen, size:mini}}" shorten_filename=True exclude_used_q_i_ids=True eval_every_save_step=False flash_attn_override='bypass' model.sft.model=$MODEL