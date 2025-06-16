conda deactivate
conda activate unsloth

MASTER_ADDR=$1
MASTER_PORT=$2
HOST_NODE_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
RANKPEFTCKPT=$3
TEMP=0

C1="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 torchrun --rdzv_endpoint=${HOST_NODE_ADDR} model/main_grpo_worker.py architecture.master.addr=${MASTER_ADDR} architecture.master.port=${MASTER_PORT} world_size=4 device_num=0 local_rank='1' role=$'criteria2' do_grpo=False grpo_settings.grpo_task=rankllm grpo_settings.lora_rank=128 grpo_settings.max_seq_length=1024 grpo_settings.temperature=$TEMP sft_or_grpo_rm=True save_steps=500 use_vllm=False load_peft_ckpt_path=$RANKPEFTCKPT reward_funcs_version=2 exclude_used_q_i_ids=False flash_attn_override='bypass'  model.sft.model=qwen "
C2="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 torchrun --rdzv_endpoint=${HOST_NODE_ADDR} model/main_grpo_worker.py architecture.master.addr=${MASTER_ADDR} architecture.master.port=${MASTER_PORT} world_size=4 device_num=0 local_rank='2' role='criteria3' do_grpo=False grpo_settings.grpo_task=rankllm grpo_settings.lora_rank=128 grpo_settings.max_seq_length=1024 grpo_settings.temperature=$TEMP sft_or_grpo_rm=True save_steps=500 use_vllm=False load_peft_ckpt_path=$RANKPEFTCKPT reward_funcs_version=2 exclude_used_q_i_ids=False flash_attn_override='bypass'  model.sft.model=qwen "
C3="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:30000 torchrun --rdzv_endpoint=${HOST_NODE_ADDR} model/main_grpo_worker.py architecture.master.addr=${MASTER_ADDR} architecture.master.port=${MASTER_PORT} world_size=4 device_num=0 local_rank='3' role='criteria4' do_grpo=False grpo_settings.grpo_task=rankllm grpo_settings.lora_rank=128 grpo_settings.max_seq_length=1024 grpo_settings.temperature=$TEMP sft_or_grpo_rm=True save_steps=500 use_vllm=False load_peft_ckpt_path=$RANKPEFTCKPT reward_funcs_version=2 exclude_used_q_i_ids=False flash_attn_override='bypass'  model.sft.model=qwen "

tmux new-session -d -s worker1 $C1
tmux new-session -d -s worker2 $C2
tmux new-session -d -s worker3 $C3 