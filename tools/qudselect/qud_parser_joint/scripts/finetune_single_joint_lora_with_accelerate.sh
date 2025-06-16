# export CUDA_VISIBLE_DEVICES=4

MODEL_SIZE=$1  # 8B or 3B
MODEL_VERS=$2  # "3.1" if 8B # 3.2 if 3B
MODEL_NAME=$3
LORA_RANK=256 # should be 256. but deepspeed cannot work and 256 for 8B gives OOM. 256 is ok for 3B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

if [ "$MODEL_NAME" == "llama" ]; then
    MODELPATH=meta-llama/Llama-${MODEL_VERS}-${MODEL_SIZE}-Instruct
elif [ "$MODEL_NAME" == "qwen" ]; then
    MODELPATH=Qwen/Qwen2.5-${MODEL_SIZE}-Instruct
else
    echo ${MODEL_NAME} "model not set up"
    exit 1 
fi

echo "Training "${MODEL_NAME}" model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# --num_machines 1 \
# --num_processes $NUM_GPUS \
# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path  $MODELPATH \
    --use_lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --lora_dropout 0.05 \
    --tokenizer_name $MODELPATH \
    --use_slow_tokenizer \
    --train_file data/processed/single_joint_train.jsonl \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/single_joint_${MODEL_NAME}_${MODEL_SIZE}_lora/ \
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path $MODELPATH \
    --lora_model_name_or_path output/single_joint_${MODEL_NAME}_${MODEL_SIZE}_lora/