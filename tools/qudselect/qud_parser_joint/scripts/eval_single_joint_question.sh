# export CUDA_VISIBLE_DEVICES=4

MODEL_SIZE=$1
MODEL_NAME=$2
python open_instruct/predict.py \
    --model_name_or_path output/single_joint_${MODEL_NAME}_${MODEL_SIZE}_lora/ \
    --input_files data/processed/single_joint_question_val.jsonl \
    --output_file data/processed/single_joint_question_val_outputs_${MODEL_NAME}_${MODEL_SIZE}.jsonl \
    --batch_size 1 \
    --load_in_8bit \
    --num_return_sequences 5 \
    --stop_sequences "Sentence ["