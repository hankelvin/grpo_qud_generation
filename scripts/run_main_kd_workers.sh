conda deactivate
conda activate unsloth

HOSTNAME=$1
tmux new-session -d -s worker1 "torchrun model/main_grpo_worker.py architecture.master.addr=$HOSTNAME role='criteria2' "
tmux new-session -d -s worker2 "torchrun model/main_grpo_worker.py architecture.master.addr=$HOSTNAME role='criteria3' "
tmux new-session -d -s worker3 "torchrun model/main_grpo_worker.py architecture.master.addr=$HOSTNAME role='criteria4' "