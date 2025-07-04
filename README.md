## Generating Questions Under Discussion with GRPO post-training
This repository holds the data, code and models in the paper, "[Generating Questions Under Discussion with Reinforcement Learning using Ranking and Scoring for Reward and Evaluation](https://github.com/hankelvin/grpo_qud_generation/tree/main/paper/grpo_qud_generation.pdf)".

### Preliminaries
#### 1. Environment set-up
```
conda create -n unsloth python==3.10 -y
conda activate unsloth
python -m pip install -r requirements.txt
python -m pip install --upgrade pillow
python -m pip install git+https://github.com/huggingface/trl.git@e5ae703d352b29537159180087ef8bd4b41bf625
python -m spacy download en_core_web_sm en_core_web_lg
```

#### 2. Data sources
- [Discourse Comprehension: A Question Answering Framework to Represent Sentence Connections](https://aclanthology.org/2022.emnlp-main.806/)  -- [repository](https://github.com/wjko2/DCQA-Discourse-Comprehension-by-Question-Answering)
- [QUDeval: Evaluating Questions Under Discussion Discourse Parsing](https://aclanthology.org/2023.emnlp-main.325/) -- [repository](https://github.com/lingchensanwen/QUDeval)
- [TED-Q: TED Talks and the Questions they Evoke](https://aclanthology.org/2020.lrec-1.141/) -- [repository](https://github.com/amore-upf/ted-q)
- [SCRS outputs we obtained for reward model knowledge distillation]()

____
### PART I:  Reference-free QUD evaluation
- Run approaches without CoT/SCRS
    ```
    ./scripts/run_main_phase1.sh False  False   False   False   'small' 'null'
    ```
- Run approaches with CoT/SCRS (if applicable)
    ```
    ./scripts/run_main_phase1.sh True   True    True    True    'small' 'null'
    ```

____ 
### PART II: GRPO for QUD generation
#### a. Obtaining SCRS outputs for reward model knowledge distillation
- cold-start with RB-NLI for reward model (replace "??" with your setting)
    ```
    HOSTNAME="??" # the IP address of the node for the master 
    ./scripts/run_main_kd.sh ${HOSTNAME} null 500 1e9 1 False ''
    ```
- continue with GPT4o for reward modeling (NOTE: 1x GPU for master node, with 3x worker processes for GPT4o API calls)
    ```
    HOSTNAME="??" # the IP address of the node for the master 
    CKPTPATH="??" # the 500th-step checkpoint at the end of the RB-NLI cold-start just above
    ./scripts/run_main_kd_workers.sh ${HOSTNAME} 
    ./scripts/run_main_kd.sh ${HOSTNAME} ${CKPTPATH} 2000 0 1 False ''
    ```

#### b. Training Qwen supervised fine-tuned reward model (Qwen SFT RM)
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python  model/main_grpo_master.py ranker_args.do_cot=True ranker_args.cot_json=True ranker_args.cot_fine=True ranker_args.add_task_decomp_cot=False do_grpo=False grpo_settings.grpo_task=rankllm grpo_settings.gen_bsz=2 grpo_settings.lora_rank=128 save_steps=500 reward_funcs_version=2 flash_attn_override='eager'  
```

#### c. Training Classifiers 
```
python model/train_qudselect_classifiers.py
```

#### d. Obtaining QUDSelect baselines
```
./scripts/train_qud_select.sh
```

#### e. Post-training for QUD generation
- Cold-start with RB-NLI for reward modeling 
    ```
    HOSTNAME="??"
    MASTERPORT="??"
    ./scripts/run_main_grpo_coldstart.sh llama  $HOSTNAME 
    ./scripts/run_main_grpo_coldstart.sh qwen   $HOSTNAME 
    ```
- With Qwen SFT RM for reward modeling
    ```
    HOSTNAME="??"
    MASTERPORT="??"
    RANKPEFTCKPT="??" # the 6000th-step checkpoint from `b. Training Qwen supervised fine-tuned reward model (Qwen SFT RM)`
    CUDA_DEVICE_MASTER=0

    # launch reward models on worker nodes
    ./scripts/run_main_grpo_workers.sh $HOSTNAME $MASTERPORT $RANKPEFTCKPT
    
    # launch GRPO post training for qwen
    QUDPEFTCKPT="" # the checkpoint at the end of the preceding RB-NLI cold-start phase (for qwen)
    ./scripts/run_main_grpo_coldstart.sh qwen   $HOSTNAME $MASTERPORT $QUDPEFTCKPT $CUDA_DEVICE_MASTER
    
    # launch GRPO post training for llama
    QUDPEFTCKPT="" # the checkpoint at the end of the preceding RB-NLI cold-start phase (for llama)
    ./scripts/run_main_grpo_coldstart.sh llama  $HOSTNAME $MASTERPORT $QUDPEFTCKPT $CUDA_DEVICE_MASTER
    ```

#### f. running evaluation
```
./scripts/evaluate_part2.sh
```

____ 
### Models 
Our trained models can be found here:
- Knowledge-distilled (SFT) reward model (Qwen): [folder](https://drive.google.com/drive/folders/1Uh-VBdo37c4O1sA4LxfVp9K5QnKd7-_m?usp=drive_link)
- QUD generation (Llama): [folder](https://drive.google.com/drive/folders/1dV4e5Ky_65PcmBgDH_MmVQWvHWWbElmn?usp=drive_link)
- QUD generation (Qwen): [folder](https://drive.google.com/drive/folders/1EJ8XuUPJfGFUMyOLfa_WLi3q8mUaRhuO?usp=drive_link)
____

If you find our work useful, please cite:
```
@misc{han-gardent-2025-generating-quds,
    title = "Generating Questions Under Discussion with Reinforcement Learning using Ranking and Scoring for Reward and Evaluation",
    author = "Han, Kelvin and Gardent, Claire",
    month = may,
    url = "https://github.com/hankelvin/grpo_qud_generation",
    year = "2025",}
```