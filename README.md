<div align="center">

# ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2403.03894-b31b1b.svg)](https://arxiv.org/abs/2403.03894)
[![ObscuraX on HuggingFace datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-ObscuraX-yellow?style=flat)](https://huggingface.co/datasets/ObscuraCoder/ObscuraX)

</div>

> **Abstract:**
>
> Language models (LMs) have become a staple of the code-writing toolbox. Their pre-training recipe has, however, remained stagnant over recent years, barring the occasional changes in data sourcing and filtering strategies. In particular, research exploring modifications to Code-LMs' pre-training objectives, geared towards improving data efficiency and better disentangling between syntax and semantics, has been noticeably sparse, especially compared with corresponding efforts in natural language LMs. In this work, we examine grounding on obfuscated code as a means of helping Code-LMs look beyond the surface-form syntax and enhance their pre-training sample efficiency. To this end, we compile ObscuraX, a dataset of approximately 55M source and obfuscated code pairs in seven languages. Subsequently, we pre-train ObscuraCoder models, ranging in size from 255M to 2.8B parameters, on a 272B-token corpus that includes ObscuraX and demonstrate that our obfuscation-based pre-training recipe leads to consistent improvements in Code-LMs' abilities compared to both vanilla autoregressive pre-training as well as existing de-obfuscation (DOBF) objectives. ObscuraCoder demonstrates sizeable gains across multiple tests of syntactic and semantic code understanding, along with improved capabilities in multilingual code completion, multilingual code commit summarization, and multi-purpose library-oriented code generation.
>
Contact person: [Indraneil Paul](mailto:indraneil.paul@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

This repo contains the code accompanying ICLR 25 submission ObscuraCoder. It includes the instructions for loading the ObscuraX dataset, the evaluation sandbox recipes, the code for continued pre-training and the zero-shot tasks evaluation. We also package our custom obfuscator whcih we used to create ObscuraX.

## Setup and Workflow

For the stages involving evaluation, one can setup the evaluation sandboxed environment using the following commands:
>
```bash
docker build -t obscuracoder-evaluation:latest - < Dockerfiles/RECIPE_NAME.Dockerfile
docker run -it --gpus all --name obscuracoder-evaluation obscuracoder-evaluation:latest
```
>
>
## Continued Pre-Training

We provide the code to run the continued pre-training of the IR-based models. The script is located in the `Train_Code+Utilities` directory. It requires a HuggingFace dataset and a HuggingFace model to run. Create a pairwise dataset of source code and obfuscated code and upload it to the HuggingFace datasets.
>
The script is named `Continued_Pretrain.py`. The script is designed to be run as follows:
>
```bash
accelerate launch --num_processes=4 --main_process_port=29699 Train_Code+Utilities/Continued_Pretrain.py \
    --dataset_name "YOUR_DATASET_NAME" \
    --token "YOUR_HF_TOKEN" \
    --wandb_token "YOUR_WANDB_TOKEN" \
    --project_name "YOUR_PROJECT_NAME" \
    --run_name "YOUR_RUN_NAME" \
    --do_eval True \
    --do_train True \
    --trust_remote_code True \
    --low_cpu_mem_usage True \
    --gradient_accumulation_step 2 \
    --optim "adamw_apex_fused" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --tf32 True \
    --logging_steps 100 \
    --logging_strategy "steps" \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --max_train_samples 256000 \
    --max_eval_samples 8192 \
    --model_name_or_path "bigcode/starcoderbase-1b" \
    --num_train_epochs 1.0 \
    --output_dir "YOUR_OUTPUT_DIR" \
    --overwrite_output_dir True \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --preprocessing_num_workers 12 \
    --report_to "wandb" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --seed 484 \
    --validation_split_percentage 10 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --llm_int8_threshold 6.0 \
    --lora_alpha 128 \
    --lora_r 256 \
    --lora_dropout 0.05 \
    --deepspeed "YOUR_DEEPSPEED_CONFIG"
```
>
>
## Zero-shot Tasks Evaluation

We take inspiration from the [vllm-code-harness](https://github.com/iNeil77/vllm-code-harness) library to run the zero-shot tasks evaluation. This allows us to speed up evaluations thus allowing for the extensive experiments in the paper.
>
We provide the code to run the zero-shot tasks evaluation. The script is located in the `Evaluation_Scripts` directory. The tasks include CodeXGLUE code to text, Multipl-E, HumanEvalPack-FixDocs and ReCode. We scripts are named `codexglue_code_to_text.sh`, `multipl_e.sh`, `huma_eval_pack_fixdocs.sh` and `recode.sh` respectively. The scripts already have the hyperparameters used in the paper and are designed to be run directly. For example, to run the CodeXGLUE code to text task, run the following command:
>
```bash
./Train+Inference_Scripts/Commit_Chronicle.sh
```
>
>
## Commit Chronicle Training and Evaluation

We provide the code to run the commit chronicle training and evaluation. It requires the runner to make the dataset available on HuggingFace datasets, split by language. The script is located in the `Train_Code+Utilities` directory. The script is named `commitchronicle_train.py`. The script is designed to be run as follows:
>
```bash
for language in "Ruby" "Objective-C" "Swift" "Rust" "Go" "C" "C++" "Python"
do
    python /Train_Code+Utilities/Commit_Chronicle.py \
        --model_name_or_path "iNeil77/codellama-7b-hf-irv-400" \
        --token "YOUR_HF_TOKEN" \
        --wandb_token "TOUR_WANDB_TOKEN" \
        --hf_data_path "YOUR_DATASET_PATH" \
        --language $language \
        --project_name "LLVM_Align" \
        --run_name "YOUR_RUN_NAME_$language" \
        --output_dir "YOUR_OUTPUT_DIR/$language" \
        --do_train True \
        --do_predict True \
        --trust_remote_code True \
        --low_cpu_mem_usage True \
        --num_train_epochs 2 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --optim "adamw_apex_fused" \
        --learning_rate 3e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 50 \
        --dataloader_drop_last True \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory True \
        --dataloader_persistent_workers True \
        --ddp_find_unused_parameters False \
        --llm_int8_threshold 6.0 \
        --lora_alpha 32 \
        --lora_r 64 \
        --lora_dropout 0.1 \
        --tf32 True \
        --model_max_length 768 \
        --max_train_samples 30720 \
        --max_eval_samples 2560 \
        --max_predict_samples 2560
done
```
>
>
## ObscuraX Dataset
>
The ObscuraX dataset comprises source code and onfuscated code pairs generated from accepted and de-duped programming contest solutions. The dataset is divided into language configs and mode splits. The language can be one of `c`, `cpp`, `go`, `java`, `python`, `rust` and `typescript`, indicating the source files' languages. Once you have submitted an access request which has been approved, loading the dataset can be done as follows:
>
```python
from datasets import load_dataset
dataset = load_dataset("ObscuraX", "c")
```
>
>
## Experimental Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
>
>
## Citation

```bib
@article{paul2025obscuracoder,
  title = {ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding},
  author = {Paul, Indraneil and Yang, Haoyi and Glava\v{s}, Goran and Kersting, Kristian and Gurevych, Iryna},
  year = 2025,
  month = feb,
  journal = {arXiv preprint},
  url = {https://arxiv.org/abs/xxxx.xxxxx},
  eprint = {xxxx.xxxxx},
  archiveprefix = {arXiv},
  primaryclass = {cs.AI},
}
```
