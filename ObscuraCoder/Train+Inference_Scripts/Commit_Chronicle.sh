mamba activate Env

for model in "BASE_CP_OBF_255M" "BASE_CP_OBF_491M" "BASE_CP_OBF_1229M"
do
    for lang in "C" "C++" "Go" "Java" "Python" "Rust" "TypeScript"
    do 
        python /Code/Commit_Chronicle.py \
            --model_name_or_path "$model" \
            --token "HF_TOKEN" \
            --wandb_token "WANDB_TOKEN" \
            --hf_data_path "commit-chronicle" \
            --language "$lang" \
            --project_name "Obscura" \
            --run_name "CC-$model-$lang" \
            --output_dir "/Outputs/CommitChronicle/$model/$lang" \
            --do_train True \
            --do_predict True \
            --trust_remote_code True \
            --low_cpu_mem_usage True \
            --num_train_epochs 2 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "epoch" \
            --save_strategy "epoch" \
            --optim "adamw_torch_fused" \
            --learning_rate 3e-4 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 200 \
            --dataloader_drop_last True \
            --dataloader_num_workers 4 \
            --dataloader_pin_memory True \
            --data_processing_workers 32 \
            --dataloader_persistent_workers True \
            --ddp_find_unused_parameters False \
            --llm_int8_threshold 6.0 \
            --lora_alpha 32 \
            --lora_r 64 \
            --lora_dropout 0.1 \
            --tf32 True \
            --model_max_length 1024 \
            --max_train_samples 256000 \
            --max_eval_samples 25600 \
            --max_predict_samples 76800
    done
done
