for model in "deepseek-ai/deepseek-coder-1.3b-base" "bigcode/starcoder2-3b"
do
    for temp in 0.1 0.8
    do
        echo "Running $model with temperature $temp"

        #Generate
        docker run --gpus '"device=6"' -e HUGGING_FACE_HUB_TOKEN=HF_TOKEN \
            -v $(pwd):/app \
            -v /Outputs/BigCodeBench:/Outputs \
            -t bigcodebench/bigcodebench-generate:latest \
            --model $model \
            --subset "complete" \
            --bs 16 \
            --temperature $temp \
            --n_samples 50 \
            --backend "vllm" \
            --tp 1 \
            --save_path "/Outputs/$model/50/$temp/generations.jsonl" \
            --trust_remote_code

        #Sanitize
        python -m bigcodebench.sanitize --samples ../Outputs/BigCodeBench/$model/50/$temp/generations.jsonl

        #Evaluate
        docker run -v $(pwd):/app -v /Outputs/BigCodeBench:/Outputs -t bigcodebench/bigcodebench-evaluate:latest \
            --subset "complete" \
            --samples "/Outputs/$model/50/$temp/generations-sanitized.jsonl" \
            --no-gt \
            --parallel 32 \
            --min-time-limit 4.5 \  
    done
done

