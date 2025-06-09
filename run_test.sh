# accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=1 src/open_r1/grpo.py \
#     --output_dir Qwen-05B-GRPO \
#     --model_name_or_path Qwen2-0.5B-Instruct \
#     --dataset_name gsm8k/main \
#     --max_prompt_length 512 \
#     --max_completion_length 1024 \
#     --per_device_train_batch_size 2 \
#     --num_generations 2 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --logging_strategy steps \
#     --learning_rate 3.0e-06 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 10 \
#     --eval_strategy no \
#     --bf16 \
#     --use_vllm \
#     # --vllm_device auto \
#     --vllm_gpu_memory_utilization 0.7



# 在 run_test.sh 中，将原来的 accelerate launch 命令替换为：
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/deepseek-r1-gsm8k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/open_r1/grpo.py \
    --output_dir Qwen-05B-GRPO \
    --model_name_or_path Qwen2-0.5B-Instruct \
    --dataset_name gsm8k/main \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --num_generations 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_strategy steps \
    --learning_rate 3.0e-06 \
    --gradient_accumulation_steps 32 \
    --logging_steps 10 \
    --eval_strategy no \
    --bf16 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.7