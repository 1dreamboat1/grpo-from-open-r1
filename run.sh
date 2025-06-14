export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/deepseek-r1-gsm8k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_DEBUG=INFO  # 获取更详细的NCCL错误信息

# 显式设置为单进程，单GPU训练
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500 
export CUDA_VISIBLE_DEVICES=0
# 运行GRPO训练
python src/open_r1/grpo.py \
    --output_dir Qwen-05B-GRPO \
    --model_name_or_path Qwen2-0.5B-Instruct \
    --dataset_name gsm8k/main \
    --max_prompt_length 256 \
    --max_completion_length 165 \
    --per_device_train_batch_size 2 \
    --num_generations 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_strategy steps \
    --learning_rate 3.0e-06 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --eval_strategy no \
    --bf16 \
    --gradient_checkpointing \
    # --use_vllm \
    # --vllm_gpu_memory_utilization 0.7
    #多次尝试后发现使用vllm进行加速推理会超显存，所以暂时不使用vllm