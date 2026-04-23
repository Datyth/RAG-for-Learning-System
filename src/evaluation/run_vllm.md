CUDA_VISIBLE_DEVICES=1 uv run python -m vllm.entrypoints.openai.api_server \
    --model /mnt/pretrained_fm/Qwen_Qwen3-4B-Instruct-2507 \
    --port 8001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --trust-remote-code