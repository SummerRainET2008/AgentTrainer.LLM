export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=5 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1
