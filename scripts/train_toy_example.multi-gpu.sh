ray stop -f

export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0,3,4 ray start --head --num-gpus=3 --dashboard-host=0.0.0.0

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
      "working_dir": ".",
      "env_vars": {
      }
    }' \
  -- python3 -m openrlhf_agent.cli.train_ppo_ray \
  --train.agent_func_path train/train_simple_agent_example.py \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node 2 \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 2 \
  --critic.num_nodes 0 \
  --critic.num_gpus_per_node 0 \
  --reward.num_nodes 1 \
  --reward.num_gpus_per_node 0 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size 1 \
  --train.colocate_actor_ref \
  --actor.model_name_or_path Qwen/Qwen3-8B \
  --ckpt.output_dir ./checkpoint/smoke-test-grpo \
  --algo.advantage.estimator group_norm \
  --rollout.n_samples_per_prompt 2 \
  --train.micro_batch_size 1 \
  --train.batch_size 2 \
  --rollout.micro_batch_size 1 \
  --rollout.batch_size 2 \
  --data.max_samples 8 \
  --train.max_epochs 1 \
  --data.max_len 512 \
  --rollout.max_new_tokens 32 \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation eager \
  --actor.adam.lr 5e-7 \
  --algo.kl.init_coef 0.01 \
  --data.prompt_dataset data.jsonl \
  --data.input_key prompt \
  --data.apply_chat_template \
  --vllm.gpu_memory_utilization 0.9 
