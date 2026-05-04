ray stop -f

CUDA_VISIBLE_DEVICES=4,5 ray start --head --num-gpus=2 --dashboard-host=0.0.0.0

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
      "working_dir": ".",
      "env_vars": {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1"
      }
    }' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node 1 \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 1 \
  --critic.num_nodes 0 \
  --critic.num_gpus_per_node 0 \
  --reward.num_nodes 1 \
  --reward.num_gpus_per_node 0 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size 1 \
  --train.colocate_actor_ref \
  --actor.model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --reward.remote_url ./examples/python/reward_func.py \
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
  --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
  --data.input_key context_messages \
  --data.apply_chat_template \
  --vllm.gpu_memory_utilization 0.5 
