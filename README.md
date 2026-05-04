# OpenRLHF AgentTrainer + AgentFlow

<p align="center">
  <em>Train multi-step LLM agents with RLHF, inspired by AgentFlow-style task decomposition.</em>
</p>

## Outline

- [What This Repo Is](#what-this-repo-is)
- [Why It Matters](#why-it-matters)
- [AgentFlow Paper (Concise)](#agentflow-paper-concise)
- [Quick Start](#quick-start)
- [Main Paths](#main-paths)

---

## What This Repo Is

This repository started as an attempt to add AgentFlow as a generic plugin inside OpenRLHF. After implementation, the better direction was to keep OpenRLHF unchanged and show a **compatible integration pattern** in a lightweight way.
I would show you how I weigh the pros and cons.

Reference paper: [AgentFlow](https://arxiv.org/pdf/2510.05592)

This project combines:

- **OpenRLHF-based training** (`openrlhf_agent/`) for scalable RLHF loops (PPO/Ray/vLLM).
- **AgentFlow-style solver logic** (`agentflow/`) with role separation:
  `Initializer -> Planner -> Executor -> Verifier`.
- **Training entrypoints** (`train/`, `scripts/`) for real multi-GPU workflows.

## Why It Matters

OpenRLHF provides the optimization backbone, while AgentFlow adds structured reasoning across multiple turns.  
Together, they support **long-horizon agent behavior** instead of one-shot responses.

## AgentFlow Paper (Concise)

The AgentFlow paper motivates breaking complex tasks into modular agent stages, where each stage has a focused responsibility and shared memory/state.  
This repository mirrors that framing in code so RL can optimize not just response quality, but **end-to-end task execution quality**.

## Quick Start

```bash
pip install "openrlhf[vllm]" --no-build-isolation
python train/quick_start.py
```

## Main Paths

- `openrlhf_agent/` - RLHF training, datasets, trainers, Ray/vLLM integration
- `agentflow/` - agent pipeline, engines, tools, solver
- `train/` - runnable training scripts and example agent envs
- `scripts/` - shell launchers for common training setups

---

<p align="center">
  Built for practical research on RLHF-powered autonomous agents.
</p>
