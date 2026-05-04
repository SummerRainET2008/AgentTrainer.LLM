"""OpenRLHF multi-turn agent that mirrors `Solver.solve()` logic.

This example customizes `AgentInstanceBase` and `MultiTurnAgentExecutor` and
follows the same high-level flow used in `agentflow/solver.py`:

1) analyze query
2) policy predicts next action (context/sub-goal/tool)  <-- trainable
3) environment generates tool command + executes tool    <-- frozen
4) environment verifies memory (STOP/CONTINUE)           <-- frozen
5) policy generates final answer and gets reward         <-- trainable
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional
import json
import torch
from openrlhf_agent.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from agentflow.models.executor import Executor
from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.utils import make_json_serializable_truncated
from agentflow.models.verifier import Verifier
from enum import Enum

class Phase(Enum):
    QUERY_ANALYSIS      = "phase:query_analysis"
    PREDICT_ACTION      = "phase:predict_action"
    GEN_FINAL_ANASWER   = "phase:final_answer"

def _score_answer(prediction: str, label: Any) -> float:
    """
    For demonstration: just call OpenAI models for a verification.
    """
    # todo
    return 0.0

class SolverStyleAgentInstance(AgentInstanceBase):
    """A stateful OpenRLHF environment instance that mirrors `Solver.solve()`."""

    def __init__(self, *args, **kwargs):
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        frozen_model_name = kwargs.get("frozen_model_name", os.getenv("FROZEN_MODEL_NAME", model_name))
        vllm_base_url = kwargs.get("vllm_base_url", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
        vllm_api_key = kwargs.get("vllm_api_key", os.getenv("VLLM_API_KEY", "dummy-token"))
        if not str(frozen_model_name).startswith("vllm-"):
            frozen_model_name = f"vllm-{frozen_model_name}"

        # Ensure all frozen engines (planner/verifier/executor internals) hit the vLLM server.
        os.environ["VLLM_BASE_URL"] = vllm_base_url
        os.environ["VLLM_API_KEY"] = vllm_api_key
        self._max_steps = int(kwargs.get("max_steps", 4))
        self._verbose = bool(kwargs.get("verbose", False))
        self._enabled_tools = kwargs.get(
            "enabled_tools",
            ["Base_Generator_Tool", "Python_Coder_Tool"],
        )
        tool_engine = kwargs.get("tool_engine", ["Default"] * len(self._enabled_tools))

        initializer = Initializer(
            enabled_tools=self._enabled_tools,
            tool_engine=tool_engine,
            verbose=self._verbose,
        )
        self._planner = Planner(
            llm_engine_name="",
            toolbox_metadata=initializer.toolbox_metadata,
            available_tools=initializer.available_tools,
            verbose=self._verbose,
            base_url=vllm_base_url,
        )
        self._verifier = Verifier(
            llm_engine_name=frozen_model_name,
            toolbox_metadata=initializer.toolbox_metadata,
            available_tools=initializer.available_tools,
            verbose=self._verbose,
            base_url=vllm_base_url,
        )
        self._executor = Executor(
            llm_engine_name=frozen_model_name,
            verbose=self._verbose,
            base_url=vllm_base_url,
            tool_instances_cache=initializer.tool_instances_cache,
        )

    async def reset(self, states: dict, **kwargs):
        # Equivalent to Solver.solve(): setup + analyze_query before action loop.
        self._question = str(states.get("observation", "")).strip()
        self._round_id = 0
        self._phase = Phase.QUERY_ANALYSIS

        return {
            "observation": self._planner.get_init_context(self._question)
        }

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        context = str(states.get("observation", "")).strip()
        action_text = str(states.get("action_text", ""))
        reward_value = 0.0
        done = False

        if self._phase is Phase.QUERY_ANALYSIS:
            self._phase = Phase.PREDICT_ACTION
            env_feedback = "\n".join([
                "The responses are done.\n\n",
                "round=1",
                "Your current task is task_name=[next_step_tool_calling], your responses are as follows:\n",
            ])

        elif self._phase is Phase.PREDICT_ACTION:
            # todo: update OpenRLHF_agent to support predefined response-format
            next_step = json.loads(action_text)
            sub_goal_context, sub_goal, tool_name = self._planner.extract_subgoal(next_step)
            if not context or not sub_goal or not tool_name:
                env_feedback = (
                    "Parsing failed. Respond using:\n"
                    "Context: ...\nSub-Goal: ...\nTool Name: ...\n\nAssistant: "
                )
            else:
                if tool_name not in self._planner.available_tools:
                    env_feedback = "\n".join([
                        f"Tool '{tool_name}' is not available."
                        f"The current tool calling is done\n\n"])
                else:
                    _, _, command = self._executor.generate_tool_command(
                        question=self._question,
                        context=sub_goal_context,
                        sub_goal=sub_goal,
                        tool_name=tool_name,
                        tool_metadata=self._planner.toolbox_metadata[tool_name],
                    )
                    result = self._executor.execute_tools(tool_name, command)
                    result = make_json_serializable_truncated(result)
                    env_feedback = "\n".join([
                        f"{command=}",
                        f"{result=}",
                        f"The current tool calling is done\n\n",
                    ])

                analysis, conclusion = self._verifier.verify(context + action_text + env_feedback)
                if conclusion == "STOP" or self._round_id >= self._max_steps:
                    self._phase = Phase.GEN_FINAL_ANASWER
                    env_feedback = f"Your current task is task_name=[generate_final_output], your responses are as follows:\n"

                else:
                    self._phase = Phase.PREDICT_ACTION
                    env_feedback = "\n".join([
                        f"round={self._round_id + 1}",
                        f"Your current task is task_name=[next_step_tool_calling], your responses are as follows:\n",
                    ])

        elif self._phase is Phase.GEN_FINAL_ANASWER:
            reward_value = _score_answer(action_text, states.get("label", ""))
            done = True
            env_feedback = ""

        else:
            done = True
            env_feedback  = "\n\nHuman: [ERROR] Invalid phase\n"

        reward = torch.tensor(reward_value, dtype=torch.float32)
        self._round_id += 1
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": env_feedback ,
            "done": done,
            "sampling_params": None,
            "extra_logs": {},
        }

class AgentExecutor(MultiTurnAgentExecutor):
    """OpenRLHF entrypoint class for --train.agent_func_path."""

    def __init__(self):
        super().__init__(SolverStyleAgentInstance)
