import argparse
import time
import json
from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.models.executor import Executor
from agentflow.models.utils import make_json_serializable_truncated


class Solver:
    def __init__(self,
                 planner,
                 verifier,
                 executor,
                 output_types: str = "base,final,direct",
                 max_steps: int = 10,
                 max_time: int = 300,
                 max_tokens: int = 4000,
                 root_cache_dir: str = "cache",
                 verbose: bool = True,
                 temperature: float = .0):
        self._planner = planner
        self._verifier = verifier
        self._executor = executor
        self._max_steps = max_steps
        self._max_time = max_time
        self._max_tokens = max_tokens
        self._root_cache_dir = root_cache_dir

        self._output_types = output_types.lower().split(',')
        self._temperature = temperature
        assert all(
            output_type in ["base", "final", "direct"]
            for output_type in self._output_types
        ), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self._verbose = verbose

    def solve(self, question: str):
        json_data = {"query": question, "image": None}
        print(f"{question=}")

        context = self._planner.get_init_context(question)
        query_start_time = time.time()

        context = self._planner.analyze_query(context)

        for step_id in range(1, self._max_steps + 1):
            if time.time() - query_start_time >= self._max_time:
                break

            step_start_time = time.time()
            local_start_time = time.time()

            next_step = self._planner.gen_next_step(context)
            sub_goal_context, sub_goal, tool_name = self._planner.extract_subgoal(next_step)
            print(f"Step {step_id}: Action Prediction ({tool_name})\n")
            print(f"[Context]: {sub_goal_context}\n"
                  f"[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
            print(f"[Time]: {time.time() - local_start_time}s")

            if tool_name is None or tool_name not in self._planner.available_tools:
                print(f"\nError: Tool '{tool_name}' is not available or not found.")
                command = "No command was generated because the tool was not found."
                result = "No result was generated because the tool was not found."

            else:
                local_start_time = time.time()
                analysis, explanation, command = self._executor.generate_tool_command(
                    question=question,
                    context=sub_goal_context,
                    sub_goal=sub_goal,
                    tool_name=tool_name,
                    tool_metadata=self._planner.toolbox_metadata[tool_name])

                print(f"\n==> Step {step_id}: Command Generation ({tool_name})\n")
                print(f"[Analysis]: {analysis}\n"
                      f"[Explanation]: {explanation}\n[Command]: {command}")
                print(f"[Time]: {time.time() - local_start_time}s")

                local_start_time = time.time()
                result = self._executor.execute_tools(tool_name, command)
                result = make_json_serializable_truncated(result)
                json_data[f"tool_result_{step_id}"] = result
                context = "\n".join([
                    context,
                    f"{next_step=}",
                    f"{command=}",
                    f"{result=}",
                    f"The current tool calling is done\n\n",
                ])

                print(f"\n==> Step {step_id}: "
                      f"Command Execution ({tool_name})\n")
                print(f"[Result]:\n{json.dumps(result, indent=4)}")
                print(f"[Time]: {time.time() - local_start_time}s")

            local_start_time = time.time()
            verification_analysis, conclusion = self._verifier.verify(
                context)
            print(f"\n==> Step {step_id}: Context Verification\n")
            print(f"[Analysis]: {verification_analysis}\n"
                  f"[Conclusion]: {conclusion}")
            print(f"[Time]: {time.time() - local_start_time}s")

            execution_time_step = time.time() - step_start_time
            print(f"Step[{step_id}]: takes {execution_time_step} seconds.")

            if conclusion == 'STOP':
                context = "\n".join([
                    context,
                    f"Your current task is task_name=[generate_final_output], your responses are as follows:\n",
                ])
                break
            else:
                context = "\n".join([
                    context,
                    f"round={step_id + 1}",
                    f"Your current task is task_name=[next_step_tool_calling], your responses are as follows:\n",
                ])

        final_output = self._planner.gen_final_response(context)
        print(f"\n[Total Time]: {time.time() - query_start_time}s")
        print(f"\n==> Query Solved!")

        return final_output

def construct_solver(
        llm_engine_name: str = "gpt-4o",
        enabled_tools: list[str] = ["all"],
        tool_engine: list[str] = ["Default"],
        model_engine: list[str] = [
            "trainable", "gpt-4o", "gpt-4o", "gpt-4o"
        ],  # [planner_main, planner_fixed, verifier, executor]
        output_types: str = "final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "solver_cache",
        verbose: bool = True,
        vllm_config_path: str = None,
        base_url: str = None,
        temperature: float = 0.0):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, verifier, executor]
    # "trainable" means use llm_engine_name (the trainable model)
    planner_main_engine = llm_engine_name if model_engine[
        0] == "trainable" else model_engine[0]
    planner_fixed_engine = llm_engine_name if model_engine[
        1] == "trainable" else model_engine[1]
    verifier_engine = llm_engine_name if model_engine[
        2] == "trainable" else model_engine[2]
    executor_engine = llm_engine_name if model_engine[
        3] == "trainable" else model_engine[3]

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )

    # Instantiate Planner
    planner = Planner(llm_engine_name=planner_main_engine,
                      toolbox_metadata=initializer.toolbox_metadata,
                      available_tools=initializer.available_tools,
                      verbose=verbose,
                      base_url=base_url,
                      temperature=temperature)

    # Instantiate Verifier
    verifier = Verifier(
        llm_engine_name=verifier_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url if verifier_engine == llm_engine_name else None,
        temperature=temperature)

    # Instantiate Executor with tool instances cache
    executor = Executor(
        llm_engine_name=executor_engine,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        base_url=base_url if executor_engine == llm_engine_name else
        None,  # Only use base_url for trainable model
        temperature=temperature,
        tool_instances_cache=initializer.
        tool_instances_cache  # Pass the cached tool instances
    )

    # Instantiate Solver
    solver = Solver(planner=planner,
                    verifier=verifier,
                    executor=executor,
                    output_types=output_types,
                    max_steps=max_steps,
                    max_time=max_time,
                    max_tokens=max_tokens,
                    root_cache_dir=root_cache_dir,
                    verbose=verbose,
                    temperature=temperature)
    return solver


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name",
                        default="gpt-4o",
                        help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)")
    parser.add_argument("--enabled_tools",
                        default="Base_Generator_Tool",
                        help="List of enabled tools.")
    parser.add_argument("--root_cache_dir",
                        default="solver_cache",
                        help="Path to solver cache directory.")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=4000,
                        help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps",
                        type=int,
                        default=10,
                        help="Maximum number of steps to execute.")
    parser.add_argument("--max_time",
                        type=int,
                        default=300,
                        help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose",
                        type=bool,
                        default=True,
                        help="Enable verbose output.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    tool_engine = ["gpt-4o-mini", "gpt-4o-mini", "Default", "Default"]
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=[
            "Base_Generator_Tool", "Python_Coder_Tool", "Google_Search_Tool",
            "Wikipedia_Search_Tool"
        ],
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        # base_url="http://localhost:8080/v1",
        verbose=args.verbose,
        temperature=0.7)

    # Solve the task or problem
    solver.solve("What is the capital of France?")


if __name__ == "__main__":
    main()
