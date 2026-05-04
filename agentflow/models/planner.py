import json
import os
import re
import functools
from typing import Any, Dict, List, Tuple
from PIL import Image
from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import NextStep, QueryAnalysis

_init_prompt_tpt = '''
Given a query, available tools, and metadata for tools:
    query: {query}
    Available tools: {tools}
    Metadata for tools: {tools_metadata}

please strictly follow the instruction of using following tasks and output designated results.

task_name=[query_analysis]
    Task goal: Analyze the given query to determine necessary skills and tools.

    Instructions:
    1. Identify the main objectives in the query.
    2. List the necessary skills and tools.
    3. For each skill and tool, explain how it helps address the query.
    4. Note any additional considerations.

    Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

    Be biref and precise with insight. 
    This is the end of task_name=[query_analysis]
    
 task_name=[next_step_tool_calling]
    Task goal: Determine the optimal next step to address the query using available tools and previous steps.

    Instructions:
    1. Analyze the query, previous steps, and available tools.
    2. Select the **single best tool** for the next step.
    3. Formulate a specific, achievable **sub-goal** for that tool.
    4. Provide all necessary **context** (data, file names, variables) for the tool to function.

    Response Format:
    1.  **Justification:** Explain your choice of tool and sub-goal.
    2.  **Context:** Provide all necessary information for the tool.
    3.  **Sub-Goal:** State the specific objective for the tool.
    4.  **Tool Name:** State the exact name of the selected tool.

    Rules:
    - Select only ONE tool.
    - The sub-goal must be directly achievable by the selected tool.
    - The Context section must contain all information the tool needs to function.
    - The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.
    This is the end of task_name=[next_step_tool_calling]
    
task_name=[generate_final_output]
     Task: Generate the final output based on the query and the results from all tools used.

    Context:
    **Query:** question
    **history tools called and outputs:** 

    Instructions:
    1. Review the query and the results from all tool executions.
    2. Incorporate the relevant information to create a coherent, step-by-step final output.
    This is the end of task_name=[generate_final_output]
    
[round=0]
Your current task is task_name=[query_analysis], your responses are as follows:\n
'''

class Planner:
    def __init__(self,
                 llm_engine_name: str,
                 toolbox_metadata: dict = None,
                 available_tools: List = None,
                 verbose: bool = False,
                 base_url: str = None,
                 is_multimodal: bool = False,
                 temperature: float = .0):
        self.llm_engine_name = llm_engine_name
        self.is_multimodal = is_multimodal
        if llm_engine_name in [None, ""]:
            self._llm_engine = None
        else:
            self._llm_engine = create_llm_engine(
                model_string=llm_engine_name,
                is_multimodal=False,
                base_url=base_url,
                temperature=temperature)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []

        self.verbose = verbose

    def get_init_context(self, query):
        return _init_prompt_tpt.format(query=query, tools=self.available_tools,
                                       tools_metadata=self.toolbox_metadata)

    def analyze_query(self, context) -> str:
        query_analysis = self._llm_engine([context], response_format=QueryAnalysis)
        query_analysis = str(query_analysis).strip()
        print(f"{query_analysis=}\n")

        return "\n".join([
            context,
            f"{query_analysis}",
            "The responses are done.\n\n",
            "round=1",
            "Your current task is task_name=[next_step_tool_calling], your responses are as follows:\n",
            ])

    def extract_subgoal(self, response: Any) -> Tuple[str, str, str]:
        def normalize_tool_name(tool_name: str) -> str:
            """
            Normalizes a tool name robustly using regular expressions.
            It handles any combination of spaces and underscores as separators.
            """

            def to_canonical(name: str) -> str:
                # Split the name by any sequence of one or more spaces or underscores
                parts = re.split('[ _]+', name)
                # Join the parts with a single underscore and convert to lowercase
                return "_".join(part.lower() for part in parts)

            normalized_input = to_canonical(tool_name)

            for tool in self.available_tools:
                if to_canonical(tool) == normalized_input:
                    return tool

            return f"No matched tool given: {tool_name}"

        try:
            if isinstance(response, str):
                # Attempt to parse the response as JSON
                try:
                    response_dict = json.loads(response)
                    response = NextStep(**response_dict)
                except Exception as e:
                    print(f"Failed to parse response as JSON: {str(e)}")
            if isinstance(response, NextStep):
                print("arielg 1")
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                print("arielg 2")
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"

                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()
            tool_name = normalize_tool_name(tool_name)
        except Exception as e:
            print(
                f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

        return context, sub_goal, tool_name

    def gen_next_step(self, context):
        next_step = self._llm_engine(context, response_format=NextStep)
        return next_step

    def gen_final_response(self, context):
        final_output = self._llm_engine(context)

        return final_output
