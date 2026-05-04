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

This repository started as an attempt to add AgentFlow as a generic plugin inside OpenRLHF. After implementation, a better direction emerged: keep OpenRLHF unchanged and show a **compatible integration pattern** in a lightweight way.
I also share how I weighed the pros and cons.

Reference paper: [AgentFlow](https://arxiv.org/pdf/2510.05592), an online reinforcement learning implementation that motivates breaking complex tasks into modular agent stages, where each stage has a focused responsibility and shared memory/state.

This project combines:

- **OpenRLHF-based training** (`openrlhf_agent/`) for scalable RLHF loops (PPO/Ray/vLLM).
- **AgentFlow-style solver logic** (`agentflow/`) with role separation:
  `Initializer -> Planner -> Executor -> Verifier`. To better demonstrate the idea, I simplified AgentFlow while keeping the core code.
- **Training entrypoints** (`train/`, `scripts/`) for real multi-GPU workflows. I demonstrate how to train in OpenRLHF without modifying fundamental code, rather than relying on heavy custom code from the original AgentFlow implementation.

## Why It Matters

I really like the concept of agents from both the user and engineering perspectives. This motivated me to implement it as a generic approach that OpenRLHF users can try directly.

### Prompt Comparison

To make the distinction easier to understand, I first explain the difference in training prompt design. Traditional multi-turn reinforcement learning (multi-step RL) supports tools and task decomposition, just as AgentFlow does. Multi-step RL training keeps appending new outputs, such as CoT, tool-calling commands, tool-calling results, and other intermediate information, to the current prompt to form the next-turn LLM prompt. Because all historical information is packed into a flat string, I call this the unstructured style.

In comparison, an agent system maintains important information in a key-mapped structure, called memory, during both training and inference, and I call this the structured style. Different agents generate distinct prompts, which helps the LLM understand historical information more effectively.

__Example__: A planner's prompt. A planner agent encapsulates history with planning-related instructions, without instructions for a verifier agent or a solution-generating agent.

    Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.
      Available tools: {available_tools}
      Metadata for the tools: {toolbox_metadata}
      Image: {image_info}
      Query: {question}

    Instructions:
      1. Carefully read and understand the query and any accompanying inputs.
      2. Identify the main objectives or tasks within the query.
      3. List the specific skills that would be necessary to address the query comprehensively.
      4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. 
         Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
      5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

    Your response should include:
      1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
      2. A list of required skills, with a brief explanation for each.
      3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
      4. Any additional considerations that might be important for addressing the query effectively.

    Please present your analysis in a clear, structured format.



___Example___: A verifier's prompt. A verifier agent focuses only on checking whether the current reasoning and results are grounded and sufficient to generate the final result.

    Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

    Context:
    Query: {question}
    Image: {image_info}
    Available Tools: {available_tools}
    Toolbox Metadata: {toolbox_metadata}
    Initial Analysis: {query_analysis}
    Memory (tools used and results): {memory}

    Detailed Instructions:
      1. Carefully analyze the query, initial analysis, and image (if provided):
         - Identify the main objectives of the query.
         - Note any specific requirements or constraints mentioned.
         - If an image is provided, consider its relevance and what information it contributes.

      2. Review the available tools and their metadata:
         - Understand the capabilities and limitations and best practices of each tool.
         - Consider how each tool might be applicable to the query.

      3. Examine the memory content in detail:
         - Review each tool used and its execution results.
         - Assess how well each tool's output contributes to answering the query.

      4. Critical Evaluation (address each point explicitly):
         a) Completeness: Does the memory fully address all aspects of the query?
            - Identify any parts of the query that remain unanswered.
            - Consider if all relevant information has been extracted from the image (if applicable).

         b) Unused Tools: Are there any unused tools that could provide additional relevant information?
            - Specify which unused tools might be helpful and why.

         c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
            - If yes, explain the inconsistencies and suggest how they might be resolved.

         d) Verification Needs: Is there any information that requires further verification due to tool limitations?
            - Identify specific pieces of information that need verification and explain why.

         e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
            - Point out specific ambiguities and suggest which tools could help clarify them.

      5. Final Determination:
         Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

    Response Format:

      If the memory is complete, accurate, AND verified:
      Explanation:
      <Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

      Conclusion: STOP
      If the memory is incomplete, insufficient, or requires further verification:
      Explanation:
      <Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

      Conclusion: CONTINUE
      IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.


Let's look back at classic ___Multi-step RL training___, where the required prompt looks like this:


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
  

In this holistic prompt, each task_name=[...] actually corresponds to an agent in AgentFlow. The expected LLM output is the next-step task (or agent).

    [round=0]
    Your current task is task_name=[query_analysis], your responses are as follows:\n


This prompt keeps appending new responses, including instructions to call an agent and the agent outputs, in an incremental style.

### Training Comparison

### Pros and Cons


OpenRLHF provides the optimization backbone, while AgentFlow adds structured reasoning across multiple turns.  
Together, they support **long-horizon agent behavior** instead of one-shot responses.

## AgentFlow Paper (Concise)

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
