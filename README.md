# Train multi-turn Agents with RL in RLHF

<p align="center">
  inspired by AgentFlow</em>
</p>

## Outline

- [What This Repo Is](#what-this-repo-is)
- [Why It Matters](#why-it-matters)
  - [Prompt Comparison](#prompt-comparison)
  - [Training Comparison](#training-comparison)
  - [Pros and Cons](#pros-and-cons)
- [How to modify OpenRLHF to support AgentFlow](#how-to-modify-openrlhf-to-support-agentflow)
- [How to run](#how-to-run)

---

## What This Repo Is

This repository started as an attempt to add AgentFlow as a generic plugin inside OpenRLHF. After implementation, a better direction emerged: keep OpenRLHF unchanged and show a **compatible integration pattern** in a lightweight way.
I also share how I weighed the pros and cons.

Reference paper: [AgentFlow](https://arxiv.org/pdf/2510.05592), an online reinforcement learning implementation that motivates breaking complex tasks into modular agent stages, where each stage has a focused responsibility and shared memory/state.

Some concepts:
- __Multi-turn RL training__ - One trainable LLM serves multiple roles.
  - Such as tool calling (external tools return results), decision-making (whether to stop), final response generation.
    
- **AgentFlow**
  - A set of agents collobarating to achieve one goal. 
  - The training utilizes a modified multi-turn RL and requires significant modification of popular framwork, OpenRLHF, VeRL.
    
- __Multi-turn Agent RL training__
  - Borrowing the concept of agent roles in __AgentFLow__, using the __Multi-turn RL training__. 

This project combines:

- **OpenRLHF-based training** (`openrlhf_agent/`) for scalable RLHF loops (PPO/Ray/vLLM), fixing minor bugs.
- **AgentFlow-style solver logic** (`agentflow/`) 
  - with role separation:
  `Initializer -> Planner -> Executor -> Verifier-->Generalist`. 
  - I simplied and modified AgentFlow inference codes into __Multi-turn Agent RL trainig__ compatible.
- **Training entrypoints** (`train/`, `scripts/`) 
  - demonstrating how to train in OpenRLHF without modifying the framework.

## Why It Matters

I really like the concept of agents from both the user and engineering perspectives. This motivated me to implement it as a generic approach that OpenRLHF users can try directly.

### Prompt Comparison

To make the distinction easier to understand, I first explain the difference in training prompt design. Traditional multi-turn reinforcement learning (multi-turn RL) supports tools and task decomposition, just as AgentFlow does. multi-turn RL training keeps appending new outputs, such as CoT, tool-calling commands, tool-calling results, and other intermediate information, to the current prompt to form the next-turn LLM prompt. Because all historical information is packed into a flat string, I call this the unstructured style.

In comparison, an agent system maintains important information in a key-mapped structure, called memory, during both training and inference, and I call this the structured style. Different agents generate distinct prompts, which helps the LLM understand historical information more effectively.

__Example: a planner's prompt__. 

A planner agent encapsulates history with planning-related instructions, without information for a verifier agent or a solution-generating agent.


```
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.
      Available tools: {available_tools}
      Metadata for the tools: {toolbox_metadata}
      Image: {image_info}
      Query: {question}
      
Instructions:
  1. Carefully read and understand the query and any accompanying inputs.
  2. Identify the main objectives or tasks within the query.
  3. List the specific skills that would be necessary to address the query comprehensively.
  ... 

Your response should include:
  1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
  2. A list of required skills, with a brief explanation for each.
  ...
```

__Example: a verifier's prompt__. 

A verifier agent focuses only on checking whether the current reasoning and results are grounded and sufficient to generate the final result.

```
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
  ...   
  
Response Format:
  ...
  Conclusion: CONTINUE
  IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. 
```      

Let's look back at classic __multi-turn RL training__, where the holistic prompt looks like this:

```
Given a query, available tools, and metadata for tools:
query: {query}
Available tools: {tools}
Metadata for tools: {tools_metadata}

please strictly follow the instruction of using following tasks and output designated results.

task_name=[query_analysis]
Task goal: Analyze the given query to determine necessary skills and tools.

    Instructions:
    1. Identify the main objectives in the query.
    ...

task_name=[next_step_tool_calling]
Task goal: Determine the optimal next step to address the query using available tools and previous steps.

    Instructions:
    1. Analyze the query, previous steps, and available tools.
    ...

task_name=[generate_final_output]
Task: Generate the final output based on the query and the results from all tools used.

    Instructions:
    1. Review the query and the results from all tool executions.
    ...
```

In each turn, we invoke an anxiliary agents (executor, verifier and generalist solution generator) designated by the planner agent to respond to its output. The planner agent is the exclusive one to be __trainable__. The new analysis, execution results, as well as the __next-turn instruction__, are appended to the prompt, which in turn becomes the prompt of the next-turn agent.

```
{analysis, tool execution ... }

[round=?]
Your new task is task_name=[?], your responses are as follows:\n
```    

Hence, each __task_name=[?]__ actually corresponds to an agent transition in AgentFlow. 

In comparision, in AgentFLow the prompt of agent is a __transformation__ of these incremental-style history, such as well-organized history (by maintaining a key-value cache, called __memory__), extraction of the current agent related instructions (each agent's prompt is __distinctive__).

### Training Comparison
We first look at the standard multi-turn inference procedure.
```
prompt1 --> response1 --> prompt2 --> response2 --> prompt3 --> response3 ... promptN --> responseN
```
The __prompt__ is any of the initial prompt (question), tool calling result and other intermediate information, which are not trainable.
The __response__ is direct output from a trainable LLM.

#todo


### Pros and Cons

AgentFlow over multi-turn RL training

#### Pros
- The __first__ one, the AgentFlow is quite __a natural way__ to manupilate an agent in traning and inference, such as a planner, verifier, executor. 
  - See the prompts exemplified above. 

- The __second__ one, the history information is organized into a structural __memory__, which is more efficient for the LLM to manipulate. 
  - For example, when the history is too long, it is straightforward to do __information compression and summarization__ on some key-value information (e.g. tool calling history), without __touching/destroying__ critical key-value information (e.g. __users' prompt and hard requirements__).  
  - In the __multi-turn RL training__, all context information is organized as an unstructural string, which makes LLM harder to distinguish important information from nosie.

- The __third__ one, AgentFlow allows auxiliary agents to alleviate the __workload__ of the planner. 
  - The planner only serves the role of decision maker, not serving verification like a verfier agent does, or generating executor commands like an exeutor agent does, or constructing the final answer like a generalist agent does.
  - In the multi-turn RL training, one trainable LLM serves multiple roles, which proves an ineffective practice. 

#### Cons
- The __first__ one, popular RL frameworks do not naturally support it, requiring heavy engineering work.
- The __second__ one, more GPU cost, due to low KV cache. 
    - When the running turns are not big enough, I conjecture no significant difference with the __multi-turn Agent training__ in performance.

## From Multi-turn RL and AgentFlow to Multi-turn Agent RL 

## How to modify OpenRLHF to support AgentFlow 

todo

## How to run

- `openrlhf_agent/` - RLHF training, datasets, trainers, Ray/vLLM integration
- `agentflow/` - agent pipeline, engines, tools, solver
- `train/` - runnable training scripts and example agent envs
- `scripts/` - shell launchers for common training setups
  

```bash
  >> uv sync
  >> python3 train/quick_start.py
```


---

<p align="center">
  Built for practical research on RLHF-powered autonomous agents.
</p>
