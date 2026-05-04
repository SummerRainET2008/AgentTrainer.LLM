from pydantic import BaseModel

_prompt_tpt = '''
    Concise Summary: {concise_summary}

    Required Skills: {required_skills}

    Relevant Tools: {relevant_tools}

    Additional Considerations: {additional_considerations}
'''

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return _prompt_tpt.format(
            concise_summary=self.concise_summary,
            required_skills=self.required_skills,
            relevant_tools=self.relevant_tools,
            additional_considerations=self.additional_considerations,
        )

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str


# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool


# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
