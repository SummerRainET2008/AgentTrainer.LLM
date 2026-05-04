import os
from agentflow.solver import construct_solver

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY before running quick_start.py")

    solver = construct_solver(
        llm_engine_name="gpt-4o",
        output_types="final,direct",
        max_time=300_000,
    )

    # Use a non-math query so the example highlights end-to-end planning + synthesis.
    # query = "What is the capital of France?"
    query = "What is the date of death of Sobhuza Ii's father?"

    output = solver.solve(query)
    print(f"Final Output: {output}")

if __name__ == "__main__":
    main()