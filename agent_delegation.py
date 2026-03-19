from typing import Awaitable, Callable
from pydantic_ai import RunContext


async def default_run_agent_safely(agent, query: str, role_label: str, retries: int = 1) -> str:
    response = await agent.run(query)
    return response.output if hasattr(response, "output") else str(response)


def attach_research_delegation_tools(
    research_agent,
    report_agent,
    coding_agent,
    build_context: Callable[[], str],
    store_conversation: Callable[[str], None],
    on_instruction: Callable[[str, str], None],
    on_result: Callable[[str, str], None],
    run_agent_safely: Callable[[object, str, str, int], Awaitable[str]] = default_run_agent_safely,
):
    @research_agent.tool
    async def report_agent_tool(ctx: RunContext, panel_id: str, description: str) -> str:
        instruction = (
            "COMMAND: Generate a standardized research report for "
            f"panel_id='{panel_id}' and description='{description}'."
        )
        on_instruction("Research Agent", instruction)

        full_query = build_context()
        store_conversation(full_query)
        out = await run_agent_safely(report_agent, full_query, "Report Agent", retries=1)
        on_result("Report Agent", out)
        return out

    @research_agent.tool
    async def coding_agent_tool(ctx: RunContext, code_str: str = "", task_description: str = "") -> str:
        if code_str.strip() and not task_description.strip():
            instruction = f"COMMAND: Execute Python code:\n{code_str}"
        else:
            instruction = (
                "COMMAND: Write and execute Python code for this task:\n"
                f"{task_description.strip()}\n\n"
                "Starter code (optional):\n"
                f"{code_str.strip()}"
            )

        on_instruction("Research Agent", instruction)
        full_query = build_context()
        store_conversation(full_query)
        out = await run_agent_safely(coding_agent, instruction, "Coding Agent", retries=1)
        on_result("Coding Agent", out)
        return out
