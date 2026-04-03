# python agent_cli.py [test_name]
# (c) Terence Lim
import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior

from agent_delegation import attach_research_delegation_tools
from shared_agents import create_agents
from utils import OUTPUT

TESTS = Path('tests')
K = 5
RETRIES = 2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Quant Research Agent CLI")
    parser.add_argument(
        "test",
        nargs="?",
        help="Optional name of test prompt and outfile",
    )
    return parser.parse_args()


def append_evaluation_log(response: str, evaluation_stem: str, mode: str) -> None:
    payload = {
        "date": str(datetime.now())[:19],
        "response": response,
    }
    evaluation_path = OUTPUT / (evaluation_stem + ".responses")
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_path.open(mode, encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


def build_conversation_context(messages: list[dict[str, str]], max_messages: int = 50) -> str:
    tail = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = ["<conversation_history>"]
    for i, m in enumerate(tail, start=1):
        lines.append(f"  <message index=\"{i}\" role=\"{m['role']}\">")
        lines.append("```text")
        lines.append(m["content"])
        lines.append("```")
        lines.append("  </message>")
    lines.append("</conversation_history>")
    return "\n".join(lines)


async def run_agent_safely(agent: Any, query: str, retries: int = RETRIES) -> str:
    attempts = 0
    while True:
        try:
            response = await agent.run(query)
            return response.output if hasattr(response, "output") else str(response)
        except UnexpectedModelBehavior:
            attempts += 1
            if attempts > retries:
                return "The agent could not complete the request due to model behavior errors."
            await asyncio.sleep(0.4)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(query)
            return f"The agent failed with an unexpected error: {exc}"


async def main() -> None:
    args = parse_args()
    bundle = create_agents()
    research_agent = bundle["research_agent"]
    report_agent = bundle["report_agent"]
    coding_agent = bundle["coding_agent"]

    messages: list[dict[str, str]] = []

    def on_instruction(role: str, message: str) -> None:
        messages.append({"role": role, "content": message})

    def on_result(role: str, message: str) -> None:
        messages.append({"role": role, "content": message})

    attach_research_delegation_tools(
        research_agent=research_agent,
        report_agent=report_agent,
        coding_agent=coding_agent,
        build_context=lambda: build_conversation_context(messages),
        store_conversation=lambda _: None,
        on_instruction=on_instruction,
        on_result=on_result,
        run_agent_safely=lambda a, q, _role, retries=RETRIES: run_agent_safely(a, q, retries=retries),
    )

    if args.test is None:
        print("Quant Research Agent CLI (type 'exit' to quit)")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            prompt = build_conversation_context(messages)
            reply = await run_agent_safely(research_agent, prompt)
            messages.append({"role": "assistant", "content": reply})
            print(f"\nAgent: {reply}")
        return

    print(f"Quant Research Agent CLI (single-query mode: running {K} times)")
    with open(TESTS / (args.test + ".query"), "r", encoding="utf-8") as prompt_file:
        query = prompt_file.read().strip()
    print(f"\nRunning query from {args.test}.query:\n{query}\n")

    mode = 'w'
    for run_index in range(1, K + 1):
        messages = [{"role": "user", "content": query}]
        prompt = build_conversation_context(messages)
        for tries in range(RETRIES):
            reply = await run_agent_safely(research_agent, prompt)
            if any(word not in reply for word in ["MODEL", "ERROR"]):
                break
            print(f'Retrying {tries}/{RETRIES}')
        messages.append({"role": "assistant", "content": reply})
        append_evaluation_log(reply, args.test, mode=mode)
        mode = 'a'
        print(f"\nAgent [{run_index}/{K}]: {reply}")

if __name__ == "__main__":
    asyncio.run(main())
