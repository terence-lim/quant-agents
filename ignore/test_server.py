# python performance_server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, dates_, log_message
from pydantic_ai import Agent   # use Agent within a tool call
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import json
import asyncio
import logging
from pydantic_ai.exceptions import UnexpectedModelBehavior
    
from dotenv import load_dotenv
load_dotenv()

# Create an MCP server

port = 8001
mcp = FastMCP("performance-server", host="0.0.0.0", port=port)

#model = "gemini-2.5-flash"
#model = OpenAIChatModel(model_name="gpt-4.1-mini")
#server_agent = Agent(model=model,
#                     system_prompt='You are a helpful assistant specialized in quantitative finance and portfolio performance evaluation.')

async def run_agent_with_timeout(agent, prompt: str, timeout_s: int = 120) -> str:
    """
    Run the pydantic-ai agent with a timeout + resilient error handling.
    Distinguish between client cancellations and model glitches.
    """
    try:
        # Give OpenAI enough time; keep this larger than your MCP tool timeout.
        res = await asyncio.wait_for(agent.run(prompt=prompt), timeout=timeout_s)
        return res.output if hasattr(res, "output") else str(res)

    except asyncio.CancelledError:
        # The MCP client or framework cancelled the tool call (rerun/timeout/disconnect).
        logging.warning("Agent run was cancelled by the caller (tool timeout/rerun/disconnect).")
        # Re-raise if you want the tool to hard-cancel, OR return a friendly string:
        return "Request was cancelled by the caller (timeout or rerun). Please try again."

    except UnexpectedModelBehavior as e:
        # Gemini/OpenAI occasionally return empty payloads; surface a readable message.
        logging.warning("Model returned unexpected payload: %s", e)
        return ("Agent could not complete the step because the model returned an empty payload. "
                "Please try again or switch models.")

    except Exception as e:
        logging.exception("Agent run failed:")
        return f"Agent error: {e}"


@mcp.tool(timeout=180)
async def Panel_factor_evaluate(factor_panel_id: str, description: str = '') -> str:
    """Provides a prompt with instructions and context to generate a research report 
    for evaluating a Panel of factor characteristic values for predicting stock returns.

    Args:
        factor_panel_id (str): The id of the panel data set for factor characteristic values.
        description (str): A full description and definition of the factor.
    Returns:
        str: A prompt containing instructions and context for generating a research report evaluating the factor.
    """
    # TO DO: 
    # (1) save factor_evaluation to a temp folder 
    # (2) return path as json 
    # (3) include prompt to save report after generation
    description = description.replace('"', "'")  # sanitize quotes
    code = f"""
import json
from qrafti import Panel, factor_evaluate, MEDIA, research_prompt
panel = Panel('{factor_panel_id}', **{dates_})
evaluate_str = factor_evaluate(panel)
#subfolder = 'folder_0'
#output_folder = MEDIA / subfolder
#output_folder.mkdir(parents=True, exist_ok=True)
#with open(output_folder / 'research_memo.md', 'w') as f:
#    f.write(evaluate_str)
#prompt_string = "{description}" + "\\n---------\\n" + research_prompt + "\\n\\n" + evaluate_str
#print(json.dumps(dict(prompt_string=prompt_string, output_folder=subfolder)))
print(evaluate_str)
"""
    # payload = _log_and_execute('Panel_factor_evaluate', code)
    # if 'prompt_string' in payload:
    #     result = json.loads(payload)
    #     prompt_string = result['prompt_string']
    #     log_message('Server Calling Agent With Prompt', 'Panel_factor_evaluate', prompt_string)
    #     #response = await run_agent_with_timeout(server_agent, prompt_string, timeout_s=180)
    #     #response = await server_agent.run(prompt=prompt_string)
    #     response = await asyncio.wait_for(server_agent.run(prompt=prompt), timeout=timeout_s)
    #     response = response.output if hasattr(response, "output") else str(response)
    #     log_message('Server Agent Returned Payload', 'Panel_factor_evaluate', response)
    #     return f"Research Memo:\n\n{response}"
    # else:
    #     return payload        