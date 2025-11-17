# streamlit run st_client.py --server.fileWatcherType="poll"
import os
import streamlit as st
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.mcp import MCPServerStreamableHTTP
#import logfire

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load env vars
load_dotenv()

# Configure logging
#logfire.configure()
#logfire.instrument_pydantic_ai()
import logging
logging.basicConfig(level=logging.DEBUG)

def log_conversation(debug_text: str = '', mode: str = "a"):
    if mode == "a":
        with (open("debug_query.txt", "a") as f):
            f.write(f"--- {str(datetime.now())} ---\n")
            f.write(debug_text.replace("\nUser:", f"\n\n{'-' * 40}\nUser:"))
            f.flush()
    else:
        with (open("debug_query.txt", "w") as f):
            f.write(f"=== {str(datetime.now())} ===\n")
            f.write(debug_text.replace("\nUser:", f"\n\n{'-' * 40}\nUser:"))
            f.flush()
log_conversation(mode="w")

# Settings
always_show_manager = False

# Cheatsheet and examples for using Panel API, to be injected into coding_agent prompt
PANEL_CHEATSHEET = """
You are writing Python that manipulates a custom `Panel` API for cross-sectional/time-series equity data,
and executing Python code safely in a sandbox using the tools provided to you.
At the end of any code you write, always return either the name of the final `Panel` you constructed,
or a filename in the MEDIA folder path (imported from qrafti package) of the image you plotted.
A `Panel` wraps a `pandas.DataFrame` indexed by `(date, stock)` (2-level), or by `date` only (1-level),
and exposes vectorized operators and groupwise helpers designed for factor research and portfolio construction.
Use the following reference.

CORE OBJECT
- `Panel(name: str='')` → loads a cached dataset by `name`, optionally date-filtered.
  `str(p)` returns panel name in a JSON string which can be printed to standard output to return to the caller.
  `p.frame` is the underlying DataFrame or scalar for 0-level.
  Index names: `DATE_NAME='eom'`, `STOCK_NAME='permno'`

CONSTRUCTION & PERSISTENCE
- `p.copy()`,
  `p.set(value, index: Panel|None)` (broadcast value over `index`),
  `p.set_frame(df, append=False)` (sorts, de-dups, enforces index names),
  `p.persist(name='')` (persists on disk and returns self; name auto-generated if blank).

ARITHMETIC / LOGICAL / MATRIX OPS (AUTO-ALIGN)
- Binary: `+ - * /` (with sensible join/fill),
  comparisons `== != > >= < <=` (inner join),
  logical `|` (outer) and `&` (inner).
  Unary: `-p` (negate), `~p` (boolean NOT).
- Dot product by date: `p1 @ p2` → per-date sum over stocks of first columns (used for weights × returns).

TIME, FILTERING, PLOTTING
- `p.shift(k)` shifts dates using a calendar.
- `p.filter(..., mask=..., index=...)` slices/_masks_.
- `p.plot(other_panel=None, **kwargs)` (joins when needed),
   or use `.frame` with pandas plotting. (See examples below for scatter/cumsum.)

JOIN & APPLY (DATE-GROUP AWARE)
- `p.join_frame(other, fill_value, how)` aligns another `Panel` or scalar.
- `p.apply(func, reference: Panel=None, fill_value=0, how='left', **kwargs)`
   groups by date for 2-level panels and applies `func(DataFrame)->Series`;
   Use `reference` to join an auxiliary column before applying.

EXAMPLE OF HELPER (used with `apply` / `trend`)
- `portfolio_weights(x, leverage=1.0, net=True)` → scales long/short weights to target leverage; last column is inclusion mask.
```python
def portfolio_weights(x, leverage: float = 1.0, net: bool = True) -> pd.Series:
    x.loc[~x.iloc[:, 1].astype(bool), x.columns[0]] = 0.0
    long_weight = x.loc[x.iloc[:,0] > 0, x.columns[0]].sum()
    short_weight = x.loc[x.iloc[:,0] < 0, x.columns[0]].sum()
    if net:
        total_weight = abs(long_weight + short_weight)
    else:
        total_weight = (abs(long_weight) + abs(short_weight)) / 2
    if total_weight == 0:
        return x.iloc[:, 0].rename(x.columns[0])
    return x.iloc[:, 0].mul(abs(leverage)).div(total_weight).rename(x.columns[0])
```
- `digitize(x, cuts, ascending=True)` → quantile/bin labels using mask in last column.
```python
def digitize(x, cuts: int | List[float], ascending: bool = True) -> pd.Series:
    if is_list_like(cuts):
        q = np.concatenate([[0], cuts, [1]])
    else:
        q = np.linspace(0, 1, cuts + 1)
    breakpoints = x.loc[x.iloc[:,1].astype(bool), x.columns[0]].quantile(q=q).values
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ranks = pd.cut(x.iloc[:,0], bins=breakpoints, labels=range(1, len(breakpoints)), include_lowest =True)
    if not ascending:
        ranks = len(breakpoints) - ranks.astype(int) + 1
    return ranks.astype(int)
```

CANONICAL USAGE EXAMPLES (WRITE CODE LIKE THIS)

1) Bucket a signal into terciles and form a value-weighted long-short spread:
```python
from qrafti import Panel
import numpy as np
signal = Panel("ret_12_1")
quantiles = signal.apply(digitize, fill_value=True, cuts=3)
capvw = Panel("CAPVW", **dates).filter(index=signal)
long_w = capvw.apply(portfolio_weights, reference=(quantiles==3), how="right")
short_w = capvw.apply(portfolio_weights, reference=(quantiles==1), how="right")
portfolio = (long_w - short_w).persist()
return str(portfolio)  # returns saved name of the panel as a JSON string
````

2) Plot cumulative graph with pandas and matplotlib
```python
from qrafti import Panel, MEDIA   # Path to save images
import matplotlib.pyplot as plt
panel_id = 'HML'
Panel(panel).frame.cumsum().plot(kind="line")
savefig = MEDIA / f"plot_{panel_id}.png"
fig.savefig(savefig)
payload = dict(image_path_name='file://' + str(savefig))
return json.dumps(payload)
```

3) Write and `apply` a custom helper function to winsorize values cross-sectionally by date
```python
panel_id = 'me' 
from qrafti import Panel
import pandas as pd
p1 = Panel(f'{panel_id}') 
p2 = Panel('crsp_exchcd').apply(pd.DataFrame.isin, values=[1, '1'])  # create mask for NYSE stocks
def winsorize(x, lower=0.0, upper=1.0) -> pd.Series: 
   # Helper function to apply winsorization to rows using mask in second column
   lower, upper = x.loc[x.iloc[:,1].astype(bool), x.columns[0]].quantile([lower, upper]).values 
   return x.iloc[:, 0].clip(lower=lower, upper=upper)
p3 = p1.apply(winsorize, 1 if p2 is None else p2, lower=0, upper=0.8, fill_value=False).persist()
print(str(p3))  # returns saved name of the panel as JSON string
```

AUTHORING GUIDELINES FOR THE AGENT

* Prefer `Panel.apply(...)` with a `reference` mask/weights to keep operations group-by-date and index-aligned.
* Use arithmetic/logic operators between `Panel`s; the library auto-joins and fills appropriately.
* When forming returns: compute weights (`portfolio_weights`),
  then `portfolio_returns(weights)`, then metrics/regressions/plots.
* When combining universes or cleaning data, use `p.filter(mask=mask)` or `p.filter(index=index)`
* Access raw arrays only via `p.frame` or `p.values` when absolutely necessary,
  and use `p.set_frame` to bind back to a Panel;
  otherwise keep within the `Panel` algebra to preserve index alignment.
"""

# Choose model
model = "gemini-2.5-flash"
model_parameters = {}

# -- Ollama local model --
# ollama_client = OpenAIProvider(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # dummy; Ollama ignores auth but the SDK requires a value
# )
# ollama_model = OpenAIChatModel(
#     model_name="qwen3:14b",
#     provider=ollama_client,
# )

# -- OpenAI models --
# model = OpenAIChatModel(model_name="gpt-5-mini")
# model_parameters={
#     "verbosity": "low",            # low / medium / high                                                           
#     "reasoning_effort": "minimal", # minimal / low / medium / high   
# }
# model = OpenAIChatModel(model_name="gpt-4.1-mini")

# Streamlit page configuration
st.set_page_config(layout="wide")

def build_conversation_context() -> str:
    conversation_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )
    return conversation_text

# MCP server connections
factor_server = MCPServerStreamableHTTP(url="http://localhost:8000/mcp")
performance_server = MCPServerStreamableHTTP(url="http://localhost:8001/mcp")
common_server = MCPServerStreamableHTTP(url="http://localhost:8002/mcp")
coding_server = MCPServerStreamableHTTP(url="http://localhost:8003/mcp")

# Create the Manager Agent with its own set of tools
manager_agent = Agent(
    name="Research Manager Agent",
    model=model,
    system_prompt="""
You are a Research Manager Agent who interacts with the user, answers general knowledge questions,
and uses the following tools to perform quantitative analysis tasks on Panel data sets provided to you.
When you are asked by the user to either write or execute Python, or both, 
you should call the coding_agent_tool to do so and fulfill the request based on its response.
# Tools:
* Use factor_agent_tool for loading and manipulating stock characteristics,
constructing and measuring factor portfolio weights and returns, and producing plots.
* Use performance_agent_tool to write a research report on a Panel data set of stock characteristic values;
  simply pass the Panel identifier to be analyzed to this tool, there is no need to first perform
  any other manipulations to the Panel of characteristics values.
* Use coding_agent_tool to write and execute Python code to manipulate Panel data safely in a sandbox;
  before delegating any coding task, review the Panel API cheat-sheet provided to you and include it in the
  context you send to the coding agent so it can follow those guidelines;
* Use planner_agent_tool to request a step-by-step plan for the user's query.
# Guidelines:
To execute the plan, you should sequentially execute each step by calling the appropriate agent tool.
Do not use or assume any data or panels that you were not given or was not generated by you or a specialized agent.
You may use the get_variables_descriptions tool to look for Panel ids of stocks data.
You must explain the steps you took, the planner output you received and the agent tools you used for each step;
do not output in JSON format or python code, but you should use bulleted points or narrative format for clarity.
""".strip(),
    model_settings={'temperature': 0.0, **model_parameters},
    toolsets=[common_server]
)

# ensure the coding agent executes its Python through the execute_python tool and reports the outcome back to you.

# Do not perform any data analysis or factor construction yourself.
# , which provides an agent tool and description for each step in JSON format,
# Review the full conversation and produce a JSON array of ordered steps.
# Each step must be a JSON object with the keys 'step number', 'description' and 'agent tool'.
# Increment 'step number' starting at 1.
# If no steps are required, return an empty JSON array.

# Do not use or assume any data, characteristics or definitions that you were not given or you did not generate from a specialized agent.

# For each step in the plan, delegate only that step and its description to the specialized agent specified in the 'agent tool' field of the step; 
# then delegate the next step to the specialized agent specified in the next step, and so on.
# Do not perform any steps that were assigned to other agent tools.
# If a specialized agent tool could not complete a step because it was assigned to a different agent tool, 
# then you should delegate the step to to the specialized agent it was assigned to.

planner_agent = Agent(
    name="Research Planner Agent",
    model=model,
    system_prompt="""
You are a planning specialist who designs execution plans for quantitative research requests.
Review the full conversation and produce a sequence of steps,
where each step includes a 'step number', 'description',  and 'agent tool' to be used by each agent.
Each step's 'description' should clearly explain the task to be performed.
You may call get_variables_descriptions when you need to understand available variables, before designing your plan.
The 'agent tool' value must be either 'factor_agent_tool' or 'performance_agent_tool', matching the agent that will
perform the step.
Do not use or assume any agents or tools that you were not given.
Use 'factor_agent_tool' for tasks involving characteristic preparation, factor construction, quantile sorting,
portfolio weighting, generation or evaluation of portfolio or factor returns, plotting, and any computation that relies on the factor agent's tools.
Use `performance_agent_tool` only to write a research report on a Panel data set of stock characteristic values.
""".strip(),
    model_settings={'temperature': 0.0, **model_parameters},
    toolsets=[common_server]
)

# and list of tools
# Call get_specialized_agent_tools whenever you need to confirm which capabilities the agents expose.

# Create the agent and attach MCP server
factor_agent = Agent(
    name="Factor Portfolio Construction Agent",
    model=model,
    system_prompt="""
Use the tools provided to construct, manipulate or measure factor or portfolio 
characteristics, weights or returns in Panel data sets.
Be sure to include supporting reference Panels where required in your tool calls to ensure
all information in the query is captured accurately.
Do not perform any steps that were assigned to other agent tools.
If a tool call returned an unexpected error, try calling once more with the same parameters.
Do not use or assume any data or panels that you were not given or you did not generate with a tool.
Do not use any tools that are not part of your assigned toolset.
You may use the get_variables_descriptions tool to look for Panel ids of stocks data.
You must explain in detail every step you took and all the tools you used.
""".strip(),
    toolsets=[factor_server, common_server],
    model_settings={'temperature': 0.0, **model_parameters}  # 0.1
)

performance_agent = Agent(
    name="Performance Agent",
    model=model, 
    system_prompt="""
Use the tools to write a research memo on a Panel data set of stock characteristic values.
Before drafting the memo you must call Panel_factor_evaluate with the Panel identifier that you
were asked to analyze. Parse the JSON payload returned by the tool and use the memo_prompt field
as the authoritative writing instructions. The memo_prompt contains the baseline memo guidelines
with the panel's evaluation statistics appended; follow those instructions exactly and base the
Results section strictly on the provided statistics.
""".strip(),
    toolsets=[performance_server, common_server],
    model_settings={'temperature': 0.0, **model_parameters}  # 0.1
)

coding_agent = Agent(
    name="Coding Agent",
    model=model,
    system_prompt=f"""
You are writing Python that manipulates a custom `Panel` API for cross-sectional/time-series equity data.
If requested, you may execute Python code safely in a sandbox using the tools provided to you.
When you are asked to perform a computational task, that means you should write Python to do so and
return the code you wrote, and then execute the code if also asked to.
Always consult the Panel API cheat-sheet below before generating or executing code and follow it precisely.

{PANEL_CHEATSHEET}

When responding to the manager:
- If only asked to write the code but not execute it, then return the code as a JSON text block without executing it.
- If asked to execute the code, then call the execute_python tool exactly once per attempt and report the result:
  your code should return in as a JSON str either the name of the final `Panel` you constructed 
  or a MEDIA file path for any plots by printing it to standard output.
""".strip(),
    model_settings={'temperature': 0.0, **model_parameters},
    toolsets=[coding_server]
)
# You execute python code safely in a sandbox by using the tools provided.

async def run_agent_safely(agent, query: str, role_label: str, retries: int = 1) -> str:
    """
    Run a pydantic-ai agent with retries to handle Gemini empty-content glitches.
    Returns a user-friendly string instead of raising.
    """
    attempt = 0
    while True:
        try:
            response = await agent.run(query)
            return response.output if hasattr(response, "output") else str(response)
        except UnexpectedModelBehavior as e:
            attempt += 1
            logging.warning("Gemini returned empty content for %s (attempt %d): %s for query '%s'",
                            role_label, attempt, e, query)
            if attempt > retries:
                return (f"{role_label} could not complete the step because the Gemini model "
                        f"returned an empty payload. Please try again or switch models.")
            # brief backoff before retry
            await asyncio.sleep(0.4)
        except Exception as e:
            logging.exception("Unexpected error in %s:", role_label)
            return f"{role_label} failed with an unexpected error: {e}"

@manager_agent.tool
async def performance_agent_tool(ctx: RunContext, query: str) -> str:
    """
    Tool to delegate tasks to the Performance Agent.
    """
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    log_conversation(full_query)
    # print("\n**Full query to Performance tool>> ", full_query, '>>')
    # response = await performance_agent.run(full_query)
    # out = response.output if hasattr(response, "output") else str(response)
    out = await run_agent_safely(performance_agent, full_query, "Performance Agent", retries=1)
    st.session_state.messages.append({"role": "Performance Agent", "content": f"{out}"})
    return out  # << return TEXT, not RunResult


@manager_agent.tool
async def factor_agent_tool(ctx: RunContext, query: str) -> str:
    """
    Tool to delegate tasks to the Factor Portfolio Construction Agent.
    """
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    print("\n**Full query to Factor tool>> ", full_query, '>>')
    # response = await factor_agent.run(full_query)
    # out = response.output if hasattr(response, "output") else str(response)
    out = await run_agent_safely(factor_agent, full_query, "Factor Agent", retries=1)
    st.session_state.messages.append({"role": "Factor Agent", "content": f"{out}"})
    return out  # << return TEXT


@manager_agent.tool
async def planner_agent_tool(ctx: RunContext, query: str) -> str:
    """Tool to request a structured execution plan from the Planner Agent."""
    full_query = build_conversation_context()
    if query:
        full_query = f"{full_query}\n\nManager instructions: {query}"
    log_conversation(full_query)
    out = await run_agent_safely(planner_agent, full_query, "Planner Agent", retries=1)
    st.session_state.messages.append({"role": "Planner Agent", "content": f"{out}"}) #
    return out

@manager_agent.tool
async def coding_agent_tool(ctx: RunContext, query: str) -> str:
    """Tool to execute python code safely in a sandbox"""
    full_query = build_conversation_context()
    if query:
        full_query = f"{full_query}\n\nManager instructions: {query}"
    full_query = f"{full_query}\n\nPanel API cheat-sheet:\n{PANEL_CHEATSHEET}"
    log_conversation(full_query)
    out = await run_agent_safely(coding_agent, full_query, "Code Agent", retries=1)
    st.session_state.messages.append({"role": "Coding Agent", "content": f"{out}"})
    return out

st.title("💬 Quant Research Agents")
st.markdown(
    """
    <style>
    /* === General font scaling === */
    html, body, [class*="block-container"] {
        font-size: 28px !important;
        line-height: 1.0 !important;
    }

    /* === Chat message bubbles === */
    div[data-testid="stChatMessage"] {
        font-size: 28px !important;
        line-height: 1.0 !important;
    }
    div[data-testid="stChatMessage"] p {
        font-size: 28px !important;
        line-height: 1.0 !important;
    }

    /* === Chat message headers (You / Agent) === */
    div[data-testid="stChatMessage"] strong {
        font-size: 28px !important;
    }

    /* === Chat input box === */
    textarea[data-testid="stChatInputTextArea"] {
        font-size: 28px !important;
        line-height: 1.5 !important;
    }

    /* === Sidebar & titles === */
    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }

    h1, h2, h3 {
        font-size: 48px !important;
    }

    /* Make font larger only inside the chat window
    div[data-testid="stChatMessage"] p {
        font-size: 20px;
        line-height: 1.5;
    } */
    
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "manager_delegated" not in st.session_state:
    st.session_state.manager_delegated = False  # Track if manager delegated a task


# Display past messages
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else f"🤖 {msg['role']}"
    with st.chat_message(msg["role"]):
        st.markdown(f"**{role}:** {msg['content']}")

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    message_length = len(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {prompt}")

    # Get agent response with context (include full history)
    full_query = build_conversation_context()
    print("\n**Full query to Manager agent>> ", full_query, '>>')

    st.session_state.manager_delegated = False  # reset before each run
    try:
        response = manager_agent.run_sync(full_query)
    except UnexpectedModelBehavior:
        st.session_state.messages.append({
            "role": "Manager Agent",
            "content": ("The Manager Agent hit a temporary issue with the model (empty response). "
                        "Please try again or switch models.")
        })
        response = type("Obj", (), {"output": st.session_state.messages[-1]["content"]})()
    except Exception as e:
        st.session_state.messages.append({
            "role": "Manager Agent",
            "content": f"Manager Agent failed with an unexpected error: {e}"
        })
        response = type("Obj", (), {"output": st.session_state.messages[-1]["content"]})()

    # Append manager agent reply if not delegated
    if always_show_manager or not st.session_state.manager_delegated:
        st.session_state.messages.append({"role": "Manager Agent", "content": str(response.output)})

    with st.chat_message("agent"):

        if isinstance(response.output, str) and os.path.exists(response.output):
            ext = os.path.splitext(response.output)[1].lower()

            # Display image files inline
            if ext in [".png", ".jpg", ".jpeg"]:
                st.image(response.output, caption=f"Generated file: {os.path.basename(response.output)}")

            # Offer download button for docx or other file types
            elif ext == ".docx":
                with open(response.output, "rb") as f:
                    st.download_button(
                        label=f"📄 Download {os.path.basename(response.output)}",
                        data=f,
                        file_name=os.path.basename(response.output),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            else:
                st.markdown(f"File created: `{response.output}` (unsupported for inline preview)")
        else:
            for msg in range(message_length, len(st.session_state.messages)):
                with st.chat_message(st.session_state.messages[msg]["role"]):
                    st.markdown(f"**🤖 {st.session_state.messages[msg]['role']}:** {st.session_state.messages[msg]['content']}")
#            st.markdown(f"**🤖 {st.session_state.messages[-1]['role']}:** {response.output}")
        debug_text = build_conversation_context()
        log_conversation(debug_text)
#        with (open("debug_query.txt", "w") as f):
#            f.write(debug_text.replace("\nUser:", f"\n\n{'-' * 40}\nUser:"))
#            f.flush()
