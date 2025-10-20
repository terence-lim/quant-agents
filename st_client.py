# streamlit run st_client.py --server.fileWatcherType=none
import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP
#import logfire

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load env vars
load_dotenv()

# Configure logging
#logfire.configure()
#logfire.instrument_pydantic_ai()

# Settings
always_show_manager = False

# Choose model
model="gemini-2.5-flash"

if False:
    # Create an OpenAIProvider instance pointing to Ollama
    ollama_client = OpenAIProvider(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # dummy; Ollama ignores auth but the SDK requires a value
    )
    ollama_model = OpenAIChatModel(
        model_name="qwen3:14b",
        provider=ollama_client,
    )

# Streamlit page configuration
st.set_page_config(layout="wide")

def build_conversation_context() -> str:
    conversation_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )
    return conversation_text

# Define MCP server connection
factor_server = MCPServerStreamableHTTP(url="http://localhost:8000/mcp")
risk_server = MCPServerStreamableHTTP(url="http://localhost:8001/mcp")
metadata_server = MCPServerStreamableHTTP(url="http://localhost:8002/mcp")

# Create the Manager Agent with its own set of tools
manager_agent = Agent(
    name="Research Manager Agent",
    model=model,
    system_prompt="""
You are a Research Manager Agent overseeing quantitative research tasks.
Do not perform any data analysis or factor construction yourself.
Use factor_agent_tool for loading stock characteristics and constructing factor portfolio weights.
Use risk_agent_tool for constructing factor portfolio returns, and evaluating performance and risk of
factor and portfolio returns.
Use planner_agent_tool to request a step-by-step plan
for the user's latest query.
To execute the plan, you should sequentially execute each step by calling the appropriate agent tool.
For each step in the plan, delegate only that step and its description to the specialized agent specified in the 'agent tool' field of the step; 
then delegate the next step to the specialized agent specified in the next step, and so on.
Do not perform any steps that were assigned to other agent tools.
You may use the get_variables_descriptions tool to look for PanelFrame ids of stocks data.
Do not use or assume use any data, characteristics or definitions that
you were not given or you did not generate.
Explain the steps you took, the planner output you received, and the agent tools you used for each step.
""".strip(),
    model_settings={'temperature': 0.0},
    toolsets=[metadata_server]
)
# , which provides an agent tool and description for each step in JSON format,
planner_agent = Agent(
    name="Research Planner Agent",
    model=model,
    system_prompt="""
You are a planning specialist who designs execution plans for quantitative research requests.
Review the full conversation and produce a JSON array of ordered steps.
Each step must be a JSON object with the keys 'step number', 'description' and 'agent tool'.
Increment 'step number' starting at 1.
The 'agent tool' value must be either 'factor_agent_tool' or 'risk_agent_tool', matching the agent that will
perform the step.
Use 'factor_agent_tool' for tasks involving characteristic preparation, factor construction, quantile sorting,
portfolio weighting, and any operations available from the Factor Portfolio Construction Agent such as
panelframe_isin, panelframe_winsorize, panelframe_quantiles, panelframe_spread_portfolios, and
get_variables_descriptions.
Use 'risk_agent_tool' for tasks involving portfolio return generation, matrix operations like panelframe_matmul,
date alignment with panelframe_shift_dates, performance evaluation via panelframe_performance_evaluation,
plotting with panelframe_plot, and access to get_variables_descriptions.
If no steps are required, return an empty JSON array.
You may call get_variables_descriptions when you need to understand available variables, but do not delegate
tasks yourself.
""".strip(),
    model_settings={'temperature': 0.0},
    toolsets=[metadata_server]
)


# Describe the computation field with enough detail for the executing agent to know which tool call and
# parameters are needed.


# Create the agent and attach MCP server
factor_agent = Agent(
    name="Factor Portfolio Construction Agent",
    model=model,
    system_prompt="""
Use the tools provided to perform factor portfolio construction tasks
on the PanelFrame data.
Do not perform any steps that were assigned to other agent tools.
Do not use or assume use any data, characteristics or definitions that
you were not given or you did not generate. 
You may use the get_variables_descriptions tool to look for PanelFrame ids of stocks data.
Explain the steps you took and the tools you used.
""".strip(),
    toolsets=[factor_server, metadata_server],
    model_settings={'temperature': 0.0}  # 0.1
)

risk_agent = Agent(
    name="Risk Agent",
    model=model, 
    system_prompt="""
Use the tools provided to perform factor returns performance and risk analysis tasks
on the PanelFrame data.
Do not perform any steps that were assigned to other agent tools.
Do not use or assume use any data, characteristics or definitions that
you were not given or you did not generate.
You may use the get_variables_descriptions tool to look for PanelFrame ids of stocks data.
Explain the steps you took and the tools you used.
""".strip(),
    toolsets=[risk_server, metadata_server],
    model_settings={'temperature': 0.0}  # 0.1
)

@manager_agent.tool
async def risk_agent_tool(ctx: RunContext, query: str) -> str:
    """
    Tool to delegate tasks to the Risk Agent.
    """
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    print("\n**Full query to Risk tool>> ", full_query, '>>')
    response = await risk_agent.run(full_query)
    out = response.output if hasattr(response, "output") else str(response)
    st.session_state.messages.append({"role": "Risk Agent", "content": f"{out}"})
    return out  # << return TEXT, not RunResult


@manager_agent.tool
async def factor_agent_tool(ctx: RunContext, query: str) -> str:
    """
    Tool to delegate tasks to the Factor Portfolio Construction Agent.
    """
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    print("\n**Full query to Factor tool>> ", full_query, '>>')
    response = await factor_agent.run(full_query)
    out = response.output if hasattr(response, "output") else str(response)
    st.session_state.messages.append({"role": "Factor Agent", "content": f"{out}"})
    return out  # << return TEXT


@manager_agent.tool
async def planner_agent_tool(ctx: RunContext, query: str) -> str:
    """Tool to request a structured execution plan from the Planner Agent."""
    full_query = build_conversation_context()
    if query:
        full_query = f"{full_query}\n\nManager instructions: {query}"
    response = await planner_agent.run(full_query)
    out = response.output if hasattr(response, "output") else str(response)
#    st.session_state.messages.append({"role": "Planner Agent", "content": f"{out}"})
    return out

# Streamlit app title
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
    response = manager_agent.run_sync(full_query)

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
        with (open("debug_query.txt", "w") as f):
            f.write(debug_text.replace("\nUser:", f"\n\n{'-' * 40}\nUser:"))
