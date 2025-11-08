import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Configure logging
#logfire.configure()
#logfire.instrument_pydantic_ai()
import logging
logging.basicConfig(level=logging.DEBUG)

# Choose model
model = "gemini-2.5-flash"
model_parameters = {}

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
############
#
# OpenAI model setup
#
############

#openai_provider = OpenAIProvider(
#    api_key=os.environ["OPENAI_API_KEY"],
#    #base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
#)
#model = OpenAIChatModel(model_name="openai:gpt-5-mini", provider=openai_provider)
#model = OpenAIChatModel(model_name="gpt-5-mini")
model = OpenAIChatModel(model_name="gpt-4.1-mini")
#model_parameters={
#    "verbosity": "low",            # low / medium / high                                                           
#    "reasoning_effort": "minimal", # minimal / low / medium / high   
#}

# ================================
# STREAMLIT PAGE CONFIGURATION
# ================================
st.set_page_config(layout="wide", page_title="Quant Research Agents")
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

# ================================
# SESSION STATE INITIALIZATION
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "manager_delegated" not in st.session_state:
    st.session_state.manager_delegated = False

# ================================
# LAZY MCP SERVER INITIALIZATION
# ================================
if "cache_server" not in st.session_state:
    with st.spinner("Connecting to MCP servers..."):
        st.session_state.cache_server = MCPServerStreamableHTTP(url="http://localhost:8000/mcp")
        st.session_state.performance_server = MCPServerStreamableHTTP(url="http://localhost:8001/mcp")
        st.session_state.metadata_server = MCPServerStreamableHTTP(url="http://localhost:8002/mcp")

# ================================
# LAZY AGENT INITIALIZATION
# ================================
if "manager_agent" not in st.session_state:
    with st.spinner("Initializing GPT-4.1-mini agents (one-time setup)..."):

        # --- Manager Agent ---
        st.session_state.manager_agent = Agent(
            name="Research Manager Agent",
            model=model,
            system_prompt="""
You are a Research Manager Agent who interacts with the user and oversees quantitative research tasks.
Use factor_agent_tool for loading stock characteristics and constructing factor portfolio weights and returns.
Use performance_agent_tool for evaluating performance and risk of factor and portfolio returns.
Use planner_agent_tool to request a step-by-step plan for the user's query.
To execute the plan, you should sequentially execute each step by calling the appropriate agent tool.
For each step in the plan, delegate only that step and its description to the specialized agent specified in the 'agent tool' field of the step; 
then delegate the next step to the specialized agent specified in the next step, and so on.
Do not perform any steps that were assigned to other agent tools.
If a specialized agent tool could not complete a step because it was assigned to a different agent tool, 
then you should delegate the step to to the specialized agent it was assigned to.
You may use the get_variables_descriptions tool to look for Panel ids of stocks data.
Do not use or assume any data, characteristics or definitions that you were not given or you did not generate from a specialized agent.
You must explain the steps you took, the planner output you received and the agent tools you used for each step;
do not output in JSON format or python code, but you should use bulleted points or narrative format for clarity.
""".strip(),
            model_settings={"temperature": 0.0},
        )

        # --- Planner Agent ---
        st.session_state.planner_agent = Agent(
            name="Planner Agent",
            model=model,
            system_prompt="""
You are a planning specialist who designs execution plans for quantitative research requests.
Review the full conversation and produce a sequence of steps,
where each step includes a 'step number', 'description', 'agent tool' and list of tools to be used by each agent.
Each step's 'description' should clearly explain the task to be performed.
You may call get_variables_descriptions when you need to understand available variables, before designing your plan.
The 'agent tool' value must be either 'factor_agent_tool' or 'performance_agent_tool', matching the agent that will
perform the step.
Call get_specialized_agent_tools whenever you need to confirm which capabilities the agents expose.
Do not use or assume any agents or tools that you were not given.

Use 'factor_agent_tool' for tasks involving characteristic preparation, factor construction, quantile sorting,
portfolio weighting, portfolio or factor returns generation, and any computation that relies on the factor agent's tools.
Use 'performance_agent_tool' for tasks involving portfolio or factor returns evaluation, risk reporting,
plotting, and any computation that relies on the performance agent's tools.
""".strip(),
            model_settings={"temperature": 0.0},
        )

        # --- Factor Construction Agent ---
        st.session_state.factor_agent = Agent(
            name="Factor Construction Agent",
            model=model,
            system_prompt="""
Use the tools provided to construct factor or portfolio characteristics, weights or returns on the Panel data.
Be sure to include supporting reference Panels where required in your tool calls to ensure
all information in the query is captured accurately.
Do not perform any steps that were assigned to other agent tools.
If a tool call returned an unexpected error, try calling once more with the same parameters.
Do not use or assume any data, characteristics or definitions that you were not given or you did not generate with a tool
You may use the get_variables_descriptions tool to look for Panel ids of stocks data.
You must explain in detail every step you took and all the tools you used.
""".strip(),
            toolsets=[st.session_state.cache_server],
            model_settings={"temperature": 0.0},
        )

        # --- Performance Evaluation Agent ---
        st.session_state.performance_agent = Agent(
            name="Performance Evaluation Agent",
            model=model,
            system_prompt="""
Use the tools provided to perform performance evaluation and risk analysis tasks
on factor and portfolio returns Panel data.
Be sure to include supporting reference Panels where required in your tool calls to ensure
all information in the query is captured accurately.
Do not perform any steps that were assigned to other agent tools.
If a tool call returned an unexpected error, try calling once more with the same parameters.
Do not use or assume use any data, characteristics or definitions that you were not given or you did not generate with a tool.
You may use the get_variables_descriptions tool to look for Panel ids of stocks data.
You must explain in detail every step you took and all the tools you used.
""".strip(),
            toolsets=[st.session_state.performance_server],
            model_settings={"temperature": 0.0},
        )

        # --- Metadata Agent ---
        st.session_state.metadata_agent = Agent(
            name="Metadata Agent",
            model=model,
            system_prompt=(
                "You extract metadata from cached datasets, assist in contextual reasoning, "
                "and help ensure consistent data handling across agents."
            ),
            toolsets=[st.session_state.metadata_server],
            model_settings={"temperature": 0.0},
        )

# ================================
# HELPER FUNCTION
# ================================
def build_conversation_context() -> str:
    """Combine prior messages into a conversation context string."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)


# ================================
# AGENT SHORTCUTS
# ================================
manager_agent = st.session_state.manager_agent
planner_agent = st.session_state.planner_agent
factor_agent = st.session_state.factor_agent
performance_agent = st.session_state.performance_agent

# ================================
# TOOL DEFINITIONS
# ================================
@manager_agent.tool
def factor_tool(ctx, query: str) -> str:
    """Delegate tasks to the Factor Agent."""
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    st.write("🧮 **Delegating to Factor Agent...**")
    response = factor_agent.run_sync(full_query)
    out = getattr(response, "output", str(response))
    st.session_state.messages.append({"role": "🧮 Factor Agent", "content": out})
    return out


@manager_agent.tool
def performance_tool(ctx, query: str) -> str:
    """Delegate tasks to the Performance Agent."""
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    st.write("📊 **Delegating to Performance Agent...**")
    response = performance_agent.run_sync(full_query)
    out = getattr(response, "output", str(response))
    st.session_state.messages.append({"role": "📊 Performance Agent", "content": out})
    return out


@manager_agent.tool
def planner_tool(ctx, query: str) -> str:
    """Delegate tasks to the Planner Agent."""
    st.session_state.manager_delegated = True
    full_query = build_conversation_context()
    st.write("🗂️ **Delegating to Planner Agent...**")
    response = planner_agent.run_sync(full_query)
    out = getattr(response, "output", str(response))
    st.session_state.messages.append({"role": "🗂️ Planner Agent", "content": out})
    return out


# ================================
# MAIN CHAT INTERFACE
# ================================
st.write("Type below to query the Manager Agent. Other agents will collaborate automatically.")

user_input = st.chat_input("Ask a research or portfolio question...")

if user_input:
    st.session_state.messages.append({"role": "👤 User", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("🤖 Manager Agent thinking..."):
        response = manager_agent.run_sync(user_input)

    output = getattr(response, "output", str(response))
    st.session_state.messages.append({"role": "🧠 Manager Agent", "content": output})

    with st.chat_message("assistant"):
        st.write(output)

# ================================
# DISPLAY CONVERSATION HISTORY
# ================================
if st.session_state.messages:
    st.write("### Conversation History")
    for msg in st.session_state.messages:
        role, content = msg["role"], msg["content"]
        st.markdown(f"**{role}:** {content}")

    debug_text = build_conversation_context()
    with (open("debug_query.txt", "w") as f):
        f.write(debug_text.replace("\nUser:", f"\n\n{'-' * 40}\nUser:"))
        f.flush()
