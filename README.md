# Quantitative Research Agents with Financial Tools and Intelligence

Python modules for agentically running quantitative research workflows, including data/factor services, user access interface, and empirical analysis tooling.

## Project Structure

### 1) Research Service Layer
**Files:** `factor_server.py`, `coding_server.py`, `report_server.py`, `server_utils.py`

Provides the service-side components for exposing factor computations, custom tool coding, and report generation capabilities. Shared server helpers and common logic live in `server_utils.py`, while the MCP server entry points are separated by specialization (`factor_server.py`, `coding_server.py` and `report_server.py`).

### 2) Client Access Layer
**Files:** `st_client.py`, `client_utils.py`

Contains the client-side interface used to connect to and interact with the service layer. `client_utils.py` centralizes reusable client helpers, while `st_client.py` serves as the primary client implementation in Pydantic-ai and Streamlit.

### 3) Financial Intelligence Toolkit
**Files:** `qrafti.py`, `utils.py`, `portfolio.py`, `rag.py`, `reboot.py`

Contains core analytical and orchestration modules for portfolio-focused research workflows, including shared utilities, retrieval-augmented functionality, and operational/restart helpers.

