"""Shared MCP server exposing metadata utilities for research agents."""
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables

mcp = FastMCP("metadata-server", host="0.0.0.0", port=8002)


@mcp.tool()
def get_variables_descriptions() -> dict:
    """Return a mapping of PanelFrame identifiers to their descriptions."""
    df = load_variables()
    if "Description" not in df.columns:
        return {}
    return df["Description"].to_dict()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
