"""MCP client for connecting to a remote kdream service."""
from __future__ import annotations

import asyncio
from typing import Any


async def _call_tool(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    """Connect to a remote MCP server and call a single tool."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)

    # result.content is a list of TextContent / ImageContent etc.
    if not result.content:
        return {}

    # FastMCP returns JSON-encoded text for dict/list results
    import json
    text_parts = [c.text for c in result.content if hasattr(c, "text")]
    if not text_parts:
        return {}

    raw = "\n".join(text_parts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"output": raw}


def call_tool(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    """Synchronous wrapper around _call_tool."""
    return asyncio.run(_call_tool(url, tool, arguments))
