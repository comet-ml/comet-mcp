#!/usr/bin/env python3
"""
Simple demo MCP server for testing the multi-server client.
This server provides basic math operations as tools.
"""

import asyncio
from typing import Any, Dict, List

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Import our tools module and registry
from . import tools  # This ensures tools are registered
from .utils import registry

# Create the server
server = Server("comet-mcp-server")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return registry.get_tools()

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    return registry.call_tool(name, arguments)

async def main():
    """Run the server."""
    print("🚀 Comet MCP Server Starting...")
    print(f"Available tools: {registry.get_tools()}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

def main_sync():
    """Synchronous entry point for the comet-mcp command."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
