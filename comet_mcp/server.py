#!/usr/bin/env python3
"""
Simple demo MCP server for testing the multi-server client.
This server provides basic math operations as tools.
"""

import asyncio
import os
import signal
import sys
from typing import Any, Dict, List

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Import our tools module and registry
from . import tools  # This ensures tools are registered
from .utils import registry
from .session import initialize_session

# Create the server
server = Server("comet-mcp-server")

# Global variable to track server state for clean shutdown
_server_task = None

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return registry.get_tools()

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    return registry.call_tool(name, arguments)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Received shutdown signal, cleaning up...")
    if _server_task and not _server_task.done():
        _server_task.cancel()
    # Force immediate exit to avoid waiting for stdin
    os._exit(0)

async def main():
    """Run the server."""
    global _server_task
    
    print("🚀 Comet MCP Server Starting...")
    
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize comet_ml session context
    try:
        initialize_session()
        print("✓ Comet ML session initialized")
    except Exception as e:
        print(f"⚠️  Comet ML session initialization failed: {e}")
        print("   Tools requiring comet_ml.API() will not be available")
    
    print(f"🔧 Available Comet ML tools: {[tool.name for tool in registry.get_tools()]}")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            _server_task = asyncio.create_task(
                server.run(read_stream, write_stream, server.create_initialization_options())
            )
            await _server_task
    except asyncio.CancelledError:
        print("🛑 Server shutdown completed")
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down...")
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

def main_sync():
    """Synchronous entry point for the comet-mcp command."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
