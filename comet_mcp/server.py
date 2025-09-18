#!/usr/bin/env python3
"""
Comet ML MCP Server

A Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API.
This server allows users to manage experiments, projects, and data through Comet ML's platform.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    ServerCapabilities,
)

from .config import get_config
from .comet_client import CometClient
from .tool_handler import get_comet_tools, CometToolHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CometMCPServer:
    """MCP Server for Comet ML API integration."""
    
    def __init__(self):
        self.server = Server("comet-ml")
        self.comet_client: Optional[CometClient] = None
        self.tool_handler: Optional[CometToolHandler] = None
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        self.server.list_tools()(self._handle_list_tools)
        self.server.call_tool()(self._handle_call_tool)
    
    async def _handle_list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """Handle list tools request."""
        if not self.tool_handler:
            # Return empty tools list if not initialized
            return ListToolsResult(tools=[])
        tools = self.tool_handler.get_tool_definitions()
        return ListToolsResult(tools=tools)
    
    async def _handle_call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool call request."""
        if not self.tool_handler:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: Comet ML client not initialized. Please check your API key configuration."
                )],
                isError=True
            )
        
        try:
            result = await self.tool_handler.handle_tool_call(request.params.name, request.params.arguments)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result
                )]
            )
        except Exception as e:
            logger.error(f"Error calling tool {request.params.name}: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )],
                isError=True
            )
    
    async def initialize(self):
        """Initialize the MCP server with Comet ML configuration."""
        try:
            comet_config, server_config = get_config()
            self.comet_client = CometClient(comet_config)
            self.tool_handler = CometToolHandler(self.comet_client)
            
            print("=" * 60)
            print("🚀 Comet ML MCP Server Starting Up")
            print("=" * 60)
            print(f"📡 Server Name: {server_config.server_name}")
            print(f"📦 Server Version: {server_config.server_version}")
            print(f"🔗 Comet ML API: {comet_config.base_url}")
            print(f"🏢 Workspace: {comet_config.workspace or 'Default'}")
            print("📋 Communication: MCP Protocol via stdio")
            print("⚡ Status: Ready to accept MCP client connections")
            print("=" * 60)
            
            logger.info("Comet ML MCP Server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Comet ML MCP Server: {e}")
            raise
    
    async def run(self):
        """Run the MCP server."""
        await self.initialize()
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="comet-ml",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    )
                )
            )


async def main():
    """Main entry point."""
    server = CometMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

