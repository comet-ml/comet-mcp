"""
Comet ML MCP Server Package

A Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API.
This server allows users to manage experiments, projects, and data through Comet ML's platform.
"""

__version__ = "1.0.0"
__author__ = "Comet ML Team"
__description__ = "MCP server for Comet ML API integration"

from .comet_client import CometClient, CometAPIError
from .config import CometConfig, ServerConfig, get_config
from .tool_handler import get_comet_tools, CometToolHandler
from .server import CometMCPServer

__all__ = [
    "CometClient",
    "CometAPIError", 
    "CometConfig",
    "ServerConfig",
    "get_config",
    "get_comet_tools",
    "CometToolHandler",
    "CometMCPServer",
]

