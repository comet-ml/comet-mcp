#!/usr/bin/env python3
"""
Entry point for Comet ML MCP Server

This script provides a command-line interface to run the Comet ML MCP Server.
Can be run as: python -m comet_mcp
"""

import asyncio
import sys
from . import CometMCPServer


async def main():
    """Main entry point for the MCP server."""
    try:
        server = CometMCPServer()
        await server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

