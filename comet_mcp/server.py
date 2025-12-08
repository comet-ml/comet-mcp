#!/usr/bin/env python3
"""
Comet ML MCP Server - Model Context Protocol server for Comet ML integration.
Provides tools for experiment management, project operations, and data analysis.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from typing import Any, Dict, List

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextResourceContents
from opentelemetry import trace

# Import our tools module and registry
from . import tool_loader  # This ensures tools are registered
from .utils import registry
from .session import initialize_session
from .resources import get_resource_manager
from .telemetry import initialize_telemetry, shutdown_telemetry, get_tracer

# SSE transport imports
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# Create the server
server = Server("comet-mcp")

# Global variable to track server state for clean shutdown
_server_task = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Comet ML MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport method to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    return parser.parse_args()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    tracer = get_tracer("comet-mcp.server")
    with tracer.start_as_current_span("mcp.list_tools") as span:
        try:
            tools = registry.get_tools()
            span.set_attribute("tool.count", len(tools))
            span.set_attribute("tool.names", [tool.name for tool in tools])
            return tools
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    tracer = get_tracer("comet-mcp.server")
    with tracer.start_as_current_span("mcp.call_tool") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.arguments", json.dumps(arguments))

        try:
            result = registry.call_tool(name, arguments)
            span.set_attribute("tool.success", True)
            span.set_attribute(
                "tool.result_count", len(result) if isinstance(result, list) else 1
            )
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.set_attribute("tool.success", False)
            raise


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources (generated files)."""
    tracer = get_tracer("comet-mcp.server")
    with tracer.start_as_current_span("mcp.list_resources") as span:
        try:
            resource_manager = get_resource_manager()
            resources_dict = resource_manager.list_resources()

            resources = []
            for uri, metadata in resources_dict.items():
                resources.append(
                    Resource(
                        uri=uri,
                        name=metadata["name"],
                        mimeType=metadata["mimeType"],
                        description=metadata.get("description", ""),
                    )
                )

            span.set_attribute("resource.count", len(resources))
            return resources
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


@server.read_resource()
async def read_resource(uri: str) -> TextResourceContents:
    """Read a resource by URI."""
    tracer = get_tracer("comet-mcp.server")
    with tracer.start_as_current_span("mcp.read_resource") as span:
        span.set_attribute("resource.uri", uri)

        try:
            resource_manager = get_resource_manager()
            file_path = resource_manager.get_file_path(uri)

            if file_path is None or not file_path.exists():
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, "Resource not found")
                )
                raise ValueError(f"Resource not found: {uri}")

            # Read file content as text
            content = file_path.read_text(encoding="utf-8")

            # Determine MIME type
            mime_type = resource_manager._guess_mime_type(file_path)

            span.set_attribute("resource.mime_type", mime_type)
            span.set_attribute("resource.size_bytes", len(content.encode("utf-8")))

            return TextResourceContents(
                uri=uri,
                mimeType=mime_type,
                text=content,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


# SSE transport implementation
app = FastAPI(title="Comet ML MCP Server", version="1.0.0")

# Global variables for SSE communication
_sse_clients = set()
_message_queue = asyncio.Queue()


@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for server-to-client communication."""

    async def event_generator():
        # Add client to the set
        client_id = id(asyncio.current_task())
        _sse_clients.add(client_id)
        print(f"🔌 SSE client connected: {client_id}")

        try:
            while True:
                # Wait for messages from the message queue
                try:
                    message = await asyncio.wait_for(_message_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except asyncio.CancelledError:
            print(f"🔌 SSE client disconnected: {client_id}")
        finally:
            _sse_clients.discard(client_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@app.post("/messages")
async def message_endpoint(request: Request):
    """HTTP POST endpoint for client-to-server communication."""
    try:
        data = await request.json()
        print(f"📨 Received message: {data}")

        # Process the message through the MCP server
        # This is a simplified implementation - in a real scenario,
        # you'd need to properly handle MCP protocol messages
        response = {"status": "received", "data": data}

        # Send response back via SSE
        await _message_queue.put(response)

        return {"status": "success", "message": "Message processed"}
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "transport": "sse"}


async def start_sse_server(host: str, port: int):
    """Start the SSE server."""
    print(f"🚀 Starting SSE server on {host}:{port}")
    print(f"📡 SSE endpoint: http://{host}:{port}/sse")
    print(f"📨 Messages endpoint: http://{host}:{port}/messages")
    print(f"🏥 Health check: http://{host}:{port}/health")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


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

    # Initialize OpenTelemetry
    initialize_telemetry()
    tracer = get_tracer("comet-mcp.server")

    with tracer.start_as_current_span("server.startup") as startup_span:
        # Parse command line arguments
        args = parse_args()

        startup_span.set_attribute("server.transport", args.transport)
        startup_span.set_attribute(
            "server.host", args.host if args.transport == "sse" else "stdio"
        )
        startup_span.set_attribute(
            "server.port", args.port if args.transport == "sse" else 0
        )

        print("🚀 Comet MCP Server Starting...")
        print(f"🚌 Transport: {args.transport}")

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize comet_ml session context
        with tracer.start_as_current_span("server.session_init") as session_span:
            try:
                initialize_session()
                session_span.set_attribute("session.initialized", True)
                print("✓ Comet ML session initialized")
            except Exception as e:
                session_span.record_exception(e)
                session_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                session_span.set_attribute("session.initialized", False)
                print(f"⚠️  Comet ML session initialization failed: {e}")
                print("   Tools requiring comet_ml.API() will not be available")

        tools = registry.get_tools()
        startup_span.set_attribute("server.tool_count", len(tools))
        startup_span.set_attribute("server.tool_names", [tool.name for tool in tools])
        print(f"🔧 Available Comet ML tools: {[tool.name for tool in tools]}")

        try:
            with tracer.start_as_current_span("server.run") as run_span:
                run_span.set_attribute("server.transport", args.transport)

                if args.transport == "stdio":
                    # Use stdio transport (default)
                    async with stdio_server() as (read_stream, write_stream):
                        _server_task = asyncio.create_task(
                            server.run(
                                read_stream,
                                write_stream,
                                server.create_initialization_options(),
                            )
                        )
                        await _server_task
                elif args.transport == "sse":
                    # Use SSE transport
                    await start_sse_server(args.host, args.port)
                else:
                    run_span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            f"Unknown transport: {args.transport}",
                        )
                    )
                    print(f"❌ Unknown transport: {args.transport}")
                    sys.exit(1)
        except asyncio.CancelledError:
            print("🛑 Server shutdown completed")
            # Force immediate exit to avoid waiting for stdin
            os._exit(0)
        except KeyboardInterrupt:
            print("\n🛑 Received keyboard interrupt, shutting down...")
            # Force immediate exit to avoid waiting for stdin
            os._exit(0)
        except Exception as e:
            startup_span.record_exception(e)
            startup_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            print(f"❌ Server error: {e}")
            raise
        finally:
            # Shutdown telemetry
            with tracer.start_as_current_span("server.shutdown") as shutdown_span:
                shutdown_span.set_attribute("server.shutdown_reason", "normal")
                shutdown_telemetry()


def main_sync():
    """Synchronous entry point for the comet-mcp command."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
