#!/usr/bin/env python3
"""
Tool definitions for MCP server.
"""

from .utils import tool


# Register tools using the decorator
@tool
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    return f"{a} + {b} = {result}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together."""
    result = a * b
    return f"{a} × {b} = {result}"


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! 👋"
