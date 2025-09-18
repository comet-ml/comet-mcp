"""
MCP tools for Comet ML API integration.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Union, get_type_hints

from mcp.types import Tool
from .comet_client import CometClient, CometAPIError

logger = logging.getLogger(__name__)


def _generate_tool_from_method(method_name: str, method: callable) -> Tool:
    """Generate a Tool definition from a method's signature and type hints."""
    # Get method signature
    sig = inspect.signature(method)
    
    # Parse docstring for description
    docstring = method.__doc__ or f"Execute {method_name}"
    description = docstring.strip().split('\n')[0] if docstring else f"Execute {method_name}"
    
    # Build input schema from method parameters
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        # Get parameter type hint
        type_hint = param.annotation if param.annotation != inspect.Parameter.empty else Any
        
        # Determine JSON schema type from Python type
        json_type = _python_type_to_json_schema_type(type_hint)
        
        # Build property definition
        prop_def = {"type": json_type}
        
        # Add default value if present
        if param.default != inspect.Parameter.empty:
            prop_def["default"] = param.default
        else:
            # No default value means it's required
            required.append(param_name)
        
        properties[param_name] = prop_def
    
    input_schema = {
        "type": "object",
        "properties": properties
    }
    
    if required:
        input_schema["required"] = required
    
    return Tool(
        name=method_name,
        description=description,
        inputSchema=input_schema
    )


def _python_type_to_json_schema_type(type_hint: Any) -> str:
    """Convert Python type hint to JSON schema type."""
    # Handle direct type matches
    if type_hint == str:
        return "string"
    elif type_hint == int:
        return "integer"
    elif type_hint == float:
        return "number"
    elif type_hint == bool:
        return "boolean"
    elif type_hint == list:
        return "array"
    elif type_hint == dict:
        return "object"
    
    # Handle Optional types
    if hasattr(type_hint, '__origin__'):
        if type_hint.__origin__ is Union:
            # Check if it's Optional[SomeType] (which is Union[SomeType, None])
            args = type_hint.__args__
            if len(args) == 2 and type(None) in args:
                # This is Optional[SomeType], extract the actual type
                actual_type = args[0] if args[1] is type(None) else args[1]
                return _python_type_to_json_schema_type(actual_type)
            else:
                # Regular Union, default to string
                return "string"
        else:
            # Other generic types, default to string
            return "string"
    
    # Default fallback
    return "string"


def get_comet_tools() -> List[Tool]:
    """Get all available Comet ML tools."""
    # Create a temporary client instance to generate tools dynamically
    from .config import CometConfig
    temp_config = CometConfig(api_key="temp", workspace="temp")
    temp_client = CometClient(temp_config)
    
    tools = []
    for tool_name, tool_method in temp_client.get_tool_registry().items():
        tools.append(_generate_tool_from_method(tool_name, tool_method))
    return tools


class CometToolHandler:
    """Handler for Comet ML tool execution."""
    
    def __init__(self, client: CometClient):
        self.client = client
    
    def get_tool_definitions(self) -> List[Tool]:
        """Get all available tool definitions."""
        tools = []
        for tool_name, tool_method in self.client.get_tool_registry().items():
            tools.append(_generate_tool_from_method(tool_name, tool_method))
        return tools
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Handle a tool call and return formatted result."""
        return await self.client.handle_tool_call(tool_name, arguments)
    

