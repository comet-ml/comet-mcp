#!/usr/bin/env python3
"""
Comprehensive pytest test suite for Comet ML MCP Server.

This test suite covers:
- Configuration loading and validation
- Comet ML API client functionality
- MCP server tool handling
- Error handling and edge cases
- Integration testing
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the modules we're testing
from comet_mcp import (
    CometConfig, ServerConfig, get_config,
    CometClient, CometAPIError,
    get_comet_tools, CometToolHandler,
    CometMCPServer
)


class TestCometConfig:
    """Test configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CometConfig(api_key="test_key_123")
        assert config.api_key == "test_key_123"
        assert config.workspace is None
        assert config.base_url == "https://www.comet.ml/api/rest/v2"
        
        # Config with workspace
        config = CometConfig(api_key="test_key_123", workspace="test_workspace")
        assert config.workspace == "test_workspace"
    
    def test_config_validation_errors(self):
        """Test configuration validation errors."""
        # Empty API key
        with pytest.raises(ValueError, match="API key cannot be empty"):
            CometConfig(api_key="")
        
        # Whitespace-only API key
        with pytest.raises(ValueError, match="API key cannot be empty"):
            CometConfig(api_key="   ")
    
    @patch.dict(os.environ, {"COMET_API_KEY": "test_key_123", "COMET_WORKSPACE": "test_workspace"})
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        comet_config, server_config = get_config()
        assert comet_config.api_key == "test_key_123"
        assert comet_config.workspace == "test_workspace"
        assert isinstance(server_config, ServerConfig)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_config_from_env_missing_key(self):
        """Test error when API key is missing from environment."""
        with pytest.raises(ValueError, match="COMET_API_KEY environment variable is required"):
            get_config()


class TestCometClient:
    """Test Comet ML API client."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return CometConfig(api_key="test_key_123", workspace="test_workspace")
    
    @pytest.fixture
    def mock_api(self):
        """Create a mock Comet ML API."""
        mock_api = Mock()
        return mock_api
    
    @patch('comet_mcp.comet_client.API')
    def test_client_initialization(self, mock_api_class, mock_config):
        """Test client initialization."""
        mock_api_instance = Mock()
        mock_api_class.return_value = mock_api_instance
        
        client = CometClient(mock_config)
        assert client.config == mock_config
        assert client.api == mock_api_instance
        mock_api_class.assert_called_once_with(api_key="test_key_123")
    
    @pytest.mark.asyncio
    async def test_get_workspaces_success(self, mock_config, mock_api):
        """Test successful workspace retrieval."""
        # Mock API response
        mock_workspaces = ["workspace1", "workspace2", "workspace3"]
        mock_api.get_workspaces.return_value = mock_workspaces
        
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            result = await client.get_workspaces()
        
        # Should return formatted string, not structured data
        assert isinstance(result, str)
        assert "workspace1" in result
        assert "workspace2" in result
        assert "workspace3" in result
        mock_api.get_workspaces.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_workspaces_error(self, mock_config, mock_api):
        """Test workspace retrieval error handling."""
        mock_api.get_workspaces.side_effect = Exception("API Error")
        
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            
            with pytest.raises(CometAPIError, match="Failed to get workspaces"):
                await client.get_workspaces()
    
    @pytest.mark.asyncio
    async def test_get_projects_success(self, mock_config, mock_api):
        """Test successful project retrieval."""
        mock_projects = ["project1", "project2"]
        mock_api.get_projects.return_value = mock_projects
        
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            result = await client.get_projects(workspace="test_workspace")
        
        # Should return formatted string, not structured data
        assert isinstance(result, str)
        assert "project1" in result
        assert "project2" in result
        mock_api.get_projects.assert_called_once_with(workspace="test_workspace")
    
    @pytest.mark.asyncio
    async def test_get_experiments_success(self, mock_config, mock_api):
        """Test successful experiment retrieval."""
        # Mock experiment objects
        mock_exp1 = Mock()
        mock_exp1.key = "exp1_key"
        mock_exp1.name = "Experiment 1"
        mock_exp1.get_state.return_value = "completed"
        mock_exp1.start_server_timestamp = 1234567890
        mock_exp1.end_server_timestamp = 1234567891
        mock_exp1.workspace = "test_workspace"
        mock_exp1.project_name = "test_project"
        
        mock_exp2 = Mock()
        mock_exp2.key = "exp2_key"
        mock_exp2.name = "Experiment 2"
        mock_exp2.get_state.return_value = "running"
        mock_exp2.start_server_timestamp = 1234567892
        mock_exp2.end_server_timestamp = 1234567893
        mock_exp2.workspace = "test_workspace"
        mock_exp2.project_name = "test_project"
        
        mock_api.get_experiments.return_value = [mock_exp1, mock_exp2]
        
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            result = await client.get_experiments(workspace="test_workspace", project="test_project", limit=2)
        
        # Should return formatted string, not structured data
        assert isinstance(result, str)
        assert "exp1_key" in result
        assert "exp2_key" in result
        assert "Experiment 1" in result
        assert "Experiment 2" in result
        mock_api.get_experiments.assert_called_once_with(workspace="test_workspace", project_name="test_project")
    
    @pytest.mark.asyncio
    async def test_get_experiment_details_success(self, mock_config, mock_api):
        """Test successful experiment details retrieval."""
        mock_exp = Mock()
        mock_exp.key = "exp_key"
        mock_exp.name = "Test Experiment"
        mock_exp.state = "completed"
        mock_exp.created_at = 1234567890
        mock_exp.updated_at = 1234567891
        mock_exp.workspace = "test_workspace"
        mock_exp.project_name = "test_project"
        
        # Mock attributes that might not exist
        mock_exp.description = "Test description"
        mock_exp.tags = ["tag1", "tag2"]
        mock_exp.metrics = [Mock(), Mock()]  # 2 metrics
        mock_exp.parameters = [Mock()]  # 1 parameter
        mock_exp.assets = []  # 0 assets
        
        mock_api.get_experiment_by_key.return_value = mock_exp
        
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            result = await client.get_experiment_details("exp_key")
        
        # Should return formatted string, not structured data
        assert isinstance(result, str)
        assert "exp_key" in result
        assert "Test Experiment" in result
        assert "Test description" in result
        assert "tag1" in result
        assert "tag2" in result
        mock_api.get_experiment_by_key.assert_called_once_with("exp_key")
    
    @pytest.mark.asyncio
    async def test_get_experiment_details_missing_key(self, mock_config, mock_api):
        """Test experiment details with missing experiment key."""
        with patch('comet_mcp.comet_client.API', return_value=mock_api):
            client = CometClient(mock_config)
            
            with pytest.raises(ValueError, match="experiment_key is required"):
                await client.get_experiment_details("")


class TestCometToolHandler:
    """Test tool handler functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Comet client."""
        return AsyncMock(spec=CometClient)
    
    @pytest.fixture
    def tool_handler(self, mock_client):
        """Create a tool handler with mock client."""
        return CometToolHandler(mock_client)
    
    @pytest.mark.asyncio
    async def test_handle_get_workspaces(self, tool_handler, mock_client):
        """Test handling get_workspaces tool call."""
        mock_response = "• workspace1 (Workspace 1)\n  Created: None\n\n• workspace2 (Workspace 2)\n  Created: None"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("get_workspaces", {})
        
        assert "workspace1" in result
        assert "workspace2" in result
        mock_client.handle_tool_call.assert_called_once_with("get_workspaces", {})
    
    @pytest.mark.asyncio
    async def test_handle_get_projects(self, tool_handler, mock_client):
        """Test handling get_projects tool call."""
        mock_response = "• project1\n  Description: Test project\n  Experiments: 5 | Created: None"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("get_projects", {"workspace": "test_workspace"})
        
        assert "project1" in result
        assert "Test project" in result
        mock_client.handle_tool_call.assert_called_once_with("get_projects", {"workspace": "test_workspace"})
    
    @pytest.mark.asyncio
    async def test_handle_get_experiments(self, tool_handler, mock_client):
        """Test handling get_experiments tool call."""
        mock_response = "• exp1: Test Experiment\n  State: completed | Created: 1234567890"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("get_experiments", {
            "workspace": "test_workspace",
            "project": "test_project",
            "limit": 10
        })
        
        assert "exp1" in result
        assert "Test Experiment" in result
        mock_client.handle_tool_call.assert_called_once_with("get_experiments", {
            "workspace": "test_workspace",
            "project": "test_project",
            "limit": 10
        })
    
    @pytest.mark.asyncio
    async def test_handle_get_experiment_details(self, tool_handler, mock_client):
        """Test handling get_experiment_details tool call."""
        mock_response = "**Experiment Details**\nKey: exp1\nName: Test Experiment\nState: completed\nCreated: 1234567890\nUpdated: 1234567891\nDescription: Test description\nTags: tag1\nMetrics: 5 metrics logged\nParameters: 3 parameters\nAssets: 2 files"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("get_experiment_details", {"experiment_key": "exp1"})
        
        assert "exp1" in result
        assert "Test Experiment" in result
        assert "Test description" in result
        mock_client.handle_tool_call.assert_called_once_with("get_experiment_details", {"experiment_key": "exp1"})
    
    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, tool_handler, mock_client):
        """Test handling unknown tool call."""
        mock_response = "Unknown tool: unknown_tool"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("unknown_tool", {})
        
        assert "Unknown tool: unknown_tool" in result
        mock_client.handle_tool_call.assert_called_once_with("unknown_tool", {})
    
    @pytest.mark.asyncio
    async def test_handle_tool_error(self, tool_handler, mock_client):
        """Test handling tool call errors."""
        mock_response = "Comet API Error: API Error"
        mock_client.handle_tool_call.return_value = mock_response
        
        result = await tool_handler.handle_tool_call("get_workspaces", {})
        
        assert "Comet API Error: API Error" in result
        mock_client.handle_tool_call.assert_called_once_with("get_workspaces", {})


class TestCometMCPServer:
    """Test MCP server functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configurations."""
        comet_config = CometConfig(api_key="test_key_123", workspace="test_workspace")
        server_config = ServerConfig()
        return comet_config, server_config
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Comet client."""
        return AsyncMock(spec=CometClient)
    
    @pytest.fixture
    def mock_tool_handler(self):
        """Create a mock tool handler."""
        return AsyncMock(spec=CometToolHandler)
    
    @patch('comet_mcp.server.get_config')
    @patch('comet_mcp.server.CometClient')
    @patch('comet_mcp.server.CometToolHandler')
    def test_server_initialization(self, mock_handler_class, mock_client_class, mock_get_config, mock_config):
        """Test server initialization."""
        mock_get_config.return_value = mock_config
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        server = CometMCPServer()
        
        assert server.comet_client is None
        assert server.tool_handler is None
        assert server.server is not None
    
    @pytest.mark.asyncio
    @patch('comet_mcp.server.get_config')
    @patch('comet_mcp.server.CometClient')
    @patch('comet_mcp.server.CometToolHandler')
    async def test_server_initialize(self, mock_handler_class, mock_client_class, mock_get_config, mock_config):
        """Test server initialization with config."""
        mock_get_config.return_value = mock_config
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        server = CometMCPServer()
        await server.initialize()
        
        assert server.comet_client == mock_client_instance
        assert server.tool_handler == mock_handler_instance
        mock_client_class.assert_called_once_with(mock_config[0])
        mock_handler_class.assert_called_once_with(mock_client_instance)
    
    @pytest.mark.asyncio
    async def test_handle_list_tools(self):
        """Test listing available tools."""
        server = CometMCPServer()
        
        # Initialize the tool handler for the test
        from comet_mcp.comet_client import CometClient
        from comet_mcp.config import CometConfig
        from comet_mcp.tool_handler import CometToolHandler
        mock_config = CometConfig(api_key="test_key", workspace="test_workspace")
        server.comet_client = CometClient(mock_config)
        server.tool_handler = CometToolHandler(server.comet_client)
        
        # Mock the request
        from mcp.types import ListToolsRequest
        request = ListToolsRequest(method="tools/list")
        
        result = await server._handle_list_tools(request)
        
        assert len(result.tools) == 9
        tool_names = [tool.name for tool in result.tools]
        expected_tools = [
            "get_experiments", "get_experiment_details", "get_experiment_metrics",
            "get_experiment_parameters", "get_projects", "get_workspaces",
            "search_experiments", "get_experiment_assets", "get_experiment_logs"
        ]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, mock_tool_handler):
        """Test successful tool call handling."""
        mock_tool_handler.handle_tool_call.return_value = "Test result"
        
        server = CometMCPServer()
        server.tool_handler = mock_tool_handler
        
        from mcp.types import CallToolRequest
        request = CallToolRequest(
            method="tools/call",
            params={"name": "get_workspaces", "arguments": {}}
        )
        
        result = await server._handle_call_tool(request)
        
        assert result.content[0].text == "Test result"
        assert not result.isError
        mock_tool_handler.handle_tool_call.assert_called_once_with("get_workspaces", {})
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_no_handler(self):
        """Test tool call when handler is not initialized."""
        server = CometMCPServer()
        server.tool_handler = None
        
        from mcp.types import CallToolRequest
        request = CallToolRequest(
            method="tools/call",
            params={"name": "get_workspaces", "arguments": {}}
        )
        
        result = await server._handle_call_tool(request)
        
        assert "Comet ML client not initialized" in result.content[0].text
        assert result.isError
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_error(self, mock_tool_handler):
        """Test tool call error handling."""
        mock_tool_handler.handle_tool_call.side_effect = Exception("Test error")
        
        server = CometMCPServer()
        server.tool_handler = mock_tool_handler
        
        from mcp.types import CallToolRequest
        request = CallToolRequest(
            method="tools/call",
            params={"name": "get_workspaces", "arguments": {}}
        )
        
        result = await server._handle_call_tool(request)
        
        assert "Error: Test error" in result.content[0].text
        assert result.isError


class TestTools:
    """Test tool definitions and schemas."""
    
    def test_get_comet_tools(self):
        """Test that all expected tools are defined."""
        tools = get_comet_tools()
        
        assert len(tools) == 9
        
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "get_experiments", "get_experiment_details", "get_experiment_metrics",
            "get_experiment_parameters", "get_projects", "get_workspaces",
            "search_experiments", "get_experiment_assets", "get_experiment_logs"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_tool_schemas(self):
        """Test that tool schemas are properly defined."""
        tools = get_comet_tools()
        
        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"
    
    def test_get_workspaces_tool_schema(self):
        """Test get_workspaces tool schema."""
        tools = get_comet_tools()
        workspaces_tool = next(tool for tool in tools if tool.name == "get_workspaces")
        
        assert workspaces_tool.description == "Get available workspaces."
        assert workspaces_tool.inputSchema["properties"] == {}
    
    def test_get_experiment_details_tool_schema(self):
        """Test get_experiment_details tool schema."""
        tools = get_comet_tools()
        details_tool = next(tool for tool in tools if tool.name == "get_experiment_details")
        
        assert "experiment_key" in details_tool.inputSchema["properties"]
        assert details_tool.inputSchema["required"] == ["experiment_key"]
        assert details_tool.inputSchema["properties"]["experiment_key"]["type"] == "string"


class TestIntegration:
    """Integration tests with real API (requires valid API key)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_connection(self):
        """Test connection to real Comet ML API (requires valid API key)."""
        # Skip if no API key is available
        if not os.getenv("COMET_API_KEY"):
            pytest.skip("COMET_API_KEY not set, skipping integration test")
        
        try:
            comet_config, _ = get_config()
            client = CometClient(comet_config)
            
            # Test getting workspaces
            workspaces = await client.get_workspaces()
            assert isinstance(workspaces, str)
            assert len(workspaces) > 0
            
            # Test getting projects if workspace is configured
            if comet_config.workspace:
                projects = await client.get_projects(workspace=comet_config.workspace)
                assert isinstance(projects, str)
                assert len(projects) > 0
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_tool_handler(self):
        """Test tool handler with real API (requires valid API key)."""
        if not os.getenv("COMET_API_KEY"):
            pytest.skip("COMET_API_KEY not set, skipping integration test")
        
        try:
            comet_config, _ = get_config()
            client = CometClient(comet_config)
            handler = CometToolHandler(client)
            
            # Test get_workspaces
            result = await handler.handle_tool_call("get_workspaces", {})
            assert isinstance(result, str)
            assert len(result) > 0
            
            # Test get_projects if workspace is configured
            if comet_config.workspace:
                result = await handler.handle_tool_call("get_projects", {"workspace": comet_config.workspace})
                assert isinstance(result, str)
                assert len(result) > 0
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
