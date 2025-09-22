#!/usr/bin/env python3
"""
End-to-end tests for comet-mcp-chatbot functionality.
These tests require a properly configured comet_ml.API() that can access real data.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock
from datetime import datetime
from typing import List, Dict, Any

# Import the tools and chatbot
from comet_mcp.tools import (
    list_experiments,
    get_experiment_details,
    list_projects,
    get_session_info,
    search_experiments,
    list_project_experiments,
    count_project_experiments,
)
from comet_mcp.chatbot import MCPChatbot
from comet_mcp.session import initialize_session, get_session_context


class TestE2ETools:
    """End-to-end tests for individual tools with real Comet ML API."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Initialize the session for real API access
        initialize_session()
        session_context = get_session_context()

        # Skip tests if session is not properly initialized
        if not session_context.is_initialized():
            pytest.skip(
                "Comet ML session not initialized - check API key configuration"
            )

    def test_get_session_info_e2e(self):
        """Test session info with real API connection."""
        result = get_session_info()

        # Verify basic structure
        assert isinstance(result, dict)
        assert "initialized" in result
        assert "api_status" in result
        assert "workspace" in result
        assert "error" in result

        # Should be initialized and connected
        assert result["initialized"] is True
        assert result["api_status"] in ["Connected", "Error"]

        print(f"Session info: {result}")

    def test_list_projects_e2e(self):
        """Test listing projects with real API."""
        result = list_projects()

        # Verify structure
        assert isinstance(result, list)

        if result:  # If projects exist
            project = result[0]
            assert isinstance(project, dict)
            assert "name" in project
            assert "workspace" in project
            assert "created_at" in project
            assert "description" in project

            print(f"Found {len(result)} projects")
            print(
                f"First project: {project['name']} in workspace {project['workspace']}"
            )

    def test_list_experiments_e2e(self):
        """Test listing experiments with real API."""
        result = list_experiments()

        # Verify structure
        assert isinstance(result, list)

        if result:  # If experiments exist
            exp = result[0]
            assert isinstance(exp, dict)
            assert "id" in exp
            assert "name" in exp
            assert "status" in exp
            assert "created_at" in exp
            assert "description" in exp

            print(f"Found {len(result)} experiments")
            print(f"First experiment: {exp['name']} ({exp['status']})")

    def test_list_experiments_with_project_e2e(self):
        """Test listing experiments filtered by project."""
        # First get available projects
        projects = list_projects()

        if not projects:
            pytest.skip("No projects available for testing")

        project_name = projects[0]["name"]
        result = list_experiments(project_name=project_name)

        # Verify structure
        assert isinstance(result, list)

        print(f"Found {len(result)} experiments in project '{project_name}'")

        # All experiments should be from the specified project
        for exp in result:
            assert isinstance(exp, dict)
            assert "id" in exp
            assert "name" in exp

    def test_count_project_experiments_e2e(self):
        """Test counting experiments in a project."""
        # First get available projects
        projects = list_projects()

        if not projects:
            pytest.skip("No projects available for testing")

        project_name = projects[0]["name"]
        result = count_project_experiments(project_name)

        # Verify structure
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "workspace" in result
        assert "experiment_count" in result
        assert "experiments" in result

        assert result["project_name"] == project_name
        assert isinstance(result["experiment_count"], int)
        assert isinstance(result["experiments"], list)
        assert result["experiment_count"] == len(result["experiments"])

        print(f"Project '{project_name}' has {result['experiment_count']} experiments")

    def test_list_project_experiments_e2e(self):
        """Test listing experiments in a specific project."""
        # First get available projects
        projects = list_projects()

        if not projects:
            pytest.skip("No projects available for testing")

        project_name = projects[0]["name"]
        result = list_project_experiments(project_name)

        # Verify structure
        assert isinstance(result, list)

        print(f"Found {len(result)} experiments in project '{project_name}'")

        # Verify experiment structure
        for exp in result:
            assert isinstance(exp, dict)
            assert "id" in exp
            assert "name" in exp
            assert "status" in exp
            assert "created_at" in exp
            assert "description" in exp

    def test_search_experiments_e2e(self):
        """Test searching experiments with real API."""
        # Search for common terms
        search_terms = ["test", "experiment", "model", "training"]

        for term in search_terms:
            result = search_experiments(term)

            # Verify structure
            assert isinstance(result, dict)
            assert "query" in result
            assert "count" in result
            assert "experiments" in result

            assert result["query"] == term
            assert isinstance(result["count"], int)
            assert isinstance(result["experiments"], list)
            assert result["count"] == len(result["experiments"])

            if result["count"] > 0:
                print(f"Search '{term}' found {result['count']} experiments")
                break
        else:
            print("No experiments found for any search terms")

    def test_search_experiments_with_project_e2e(self):
        """Test searching experiments within a specific project."""
        # First get available projects
        projects = list_projects()

        if not projects:
            pytest.skip("No projects available for testing")

        project_name = projects[0]["name"]
        result = search_experiments("test", project_name=project_name)

        # Verify structure
        assert isinstance(result, dict)
        assert "query" in result
        assert "count" in result
        assert "experiments" in result

        print(
            f"Search 'test' in project '{project_name}' found {result['count']} experiments"
        )

    def test_get_experiment_details_e2e(self):
        """Test getting experiment details with real API."""
        # First get available experiments
        experiments = list_experiments()

        if not experiments:
            pytest.skip("No experiments available for testing")

        experiment_id = experiments[0]["id"]
        result = get_experiment_details(experiment_id)

        # Verify structure
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "status" in result
        assert "created_at" in result
        assert "updated_at" in result
        assert "description" in result
        assert "metrics" in result
        assert "parameters" in result

        assert result["id"] == experiment_id
        assert isinstance(result["metrics"], list)
        assert isinstance(result["parameters"], list)

        print(f"Experiment details for '{result['name']}':")
        print(f"  Status: {result['status']}")
        print(f"  Metrics: {len(result['metrics'])}")
        print(f"  Parameters: {len(result['parameters'])}")


class TestE2EChatbot:
    """End-to-end tests for the MCP chatbot with real API calls."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Initialize the session for real API access
        initialize_session()
        session_context = get_session_context()

        # Skip tests if session is not properly initialized
        if not session_context.is_initialized():
            pytest.skip(
                "Comet ML session not initialized - check API key configuration"
            )

    @pytest.mark.asyncio
    async def test_chatbot_session_info(self):
        """Test chatbot can get session information."""
        # Create a minimal config for testing
        config = {
            "servers": [
                {
                    "name": "comet-mcp",
                    "description": "Comet ML MCP server for experiment management",
                    "command": "comet-mcp",
                }
            ]
        }

        # Mock the config loading
        with patch("comet_mcp.chatbot.MCPChatbot.load_config") as mock_load_config:
            mock_load_config.return_value = [
                Mock(
                    name="comet-mcp",
                    description="Comet ML MCP server",
                    command="comet-mcp",
                    args=[],
                    env=None,
                )
            ]

            # Mock the server connection
            with patch("comet_mcp.chatbot.MCPChatbot._connect_server"):
                chatbot = MCPChatbot(
                    "config.json", "openai/gpt-4o-mini", {}, max_rounds=2
                )

                # Mock the session and tools
                mock_session = Mock()
                mock_tool = Mock()
                mock_tool.name = "get_session_info"
                mock_tool.description = "Get session information"
                mock_tool.inputSchema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

                mock_tools_response = Mock()
                mock_tools_response.tools = [mock_tool]
                mock_session.list_tools.return_value = mock_tools_response

                mock_tool_result = Mock()
                mock_tool_result.content = [
                    {
                        "type": "text",
                        "text": '{"initialized": true, "api_status": "Connected", "workspace": "test-user", "error": null}',
                    }
                ]
                mock_session.call_tool.return_value = mock_tool_result

                chatbot.sessions["comet-mcp"] = mock_session

                # Test the chatbot functionality
                response = await chatbot.chat_once(
                    "What is the current session status?"
                )

                # Verify response
                assert isinstance(response, str)
                assert len(response) > 0

                print(f"Chatbot response: {response}")

    def test_tool_registry_integration(self):
        """Test that all tools are properly registered and callable."""
        from comet_mcp.utils import registry

        # Get all registered tools
        tools = registry.get_tools()
        tool_names = [tool.name for tool in tools]

        # Verify expected tools are registered
        expected_tools = [
            "list_experiments",
            "get_experiment_details",
            "list_projects",
            "get_session_info",
            "search_experiments",
            "list_project_experiments",
            "count_project_experiments",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Tool '{tool_name}' not found in registry"

        print(f"All {len(expected_tools)} expected tools are registered")
        print(f"Total tools registered: {len(tool_names)}")

    def test_tool_call_format(self):
        """Test that tools can be called through the registry."""
        from comet_mcp.utils import registry

        # Test calling get_session_info through registry
        result = registry.call_tool("get_session_info", {})

        # Verify result format
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"

        # Should contain JSON data
        import json

        data = json.loads(result[0]["text"])
        assert "initialized" in data
        assert "api_status" in data

        print("Tool call through registry successful")


class TestE2EProjectWorkflow:
    """End-to-end tests for complete project workflow scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Initialize the session for real API access
        initialize_session()
        session_context = get_session_context()

        # Skip tests if session is not properly initialized
        if not session_context.is_initialized():
            pytest.skip(
                "Comet ML session not initialized - check API key configuration"
            )

    def test_complete_project_analysis_workflow(self):
        """Test a complete workflow of analyzing a project."""
        # Step 1: Get session info
        session_info = get_session_info()
        assert session_info["initialized"] is True
        print(f"✓ Session initialized: {session_info['api_status']}")

        # Step 2: List available projects
        projects = list_projects()
        if not projects:
            pytest.skip("No projects available for testing")

        project = projects[0]
        project_name = project["name"]
        print(f"✓ Analyzing project: {project_name}")

        # Step 3: Count experiments in project
        count_result = count_project_experiments(project_name)
        experiment_count = count_result["experiment_count"]
        print(f"✓ Project has {experiment_count} experiments")

        # Step 4: List experiments in project
        experiments = list_project_experiments(project_name)
        assert len(experiments) == experiment_count
        print(f"✓ Retrieved {len(experiments)} experiment details")

        # Step 5: Search within project
        search_result = search_experiments("test", project_name=project_name)
        print(f"✓ Found {search_result['count']} experiments matching 'test'")

        # Step 6: Get details for first experiment (if any)
        if experiments:
            exp_details = get_experiment_details(experiments[0]["id"])
            print(f"✓ Retrieved details for experiment: {exp_details['name']}")
            print(f"  - Status: {exp_details['status']}")
            print(f"  - Metrics: {len(exp_details['metrics'])}")
            print(f"  - Parameters: {len(exp_details['parameters'])}")

        print("✓ Complete project analysis workflow successful")

    def test_cross_project_comparison_workflow(self):
        """Test comparing experiments across multiple projects."""
        # Get all projects
        projects = list_projects()
        if len(projects) < 2:
            pytest.skip("Need at least 2 projects for comparison")

        print("Comparing experiments across projects:")

        project_stats = []
        for project in projects[:3]:  # Limit to first 3 projects
            project_name = project["name"]
            count_result = count_project_experiments(project_name)

            stats = {
                "name": project_name,
                "workspace": project["workspace"],
                "experiment_count": count_result["experiment_count"],
                "created_at": project["created_at"],
            }
            project_stats.append(stats)

            print(f"  - {project_name}: {stats['experiment_count']} experiments")

        # Verify we got stats for multiple projects
        assert len(project_stats) >= 2
        print("✓ Cross-project comparison successful")

    def test_error_handling_workflow(self):
        """Test error handling with invalid inputs."""
        # Test with non-existent project
        result = count_project_experiments("non-existent-project-12345")
        assert result["experiment_count"] == 0
        assert result["experiments"] == []
        print("✓ Handled non-existent project gracefully")

        # Test with non-existent experiment ID
        try:
            get_experiment_details("non-existent-experiment-12345")
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "not found" in str(e).lower()
            print("✓ Handled non-existent experiment ID gracefully")

        # Test search with no results
        search_result = search_experiments(
            "very-specific-term-that-should-not-exist-12345"
        )
        assert search_result["count"] == 0
        assert search_result["experiments"] == []
        print("✓ Handled search with no results gracefully")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
