#!/usr/bin/env python3
"""
Unit tests for comet_mcp.tools module.
Tests that each tool works correctly and returns structured data.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

# Import the tools and their types
from comet_mcp.tools import (
    list_experiments,
    get_experiment_details,
    list_projects,
    get_session_info,
    list_project_experiments,
    count_project_experiments,
    ExperimentInfo,
    ExperimentDetails,
    ProjectInfo,
    SessionInfo,
)
from comet_mcp.session import session_context


class TestListExperiments:
    """Test cases for list_experiments tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Reset session context for each test
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_success(self, mock_get_api):
        """Test successful listing of experiments."""
        # Mock API response
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        
        mock_experiment1 = Mock()
        mock_experiment1.id = "exp1"
        mock_experiment1.name = "Test Experiment 1"
        mock_experiment1.get_state.return_value = "completed"
        mock_experiment1.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment1.description = "Test description 1"

        mock_experiment2 = Mock()
        mock_experiment2.id = "exp2"
        mock_experiment2.name = "Test Experiment 2"
        mock_experiment2.get_state.return_value = "running"
        mock_experiment2.start_server_timestamp = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment2.description = None

        mock_api.get_experiments.return_value = [mock_experiment1, mock_experiment2]
        mock_get_api.return_value.__enter__.return_value = mock_api

        # Call the function
        result = list_experiments()

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 2

        # Verify first experiment
        exp1 = result[0]
        assert isinstance(exp1, dict)
        assert exp1["id"] == "exp1"
        assert exp1["name"] == "Test Experiment 1"
        assert exp1["status"] == "completed"
        assert exp1["created_at"] == "2024-01-01T12:00:00"
        assert exp1["description"] == "Test description 1"

        # Verify second experiment
        exp2 = result[1]
        assert isinstance(exp2, dict)
        assert exp2["id"] == "exp2"
        assert exp2["name"] == "Test Experiment 2"
        assert exp2["status"] == "running"
        assert exp2["created_at"] == "2024-01-02T12:00:00"
        assert exp2["description"] is None

        # Verify API was called correctly
        mock_api.get_experiments.assert_called_once_with("default-workspace")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_with_workspace(self, mock_get_api):
        """Test listing experiments with specific workspace."""
        mock_api = Mock()
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_experiments(workspace="test-workspace")

        assert isinstance(result, list)
        assert len(result) == 0
        mock_api.get_experiments.assert_called_once_with("test-workspace")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_empty_result(self, mock_get_api):
        """Test listing experiments when no experiments exist."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_experiments()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_api_error(self, mock_get_api):
        """Test handling of API errors."""
        mock_get_api.side_effect = Exception("API connection failed")

        with pytest.raises(Exception) as exc_info:
            list_experiments()

        assert "Error listing experiments" in str(exc_info.value)
        assert "API connection failed" in str(exc_info.value)


class TestGetExperimentDetails:
    """Test cases for get_experiment_details tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_get_experiment_details_success(self, mock_get_api):
        """Test successful retrieval of experiment details."""
        # Mock experiment object
        mock_experiment = Mock()
        mock_experiment.id = "exp123"
        mock_experiment.name = "Detailed Experiment"
        mock_experiment.get_state.return_value = "completed"
        mock_experiment.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment.end_server_timestamp = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment.url = "https://example.com/exp123"
        mock_experiment.description = "Detailed description"

        # Mock metrics summary
        mock_metrics_summary = [
            {"name": "accuracy", "valueCurrent": 0.95},
            {"name": "loss", "valueCurrent": 0.05}
        ]

        # Mock parameters summary
        mock_params_summary = [
            {"name": "learning_rate", "valueCurrent": 0.001},
            {"name": "batch_size", "valueCurrent": 32}
        ]

        mock_experiment.get_metrics_summary.return_value = mock_metrics_summary
        mock_experiment.get_parameters_summary.return_value = mock_params_summary

        # Mock API
        mock_api = Mock()
        mock_api.get_experiment_by_key.return_value = mock_experiment
        mock_get_api.return_value.__enter__.return_value = mock_api

        # Call the function
        result = get_experiment_details("exp123")

        # Verify result structure
        assert isinstance(result, dict)
        assert result["id"] == "exp123"
        assert result["name"] == "Detailed Experiment"
        assert result["status"] == "completed"
        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["updated_at"] == "2024-01-02T12:00:00"
        assert result["description"] == "Detailed description"

        # Verify metrics
        assert isinstance(result["metrics"], list)
        assert len(result["metrics"]) == 2
        assert result["metrics"][0]["name"] == "accuracy"
        assert result["metrics"][0]["value"] == 0.95
        assert result["metrics"][1]["name"] == "loss"
        assert result["metrics"][1]["value"] == 0.05

        # Verify parameters
        assert isinstance(result["parameters"], list)
        assert len(result["parameters"]) == 2
        assert result["parameters"][0]["name"] == "learning_rate"
        assert result["parameters"][0]["value"] == 0.001
        assert result["parameters"][1]["name"] == "batch_size"
        assert result["parameters"][1]["value"] == 32

    @patch("comet_mcp.tools.get_comet_api")
    def test_get_experiment_details_not_found(self, mock_get_api):
        """Test handling when experiment is not found."""
        mock_api = Mock()
        mock_api.get_experiment_by_key.return_value = None
        mock_get_api.return_value.__enter__.return_value = mock_api

        with pytest.raises(Exception) as exc_info:
            get_experiment_details("nonexistent")

        assert "Experiment with ID 'nonexistent' not found" in str(exc_info.value)

    @patch("comet_mcp.tools.get_comet_api")
    def test_get_experiment_details_no_metrics_params(self, mock_get_api):
        """Test experiment details when no metrics or parameters exist."""
        mock_experiment = Mock()
        mock_experiment.id = "exp123"
        mock_experiment.name = "Simple Experiment"
        mock_experiment.get_state.return_value = "completed"
        mock_experiment.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment.end_server_timestamp = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment.url = "https://example.com/exp123"
        mock_experiment.description = None
        mock_experiment.get_metrics_summary.return_value = None
        mock_experiment.get_parameters_summary.return_value = None

        mock_api = Mock()
        mock_api.get_experiment_by_key.return_value = mock_experiment
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = get_experiment_details("exp123")

        assert result["metrics"] == []
        assert result["parameters"] == []
        assert result["description"] is None


class TestListProjects:
    """Test cases for list_projects tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_projects_success(self, mock_get_api):
        """Test successful listing of projects."""
        mock_api = Mock()
        mock_api.get_projects.return_value = ["project1", "project2"]
        mock_api.get_default_workspace.return_value = "default-workspace"

        # Mock project details
        mock_project1_details = {
            "projectName": "project1",
            "workspaceName": "default-workspace",
            "lastUpdated": datetime(2024, 1, 1, 12, 0, 0),
            "projectDescription": "First project",
        }
        mock_project2_details = {
            "projectName": "project2",
            "workspaceName": "default-workspace",
            "lastUpdated": datetime(2024, 1, 2, 12, 0, 0),
            "projectDescription": "Second project",
        }

        mock_api.get_project.side_effect = [
            mock_project1_details,
            mock_project2_details,
        ]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_projects()

        assert isinstance(result, list)
        assert len(result) == 2

        # Verify projects are sorted by date, latest first
        # project2 should be first (2024-01-02) and project1 second (2024-01-01)
        proj1 = result[0]  # Should be project2 (latest)
        assert isinstance(proj1, dict)
        assert proj1["name"] == "project2"
        assert proj1["workspace"] == "default-workspace"
        assert proj1["created_at"] == "2024-01-02T12:00:00"
        assert proj1["description"] == "Second project"

        proj2 = result[1]  # Should be project1 (earlier)
        assert isinstance(proj2, dict)
        assert proj2["name"] == "project1"
        assert proj2["workspace"] == "default-workspace"
        assert proj2["created_at"] == "2024-01-01T12:00:00"
        assert proj2["description"] == "First project"

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_projects_with_workspace(self, mock_get_api):
        """Test listing projects with specific workspace."""
        mock_api = Mock()
        mock_api.get_projects.return_value = ["project1"]

        mock_project_details = {
            "projectName": "project1",
            "workspaceName": "test-workspace",
            "lastUpdated": datetime(2024, 1, 1, 12, 0, 0),
            "projectDescription": "Test project",
        }
        mock_api.get_project.return_value = mock_project_details
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_projects(workspace="test-workspace")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["workspace"] == "test-workspace"
        mock_api.get_projects.assert_called_once_with(workspace="test-workspace")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_projects_sorting_by_date(self, mock_get_api):
        """Test that projects are sorted by date with latest first."""
        mock_api = Mock()
        mock_api.get_projects.return_value = [
            "old_project",
            "new_project",
            "middle_project",
        ]
        mock_api.get_default_workspace.return_value = "default-workspace"

        # Mock project details with different dates
        mock_old_project = {
            "projectName": "old_project",
            "workspaceName": "default-workspace",
            "lastUpdated": datetime(2024, 1, 1, 10, 0, 0),  # Oldest
            "projectDescription": "Old project",
        }
        mock_new_project = {
            "projectName": "new_project",
            "workspaceName": "default-workspace",
            "lastUpdated": datetime(2024, 1, 3, 15, 30, 0),  # Newest
            "projectDescription": "New project",
        }
        mock_middle_project = {
            "projectName": "middle_project",
            "workspaceName": "default-workspace",
            "lastUpdated": datetime(2024, 1, 2, 12, 0, 0),  # Middle
            "projectDescription": "Middle project",
        }

        mock_api.get_project.side_effect = [
            mock_old_project,
            mock_new_project,
            mock_middle_project,
        ]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_projects()

        assert isinstance(result, list)
        assert len(result) == 3

        # Verify projects are sorted by date, latest first
        # Expected order: new_project (2024-01-03), middle_project (2024-01-02), old_project (2024-01-01)
        assert result[0]["name"] == "new_project"
        assert result[0]["created_at"] == "2024-01-03T15:30:00"

        assert result[1]["name"] == "middle_project"
        assert result[1]["created_at"] == "2024-01-02T12:00:00"

        assert result[2]["name"] == "old_project"
        assert result[2]["created_at"] == "2024-01-01T10:00:00"

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_projects_empty_result(self, mock_get_api):
        """Test listing projects when no projects exist."""
        mock_api = Mock()
        mock_api.get_projects.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_projects()

        assert isinstance(result, list)
        assert len(result) == 0


class TestGetSessionInfo:
    """Test cases for get_session_info tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()

    @patch("comet_mcp.tools.get_session_context")
    @patch("comet_mcp.tools.get_comet_api")
    def test_get_session_info_initialized_success(self, mock_get_api, mock_get_context):
        """Test session info when session is initialized and API works."""
        # Mock session context
        mock_context = Mock()
        mock_context.is_initialized.return_value = True
        mock_get_context.return_value = mock_context

        # Mock API
        mock_api = Mock()
        mock_api.get_user_info.return_value = {"username": "testuser"}
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = get_session_info()

        assert isinstance(result, dict)
        assert result["initialized"] is True
        assert result["api_status"] == "Connected"
        assert result["user"] == "testuser"
        assert result["error"] is None

    @patch("comet_mcp.tools.get_session_context")
    @patch("comet_mcp.tools.get_comet_api")
    def test_get_session_info_initialized_no_user_info(
        self, mock_get_api, mock_get_context
    ):
        """Test session info when user info is not available."""
        mock_context = Mock()
        mock_context.is_initialized.return_value = True
        mock_get_context.return_value = mock_context

        mock_api = Mock()
        mock_api.get_user_info.side_effect = AttributeError("Method not available")
        mock_api.get_default_workspace.return_value = "test-workspace"
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = get_session_info()

        assert result["initialized"] is True
        assert result["api_status"] == "Connected"
        assert result["user"] == "Connected to workspace: test-workspace"
        assert result["error"] is None

    @patch("comet_mcp.tools.get_session_context")
    @patch("comet_mcp.tools.get_comet_api")
    def test_get_session_info_api_error(self, mock_get_api, mock_get_context):
        """Test session info when API connection fails."""
        mock_context = Mock()
        mock_context.is_initialized.return_value = True
        mock_get_context.return_value = mock_context

        mock_get_api.side_effect = Exception("Connection failed")

        result = get_session_info()

        assert result["initialized"] is True
        assert result["api_status"] == "Error"
        assert result["user"] is None
        assert result["error"] == "Connection failed"

    @patch("comet_mcp.tools.get_session_context")
    def test_get_session_info_not_initialized(self, mock_get_context):
        """Test session info when session is not initialized."""
        mock_context = Mock()
        mock_context.is_initialized.return_value = False
        mock_get_context.return_value = mock_context

        result = get_session_info()

        assert result["initialized"] is False
        assert result["api_status"] == "Not initialized"
        assert result["user"] is None
        assert result["error"] == "Comet ML session is not initialized."




class TestStructuredDataTypes:
    """Test that all tools return properly structured TypedDict data."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_experiment_info_structure(self, mock_get_api):
        """Test that list_experiments returns proper ExperimentInfo structure."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        mock_experiment = Mock()
        mock_experiment.id = "exp1"
        mock_experiment.name = "Test"
        mock_experiment.get_state.return_value = "completed"
        mock_experiment.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment.description = "Test description"
        mock_api.get_experiments.return_value = [mock_experiment]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_experiments()
        exp_info = result[0]

        # Verify all required fields exist
        assert "id" in exp_info
        assert "name" in exp_info
        assert "status" in exp_info
        assert "created_at" in exp_info
        assert "description" in exp_info

        # Verify field types
        assert isinstance(exp_info["id"], str)
        assert isinstance(exp_info["name"], str)
        assert isinstance(exp_info["status"], str)
        assert isinstance(exp_info["created_at"], str)
        # description can be str or None
        assert exp_info["description"] is None or isinstance(
            exp_info["description"], str
        )

    @patch("comet_mcp.tools.get_comet_api")
    def test_experiment_details_structure(self, mock_get_api):
        """Test that get_experiment_details returns proper ExperimentDetails structure."""
        mock_api = Mock()
        mock_experiment = Mock()
        mock_experiment.id = "exp1"
        mock_experiment.name = "Test"
        mock_experiment.get_state.return_value = "completed"
        mock_experiment.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment.end_server_timestamp = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment.url = "https://example.com/exp1"
        mock_experiment.description = "Test description"
        mock_experiment.get_metrics_summary.return_value = []
        mock_experiment.get_parameters_summary.return_value = []
        mock_api.get_experiment_by_key.return_value = mock_experiment
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = get_experiment_details("exp1")

        # Verify all required fields exist
        required_fields = [
            "id",
            "name",
            "status",
            "created_at",
            "updated_at",
            "description",
            "metrics",
            "parameters",
        ]
        for field in required_fields:
            assert field in result

        # Verify field types
        assert isinstance(result["id"], str)
        assert isinstance(result["name"], str)
        assert isinstance(result["status"], str)
        assert isinstance(result["created_at"], str)
        assert isinstance(result["updated_at"], str)
        assert isinstance(result["metrics"], list)
        assert isinstance(result["parameters"], list)
        assert result["description"] is None or isinstance(result["description"], str)

    @patch("comet_mcp.tools.get_comet_api")
    def test_project_info_structure(self, mock_get_api):
        """Test that list_projects returns proper ProjectInfo structure."""
        mock_api = Mock()
        mock_api.get_projects.return_value = ["project1"]
        mock_api.get_default_workspace.return_value = "workspace1"
        mock_api.get_project.return_value = {
            "projectName": "project1",
            "workspaceName": "workspace1",
            "lastUpdated": datetime(2024, 1, 1, 12, 0, 0),
            "projectDescription": "Test project",
        }
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_projects()
        project_info = result[0]

        # Verify all required fields exist
        required_fields = ["name", "workspace", "created_at", "description"]
        for field in required_fields:
            assert field in project_info

        # Verify field types
        assert isinstance(project_info["name"], str)
        assert isinstance(project_info["workspace"], str)
        assert isinstance(project_info["created_at"], str)
        assert project_info["description"] is None or isinstance(
            project_info["description"], str
        )

    @patch("comet_mcp.tools.get_session_context")
    def test_session_info_structure(self, mock_get_context):
        """Test that get_session_info returns proper SessionInfo structure."""
        mock_context = Mock()
        mock_context.is_initialized.return_value = False
        mock_get_context.return_value = mock_context

        result = get_session_info()

        # Verify all required fields exist
        required_fields = ["initialized", "api_status", "user", "error"]
        for field in required_fields:
            assert field in result

        # Verify field types
        assert isinstance(result["initialized"], bool)
        assert isinstance(result["api_status"], str)
        assert result["user"] is None or isinstance(result["user"], str)
        assert result["error"] is None or isinstance(result["error"], str)



class TestListProjectExperiments:
    """Test cases for list_project_experiments tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_project_experiments_success(self, mock_get_api):
        """Test successful listing of experiments in a project."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        
        mock_experiment1 = Mock()
        mock_experiment1.id = "exp1"
        mock_experiment1.name = "Project Experiment 1"
        mock_experiment1.get_state.return_value = "completed"
        mock_experiment1.start_server_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment1.description = "First project experiment"

        mock_experiment2 = Mock()
        mock_experiment2.id = "exp2"
        mock_experiment2.name = "Project Experiment 2"
        mock_experiment2.get_state.return_value = "running"
        mock_experiment2.start_server_timestamp = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment2.description = "Second project experiment"

        mock_api.get_experiments.return_value = [mock_experiment1, mock_experiment2]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_project_experiments("smoke-test")

        assert isinstance(result, list)
        assert len(result) == 2

        # Verify first experiment
        exp1 = result[0]
        assert exp1["id"] == "exp1"
        assert exp1["name"] == "Project Experiment 1"
        assert exp1["status"] == "completed"
        assert exp1["created_at"] == "2024-01-01T12:00:00"
        assert exp1["description"] == "First project experiment"

        # Verify second experiment
        exp2 = result[1]
        assert exp2["id"] == "exp2"
        assert exp2["name"] == "Project Experiment 2"
        assert exp2["status"] == "running"
        assert exp2["created_at"] == "2024-01-02T12:00:00"
        assert exp2["description"] == "Second project experiment"

        # Verify API was called correctly
        mock_api.get_experiments.assert_called_once_with("default-workspace", project_name="smoke-test")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_project_experiments_with_workspace(self, mock_get_api):
        """Test listing project experiments with specific workspace."""
        mock_api = Mock()
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_project_experiments("smoke-test", workspace="test-workspace")

        assert isinstance(result, list)
        assert len(result) == 0
        mock_api.get_experiments.assert_called_once_with("test-workspace", project_name="smoke-test")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_project_experiments_empty_result(self, mock_get_api):
        """Test listing project experiments when no experiments exist."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_project_experiments("empty-project")

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_project_experiments_api_error(self, mock_get_api):
        """Test handling of API errors."""
        mock_get_api.side_effect = Exception("API connection failed")

        with pytest.raises(Exception) as exc_info:
            list_project_experiments("smoke-test")

        assert "Error listing experiments for project 'smoke-test'" in str(exc_info.value)
        assert "API connection failed" in str(exc_info.value)


class TestCountProjectExperiments:
    """Test cases for count_project_experiments tool."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_count_project_experiments_success(self, mock_get_api):
        """Test successful counting of experiments in a project."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        
        mock_experiment1 = Mock()
        mock_experiment1.id = "exp1"
        mock_experiment1.name = "Project Experiment 1"
        mock_experiment1.state = "completed"
        mock_experiment1.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment1.description = "First project experiment"

        mock_experiment2 = Mock()
        mock_experiment2.id = "exp2"
        mock_experiment2.name = "Project Experiment 2"
        mock_experiment2.state = "running"
        mock_experiment2.created_at = datetime(2024, 1, 2, 12, 0, 0)
        mock_experiment2.description = "Second project experiment"

        mock_api.get_experiments.return_value = [mock_experiment1, mock_experiment2]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = count_project_experiments("smoke-test")

        assert isinstance(result, dict)
        assert result["project_name"] == "smoke-test"
        assert result["workspace"] == "default-workspace"
        assert result["experiment_count"] == 2
        assert isinstance(result["experiments"], list)
        assert len(result["experiments"]) == 2

        # Verify API was called correctly
        mock_api.get_experiments.assert_called_once_with("default-workspace", project_name="smoke-test")

    @patch("comet_mcp.tools.get_comet_api")
    def test_count_project_experiments_with_workspace(self, mock_get_api):
        """Test counting project experiments with specific workspace."""
        mock_api = Mock()
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = count_project_experiments("smoke-test", workspace="test-workspace")

        assert isinstance(result, dict)
        assert result["project_name"] == "smoke-test"
        assert result["workspace"] == "test-workspace"
        assert result["experiment_count"] == 0
        assert result["experiments"] == []
        mock_api.get_experiments.assert_called_once_with("test-workspace", project_name="smoke-test")

    @patch("comet_mcp.tools.get_comet_api")
    def test_count_project_experiments_empty_project(self, mock_get_api):
        """Test counting experiments in an empty project."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = count_project_experiments("empty-project")

        assert isinstance(result, dict)
        assert result["project_name"] == "empty-project"
        assert result["workspace"] == "default-workspace"
        assert result["experiment_count"] == 0
        assert result["experiments"] == []

    @patch("comet_mcp.tools.get_comet_api")
    def test_count_project_experiments_api_error(self, mock_get_api):
        """Test handling of API errors."""
        mock_get_api.side_effect = Exception("API connection failed")

        with pytest.raises(Exception) as exc_info:
            count_project_experiments("smoke-test")

        assert "Error counting experiments for project 'smoke-test'" in str(exc_info.value)
        assert "API connection failed" in str(exc_info.value)


class TestUpdatedListExperiments:
    """Test cases for updated list_experiments tool with project filtering."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        session_context.reset()
        session_context.initialize()

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_with_project_filter(self, mock_get_api):
        """Test listing experiments with project filter."""
        mock_api = Mock()
        mock_api.get_default_workspace.return_value = "default-workspace"
        
        mock_experiment = Mock()
        mock_experiment.id = "exp1"
        mock_experiment.name = "Filtered Experiment"
        mock_experiment.state = "completed"
        mock_experiment.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_experiment.description = "Filtered experiment"

        mock_api.get_experiments.return_value = [mock_experiment]
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_experiments(project_name="smoke-test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "exp1"
        assert result[0]["name"] == "Filtered Experiment"

        # Verify API was called with project filter
        mock_api.get_experiments.assert_called_once_with("default-workspace", project_name="smoke-test")

    @patch("comet_mcp.tools.get_comet_api")
    def test_list_experiments_with_workspace_and_project(self, mock_get_api):
        """Test listing experiments with both workspace and project filters."""
        mock_api = Mock()
        mock_api.get_experiments.return_value = []
        mock_get_api.return_value.__enter__.return_value = mock_api

        result = list_experiments(workspace="test-workspace", project_name="smoke-test")

        assert isinstance(result, list)
        assert len(result) == 0
        mock_api.get_experiments.assert_called_once_with("test-workspace", project_name="smoke-test")




if __name__ == "__main__":
    pytest.main([__file__])
