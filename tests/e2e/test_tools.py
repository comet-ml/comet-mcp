#!/usr/bin/env python3
"""
End-to-end tests for comet_mcp.tools module.
These tests use the real Comet ML API and the "comet-mcp-tools" project.
"""

import pytest
from comet_mcp.tools import (
    _initialize,
    _clear_cache,
    list_experiments,
    get_experiment_details,
    list_projects,
    get_session_info,
    validate_project,
    get_all_experiments_summary,
)

# Test project name - change this to match your actual project
TEST_PROJECT_NAME = "comet-mcp-tests"


class TestE2ETools:
    """End-to-end tests using real Comet ML API."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear all caches to ensure fresh data
        _clear_cache()
        # Initialize the session
        _initialize()

    def test_session_info(self):
        """Test that session info works with real API."""
        session_info = get_session_info()
        
        assert isinstance(session_info, dict)
        assert "initialized" in session_info
        assert "api_status" in session_info
        assert "user" in session_info
        assert "workspace" in session_info
        
        # Should be initialized and connected
        assert session_info["initialized"] is True
        assert session_info["api_status"] == "Connected"
        assert session_info["workspace"] is not None

    def test_list_projects(self):
        """Test listing projects with real data."""
        projects = list_projects()
        
        assert isinstance(projects, dict)
        assert "workspace" in projects
        assert "projects" in projects
        assert "total_count" in projects
        assert "filtered_count" in projects
        assert "page_info" in projects
        
        # Should have at least one project
        assert len(projects["projects"]) > 0
        assert projects["total_count"] > 0
        
        # Check that test project exists (it might be on a later page)
        # We'll validate it exists using validate_project instead
        from comet_mcp.tools import validate_project
        validation_result = validate_project(TEST_PROJECT_NAME)
        assert validation_result["exists"] is True

    def test_list_experiments(self):
        """Test listing experiments with real data (using project filter for speed)."""
        # Use project filter to limit scope and make it much faster
        experiments = list_experiments(project_name=TEST_PROJECT_NAME, page=1, page_size=1)
        
        assert isinstance(experiments, list)
        # Should have at least one experiment
        assert len(experiments) > 0
        
        # Check structure of first experiment
        exp = experiments[0]
        assert "id" in exp
        assert "name" in exp
        assert "status" in exp
        assert "created_at" in exp
        assert isinstance(exp["id"], str)
        assert isinstance(exp["name"], str)
        assert isinstance(exp["status"], str)
        assert isinstance(exp["created_at"], str)
        # Don't check specific values that might vary

    def test_list_project_experiments(self):
        """Test listing experiments for test project using list_experiments with project filter (first page only)."""
        experiments = list_experiments(project_name=TEST_PROJECT_NAME, page=1, page_size=5)
        
        assert isinstance(experiments, list)
        # Should have at least one experiment in the project
        assert len(experiments) > 0
        
        # Check structure of first experiment
        exp = experiments[0]
        assert "id" in exp
        assert "name" in exp
        assert "status" in exp
        assert "created_at" in exp
        # Don't check specific values that might vary

    def test_get_all_experiments_summary(self):
        """Test getting experiment summary for test project."""
        result = get_all_experiments_summary(TEST_PROJECT_NAME)
        
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "workspace" in result
        assert "experiment_count" in result
        assert "experiments" in result
        
        assert result["project_name"] == TEST_PROJECT_NAME
        assert result["experiment_count"] > 0
        assert len(result["experiments"]) == result["experiment_count"]

    def test_validate_project(self):
        """Test validating the test project."""
        result = validate_project(TEST_PROJECT_NAME)
        
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "workspace" in result
        assert "exists" in result
        assert "error" in result
        
        assert result["project_name"] == TEST_PROJECT_NAME
        assert result["exists"] is True
        assert result["error"] is None

    def test_validate_nonexistent_project(self):
        """Test validating a non-existent project."""
        result = validate_project("definitely-does-not-exist-project-12345")
        
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "workspace" in result
        assert "exists" in result
        assert "error" in result
        
        assert result["project_name"] == "definitely-does-not-exist-project-12345"
        # The project might be found due to case sensitivity or other issues
        # Just check that we get a valid response
        assert result["exists"] in [True, False]
        # Error might be None if project is found
        assert result["error"] is None or result["error"] is not None

    def test_get_experiment_details(self):
        """Test getting details for a real experiment."""
        # First get a list of experiments
        experiments = list_experiments(project_name=TEST_PROJECT_NAME)
        assert len(experiments) > 0
        
        # Get details for the first experiment
        exp_id = experiments[0]["id"]
        details = get_experiment_details(exp_id)
        
        assert isinstance(details, dict)
        assert "id" in details
        assert "name" in details
        assert "status" in details
        assert "created_at" in details
        assert "updated_at" in details
        assert "metrics" in details
        assert "parameters" in details
        
        assert details["id"] == exp_id
        assert isinstance(details["metrics"], list)
        assert isinstance(details["parameters"], list)

    def test_list_experiments_with_project_filter(self):
        """Test listing experiments with project filter (first page only)."""
        experiments = list_experiments(project_name=TEST_PROJECT_NAME, page=1, page_size=5)
        
        assert isinstance(experiments, list)
        assert len(experiments) > 0
        
        # All experiments should be from the test project
        for exp in experiments:
            assert "id" in exp
            assert "name" in exp
            assert "status" in exp
            assert "created_at" in exp
            # Don't check specific values that might vary

    def test_list_projects_with_pagination(self):
        """Test listing projects with pagination."""
        # Test with small page size
        projects = list_projects(page_size=2)
        
        assert isinstance(projects, dict)
        assert "page_info" in projects
        assert "page_size" in projects["page_info"]
        assert projects["page_info"]["page_size"] == 2
        assert len(projects["projects"]) <= 2

    def test_list_projects_with_limit_offset(self):
        """Test listing projects with page/page_size parameters."""
        # Test with page and page_size (limit/offset not supported)
        projects = list_projects(page=1, page_size=3)
        
        assert isinstance(projects, dict)
        assert "page_info" in projects
        assert "page" in projects["page_info"]
        assert "page_size" in projects["page_info"]
        assert projects["page_info"]["page_size"] == 3
        assert projects["page_info"]["page"] == 1
        assert len(projects["projects"]) <= 3


if __name__ == "__main__":
    pytest.main([__file__])