#!/usr/bin/env python3
"""
Comet ML tools for MCP server.
These tools require access to comet_ml.API() singleton.
"""

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from .utils import tool, format_datetime
from .session import get_comet_api, get_session_context


class ExperimentInfo(TypedDict):
    """Information about a Comet ML experiment."""

    id: str
    name: str
    status: str
    created_at: str
    description: Optional[str]


class ExperimentDetails(TypedDict):
    """Detailed information about a Comet ML experiment."""

    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    description: Optional[str]
    metrics: List[Dict[str, Any]]
    parameters: List[Dict[str, Any]]


class ProjectInfo(TypedDict):
    """Information about a Comet ML project."""

    name: str
    workspace: str
    created_at: str
    description: Optional[str]


class SessionInfo(TypedDict):
    """Information about the current Comet ML session."""

    initialized: bool
    api_status: str
    user: Optional[str]
    error: Optional[str]


class SearchResults(TypedDict):
    """Results from searching experiments."""

    query: str
    count: int
    experiments: List[ExperimentInfo]


@tool
def list_experiments(workspace: Optional[str] = None, project_name: Optional[str] = None) -> List[ExperimentInfo]:
    """
    List recent experiments from Comet ML.

    Args:
        workspace: Workspace name (optional, uses default if not provided)
        project_name: Project name to filter experiments (optional)
    """
    try:
        with get_comet_api() as api:
            # Determine target workspace
            if workspace:
                target_workspace = workspace
            else:
                target_workspace = api.get_default_workspace()

            # Get experiments with optional project filtering
            if project_name:
                experiments = api.get_experiments(target_workspace, project_name=project_name)
            else:
                experiments = api.get_experiments(target_workspace)

            if not experiments:
                return []

            result = []
            for exp in experiments:
                result.append(
                    ExperimentInfo(
                        id=exp.id,
                        name=exp.name,
                        status=exp.get_state(),
                        created_at=format_datetime(exp.start_server_timestamp),
                        description=getattr(exp, 'description', None),
                    )
                )

            return result

    except Exception as e:
        raise Exception(f"Error listing experiments: {str(e)}")


@tool
def get_experiment_details(experiment_id: str) -> ExperimentDetails:
    """
    Get detailed information about a specific experiment.

    Args:
        experiment_id: The ID of the experiment to retrieve
    """
    try:
        with get_comet_api() as api:
            experiment = api.get_experiment_by_key(experiment_id)

            if not experiment:
                raise Exception(f"Experiment with ID '{experiment_id}' not found.")

            # Get metrics
            metrics = experiment.get_metrics()
            metrics_list = []
            if metrics:
                for metric in metrics[:10]:  # Show first 10 metrics
                    metrics_list.append({"name": metric.name, "value": metric.value})

            # Get parameters
            params = experiment.get_parameters()
            params_list = []
            if params:
                for param in params[:10]:  # Show first 10 parameters
                    params_list.append({"name": param.name, "value": param.value})

            return ExperimentDetails(
                id=experiment.id,
                name=experiment.name,
                status=experiment.get_state(),
                created_at=format_datetime(experiment.start_server_timestamp),
                updated_at=format_datetime(experiment.end_server_timestamp or experiment.start_server_timestamp),
                description=getattr(experiment, 'description', None),
                metrics=metrics_list,
                parameters=params_list,
            )

    except Exception as e:
        raise Exception(f"Error getting experiment details: {str(e)}")


@tool
def list_projects(workspace: Optional[str] = None) -> List[ProjectInfo]:
    """
    List projects in the Comet ML workspace.

    Args:
        workspace: Workspace name (optional, uses default if not provided)
    """
    try:
        with get_comet_api() as api:
            if workspace:
                project_names = api.get_projects(workspace=workspace)
                target_workspace = workspace
            else:
                # Use default workspace when none is provided
                target_workspace = api.get_default_workspace()
                project_names = api.get_projects(target_workspace)

            if not project_names:
                return []

            result = []
            for project_name in project_names:
                # Get detailed project information
                project_details = api.get_project(target_workspace, project_name)
                result.append(
                    ProjectInfo(
                        name=project_details.get("projectName", project_name),
                        workspace=project_details.get(
                            "workspaceName", target_workspace
                        ),
                        created_at=format_datetime(project_details.get("lastUpdated")),
                        description=project_details.get("projectDescription", ""),
                    )
                )

            # Sort by date, latest first
            result.sort(key=lambda x: x["created_at"], reverse=True)

            return result

    except Exception as e:
        raise Exception(f"Error listing projects: {str(e)}")


@tool
def get_session_info() -> SessionInfo:
    """
    Get information about the current Comet ML session.
    """
    try:
        context = get_session_context()

        if not context.is_initialized():
            return SessionInfo(
                initialized=False,
                api_status="Not initialized",
                user=None,
                error="Comet ML session is not initialized.",
            )

        # Test API connection
        try:
            with get_comet_api() as api:
                # Try to get user info to verify connection
                try:
                    user_info = api.get_user_info()
                    username = (
                        user_info.get("username", "Unknown") if user_info else "Unknown"
                    )
                except AttributeError:
                    # get_user_info method doesn't exist, try alternative approach
                    try:
                        # Try to get workspace info as a connection test
                        workspace = api.get_default_workspace()
                        username = (
                            f"Connected to workspace: {workspace}"
                            if workspace
                            else "Connected"
                        )
                    except Exception:
                        username = "Connected (user info unavailable)"

                return SessionInfo(
                    initialized=True, api_status="Connected", user=username, error=None
                )
        except Exception as e:
            return SessionInfo(
                initialized=True, api_status="Error", user=None, error=str(e)
            )

    except Exception as e:
        return SessionInfo(
            initialized=False,
            api_status="Error",
            user=None,
            error=f"Error getting session info: {str(e)}",
        )


@tool
def list_project_experiments(project_name: str, workspace: Optional[str] = None) -> List[ExperimentInfo]:
    """
    List experiments in a specific project.

    Args:
        project_name: Name of the project to get experiments from
        workspace: Workspace name (optional, uses default if not provided)
    """
    try:
        with get_comet_api() as api:
            # Determine target workspace
            if workspace:
                target_workspace = workspace
            else:
                target_workspace = api.get_default_workspace()

            # Get experiments for the specific project
            experiments = api.get_experiments(target_workspace, project_name=project_name)

            if not experiments:
                return []

            result = []
            for exp in experiments:
                result.append(
                    ExperimentInfo(
                        id=exp.id,
                        name=exp.name,
                        status=exp.get_state(),
                        created_at=format_datetime(exp.start_server_timestamp),
                        description=getattr(exp, 'description', None),
                    )
                )

            return result

    except Exception as e:
        # Handle Comet ML specific exceptions
        if "No such project" in str(e):
            return []
        raise Exception(f"Error listing experiments for project '{project_name}': {str(e)}")


@tool
def count_project_experiments(project_name: str, workspace: Optional[str] = None) -> Dict[str, Any]:
    """
    Count experiments in a specific project.

    Args:
        project_name: Name of the project to count experiments in
        workspace: Workspace name (optional, uses default if not provided)
    """
    try:
        with get_comet_api() as api:
            # Determine target workspace
            if workspace:
                target_workspace = workspace
            else:
                target_workspace = api.get_default_workspace()

            # Get experiments for the specific project
            experiments = api.get_experiments(target_workspace, project_name=project_name)

            count = len(experiments) if experiments else 0

            return {
                "project_name": project_name,
                "workspace": target_workspace,
                "experiment_count": count,
                "experiments": experiments if experiments else []
            }

    except Exception as e:
        # Handle Comet ML specific exceptions
        if "No such project" in str(e):
            return {
                "project_name": project_name,
                "workspace": target_workspace if 'target_workspace' in locals() else "unknown",
                "experiment_count": 0,
                "experiments": []
            }
        raise Exception(f"Error counting experiments for project '{project_name}': {str(e)}")


@tool
def search_experiments(query: str, workspace: Optional[str] = None, project_name: Optional[str] = None) -> SearchResults:
    """
    Search for experiments by name or description.

    Args:
        query: Search query to match against experiment names or descriptions
        workspace: Workspace name (optional, uses default if not provided)
        project_name: Project name to filter experiments (optional)
    """
    try:
        with get_comet_api() as api:
            # Determine target workspace
            if workspace:
                target_workspace = workspace
            else:
                target_workspace = api.get_default_workspace()

            # Get experiments with optional project filtering
            if project_name:
                experiments = api.get_experiments(target_workspace, project_name=project_name)
            else:
                experiments = api.get_experiments(target_workspace)

            if not experiments:
                return SearchResults(query=query, count=0, experiments=[])

            # Simple text search
            matching_experiments = []
            query_lower = query.lower()

            for exp in experiments:
                if query_lower in exp.name.lower() or (
                    exp.description and query_lower in exp.description.lower()
                ):
                    matching_experiments.append(
                        ExperimentInfo(
                            id=exp.id,
                            name=exp.name,
                            status=exp.state,
                            created_at=format_datetime(exp.created_at),
                            description=exp.description,
                        )
                    )

            return SearchResults(
                query=query,
                count=len(matching_experiments),
                experiments=matching_experiments,
            )

    except Exception as e:
        raise Exception(f"Error searching experiments: {str(e)}")
