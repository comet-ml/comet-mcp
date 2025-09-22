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
    url: str
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
    workspace: Optional[str]
    error: Optional[str]


class SearchResults(TypedDict):
    """Results from searching experiments."""

    query: str
    count: int
    experiments: List[ExperimentInfo]


@tool
def list_experiments(
    workspace: Optional[str] = None, project_name: Optional[str] = None
) -> List[ExperimentInfo]:
    """
    List recent experiments from Comet ML.

    Args:
        workspace: Workspace name (optional, uses default if not provided)
        project_name: Project name to filter experiments (optional)
    """
    with get_comet_api() as api:
        # Determine target workspace
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        # Get experiments with optional project filtering
        if project_name:
            experiments = api.get_experiments(
                target_workspace, project_name=project_name
            )
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
                    description=getattr(exp, "description", None),
                )
            )

        return result


@tool
def get_experiment_code(experiment_id: str) -> Dict[str, str]:
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)
        return {"code": experiment.get_code()}


@tool
def get_experiment_details(experiment_id: str) -> ExperimentDetails:
    """
    Get detailed information about a specific experiment.

    Args:
        experiment_id: The ID of the experiment to retrieve
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)

        if not experiment:
            raise Exception(f"Experiment with ID '{experiment_id}' not found.")

        # Get metrics
        metrics = experiment.get_metrics_summary()
        metrics_list = []
        if metrics:
            for metric in metrics[:10]:  # Show first 10 metrics
                metrics_list.append(
                    {"name": metric["name"], "value": metric["valueCurrent"]}
                )

        # Get parameters
        params = experiment.get_parameters_summary()
        params_list = []
        if params:
            for param in params[:10]:  # Show first 10 parameters
                params_list.append(
                    {"name": param["name"], "value": param["valueCurrent"]}
                )

        return ExperimentDetails(
            id=experiment.id,
            url=experiment.url,
            name=experiment.name,
            status=experiment.get_state(),
            created_at=format_datetime(experiment.start_server_timestamp),
            updated_at=format_datetime(
                experiment.end_server_timestamp or experiment.start_server_timestamp
            ),
            description=getattr(experiment, "description", None),
            metrics=metrics_list,
            parameters=params_list,
        )


@tool
def list_projects(workspace: Optional[str] = None) -> List[ProjectInfo]:
    """
    List projects in the Comet ML workspace.

    Args:
        workspace: Workspace name (optional, uses default if not provided)
    """
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
                    workspace=project_details.get("workspaceName", target_workspace),
                    created_at=format_datetime(project_details.get("lastUpdated")),
                    description=project_details.get("projectDescription", ""),
                )
            )

        # Sort by date, latest first
        result.sort(key=lambda x: x["created_at"], reverse=True)

        return result


@tool
def get_session_info() -> SessionInfo:
    """
    Get information about the current Comet ML session.
    """
    with get_comet_api() as api:
        workspace = api.get_default_workspace()
        return SessionInfo(
            initialized=True, api_status="Connected", workspace=workspace, error=None
        )


@tool
def list_project_experiments(
    project_name: str, workspace: Optional[str] = None
) -> List[ExperimentInfo]:
    """
    List experiments in a specific project.

    Args:
        project_name: Name of the project to get experiments from
        workspace: Workspace name (optional, uses default if not provided)
    """
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
                    description=getattr(exp, "description", None),
                )
            )

        return result


@tool
def count_project_experiments(
    project_name: str, workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Count experiments in a specific project.

    Args:
        project_name: Name of the project to count experiments in
        workspace: Workspace name (optional, uses default if not provided)
    """
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
            "experiments": experiments if experiments else [],
        }
