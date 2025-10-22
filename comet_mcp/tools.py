#!/usr/bin/env python3
"""
Comet ML tools for MCP server.
These tools require access to comet_ml.API() singleton.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from comet_mcp.utils import format_datetime
from comet_mcp.session import get_comet_api, get_session_context
from comet_mcp.cache import cached


def _get_state(metadata):
    if metadata["running"]:
        return "running"

    if metadata["hasCrashed"]:
        return "crashed"

    return "finished"


@cached(ttl_seconds=300)  # Cache for 5 minutes
def list_experiments(
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List recent experiments from Comet ML. Typically, don't show the
    user the experiment_id unless they ask to see it.

    Args:
        workspace: Workspace name (optional, will lookup the default workspace name if not provided)
        project_name: Project name to filter experiments (optional)
        page: get paged results, starting with page 1
        page_size: get this number of experiments at a time
        sort_by: Field to sort by. Must be "startTime" or "endTime" if provided.
        sort_order: Sort direction. Must be "asc" or "desc" if provided.
            Required when page, page_size, and sort_by are all specified.

    Returns:
        List of dictionaries containing experiment details:
        - id: Unique experiment identifier
        - name: Human-readable experiment name
        - status: Current experiment state (e.g., "running", "finished")
        - created_at: Formatted timestamp when experiment was created
        - description: Optional experiment description if available
    """
    with get_comet_api() as api:
        # Determine target workspace
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        # Get experiments for the specified workspace and project
        if project_name:
            experiments = api.get_experiments(
                workspace=target_workspace,
                project_name=project_name,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        else:
            experiments = api.get_experiments(
                workspace=target_workspace,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        if not experiments:
            return []

        result = []
        for exp in experiments:
            result.append(
                {
                    "id": exp.id,
                    "name": exp.name,
                    "status": exp.get_state(),
                    "created_at": format_datetime(exp.start_server_timestamp),
                    "description": getattr(exp, "description", None),
                }
            )

        return result


@cached(ttl_seconds=3600)  # Cache for 1 hour
def get_default_workspace() -> str:
    """
    Get the default workspace name for this user.

    Returns:
        String containing the default workspace name for the authenticated user.
        This is the workspace that will be used when no workspace is explicitly specified.
    """
    with get_comet_api() as api:
        return api.get_default_workspace()


def get_experiment_code(experiment_id: str) -> Dict[str, str]:
    """
    Get the code for a specific experiment.

    Args:
        experiment_id: The ID of the experiment to retrieve

    Returns:
        Dictionary containing:
        - code: String containing the complete source code that was logged for this experiment
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)
        return {"code": experiment.get_code()}


def get_experiment_metric_data(
    experiment_ids: List[str], metric_names: List[str], x_axis: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get multiple metric data for specific experiments. Use this tool to
    get metrics for multiple experiments at once. You must pass in at
    least one experiment ID, and at least one metric name. Only use
    this tool if you want the entire metric values.

    Args:
        experiment_ids: List of experiment IDs to retrieve metrics for
        metric_names: List of metric names to retrieve
        x_axis: The name of the x-axis to retrieve (optional). Must be: "steps", "epochs", "timestamps", or "durations".
                If not provided, will try in order of priority: steps, epochs, timestamps, durations

    Returns:
        Dictionary containing:
        - experiment_ids: List of experiment IDs that were requested
        - x_axis: The x-axis type used (either specified or auto-selected)
        - experiments: Dictionary mapping experiment IDs to their metric data:
          Each experiment contains a dictionary of metric names, where each metric has:
          - metric_name: The name of the metric
          - x_axis: The x-axis type used for this metric
          - data: List of (x, y) coordinate pairs representing the metric values over time
    """
    with get_comet_api() as api:
        data = api.get_metrics_for_chart(experiment_ids, metric_names)

        results = {}

        # Process each experiment
        for experiment_id in experiment_ids:
            if experiment_id not in data:
                continue  # Skip experiments not found in data

            experiment_data = data[experiment_id]
            if not experiment_data or experiment_data.get("empty", True):
                continue  # Skip experiments without data

            experiment_metrics = {}
            experiment_has_data = False

            # Process each metric for this experiment
            for metric_name in metric_names:
                # Find the metric in the metrics list
                metric_data = None
                for metric in experiment_data.get("metrics", []):
                    if metric.get("metricName") == metric_name:
                        metric_data = metric
                        break

                if not metric_data:
                    continue  # Skip metrics not found for this experiment

                values = metric_data["values"]

                # Handle x_axis selection with priority ordering
                if x_axis is not None:
                    # Use provided x_axis if available
                    current_x_axis = x_axis
                    if (
                        current_x_axis not in metric_data
                        or metric_data[current_x_axis] is None
                    ):
                        # Try to find an available x_axis in the order specified in docstring
                        for fallback_axis in [
                            "steps",
                            "epochs",
                            "timestamps",
                            "durations",
                        ]:
                            if (
                                fallback_axis in metric_data
                                and metric_data[fallback_axis] is not None
                            ):
                                current_x_axis = fallback_axis
                                break
                        else:
                            # If no standard x_axis is found, skip this metric
                            continue
                else:
                    # No x_axis provided, try in order of priority: steps, epochs, timestamps, durations
                    current_x_axis = None
                    for priority_axis in ["steps", "epochs", "timestamps", "durations"]:
                        if (
                            priority_axis in metric_data
                            and metric_data[priority_axis] is not None
                        ):
                            current_x_axis = priority_axis
                            break

                    if current_x_axis is None:
                        # If no standard x_axis is found, skip this metric
                        continue

                x_axis_values = metric_data[current_x_axis]

                # Convert timestamps to datetime objects if needed
                if current_x_axis == "timestamps":
                    x_axis_values = [
                        datetime.fromtimestamp(value) for value in x_axis_values
                    ]

                # Store metric data with metric name included
                experiment_metrics[metric_name] = {
                    "metric_name": metric_name,
                    "x_axis": current_x_axis,
                    "data": list(zip(x_axis_values, values)),  # (x, y) pairs
                }
                experiment_has_data = True

            # Only include experiments that have data
            if experiment_has_data:
                results[experiment_id] = experiment_metrics

        return {
            "experiment_ids": experiment_ids,
            "x_axis": x_axis or "auto-selected",
            "experiments": results,
        }


def get_experiment_details(experiment_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific experiment, including
    metric and parameter names.

    Args:
        experiment_id: The ID of the experiment to retrieve

    Returns:
        Dictionary containing:
        - id: Unique experiment identifier
        - url: Direct URL to view the experiment in Comet ML web interface
        - name: Human-readable experiment name
        - status: Current experiment state (e.g., "running", "finished")
        - created_at: Formatted timestamp when experiment was created
        - updated_at: Formatted timestamp when experiment was last updated
        - description: Optional experiment description if available
        - metrics: List of dictionaries with metric names and current values
        - parameters: List of dictionaries with parameter names and current values
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)

        if not experiment:
            raise Exception(f"Experiment with ID '{experiment_id}' not found.")

        # Get metrics
        metrics = experiment.get_metrics_summary()
        metrics_list = []
        if metrics:
            for metric in metrics:
                metrics_list.append(
                    {"name": metric["name"], "value": metric.get("valueCurrent", 0)}
                )

        # Get parameters
        params = experiment.get_parameters_summary()
        params_list = []
        if params:
            for param in params:
                params_list.append(
                    {"name": param["name"], "value": param.get("valueCurrent", "")}
                )

        return {
            "id": experiment.id,
            "url": experiment.url,
            "name": experiment.name,
            "status": experiment.get_state(),
            "created_at": format_datetime(experiment.start_server_timestamp),
            "updated_at": format_datetime(
                experiment.end_server_timestamp or experiment.start_server_timestamp
            ),
            "description": getattr(experiment, "description", None),
            "metrics": metrics_list,
            "parameters": params_list,
        }


@cached(ttl_seconds=1800)  # Cache for 30 minutes (projects change rarely)
def list_projects(
    workspace: Optional[str] = None,
    prefix: Optional[str] = None,
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
) -> Dict[str, Any]:
    """
    List project names in a Comet ML workspace with filtering and pagination support.

    IMPORTANT: This function is expensive as it fetches all projects from the workspace.
    Do NOT use this function to validate if a specific project exists. Instead, use
    validate_project() which is much faster for validation purposes.

    Args:
        workspace: Workspace name (optional, will lookup the default workspace if not provided)
        prefix: Filter projects by name prefix (optional, case-insensitive)
        page: Page number to retrieve (default: 1, starts from 1)
        page_size: Number of projects to return per page (default: 10, max: 100)

    Returns:
        Dictionary containing:
        - workspace: The workspace name that was searched
        - projects: List of project names matching the criteria (sorted alphabetically)
        - total_count: Total number of projects in the workspace
        - filtered_count: Number of projects matching the prefix filter (if prefix provided)
        - page_info: Dictionary with pagination metadata:
          - page: Current page number
          - page_size: Number of projects per page
          - has_more: Boolean indicating if more results are available
          - returned_count: Actual number of projects returned in this page
    """
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = get_default_workspace()

        # Get all projects from the workspace
        all_projects = sorted(api.get_projects(workspace=target_workspace))
        total_count = len(all_projects)

        # Apply prefix filtering if provided
        if prefix:
            prefix_lower = prefix.lower()
            filtered_projects = [
                project
                for project in all_projects
                if project.lower().startswith(prefix_lower)
            ]
        else:
            filtered_projects = all_projects

        filtered_count = len(filtered_projects)

        # Apply pagination
        # Ensure page_size doesn't exceed maximum
        page_size = min(page_size, 100)

        # Calculate pagination bounds
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Get the page of results
        page_projects = filtered_projects[start_idx:end_idx]

        # Determine if there are more results
        has_more = end_idx < filtered_count

        return {
            "workspace": target_workspace,
            "projects": page_projects,
            "total_count": total_count,
            "filtered_count": filtered_count,
            "page_info": {
                "page": page,
                "page_size": page_size,
                "has_more": has_more,
                "returned_count": len(page_projects),
            },
        }


def get_project_details(project_name: str, workspace: Optional[str]) -> Dict[str, Any]:
    """
    Get detailed information about a project.

    Note: If you need to validate that a project exists before calling this function,
    use validate_project() instead of list_projects() for better performance.

    Args:
        project_name: the name of the project of which to get details
        workspace: The workspace name

    Returns:
        Dictionary containing:
        - name: The project name
        - workspace: The workspace name where the project is located
        - created_at: Formatted timestamp when the project was created
        - description: Project description if available (empty string if none)
    """
    # Get detailed project information
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = get_default_workspace()

        project_details = api.get_project(target_workspace, project_name)
        return {
            "name": project_details.get("projectName", project_name),
            "workspace": project_details.get("workspaceName", target_workspace),
            "created_at": format_datetime(project_details.get("lastUpdated")),
            "description": project_details.get("projectDescription", ""),
        }


@cached(ttl_seconds=600)  # Cache for 10 minutes
def get_session_info() -> Dict[str, Any]:
    """
    Get information about the current Comet ML session.

    Returns:
        Dictionary containing:
        - initialized: Boolean indicating if the Comet ML session is properly initialized
        - api_status: String describing the connection status ("Connected", "Not initialized", "Error")
        - user: Username of the authenticated user, or workspace info if user info unavailable
        - workspace: Default workspace name for the user
        - error: Error message if there was a problem, None if successful
    """
    session_context = get_session_context()

    if not session_context.is_initialized():
        return {
            "initialized": False,
            "api_status": "Not initialized",
            "user": None,
            "workspace": None,
            "error": "Comet ML session is not initialized.",
        }

    try:
        with get_comet_api() as api:
            workspace = get_default_workspace()

            # Try to get user info
            try:
                user_info = api.get_user_info()
                user = user_info.get("username") if user_info else None
            except (AttributeError, Exception):
                # Fallback to workspace info if user info not available
                user = f"Connected to workspace: {workspace}"

            return {
                "initialized": True,
                "api_status": "Connected",
                "user": user,
                "workspace": workspace,
                "error": None,
            }
    except Exception as e:
        return {
            "initialized": True,
            "api_status": "Error",
            "user": None,
            "workspace": None,
            "error": str(e),
        }


def get_all_experiments_summary(
    project_name: str, workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get count and experiment summary in a specific project.

    Note: If you need to validate that a project exists before calling this function,
    use validate_project() instead of list_projects() for better performance.

    Args:
        project_name: Name of the project to count experiments in
        workspace: Workspace name (optional, uses default if not provided)

    Returns:
        Dictionary containing:
        - project_name: The name of the project that was counted
        - workspace: The workspace name where the project is located
        - experiment_count: Total number of experiments in the project
        - experiments: List of dictionaries with basic experiment details:
          - id: Unique experiment identifier
          - name: Human-readable experiment name
          - status: Current experiment state
          - created_at: Formatted timestamp when experiment was created
    """
    with get_comet_api() as api:
        # Determine target workspace
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = get_default_workspace()

        # Get experiments for the specific project; could be paged
        experiments = api._get_project_experiments(
            target_workspace,
            project_name,
        )
        count = len(experiments) if experiments else 0
        for metadatum in experiments.values():
            return {
                "project_name": project_name,
                "workspace": target_workspace,
                "experiment_count": count,
                "experiments": (
                    [
                        {
                            "id": exp["experimentKey"],
                            "name": exp["experimentName"],
                            "status": _get_state(exp),
                            "created_at": format_datetime(exp["startTimeMillis"]),
                        }
                        for exp in experiments.values()
                    ]
                    if experiments
                    else []
                ),
            }


def validate_project(
    project_name: str, workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lightweight validation to check if a project exists without listing all projects.
    This is much faster than list_projects() and should be used for validation purposes.

    Args:
        project_name: Name of the project to validate
        workspace: Workspace name (optional, will lookup the default workspace if not provided)

    Returns:
        Dictionary containing:
        - project_name: The name of the project that was validated
        - workspace: The workspace name where the project was searched
        - exists: Boolean indicating if the project exists
        - error: Error message if validation failed, None if successful
    """
    with get_comet_api() as api:
        try:
            # Determine target workspace
            if workspace:
                target_workspace = workspace
            else:
                target_workspace = get_default_workspace()

            # Try to get project details - this will fail if project doesn't exist
            project_details = api.get_project(target_workspace, project_name)

            return {
                "project_name": project_name,
                "workspace": target_workspace,
                "exists": True,
                "error": None,
            }
        except Exception as e:
            return {
                "project_name": project_name,
                "workspace": target_workspace,
                "exists": False,
                "error": str(e),
            }


def get_experiment_summary(experiment_id: str) -> Dict[str, Any]:
    """
    Get a summary of experiment performance with final/best metric values.
    Use this for performance comparison and final results analysis.

    Args:
        experiment_id: The ID of the experiment to retrieve summary for

    Returns:
        Dictionary containing:
        - id: Unique experiment identifier
        - name: Human-readable experiment name
        - status: Current experiment state
        - final_metrics: Dictionary of final/best metric values
        - best_metrics: Dictionary of best metric values achieved during training
        - created_at: Formatted timestamp when experiment was created
        - updated_at: Formatted timestamp when experiment was last updated
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)

        if not experiment:
            raise Exception(f"Experiment with ID '{experiment_id}' not found.")

        # Get final metrics
        final_metrics = {}
        best_metrics = {}

        metrics_summary = experiment.get_metrics_summary()
        if metrics_summary:
            for metric in metrics_summary:
                metric_name = metric["name"]
                final_metrics[metric_name] = metric.get("valueCurrent", 0)
                best_metrics[metric_name] = metric.get(
                    "valueMax", metric.get("valueCurrent", 0)
                )

        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.get_state(),
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "created_at": format_datetime(experiment.start_server_timestamp),
            "updated_at": format_datetime(
                experiment.end_server_timestamp or experiment.start_server_timestamp
            ),
        }


def get_experiment_training_progress(
    experiment_id: str, metric_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get detailed training progress data for an experiment.
    ⚠️ WARNING: This is an EXPENSIVE operation that fetches ALL training data.
    Use ONLY when specifically analyzing learning curves, convergence patterns,
    or investigating overfitting/underfitting with step-by-step data.

    Args:
        experiment_id: The ID of the experiment to retrieve training progress for
        metric_names: Optional list of specific metric names to retrieve.
                     If None, retrieves all available metrics.

    Returns:
        Dictionary containing:
        - id: Unique experiment identifier
        - name: Human-readable experiment name
        - training_metrics: Dictionary mapping metric names to their training data:
          Each metric contains:
          - metric_name: The name of the metric
          - x_axis: The x-axis type used (steps, epochs, timestamps, or durations)
          - data: List of (x, y) coordinate pairs representing the metric values over time
        - available_metrics: List of all available metric names for this experiment
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)

        if not experiment:
            raise Exception(f"Experiment with ID '{experiment_id}' not found.")

        # Get all available metrics if none specified
        if metric_names is None:
            metrics_summary = experiment.get_metrics_summary()
            metric_names = (
                [metric["name"] for metric in metrics_summary]
                if metrics_summary
                else []
            )

        # Get training data for specified metrics
        training_data = api.get_metrics_for_chart([experiment_id], metric_names)

        training_metrics = {}
        available_metrics = []

        if experiment_id in training_data and training_data[experiment_id]:
            experiment_data = training_data[experiment_id]

            if not experiment_data.get("empty", True):
                for metric in experiment_data.get("metrics", []):
                    metric_name = metric.get("metricName")
                    if metric_name in metric_names:
                        available_metrics.append(metric_name)

                        # Determine best x_axis to use
                        x_axis = None
                        for axis in ["steps", "epochs", "timestamps", "durations"]:
                            if axis in metric and metric[axis] is not None:
                                x_axis = axis
                                break

                        if x_axis:
                            x_values = metric[x_axis]
                            y_values = metric["values"]

                            # Convert timestamps to datetime objects if needed
                            if x_axis == "timestamps":
                                x_values = [
                                    datetime.fromtimestamp(val) for val in x_values
                                ]

                            training_metrics[metric_name] = {
                                "metric_name": metric_name,
                                "x_axis": x_axis,
                                "data": list(zip(x_values, y_values)),
                            }

        return {
            "id": experiment.id,
            "name": experiment.name,
            "training_metrics": training_metrics,
            "available_metrics": available_metrics,
        }


def get_experiment_parameters(experiment_id: str) -> Dict[str, Any]:
    """
    Get experiment parameters and configuration settings.
    Use this for configuration analysis and hyperparameter investigation.

    Args:
        experiment_id: The ID of the experiment to retrieve parameters for

    Returns:
        Dictionary containing:
        - id: Unique experiment identifier
        - name: Human-readable experiment name
        - parameters: Dictionary mapping parameter names to their values
        - parameter_count: Total number of parameters found
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)

        if not experiment:
            raise Exception(f"Experiment with ID '{experiment_id}' not found.")

        # Get parameters summary
        params_summary = experiment.get_parameters_summary()
        parameters = {}

        if params_summary:
            for param in params_summary:
                param_name = param["name"]
                param_value = param.get("valueCurrent", "")
                parameters[param_name] = param_value

        return {
            "id": experiment.id,
            "name": experiment.name,
            "parameters": parameters,
            "parameter_count": len(parameters),
        }


def _get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.

    Returns:
        Dictionary containing cache statistics and session information.
    """
    from comet_mcp.cache import get_cache_info

    return get_cache_info()


def _clear_cache(func_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear cache entries.

    Args:
        func_name: Optional function name to clear cache for. If None, clears all caches.

    Returns:
        Dictionary with operation status.
    """
    from comet_mcp.cache import cache_invalidate

    try:
        cache_invalidate(func_name)
        return {
            "status": "success",
            "message": f"Cache cleared for {'all functions' if func_name is None else func_name}",
            "func_name": func_name,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to clear cache: {str(e)}",
            "func_name": func_name,
        }


def _initialize():
    from comet_mcp.session import initialize_session

    initialize_session()
