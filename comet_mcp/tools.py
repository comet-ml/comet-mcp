#!/usr/bin/env python3
"""
Comet ML tools for MCP server.
These tools require access to comet_ml.API() singleton.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import csv
import io
import json
from functools import wraps
from comet_mcp.utils import format_datetime, supports_paged_queries
from comet_mcp.session import get_comet_api, get_session_context
from comet_mcp.cache import cached
from comet_mcp.resources import get_resource_manager
from comet_ml.query import Tag
from comet_mcp.telemetry import get_tracer
from opentelemetry import trace


SUPPORTS_PAGED_QUERIES = supports_paged_queries()

if not SUPPORTS_PAGED_QUERIES:
    print(
        "WARNING: running without paged queries; update comet_ml SDK and backend to fix",
        file=sys.stderr,
    )


def _get_state(metadata):
    if metadata["running"]:
        return "running"

    if metadata["hasCrashed"]:
        return "crashed"

    return "finished"


def _instrument_tool(func):
    """Decorator to instrument tool functions with OpenTelemetry."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer("comet-mcp.tools")
        span_name = f"tool.{func.__name__}"

        with tracer.start_as_current_span(span_name) as span:
            # Add function arguments as attributes (sanitized)
            for key, value in kwargs.items():
                if value is not None:
                    # Only add simple types to avoid large payloads
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"tool.arg.{key}", str(value))
                    elif isinstance(value, list) and len(value) < 10:
                        span.set_attribute(f"tool.arg.{key}.count", len(value))

            try:
                result = func(*args, **kwargs)

                # Add result metadata
                if isinstance(result, dict):
                    if "workspace" in result:
                        span.set_attribute("tool.result.workspace", result["workspace"])
                    if "project_name" in result:
                        span.set_attribute(
                            "tool.result.project_name", result["project_name"]
                        )
                    if "experiment_count" in result:
                        span.set_attribute(
                            "tool.result.experiment_count", result["experiment_count"]
                        )
                    if "experiments" in result and isinstance(
                        result["experiments"], list
                    ):
                        span.set_attribute(
                            "tool.result.experiment_list_count",
                            len(result["experiments"]),
                        )
                elif isinstance(result, list):
                    span.set_attribute("tool.result.count", len(result))

                span.set_attribute("tool.success", True)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("tool.success", False)
                raise

    return wrapper


@_instrument_tool
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
        project_name: Project name to filter experiments (optional, default is "general")
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
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        if project_name is None:
            project_name = "general"

        if SUPPORTS_PAGED_QUERIES:
            # Build kwargs for get_experiments, only including sort params if both are provided
            get_experiments_kwargs = {
                "workspace": target_workspace,
                "project_name": project_name,
                "page": page,
                "page_size": page_size,
            }
            # Only pass sort_by and sort_order if both are provided and valid
            if sort_by and sort_order:
                if sort_by in ["startTime", "endTime"] and sort_order.lower() in [
                    "asc",
                    "desc",
                ]:
                    get_experiments_kwargs["sort_by"] = sort_by
                    get_experiments_kwargs["sort_order"] = sort_order

            experiments = api.get_experiments(**get_experiments_kwargs)
        else:
            # Get all experiments when paged queries are not supported
            experiments = api.get_experiments(
                workspace=target_workspace,
                project_name=project_name,
            )

            # Apply manual sorting if requested
            if sort_by and sort_order:
                if sort_by in ["startTime", "endTime"]:
                    # Map sort_by parameter to actual attribute name
                    # "startTime" -> "start_server_timestamp"
                    # "endTime" -> "end_server_timestamp"
                    attr_map = {
                        "startTime": "start_server_timestamp",
                        "endTime": "end_server_timestamp",
                    }
                    attr_name = attr_map.get(sort_by)
                    if attr_name:
                        reverse = sort_order.lower() == "desc"
                        # Use a default of datetime.min for missing timestamps to sort them first
                        from datetime import datetime as dt

                        default_ts = dt.min
                        experiments.sort(
                            key=lambda exp: getattr(exp, attr_name, default_ts),
                            reverse=reverse,
                        )

            # Apply manual pagination
            if page and page_size:
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                experiments = experiments[start_idx:end_idx]

        if not experiments:
            return []

        result = {
            "workspace": target_workspace,
            "project": project_name,
            "experiments": [],
        }
        for exp in experiments:
            result["experiments"].append(
                {
                    "id": exp.id,
                    "name": exp.name,
                    "status": exp.get_state(),
                    "created_at": format_datetime(exp.start_server_timestamp),
                    "description": getattr(exp, "description", None),
                }
            )

        return result


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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
        - others: List of dictionaries with "other" names and values
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

        # Get others
        others = experiment.get_others_summary()
        others_list = []
        if others:
            for item in others:
                others_list.append(
                    {"name": item["name"], "value": item.get("valueCurrent", "")}
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
            "others": others_list,
        }


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
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


@_instrument_tool
def get_project_tags(
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """
    This function will return a list of tag texts that exist
    in this project.

    Args:
        workspace: Workspace name (optional, will lookup the default workspace name if not provided)
        project_name: Project name to filter experiments (optional, default is "general")

    Returns:
        A list of tag names.

    """
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        if project_name is None:
            project_name = "general"

        return [
            qv.rhs
            for qv in api.get_query_variables(target_workspace, project_name)
            if hasattr(qv, "rhs")
        ]


@_instrument_tool
def get_project_logged_item_names(
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """
    This function will return a list of named items that have been
    logged in this project.

    Args:
        workspace: Workspace name (optional, will lookup the default workspace name if not provided)
        project_name: Project name to filter experiments (optional, default is "general")

    Returns:
        A list of names.

    """
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        if project_name is None:
            project_name = "general"

        return [
            qv.name
            for qv in api.get_query_variables(target_workspace, project_name)
            if hasattr(qv, "name")
        ]


@_instrument_tool
def find_tagged_experiments(
    tag: str,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """
    This function will find all experiments that are tagged with the text 'tag'.

    Args:
        tag: the text of the tag
        workspace: Workspace name (optional, will lookup the default workspace name if not provided)
        project_name: Project name to filter experiments (optional, default is "general")

    Returns:
        List containing matching unique experiment identifiers
    """
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        if project_name is None:
            project_name = "general"

        valid = validate_project(project_name, target_workspace)
        if valid["error"] is not None:
            return valid

        query = Tag(tag)
        experiments = api.query(target_workspace, project_name, query)
        if experiments:
            return [experiment.id for experiment in experiments]
        else:
            return f"No experiments with the tag '{tag}' were found in workspace '{target_workspace}' and project '{project_name}'."


@_instrument_tool
def query_experiments(
    name: str,
    comparison: str,
    value: str,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """
    This function will find all experiments that have a logged item named 'name'
    using comparison to compare with 'value'.

    Experiment name is logged as "Name".

    Args:
        name: the name of the logged item; use 'TYPE:name' to distinguish between names
            where 'TYPE' can be "Environment", "Metadata", "Metric", "Other",
            or "Parameter"
        comparison: can be any of "==", ">", ">=", "<", ">=", "!=", "contains",
            "endswith", or "startswith".
        value: the value to compare to the logged value; can be any value; special values are
            "true", "false", "none", or "datetime:ISO-FORMAT-DATETIME"
        workspace: Workspace name (optional, will lookup the default workspace name if not provided)
        project_name: Project name to filter experiments (optional, default is "general")

    Returns:
        List containing matching unique experiment identifiers
    """
    with get_comet_api() as api:
        if workspace:
            target_workspace = workspace
        else:
            target_workspace = api.get_default_workspace()

        if project_name is None:
            project_name = "general"

        query = None
        valid = validate_project(project_name, target_workspace)
        if valid["error"] is None:
            results = get_project_logged_item_names(target_workspace, project_name)
            if isinstance(results, list):
                if ":" in name:
                    query_type, name = name.split(":", 1)
                    selected = [
                        qv
                        for qv in api.get_query_variables(
                            target_workspace, project_name
                        )
                        if hasattr(qv, "name")
                        and qv.name == name
                        and qv.__class__.__name__.lower() == query_type.lower()
                    ]
                else:
                    selected = [
                        qv
                        for qv in api.get_query_variables(
                            target_workspace, project_name
                        )
                        if hasattr(qv, "name") and qv.name == name
                    ]

                if len(selected) == 0:
                    return f"No experiments matching this query were found in workspace '{target_workspace}' and project '{project_name}'."
                elif len(selected) == 1:
                    query = _create_query(selected[0], comparison, value)
                else:
                    query_types = [qv.__class__.__name__ for qv in selected]
                    return f"The name '{name}' is associated with multiple logged types ({query_types}); please specify which one you want by using 'TYPE:{name}'"
            else:
                return f"The name '{name}' was not found in workspace '{target_workspace}' and project '{project_name}'."
        else:
            return valid

        experiments = api.query(target_workspace, project_name, query)
        if experiments:
            return [experiment.id for experiment in experiments]
        else:
            return f"No experiments matching this query were found in workspace '{target_workspace}' and project '{project_name}'."


def _create_query(qv, comparison, value):
    # Convert values:
    if isinstance(value, str):
        if value.lower() == "true":
            value = 1
        elif value.lower() == "false":
            value = 0
        elif value.lower() == "none":
            value = None
        elif value.startswith("datetime:"):
            _, iso_format = value.split(":", 1)
            value = datetime.fromisoformat(iso_format)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
    elif isinstance(value, bool):
        if value is True:
            value = 1
        elif value is False:
            value = 0

    # Create comparison:
    if comparison == "==":
        return qv == value
    elif comparison == "!=":
        return qv != value
    elif comparison == "<":
        return qv < value
    elif comparison == "<=":
        return qv <= value
    elif comparison == ">":
        return qv > value
    elif comparison == ">=":
        return qv >= value
    elif comparison.lower() == "contains":
        return qv.contains(value)
    elif comparison.lower() == "endswith":
        return qv.endsWith(value)
    elif comparison.lower() == "startswith":
        return qv.startsWith(value)
    else:
        raise Exception(f"Unknown comparison '{comparison}'")


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


@_instrument_tool
def experiment_spreadsheet(
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    experiment_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Export experiment details to a CSV file with workspace, project_name, experiment name,
    duration, metrics, and parameters. The CSV file is made available as an MCP resource
    that can be accessed without the LLM needing to process the data.

    Args:
        workspace: Workspace name (optional, uses default if not provided).
                   Ignored if experiment_keys is provided.
        project_name: Project name (optional, default is "general"). Ignored if experiment_keys is provided.
        experiment_keys: Optional list of experiment IDs. If provided, workspace and
                        project_name arguments are ignored and only these experiments
                        are exported.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - message: Status message
        - resource_uri: URI of the generated CSV file resource (if successful)
        - filename: Name of the generated file
        - experiment_count: Number of experiments exported
    """
    try:
        with get_comet_api() as api:
            experiment_ids = []

            # If experiment_keys is provided, use those and ignore workspace/project_name
            if experiment_keys:
                experiment_ids = experiment_keys
            else:
                # Determine target workspace
                if workspace:
                    target_workspace = workspace
                else:
                    target_workspace = get_default_workspace()

                if project_name is None:
                    project_name = "general"

                # Validate project exists
                validation = validate_project(project_name, target_workspace)
                if not validation.get("exists", False):
                    return {
                        "status": "error",
                        "message": validation.get("error", "Project not found"),
                        "resource_uri": None,
                        "filename": None,
                        "experiment_count": 0,
                    }

                # Get all experiments for the project
                experiments_metadata = api._get_project_experiments(
                    target_workspace, project_name
                )
                if experiments_metadata:
                    experiment_ids = [
                        exp.get("experimentKey", "")
                        for exp in experiments_metadata.values()
                        if exp.get("experimentKey")
                    ]
                else:
                    experiment_ids = []

            if not experiment_ids:
                return {
                    "status": "error",
                    "message": "No experiments found to export",
                    "resource_uri": None,
                    "filename": None,
                    "experiment_count": 0,
                }

            # Prepare CSV data by fetching full experiment details
            csv_rows = []
            for exp_id in experiment_ids:
                try:
                    experiment = api.get_experiment_by_key(exp_id)
                    if not experiment:
                        continue

                    # Get workspace and project_name from experiment if available
                    exp_workspace = getattr(experiment, "workspace", None)
                    exp_project = getattr(experiment, "project_name", None)

                    # If not available from experiment object, try to extract from URL
                    if not exp_workspace or not exp_project:
                        try:
                            # URL format: https://www.comet.com/{workspace}/{project_name}/...
                            url = getattr(experiment, "url", "")
                            if url and "/" in url:
                                parts = url.split("/")
                                if len(parts) >= 4:
                                    # Find the workspace and project in the URL
                                    for i, part in enumerate(parts):
                                        if "comet.com" in part or "comet.ml" in part:
                                            if i + 2 < len(parts):
                                                exp_workspace = (
                                                    exp_workspace or parts[i + 1]
                                                )
                                                exp_project = (
                                                    exp_project or parts[i + 2]
                                                )
                                            break
                        except Exception:
                            pass

                    # If still not available and we have context, use it
                    if not exp_workspace and workspace:
                        exp_workspace = workspace
                    elif not exp_workspace:
                        exp_workspace = get_default_workspace()

                    if not exp_project and project_name:
                        exp_project = project_name

                    # Calculate duration
                    start_time = experiment.start_server_timestamp
                    end_time = experiment.end_server_timestamp or start_time
                    duration_seconds = None
                    if start_time and end_time:
                        # Check if timestamps are in milliseconds (> 1e12) or seconds
                        if start_time > 1e12 or end_time > 1e12:
                            # Timestamps are in milliseconds
                            duration_seconds = (end_time - start_time) / 1000.0
                        else:
                            # Timestamps are in seconds
                            duration_seconds = end_time - start_time

                    # Get metrics
                    metrics_summary = experiment.get_metrics_summary()
                    metrics_dict = {}
                    if metrics_summary:
                        for metric in metrics_summary:
                            metric_name = metric.get("name", "")
                            metric_value = metric.get("valueCurrent", 0)
                            if metric_name:
                                metrics_dict[metric_name] = metric_value

                    # Get parameters
                    params_summary = experiment.get_parameters_summary()
                    params_dict = {}
                    if params_summary:
                        for param in params_summary:
                            param_name = param.get("name", "")
                            param_value = param.get("valueCurrent", "")
                            if param_name:
                                params_dict[param_name] = param_value

                    # Create row
                    row = {
                        "workspace": exp_workspace or "",
                        "project_name": exp_project or "",
                        "experiment_name": experiment.name or "",
                        "duration_seconds": duration_seconds,
                        "metrics": json.dumps(metrics_dict) if metrics_dict else "",
                        "parameters": json.dumps(params_dict) if params_dict else "",
                    }
                    csv_rows.append(row)

                except Exception as e:
                    # Skip experiments that fail to load
                    continue

            # Generate CSV content
            if not csv_rows:
                return {
                    "status": "error",
                    "message": "No experiment data to export",
                    "resource_uri": None,
                    "filename": None,
                    "experiment_count": 0,
                }

            # Create CSV in memory
            output = io.StringIO()
            fieldnames = [
                "workspace",
                "project_name",
                "experiment_name",
                "duration_seconds",
                "metrics",
                "parameters",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            csv_content = output.getvalue()
            output.close()

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_keys:
                filename = f"experiment_spreadsheet_{timestamp}.csv"
            else:
                safe_project = "".join(
                    c
                    for c in (project_name or "experiments")
                    if c.isalnum() or c in ("-", "_")
                )
                filename = f"experiment_spreadsheet_{safe_project}_{timestamp}.csv"

            # Register file as resource
            resource_manager = get_resource_manager()
            resource_uri = resource_manager.create_file(
                filename, csv_content.encode("utf-8")
            )

            # Get the actual file path
            file_path = resource_manager.get_file_path(resource_uri)

            return {
                "status": "success",
                "message": f"Exported {len(csv_rows)} experiments to CSV",
                "resource_uri": resource_uri,
                "filename": filename,
                "file_path": str(file_path) if file_path else None,
                "experiment_count": len(csv_rows),
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to export experiments: {str(e)}",
            "resource_uri": None,
            "filename": None,
            "experiment_count": 0,
        }


def _initialize():
    from comet_mcp.session import initialize_session

    initialize_session()
