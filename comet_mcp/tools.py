#!/usr/bin/env python3
"""
Comet ML tools for MCP server.
These tools require access to comet_ml.API() singleton.
"""

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
from comet_mcp.utils import format_datetime
from comet_mcp.session import get_comet_api, get_session_context


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
    user: Optional[str]
    workspace: Optional[str]
    error: Optional[str]


class SearchResults(TypedDict):
    """Results from searching experiments."""

    query: str
    count: int
    experiments: List[ExperimentInfo]

class ImageResult(TypedDict):
    """Result from an image-creating function."""

    type: str
    content_type: str
    image_base64: str


def list_experiments(
    workspace: Optional[str] = None, project_name: Optional[str] = None
) -> List[ExperimentInfo]:
    """
    List recent experiments from Comet ML. Typically, don't show the
    user the experiment_id unless they ask to see it.

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
            
            # Get experiments for the specified workspace and project
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
                        description=getattr(exp, "description", None),
                    )
                )

            return result
    except Exception as e:
        raise Exception(f"Error listing experiments: {e}")

def get_plot_of_xy_data(data: List[List[float]], title: str = "XY Data Plot", metric_data: Optional[Dict[str, Any]] = None) -> ImageResult:
    """
    Create a plot of a list of [x, y] data points using matplotlib.
    Can also plot multiple metrics from get_experiment_metrics.
    
    Args:
        data: List of [x, y] coordinate pairs to plot (optional if metric_data provided)
        title: Title for the plot (default: "XY Data Plot")
        metric_data: Dictionary from get_experiment_metrics containing multiple metrics to plot
        
    Returns:
        ImageResult with base64-encoded PNG image
    """
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    if metric_data and 'experiments' in metric_data:
        # Plot multiple metrics from get_experiment_metrics (new structure)
        experiments = metric_data['experiments']
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-', 'k-']  # Different colors for each metric
        markers = ['o', 's', '^', 'v', 'D', 'p', '*']  # Different markers for each metric
        color_index = 0
        
        for experiment_id, experiment_metrics in experiments.items():
            for metric_name, metric_info in experiment_metrics.items():
                if 'error' in metric_info:
                    continue  # Skip metrics with errors
                
                data_points = metric_info['data']
                if not data_points:
                    continue
                    
                x_coords = [point[0] for point in data_points]
                y_coords = [point[1] for point in data_points]
                
                color_style = colors[color_index % len(colors)]
                marker_style = markers[color_index % len(markers)]
                
                # Create label with experiment ID and metric name
                label = f"{experiment_id}: {metric_info.get('metric_name', metric_name)}"
                
                plt.plot(x_coords, y_coords, color_style, linewidth=2, 
                        marker=marker_style, markersize=4, label=label)
                color_index += 1
        
        plt.legend()
        plt.xlabel(metric_data.get('x_axis', 'X'))
        plt.ylabel('Metric Value')
    elif metric_data and 'metrics' in metric_data:
        # Plot multiple metrics from get_experiment_metrics (old structure for backward compatibility)
        metrics = metric_data['metrics']
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-', 'k-']  # Different colors for each metric
        markers = ['o', 's', '^', 'v', 'D', 'p', '*']  # Different markers for each metric
        
        for i, (metric_name, metric_info) in enumerate(metrics.items()):
            if 'error' in metric_info:
                continue  # Skip metrics with errors
            
            data_points = metric_info['data']
            if not data_points:
                continue
                
            x_coords = [point[0] for point in data_points]
            y_coords = [point[1] for point in data_points]
            
            color_style = colors[i % len(colors)]
            marker_style = markers[i % len(markers)]
            
            plt.plot(x_coords, y_coords, color_style, linewidth=2, 
                    marker=marker_style, markersize=4, label=metric_info.get('metric_name', metric_name))
        
        plt.legend()
        plt.xlabel(metric_data.get('x_axis', 'X'))
        plt.ylabel('Metric Value')
        
    elif data:
        # Plot single data series
        x_coords = [point[0] for point in data]
        y_coords = [point[1] for point in data]
        
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('X')
        plt.ylabel('Y')
    else:
        raise ValueError("Either data or metric_data must be provided")
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Save plot to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Clean up
    plt.close()
    buffer.close()
    
    return ImageResult(
        type="image_result",
        content_type="image/png",
        image_base64=image_base64
    )
    
def get_default_workspace() -> str:
    """
    Get the default workspace name for this user.
    """
    with get_comet_api() as api:
        return api.get_default_workspace()


def get_experiment_code(experiment_id: str) -> Dict[str, str]:
    """
    Get the code for a specific experiment.

    Args:
        experiment_id: The ID of the experiment to retrieve
    """
    with get_comet_api() as api:
        experiment = api.get_experiment_by_key(experiment_id)
        return {"code": experiment.get_code()}

def get_experiment_metric_data(experiment_ids: List[str], metric_names: List[str], x_axis: Optional[str] = None) -> Dict[str, Any]:
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
        Dictionary containing experiment_ids, x_axis, and metrics data with (x, y) coordinate pairs for plotting
    """
    with get_comet_api() as api:
        try:
            data = api.get_metrics_for_chart(experiment_ids, metric_names)
            
            results = {}
            
            # Process each experiment
            for experiment_id in experiment_ids:
                if experiment_id not in data:
                    continue  # Skip experiments not found in data
                
                experiment_data = data[experiment_id]
                if not experiment_data or experiment_data.get('empty', True):
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
                        if current_x_axis not in metric_data or metric_data[current_x_axis] is None:
                            # Try to find an available x_axis in the order specified in docstring
                            for fallback_axis in ["steps", "epochs", "timestamps", "durations"]:
                                if fallback_axis in metric_data and metric_data[fallback_axis] is not None:
                                    current_x_axis = fallback_axis
                                    break
                            else:
                                # If no standard x_axis is found, skip this metric
                                continue
                    else:
                        # No x_axis provided, try in order of priority: steps, epochs, timestamps, durations
                        current_x_axis = None
                        for priority_axis in ["steps", "epochs", "timestamps", "durations"]:
                            if priority_axis in metric_data and metric_data[priority_axis] is not None:
                                current_x_axis = priority_axis
                                break
                        
                        if current_x_axis is None:
                            # If no standard x_axis is found, skip this metric
                            continue
                    
                    x_axis_values = metric_data[current_x_axis]
                    
                    # Convert timestamps to datetime objects if needed
                    if current_x_axis == "timestamps":
                        x_axis_values = [datetime.fromtimestamp(value) for value in x_axis_values]
                    
                    # Store metric data with metric name included
                    experiment_metrics[metric_name] = {
                        "metric_name": metric_name,
                        "x_axis": current_x_axis,
                        "data": list(zip(x_axis_values, values))  # (x, y) pairs for plotting
                    }
                    experiment_has_data = True
                
                # Only include experiments that have data
                if experiment_has_data:
                    results[experiment_id] = experiment_metrics
            
            return {
                "experiment_ids": experiment_ids,
                "x_axis": x_axis or "auto-selected",
                "experiments": results
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get metrics {metric_names} for experiments {experiment_ids}: {e}")


def get_experiment_details(experiment_id: str) -> ExperimentDetails:
    """
    Get detailed information about a specific experiment, including
    metric and parameter names.

    Args:
        experiment_id: The ID of the experiment to retrieve
    """
    try:
        with get_comet_api() as api:
            experiment = api.get_experiment_by_key(experiment_id)

            if not experiment:
                raise Exception(f"Experiment with ID '{experiment_id}' not found.")

            # Get metrics
            metrics = experiment.get_metrics_summary()
            metrics_list = []
            if metrics:
                for metric in metrics:
                    metrics_list.append({
                        "name": metric["name"],
                        "value": metric.get("valueCurrent", 0)
                    })

            # Get parameters
            params = experiment.get_parameters_summary()
            params_list = []
            if params:
                for param in params:
                    params_list.append({
                        "name": param["name"],
                        "value": param.get("valueCurrent", "")
                    })

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
    except Exception as e:
        raise Exception(f"Error getting experiment details for '{experiment_id}': {e}")


def list_projects(workspace: Optional[str] = None) -> List[ProjectInfo]:
    """
    List the project names in a Comet ML workspace. Only use this tool
    if you need a list of all project names. Do not use to verify a
    project name unless you get an error that the project does not exist.

    Args:
        workspace: Workspace name (optional, uses default workspace if not provided)
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


def get_session_info() -> SessionInfo:
    """
    Get information about the current Comet ML session.
    """
    session_context = get_session_context()
    
    if not session_context.is_initialized():
        return SessionInfo(
            initialized=False,
            api_status="Not initialized",
            user=None,
            workspace=None,
            error="Comet ML session is not initialized."
        )
    
    try:
        with get_comet_api() as api:
            workspace = api.get_default_workspace()
            
            # Try to get user info
            try:
                user_info = api.get_user_info()
                user = user_info.get("username") if user_info else None
            except (AttributeError, Exception):
                # Fallback to workspace info if user info not available
                user = f"Connected to workspace: {workspace}"
            
            return SessionInfo(
                initialized=True,
                api_status="Connected",
                user=user,
                workspace=workspace,
                error=None
            )
    except Exception as e:
        return SessionInfo(
            initialized=True,
            api_status="Error",
            user=None,
            workspace=None,
            error=str(e)
        )


def list_project_experiments(
    project_name: str, workspace: Optional[str] = None
) -> List[ExperimentInfo]:
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
                        description=getattr(exp, "description", None),
                    )
                )

            return result
    except Exception as e:
        raise Exception(f"Error listing experiments for project '{project_name}': {e}")


def count_project_experiments(
    project_name: str, workspace: Optional[str] = None
) -> Dict[str, Any]:
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
                "experiments": [ExperimentInfo(
                    id=exp.id,
                    name=exp.name,
                    status=exp.get_state(),
                    created_at=format_datetime(exp.start_server_timestamp),
                    description=getattr(exp, "description", None),
                ) for exp in experiments] if experiments else [],
            }
    except Exception as e:
        raise Exception(f"Error counting experiments for project '{project_name}': {e}")


