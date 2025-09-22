#!/usr/bin/env python3
"""
Comet ML tools for MCP server.
These tools require access to comet_ml.API() singleton.
"""

from typing import List, Dict, Any, Optional
from .utils import tool
from .session import get_comet_api, get_session_context


@tool
def list_experiments(limit: int = 10, workspace: Optional[str] = None) -> str:
    """
    List recent experiments from Comet ML.
    
    Args:
        limit: Maximum number of experiments to return (default: 10)
        workspace: Workspace name (optional, uses default if not provided)
    """
    try:
        with get_comet_api() as api:
            if workspace:
                experiments = api.get_experiments(workspace=workspace, limit=limit)
            else:
                experiments = api.get_experiments(limit=limit)
            
            if not experiments:
                return "No experiments found."
            
            result = f"Found {len(experiments)} experiment(s):\n\n"
            for exp in experiments:
                result += f"• {exp.name} (ID: {exp.id})\n"
                result += f"  Status: {exp.state}\n"
                result += f"  Created: {exp.created_at}\n"
                if exp.description:
                    result += f"  Description: {exp.description}\n"
                result += "\n"
            
            return result
            
    except Exception as e:
        return f"Error listing experiments: {str(e)}"


@tool
def get_experiment_details(experiment_id: str) -> str:
    """
    Get detailed information about a specific experiment.
    
    Args:
        experiment_id: The ID of the experiment to retrieve
    """
    try:
        with get_comet_api() as api:
            experiment = api.get_experiment_by_key(experiment_id)
            
            if not experiment:
                return f"Experiment with ID '{experiment_id}' not found."
            
            result = f"Experiment Details:\n"
            result += f"Name: {experiment.name}\n"
            result += f"ID: {experiment.id}\n"
            result += f"Status: {experiment.state}\n"
            result += f"Created: {experiment.created_at}\n"
            result += f"Updated: {experiment.updated_at}\n"
            
            if experiment.description:
                result += f"Description: {experiment.description}\n"
            
            # Get metrics
            metrics = experiment.get_metrics()
            if metrics:
                result += f"\nMetrics:\n"
                for metric in metrics[:10]:  # Show first 10 metrics
                    result += f"  {metric.name}: {metric.value}\n"
            
            # Get parameters
            params = experiment.get_parameters()
            if params:
                result += f"\nParameters:\n"
                for param in params[:10]:  # Show first 10 parameters
                    result += f"  {param.name}: {param.value}\n"
            
            return result
            
    except Exception as e:
        return f"Error getting experiment details: {str(e)}"


@tool
def list_projects(workspace: Optional[str] = None) -> str:
    """
    List projects in the Comet ML workspace.
    
    Args:
        workspace: Workspace name (optional, uses default if not provided)
    """
    try:
        with get_comet_api() as api:
            if workspace:
                projects = api.get_projects(workspace=workspace)
            else:
                projects = api.get_projects()
            
            if not projects:
                return "No projects found."
            
            result = f"Found {len(projects)} project(s):\n\n"
            for project in projects:
                result += f"• {project.name}\n"
                result += f"  Workspace: {project.workspace}\n"
                result += f"  Created: {project.created_at}\n"
                if project.description:
                    result += f"  Description: {project.description}\n"
                result += "\n"
            
            return result
            
    except Exception as e:
        return f"Error listing projects: {str(e)}"


@tool
def get_session_info() -> str:
    """
    Get information about the current Comet ML session.
    """
    try:
        context = get_session_context()
        
        if not context.is_initialized():
            return "Comet ML session is not initialized."
        
        result = "Comet ML Session Information:\n"
        result += f"Initialized: {context.is_initialized()}\n"
        
        # Test API connection
        try:
            with get_comet_api() as api:
                # Try to get user info to verify connection
                user_info = api.get_user_info()
                result += f"API Status: Connected\n"
                result += f"User: {user_info.get('username', 'Unknown')}\n"
        except Exception as e:
            result += f"API Status: Error - {str(e)}\n"
        
        return result
        
    except Exception as e:
        return f"Error getting session info: {str(e)}"


@tool
def search_experiments(query: str, limit: int = 10) -> str:
    """
    Search for experiments by name or description.
    
    Args:
        query: Search query to match against experiment names or descriptions
        limit: Maximum number of results to return (default: 10)
    """
    try:
        with get_comet_api() as api:
            # Get all experiments and filter by query
            experiments = api.get_experiments(limit=100)  # Get more to search through
            
            if not experiments:
                return "No experiments found."
            
            # Simple text search
            matching_experiments = []
            query_lower = query.lower()
            
            for exp in experiments:
                if (query_lower in exp.name.lower() or 
                    (exp.description and query_lower in exp.description.lower())):
                    matching_experiments.append(exp)
            
            if not matching_experiments:
                return f"No experiments found matching query: '{query}'"
            
            # Limit results
            matching_experiments = matching_experiments[:limit]
            
            result = f"Found {len(matching_experiments)} experiment(s) matching '{query}':\n\n"
            for exp in matching_experiments:
                result += f"• {exp.name} (ID: {exp.id})\n"
                result += f"  Status: {exp.state}\n"
                result += f"  Created: {exp.created_at}\n"
                if exp.description:
                    result += f"  Description: {exp.description}\n"
                result += "\n"
            
            return result
            
    except Exception as e:
        return f"Error searching experiments: {str(e)}"