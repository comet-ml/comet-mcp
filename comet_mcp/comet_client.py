"""
Comet ML API client for the MCP server.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

from comet_ml import API
from pydantic import BaseModel

from .config import CometConfig

logger = logging.getLogger(__name__)


class CometAPIError(Exception):
    """Custom exception for Comet ML API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class CometClient:
    """Client for interacting with Comet ML API using the official Python SDK."""
    
    def __init__(self, config: CometConfig):
        self.config = config
        # Initialize the Comet ML API with the provided API key
        self.api = API(api_key=config.api_key)
    
    async def get_experiments(
        self, 
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> str:
        """Get experiments from Comet ML."""
        try:
            # Use provided workspace or default from config
            target_workspace = workspace or self.config.workspace
            
            # Get experiments using the official API (no limit/offset support)
            experiments = self.api.get_experiments(
                workspace=target_workspace,
                project_name=project
            )
            
            # Convert to list format for consistency
            experiments_list = []
            for exp in experiments:
                experiments_list.append({
                    "experimentKey": exp.key,
                    "experimentName": exp.name,
                    "state": exp.get_state(),
                    "createdAt": exp.start_server_timestamp,
                    "updatedAt": exp.end_server_timestamp,
                    "workspaceName": exp.workspace,
                    "projectName": exp.project_name
                })
            
            # Apply limit and offset manually
            if offset > 0:
                experiments_list = experiments_list[offset:]
            if limit > 0:
                experiments_list = experiments_list[:limit]
            
            return self._format_experiments(experiments_list)
            
        except Exception as e:
            raise CometAPIError(f"Failed to get experiments: {str(e)}")
    
    async def get_experiment_details(self, experiment_key: str) -> str:
        """Get detailed information about a specific experiment."""
        if not experiment_key:
            raise ValueError("experiment_key is required")
        
        try:
            experiment = self.api.get_experiment_by_key(experiment_key)
            experiment_data = {
                "experimentKey": experiment.key,
                "experimentName": experiment.name,
                "state": experiment.state,
                "createdAt": experiment.created_at,
                "updatedAt": experiment.updated_at,
                "workspaceName": experiment.workspace,
                "projectName": experiment.project_name,
                "description": getattr(experiment, 'description', None),
                "tags": getattr(experiment, 'tags', []),
                "metrics": len(getattr(experiment, 'metrics', [])) if hasattr(experiment, 'metrics') else 0,
                "parameters": len(getattr(experiment, 'parameters', [])) if hasattr(experiment, 'parameters') else 0,
                "assets": len(getattr(experiment, 'assets', [])) if hasattr(experiment, 'assets') else 0
            }
            return self._format_experiment_details(experiment_data)
        except Exception as e:
            raise CometAPIError(f"Failed to get experiment details: {str(e)}")
    
    async def get_experiment_metrics(self, experiment_key: str, limit: int = 10) -> str:
        """Get metrics for a specific experiment."""
        if not experiment_key:
            raise ValueError("experiment_key is required")
        
        try:
            experiment = self.api.get_experiment_by_key(experiment_key)
            metrics = getattr(experiment, 'metrics', [])
            
            metrics_list = []
            for metric in metrics:
                metrics_list.append({
                    "name": getattr(metric, 'name', 'unknown'),
                    "value": getattr(metric, 'value', None),
                    "step": getattr(metric, 'step', None),
                    "timestamp": getattr(metric, 'timestamp', None)
                })
            
            # Apply limit before formatting
            if limit > 0:
                metrics_list = metrics_list[:limit]
            
            return self._format_metrics(metrics_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get experiment metrics: {str(e)}")
    
    async def get_experiment_parameters(self, experiment_key: str) -> str:
        """Get parameters for a specific experiment."""
        if not experiment_key:
            raise ValueError("experiment_key is required")
        
        try:
            experiment = self.api.get_experiment_by_key(experiment_key)
            parameters = getattr(experiment, 'parameters', [])
            
            params_list = []
            for param in parameters:
                params_list.append({
                    "name": getattr(param, 'name', 'unknown'),
                    "value": getattr(param, 'value', None)
                })
            
            return self._format_parameters(params_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get experiment parameters: {str(e)}")
    
    async def get_projects(self, workspace: Optional[str] = None) -> str:
        """Get projects from Comet ML."""
        try:
            # Use provided workspace or default from config
            target_workspace = workspace or self.config.workspace
            
            projects = self.api.get_projects(workspace=target_workspace)
            
            projects_list = []
            for project in projects:
                # Projects are returned as strings (project names)
                project_name = project if isinstance(project, str) else getattr(project, 'name', str(project))
                projects_list.append({
                    "projectName": project_name,
                    "description": "No description",  # Not available in the API response
                    "experimentCount": 0,  # Not available in the API response
                    "createdAt": None,  # Not available in the API response
                    "workspaceName": target_workspace
                })
            
            return self._format_projects(projects_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get projects: {str(e)}")
    
    async def get_workspaces(self) -> str:
        """Get available workspaces."""
        try:
            workspaces = self.api.get_workspaces()
            
            workspaces_list = []
            for workspace in workspaces:
                # Workspaces are returned as strings (workspace names)
                workspace_name = workspace if isinstance(workspace, str) else getattr(workspace, 'name', str(workspace))
                workspaces_list.append({
                    "workspaceName": workspace_name,
                    "displayName": workspace_name,
                    "createdAt": None  # Not available in the API response
                })
            
            return self._format_workspaces(workspaces_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get workspaces: {str(e)}")
    
    async def search_experiments(
        self,
        query: str,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """Search experiments using a query string."""
        try:
            # Use provided workspace or default from config
            target_workspace = workspace or self.config.workspace
            
            # Use the query method from the API
            experiments = self.api.query(
                query=query,
                workspace=target_workspace,
                project_name=project,
                limit=limit
            )
            
            # Convert to list format for consistency
            experiments_list = []
            for exp in experiments:
                experiments_list.append({
                    "experimentKey": exp.key,
                    "experimentName": exp.name,
                    "state": exp.get_state(),
                    "createdAt": exp.start_server_timestamp,
                    "updatedAt": exp.end_server_timestamp,
                    "workspaceName": exp.workspace,
                    "projectName": exp.project_name
                })
            
            return self._format_experiments(experiments_list)
        except Exception as e:
            raise CometAPIError(f"Failed to search experiments: {str(e)}")
    
    async def get_experiment_assets(self, experiment_key: str, limit: int = 10) -> str:
        """Get assets (files) for a specific experiment."""
        if not experiment_key:
            raise ValueError("experiment_key is required")
        
        try:
            experiment = self.api.get_experiment_by_key(experiment_key)
            assets = getattr(experiment, 'assets', [])
            
            assets_list = []
            for asset in assets:
                assets_list.append({
                    "fileName": getattr(asset, 'name', 'unknown'),
                    "fileSize": getattr(asset, 'size', None),
                    "fileType": getattr(asset, 'type', 'unknown')
                })
            
            # Apply limit before formatting
            if limit > 0:
                assets_list = assets_list[:limit]
            
            return self._format_assets(assets_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get experiment assets: {str(e)}")
    
    async def get_experiment_logs(self, experiment_key: str, limit: int = 20) -> str:
        """Get logs for a specific experiment."""
        if not experiment_key:
            raise ValueError("experiment_key is required")
        
        try:
            experiment = self.api.get_experiment_by_key(experiment_key)
            logs = getattr(experiment, 'logs', [])
            
            logs_list = []
            for log in logs:
                logs_list.append({
                    "timestamp": getattr(log, 'timestamp', None),
                    "level": getattr(log, 'level', 'INFO'),
                    "message": getattr(log, 'message', '')
                })
            
            # Apply limit before formatting (show last N entries)
            if limit > 0:
                logs_list = logs_list[-limit:]
            
            return self._format_logs(logs_list)
        except Exception as e:
            raise CometAPIError(f"Failed to get experiment logs: {str(e)}")
    
    def _format_experiments(self, experiments: List[Dict[str, Any]]) -> str:
        """Format experiments list for display."""
        if not experiments:
            return "No experiments found."
        
        formatted = []
        for exp in experiments:
            key = exp.get('experimentKey', 'N/A')
            name = exp.get('experimentName', 'Unnamed')
            state = exp.get('state', 'Unknown')
            created = exp.get('createdAt', 'N/A')
            
            formatted.append(
                f"• {key}: {name}\n"
                f"  State: {state} | Created: {created}"
            )
        
        return "\n\n".join(formatted)
    
    def _format_experiment_details(self, experiment: Dict[str, Any]) -> str:
        """Format experiment details for display."""
        details = []
        details.append(f"**Experiment Details**")
        details.append(f"Key: {experiment.get('experimentKey', 'N/A')}")
        details.append(f"Name: {experiment.get('experimentName', 'Unnamed')}")
        details.append(f"State: {experiment.get('state', 'Unknown')}")
        details.append(f"Created: {experiment.get('createdAt', 'N/A')}")
        details.append(f"Updated: {experiment.get('updatedAt', 'N/A')}")
        
        if experiment.get('description'):
            details.append(f"Description: {experiment['description']}")
        
        if experiment.get('tags'):
            details.append(f"Tags: {', '.join(experiment['tags'])}")
        
        if experiment.get('metrics'):
            metrics_count = len(experiment['metrics']) if isinstance(experiment['metrics'], (list, dict)) else experiment['metrics']
            details.append(f"Metrics: {metrics_count} metrics logged")
        
        if experiment.get('parameters'):
            params_count = len(experiment['parameters']) if isinstance(experiment['parameters'], (list, dict)) else experiment['parameters']
            details.append(f"Parameters: {params_count} parameters")
        
        if experiment.get('assets'):
            assets_count = len(experiment['assets']) if isinstance(experiment['assets'], (list, dict)) else experiment['assets']
            details.append(f"Assets: {assets_count} files")
        
        return "\n".join(details)
    
    def _format_metrics(self, metrics: List[Dict[str, Any]]) -> str:
        """Format metrics for display."""
        if not metrics:
            return "No metrics found."
        
        formatted = []
        for metric in metrics:
            name = metric.get('name', 'N/A')
            value = metric.get('value', 'N/A')
            step = metric.get('step', 'N/A')
            timestamp = metric.get('timestamp', 'N/A')
            
            formatted.append(f"• {name}: {value} (step: {step}, time: {timestamp})")
        
        return "\n".join(formatted)
    
    def _format_parameters(self, parameters: List[Dict[str, Any]]) -> str:
        """Format parameters for display."""
        if not parameters:
            return "No parameters found."
        
        formatted = []
        for param in parameters:
            name = param.get('name', 'N/A')
            value = param.get('value', 'N/A')
            formatted.append(f"• {name}: {value}")
        
        return "\n".join(formatted)
    
    def _format_projects(self, projects: List[Dict[str, Any]]) -> str:
        """Format projects list for display."""
        if not projects:
            return "No projects found."
        
        formatted = []
        for project in projects:
            name = project.get('projectName', 'N/A')
            description = project.get('description', 'No description')
            exp_count = project.get('experimentCount', 0)
            created = project.get('createdAt', 'N/A')
            
            formatted.append(
                f"• {name}\n"
                f"  Description: {description}\n"
                f"  Experiments: {exp_count} | Created: {created}"
            )
        
        return "\n\n".join(formatted)
    
    def _format_workspaces(self, workspaces: List[Dict[str, Any]]) -> str:
        """Format workspaces list for display."""
        if not workspaces:
            return "No workspaces found."
        
        formatted = []
        for workspace in workspaces:
            name = workspace.get('workspaceName', 'N/A')
            display_name = workspace.get('displayName', 'No display name')
            created = workspace.get('createdAt', 'N/A')
            
            formatted.append(
                f"• {name} ({display_name})\n"
                f"  Created: {created}"
            )
        
        return "\n\n".join(formatted)
    
    def _format_assets(self, assets: List[Dict[str, Any]]) -> str:
        """Format assets for display."""
        if not assets:
            return "No assets found."
        
        formatted = []
        for asset in assets:
            name = asset.get('fileName', 'N/A')
            size = asset.get('fileSize', 'N/A')
            type_info = asset.get('fileType', 'N/A')
            
            formatted.append(f"• {name} ({type_info}, {size} bytes)")
        
        return "\n".join(formatted)
    
    def _format_logs(self, logs: List[Dict[str, Any]]) -> str:
        """Format logs for display."""
        if not logs:
            return "No logs found."
        
        formatted = []
        for log in logs:
            timestamp = log.get('timestamp', 'N/A')
            level = log.get('level', 'INFO')
            message = log.get('message', 'N/A')
            
            formatted.append(f"[{timestamp}] {level}: {message}")
        
        return "\n".join(formatted)
    
    def get_tool_registry(self) -> Dict[str, callable]:
        """Get all available tools registered in this client."""
        return {
            "get_experiments": self.get_experiments,
            "get_experiment_details": self.get_experiment_details,
            "get_experiment_metrics": self.get_experiment_metrics,
            "get_experiment_parameters": self.get_experiment_parameters,
            "get_projects": self.get_projects,
            "get_workspaces": self.get_workspaces,
            "search_experiments": self.search_experiments,
            "get_experiment_assets": self.get_experiment_assets,
            "get_experiment_logs": self.get_experiment_logs,
        }
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Handle a tool call and return formatted result."""
        try:
            tool_registry = self.get_tool_registry()
            if tool_name not in tool_registry:
                return f"Unknown tool: {tool_name}"
            
            tool_method = tool_registry[tool_name]
            
            # Get method signature to properly bind arguments with defaults
            sig = inspect.signature(tool_method)
            bound_args = sig.bind(**arguments)
            bound_args.apply_defaults()
            
            # Call the tool method with properly bound arguments
            return await tool_method(*bound_args.args, **bound_args.kwargs)
        
        except CometAPIError as e:
            logger.error(f"Comet API error in {tool_name}: {e}")
            return f"Comet API Error: {e.message}"
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return f"Error: {str(e)}"

