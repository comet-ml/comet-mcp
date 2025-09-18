"""
Configuration management for Comet ML MCP Server.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CometConfig(BaseModel):
    """Configuration for Comet ML API."""
    
    api_key: str = Field(..., description="Comet ML API key")
    workspace: Optional[str] = Field(None, description="Default workspace name")
    base_url: str = Field(
        default="https://www.comet.ml/api/rest/v2", 
        description="Comet ML API base URL"
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("API key cannot be empty")
        return v.strip()
    
    @validator('workspace')
    def validate_workspace(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else None
    
    @classmethod
    def from_env(cls) -> 'CometConfig':
        """Create configuration from environment variables."""
        api_key = os.getenv("COMET_API_KEY")
        if not api_key:
            raise ValueError(
                "COMET_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        workspace = os.getenv("COMET_WORKSPACE")
        
        return cls(
            api_key=api_key,
            workspace=workspace
        )


class ServerConfig(BaseModel):
    """Configuration for the MCP server."""
    
    server_name: str = Field(default="comet-ml", description="MCP server name")
    server_version: str = Field(default="1.0.0", description="MCP server version")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


def get_config() -> tuple[CometConfig, ServerConfig]:
    """Get both Comet and server configurations."""
    comet_config = CometConfig.from_env()
    server_config = ServerConfig()
    return comet_config, server_config

