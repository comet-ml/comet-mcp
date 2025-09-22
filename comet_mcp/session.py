#!/usr/bin/env python3
"""
Session management for MCP server with comet_ml.API() singleton access.
"""

import os
from contextlib import contextmanager
from typing import Optional, Generator
import comet_ml
from comet_ml import API


class SessionContext:
    """Context manager for comet_ml.API() singleton access."""
    
    def __init__(self):
        self._api: Optional[API] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the comet_ml.API() instance with configuration."""
        if self._initialized:
            return
        
        # Get API key from parameter or environment
        self._api = API()
        self._initialized = True
    
    @property
    def api(self) -> API:
        """Get the comet_ml.API() instance."""
        if not self._initialized or self._api is None:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._api
    
    
    def is_initialized(self) -> bool:
        """Check if the session is initialized."""
        return self._initialized
    
    def reset(self):
        """Reset the session (useful for testing or reconfiguration)."""
        self._api = None
        self._initialized = False


# Global session context instance
session_context = SessionContext()


@contextmanager
def get_comet_api() -> Generator[API, None, None]:
    """
    Context manager to get the comet_ml.API() instance.
    
    Usage:
        with get_comet_api() as api:
            experiments = api.get_experiments()
    """
    if not session_context.is_initialized():
        raise RuntimeError("Session not initialized. Call session_context.initialize() first.")
    
    yield session_context.api


def initialize_session():
    """Initialize the global session context."""
    session_context.initialize()


def get_session_context() -> SessionContext:
    """Get the global session context instance."""
    return session_context
