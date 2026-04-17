"""Asset handler registry and auto-discovery for Comet MCP.

To add a new handler, drop a module into this package that defines:
  MATCH_PATTERN: str   — fnmatch pattern matched against asset filenames
  handle(content: bytes, asset_name: str) -> Dict[str, Any]
"""

import fnmatch
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AssetHandlerFn = Callable[[bytes, str], Dict[str, Any]]

_registry: List[Tuple[str, AssetHandlerFn]] = []
_loaded = False


def register_handler(pattern: str, handler: AssetHandlerFn) -> None:
    """Register a handler function for assets whose names match *pattern*."""
    _registry.append((pattern, handler))


def get_handler(asset_name: str) -> Optional[AssetHandlerFn]:
    """Return the first registered handler whose pattern matches *asset_name*."""
    _ensure_loaded()
    for pattern, handler in _registry:
        if fnmatch.fnmatch(asset_name, pattern):
            return handler
    return None


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    _discover_handlers()


def _discover_handlers() -> None:
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(
                f"comet_mcp.asset_handlers.{module_info.name}"
            )
            pattern: Optional[str] = getattr(module, "MATCH_PATTERN", None)
            handler: Optional[AssetHandlerFn] = getattr(module, "handle", None)
            if pattern and handler and callable(handler):
                register_handler(pattern, handler)
        except Exception:
            pass
