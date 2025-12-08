#!/usr/bin/env python3
"""
OpenTelemetry instrumentation for Comet MCP Server.

Provides distributed tracing and structured logging with support for:
- File export (JSON Lines format)
- Opik export (OTLP HTTP endpoint)
"""

import json
import os
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode

# Try to import logging SDK components (may not be available in all versions)
try:
    from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor
    from opentelemetry.sdk._logs.export import LogExporter, LogRecord
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    LOGS_AVAILABLE = True
except ImportError:
    LOGS_AVAILABLE = False

# Global state
_tracer_provider: Optional[TracerProvider] = None
_logger_provider: Optional[Any] = None
_initialized = False
_lock = threading.Lock()


def _safe_print(message: str) -> None:
    """Safely print a message, handling closed stdout/stderr."""
    try:
        print(message)
    except (ValueError, OSError):
        # stdout/stderr may be closed during shutdown, ignore
        pass


class FileSpanExporter(SpanExporter):
    """Custom span exporter that writes spans to a JSON Lines file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file_lock = threading.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure the file exists and is writable."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(
                (
                    os.path.dirname(self.file_path)
                    if os.path.dirname(self.file_path)
                    else "."
                ),
                exist_ok=True,
            )
            # Touch the file
            with open(self.file_path, "a"):
                pass
        except Exception as e:
            print(f"Warning: Failed to create trace file {self.file_path}: {e}")

    def export(self, spans: List[Span]) -> None:
        """Export spans to file as JSON Lines."""
        if not spans:
            return

        try:
            with self._file_lock:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    for span in spans:
                        # Convert span to OTLP JSON format
                        span_data = self._span_to_dict(span)
                        f.write(json.dumps(span_data) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write trace to file {self.file_path}: {e}")

    def _span_to_dict(self, span: Span) -> Dict[str, Any]:
        """Convert a span to OTLP JSON format."""
        # Get span context
        span_context = span.get_span_context()

        # Build OTLP span representation
        span_dict = {
            "trace_id": format(span_context.trace_id, "032x"),
            "span_id": format(span_context.span_id, "016x"),
            "name": span.name,
            "kind": span.kind.name if hasattr(span.kind, "name") else str(span.kind),
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time if span.end_time else None,
            "status": {
                "code": (
                    span.status.status_code.name
                    if hasattr(span.status.status_code, "name")
                    else str(span.status.status_code)
                ),
                "message": span.status.description or "",
            },
            "attributes": dict(span.attributes) if span.attributes else {},
            "events": [
                {
                    "time_unix_nano": event.timestamp,
                    "name": event.name,
                    "attributes": dict(event.attributes) if event.attributes else {},
                }
                for event in span.events
            ],
        }

        # Add parent span if available
        if span.parent:
            # span.parent is already a SpanContext
            span_dict["parent_span_id"] = format(span.parent.span_id, "016x")

        return span_dict

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class FileLogExporter:
    """Custom log exporter that writes logs to a JSON Lines file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file_lock = threading.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure the file exists and is writable."""
        try:
            os.makedirs(
                (
                    os.path.dirname(self.file_path)
                    if os.path.dirname(self.file_path)
                    else "."
                ),
                exist_ok=True,
            )
            with open(self.file_path, "a"):
                pass
        except Exception as e:
            print(f"Warning: Failed to create log file {self.file_path}: {e}")

    def export(self, log_records: List[Any]) -> None:
        """Export log records to file as JSON Lines."""
        if not log_records:
            return

        try:
            with self._file_lock:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    for record in log_records:
                        log_data = self._log_record_to_dict(record)
                        f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write log to file {self.file_path}: {e}")

    def _log_record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert a log record to JSON format."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": getattr(record, "severity_text", "INFO"),
            "body": str(getattr(record, "body", "")),
            "attributes": dict(getattr(record, "attributes", {})),
            "resource": (
                dict(getattr(record, "resource", {}).attributes)
                if hasattr(getattr(record, "resource", None), "attributes")
                else {}
            ),
        }

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


def _get_config() -> Dict[str, Any]:
    """Get telemetry configuration from environment variables."""
    return {
        "enabled": os.getenv("OTEL_ENABLED", "true").lower() == "true",
        "service_name": os.getenv("OTEL_SERVICE_NAME", "comet-mcp"),
        "service_version": os.getenv("OTEL_SERVICE_VERSION", "1.2.0"),
        "traces_file": os.getenv("OTEL_TRACES_FILE", "traces.jsonl"),
        "logs_file": os.getenv("OTEL_LOGS_FILE", "logs.jsonl"),
        "opik_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("OPIK_ENDPOINT"),
        "opik_headers": os.getenv("OTEL_EXPORTER_OTLP_HEADERS"),
        "opik_api_key": os.getenv("OPIK_API_KEY"),
        "opik_project_name": os.getenv("OPIK_PROJECT_NAME"),
        "opik_workspace": os.getenv("OPIK_WORKSPACE"),
    }


def _build_opik_headers(config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Build headers for Opik export."""
    headers = {}

    # Check if headers are provided directly
    if config["opik_headers"]:
        # Parse header string (format: "key1=value1,key2=value2")
        for header_pair in config["opik_headers"].split(","):
            if "=" in header_pair:
                key, value = header_pair.split("=", 1)
                headers[key.strip()] = value.strip()
    else:
        # Build headers from individual variables
        if config["opik_api_key"]:
            headers["Authorization"] = config["opik_api_key"]
        if config["opik_project_name"]:
            headers["projectName"] = config["opik_project_name"]
        if config["opik_workspace"]:
            headers["Comet-Workspace"] = config["opik_workspace"]

    return headers if headers else None


def initialize_telemetry() -> None:
    """Initialize OpenTelemetry tracing and logging."""
    global _tracer_provider, _logger_provider, _initialized

    with _lock:
        if _initialized:
            return

        config = _get_config()

        if not config["enabled"]:
            _safe_print("OpenTelemetry is disabled via OTEL_ENABLED=false")
            return

        # Create resource
        resource = Resource.create(
            {
                SERVICE_NAME: config["service_name"],
                SERVICE_VERSION: config["service_version"],
            }
        )

        # Initialize TracerProvider
        _tracer_provider = TracerProvider(resource=resource)

        # Add file exporter if configured
        if config["traces_file"]:
            try:
                file_exporter = FileSpanExporter(config["traces_file"])
                _tracer_provider.add_span_processor(BatchSpanProcessor(file_exporter))
                _safe_print(f"OpenTelemetry: Writing traces to {config['traces_file']}")
            except Exception as e:
                print(f"Warning: Failed to initialize file trace exporter: {e}")

        # Add Opik exporter if configured
        if config["opik_endpoint"]:
            try:
                headers = _build_opik_headers(config)
                opik_exporter = OTLPSpanExporter(
                    endpoint=config["opik_endpoint"],
                    headers=headers,
                )
                _tracer_provider.add_span_processor(BatchSpanProcessor(opik_exporter))
                _safe_print(
                    f"OpenTelemetry: Sending traces to Opik at {config['opik_endpoint']}"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Opik trace exporter: {e}")

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Initialize logging if available
        # Note: OpenTelemetry logging SDK is still experimental and may not be available
        # For now, we focus on tracing which is more stable
        if LOGS_AVAILABLE and False:  # Disabled until log SDK is more stable
            try:
                from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
                from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

                _logger_provider = SDKLoggerProvider(resource=resource)

                # Add file log exporter if configured
                if config["logs_file"]:
                    try:
                        file_log_exporter = FileLogExporter(config["logs_file"])
                        # Create a simple processor for file export
                        # Note: This is a simplified implementation
                        # Full log SDK integration would require more setup
                        print(f"OpenTelemetry: Writing logs to {config['logs_file']}")
                    except Exception as e:
                        print(f"Warning: Failed to initialize file log exporter: {e}")

                # Add Opik log exporter if configured
                if config["opik_endpoint"]:
                    try:
                        headers = _build_opik_headers(config)
                        opik_log_exporter = OTLPLogExporter(
                            endpoint=config["opik_endpoint"].replace(
                                "/otel", "/otel/v1/logs"
                            ),
                            headers=headers,
                        )
                        _logger_provider.add_log_record_processor(
                            BatchLogRecordProcessor(opik_log_exporter)
                        )
                        print(
                            f"OpenTelemetry: Sending logs to Opik at {config['opik_endpoint']}"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to initialize Opik log exporter: {e}")

                set_logger_provider(_logger_provider)
            except Exception as e:
                print(f"Warning: Failed to initialize logging: {e}")

        _initialized = True
        _safe_print("OpenTelemetry initialized successfully")


def get_tracer(name: str = "comet-mcp"):
    """Get a tracer instance."""
    if not _initialized:
        initialize_telemetry()
    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    """Shutdown telemetry and flush all exports."""
    global _tracer_provider, _logger_provider, _initialized

    with _lock:
        if not _initialized:
            return

        try:
            if _tracer_provider:
                _tracer_provider.shutdown()

            if _logger_provider and hasattr(_logger_provider, "shutdown"):
                _logger_provider.shutdown()

            _initialized = False
            _safe_print("OpenTelemetry shut down successfully")
        except Exception as e:
            _safe_print(f"Warning: Error during telemetry shutdown: {e}")
