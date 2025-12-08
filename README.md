# Comet ML MCP Server

A comprehensive Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API. This server enables seamless integration with Comet ML's experiment tracking platform through a standardized protocol.

## Features

- **🔧 MCP Server**: Full Model Context Protocol implementation for tool integration
- **📊 Experiment Management**: List, search, and analyze experiments with detailed metrics
- **📁 Project Management**: Organize and explore projects across workspaces
- **🔍 Advanced Search**: Search experiments by name, description, and project
- **📈 Session Management**: Singleton `comet_ml.API()` instance with robust error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- Comet ML account and API key

### Install from Source

```bash
pip install comet-mcp --upgrade
```

### Docker Installation (Alternative)

You can run the Comet MCP server using Docker to avoid installing Python dependencies on your system.

1. **Build the Docker image:**
   ```bash
   docker build -t comet-mcp .
   ```

2. **Configure your MCP client** (see Usage section below for configuration examples)

## Configuration

The server uses standard comet_ml configuration:

1. Using `comet init`; or
2. Using environment variables

Example:

```bash
export COMET_API_KEY=your_comet_api_key_here

# Optional: Set default workspace (if not provided, uses your default)
export COMET_WORKSPACE=your_workspace_name
```

## Available Tools

### Core Comet ML Tools

- **`list_experiments(workspace, project_name)`** - List recent experiments with optional filtering
- **`get_experiment_details(experiment_id)`** - Get comprehensive experiment information including metrics and parameters
- **`get_experiment_code(experiment_id)`** - Retrieve source code from experiments
- **`get_experiment_metric_data(experiment_ids, metric_names, x_axis)`** - Get metric data for multiple experiments
- **`get_default_workspace()`** - Get the default workspace name for the current user
- **`list_projects(workspace)`** - List all projects in a workspace
- **`list_project_experiments(project_name, workspace)`** - List experiments within a specific project
- **`count_project_experiments(project_name, workspace)`** - Count and analyze experiments in a project
- **`get_session_info()`** - Get current session status and connection information

### Tool Features

- **Structured Data**: All tools return properly typed data structures
- **Error Handling**: Graceful handling of API failures and missing data
- **Flexible Filtering**: Filter by workspace, project, or search terms
- **Rich Metadata**: Includes timestamps, descriptions, and status information
- **File Resources**: Some tools (like `experiment_spreadsheet`) create CSV files that are available as MCP resources

### MCP Resources

The server provides access to generated files (like CSV exports) through the MCP resources API. When a tool creates a file, it returns a resource URI that can be accessed using the MCP `read_resource` method.

**Accessing Resources:**
- Tools that create files will return a `resource_uri` in their response
- Use the MCP `read_resource` method with the URI to read the file content
- Resources are stored on the server and can be accessed without processing all content through the LLM

**Example:**
```python
# After calling experiment_spreadsheet, you'll get a resource_uri
# Access it using:
read_resource(uri="file://comet-mcp/experiment_spreadsheet_20251206_103508.csv")
```

Most MCP clients (like Claude Desktop, Cursor, etc.) will automatically handle resource access when you reference the resource URI in your conversation.

## Usage

### 1. MCP Server Mode

Run the server to provide tools to MCP clients:

```bash
# Start the MCP server
comet-mcp
```

The server will:
- Initialize Comet ML session
- Register all available tools
- Listen for MCP client connections via stdio

### 2. Configuration File

Create a configuration for your AI system. For example:

**Local Installation:**
```json
{
  "servers": [
    {
      "name": "comet-mcp",
      "description": "Comet ML MCP server for experiment management",
      "command": "comet-mcp",
      "env": {
        "COMET_API_KEY": "${COMET_API_KEY}"
      }
    }
  ]
}
```

**Docker Installation (Alternative):**
```json
{
  "mcpServers": {
    "comet-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "COMET_API_KEY",
        "-e",
        "COMET_WORKSPACE",
        "comet-mcp",
        "comet-mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "COMET_API_KEY": "your_api_key_here",
        "COMET_WORKSPACE": "your_workspace_name"
      }
    }
  }
}
```

`comet-mcp` supports "stdio" and "sse" transport modes.


## 4. Command line options

```
usage: comet-mcp [-h] [--transport {stdio,sse}] [--host HOST] [--port PORT]

Comet ML MCP Server

options:
  -h, --help            show this help message and exit
  --transport {stdio,sse}
                        Transport method to use (default: stdio)
  --host HOST           Host for SSE transport (default: localhost)
  --port PORT           Port for SSE transport (default: 8000)
```

## 5. OpenTelemetry Observability

The Comet MCP server includes built-in OpenTelemetry instrumentation for distributed tracing and structured logging. This provides visibility into server operations, tool calls, and Comet ML API interactions.

### Features

- **Distributed Tracing**: Track requests across server operations, tool calls, and API interactions
- **Structured Logging**: Capture detailed log events with context
- **Dual Export**: Export telemetry data to both files and Opik (Comet's observability platform)
- **Low Overhead**: Minimal performance impact with async-friendly instrumentation

### Configuration

Telemetry is enabled by default but can be configured via environment variables.

#### General Configuration

```bash
# Enable/disable telemetry (default: true)
export OTEL_ENABLED=true

# Service name (default: comet-mcp)
export OTEL_SERVICE_NAME=comet-mcp

# Service version (default: 1.2.0)
export OTEL_SERVICE_VERSION=1.2.0
```

#### File Export Configuration

Export traces and logs to local files in JSON Lines format:

```bash
# Path for trace export file (default: traces.jsonl, empty to disable)
export OTEL_TRACES_FILE=traces.jsonl

# Path for log export file (default: logs.jsonl, empty to disable)
export OTEL_LOGS_FILE=logs.jsonl
```

**File Format:**
- Traces: OTLP JSON format, one span per line
- Logs: Structured JSON format, one log record per line
- Files are append-only and can be rotated externally

**Example: Reading trace files:**
```python
import json

with open("traces.jsonl", "r") as f:
    for line in f:
        span = json.loads(line)
        print(f"Span: {span['name']}, Duration: {span['end_time_unix_nano'] - span['start_time_unix_nano']}")
```

#### Opik Export Configuration

Export traces and logs to Opik (Comet's observability platform) for cloud-based observability.

**Option 1: Using OTLP Environment Variables**

```bash
# Opik endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="https://www.comet.com/opik/api/v1/private/otel"

# Headers (comma-separated key=value pairs)
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=your-api-key,projectName=your-project,Comet-Workspace=your-workspace"
```

**Option 2: Using Individual Variables**

```bash
# Opik endpoint (defaults to Comet Cloud if not set)
export OPIK_ENDPOINT="https://www.comet.com/opik/api/v1/private/otel"

# Opik API key
export OPIK_API_KEY=your-api-key

# Opik project name
export OPIK_PROJECT_NAME=your-project

# Comet workspace name
export OPIK_WORKSPACE=your-workspace
```

**For Self-Hosted Opik:**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:5173/api/v1/private/otel"
```

**For Enterprise Deployment:**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://<comet-deployment-url>/opik/api/v1/private/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=your-api-key,projectName=your-project,Comet-Workspace=your-workspace"
```

### Configuration Examples

**File-only export:**
```bash
export OTEL_TRACES_FILE=traces.jsonl
export OTEL_LOGS_FILE=logs.jsonl
# Opik export disabled (no endpoint configured)
```

**Opik-only export:**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://www.comet.com/opik/api/v1/private/otel"
export OPIK_API_KEY=your-api-key
export OPIK_PROJECT_NAME=your-project
export OPIK_WORKSPACE=your-workspace
# File export disabled (empty file paths)
export OTEL_TRACES_FILE=""
export OTEL_LOGS_FILE=""
```

**Both file and Opik export:**
```bash
# File export
export OTEL_TRACES_FILE=traces.jsonl
export OTEL_LOGS_FILE=logs.jsonl

# Opik export
export OTEL_EXPORTER_OTLP_ENDPOINT="https://www.comet.com/opik/api/v1/private/otel"
export OPIK_API_KEY=your-api-key
export OPIK_PROJECT_NAME=your-project
export OPIK_WORKSPACE=your-workspace
```

### What Gets Instrumented

The following operations are automatically instrumented:

- **Server Lifecycle**: Startup, shutdown, session initialization
- **Tool Operations**: All MCP tool calls (`list_tools`, `call_tool`, `list_resources`, `read_resource`)
- **Comet ML API Calls**: All tool functions that interact with Comet ML API
- **Cache Operations**: Cache hits, misses, and writes
- **Session Management**: Session initialization and API access

### Viewing Traces

**In Opik:**
1. Navigate to your Opik project
2. Open the Traces view
3. Filter by service name: `comet-mcp`
4. Explore trace spans and their relationships

**From Files:**
- Use tools like `jq` to parse JSON Lines files:
  ```bash
  cat traces.jsonl | jq '.name, .attributes'
  ```
- Import into analysis tools that support OTLP JSON format
- Use log aggregation tools for log files

### Troubleshooting

**Telemetry not appearing:**
- Check that `OTEL_ENABLED=true` (or not set, defaults to true)
- Verify file paths are writable (for file export)
- Check network connectivity (for Opik export)
- Review server logs for telemetry initialization messages

**Opik export errors:**
- Verify API key and endpoint are correct
- Check that project name and workspace match your Opik configuration
- Ensure you're using HTTP endpoint (not gRPC)
- Network errors are logged but don't crash the server

For more information about Opik, see the [Opik OpenTelemetry documentation](https://www.comet.com/docs/opik/integrations/opentelemetry).

## 6. Integration with Opik for use, testing, and optimization

For complete details on testing this (or any MCP server) see [examples/README](https://github.com/comet-ml/comet-mcp/blob/main/examples/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/comet-ml/comet-mcp)
- **Issues**: [GitHub Issues](https://github.com/comet-ml/comet-mcp/issues)
- **Comet ML**: [Comet ML Documentation](https://www.comet.ml/docs/)

