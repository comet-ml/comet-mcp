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

## 5. Integration with Opik for use, testing, and optimization

For complete details on testing this (or any MCP server) see [examples/README](https://github.com/comet-ml/comet-mcp/blob/main/examples/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/comet-ml/comet-mcp)
- **Issues**: [GitHub Issues](https://github.com/comet-ml/comet-mcp/issues)
- **Comet ML**: [Comet ML Documentation](https://www.comet.ml/docs/)

