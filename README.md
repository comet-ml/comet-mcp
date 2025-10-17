# Comet ML MCP Server

A comprehensive Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API. This server enables seamless integration with Comet ML's experiment tracking platform through a standardized protocol.

## Features

- **🔧 MCP Server**: Full Model Context Protocol implementation for tool integration
- **📊 Experiment Management**: List, search, and analyze experiments with detailed metrics
- **📁 Project Management**: Organize and explore projects across workspaces
- **🔍 Advanced Search**: Search experiments by name, description, and project
- **📈 Session Management**: Singleton `comet_ml.API()` instance with robust error handling
- **🧪 Comprehensive Testing**: Unit tests for all functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- Comet ML account and API key

### Install from Source

```bash
# Clone the repository
git clone https://github.com/comet-ml/comet-mcp.git
cd comet-mcp

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```


## Configuration

The server uses environment variables for Comet ML configuration:

```bash
# Required: Your Comet ML API key
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

Create a `config.json` file for custom server configurations:

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

## Development

### Project Structure

```
comet-mcp/
├── comet_mcp/           # Main package
│   ├── server.py        # MCP server implementation
│   ├── tools.py         # Comet ML tools
│   ├── session.py       # Session management
│   └── utils.py         # Utilities and tool registry
├── tests/               # Test suite
│   └── unit/           # Unit tests
├── pyproject.toml      # Project configuration
└── LICENSE            # License file
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/


# Run with coverage
pytest --cov=comet_mcp
```

### Code Quality

The project includes comprehensive tooling for code quality:

```bash
# Format code
black comet_mcp/ tests/

# Sort imports
isort comet_mcp/ tests/

# Lint code
flake8 comet_mcp/ tests/

# Type checking
mypy comet_mcp/
```

## Architecture

### Session Management

The server uses a robust session context manager that provides:

- **Singleton Pattern**: Single `comet_ml.API()` instance per server session
- **Thread Safety**: Safe concurrent access to Comet ML API
- **Error Recovery**: Graceful handling of API initialization failures
- **Configuration Management**: Centralized API key and workspace management

### Tool Registry

The tool registry system provides:

- **Automatic Registration**: Decorator-based tool registration
- **Schema Generation**: Automatic JSON schema generation from Python functions
- **Type Safety**: Full type hints and validation
- **Error Handling**: Comprehensive error reporting and recovery

### MCP Integration

The server implements the full MCP specification:

- **Tool Discovery**: Dynamic tool listing and metadata
- **Tool Execution**: Asynchronous tool calling with proper error handling
- **Protocol Compliance**: Full compatibility with MCP clients
- **Extensibility**: Easy addition of new tools and capabilities

## Examples

### List Recent Experiments

```python
# Through MCP client
result = await client.call_tool("list_experiments", {"workspace": "my-workspace"})

# Through chatbot
# User: "Show me my recent experiments"
```

### Search Experiments

```python
# Search across all experiments
result = await client.call_tool("search_experiments", {
    "query": "machine learning",
    "workspace": "research"
})

# Search within a specific project
result = await client.call_tool("search_experiments", {
    "query": "neural network",
    "project_name": "deep-learning"
})
```

### Analyze Project

```python
# Get project statistics
count_result = await client.call_tool("count_project_experiments", {
    "project_name": "my-project"
})

# List all experiments in project
experiments = await client.call_tool("list_project_experiments", {
    "project_name": "my-project"
})
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/comet-ml/comet-mcp)
- **Issues**: [GitHub Issues](https://github.com/comet-ml/comet-mcp/issues)
- **Comet ML**: [Comet ML Documentation](https://www.comet.ml/docs/)

