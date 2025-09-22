# Comet ML MCP Server

A Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API. This server allows users to manage experiments, projects, and data through Comet ML's platform using their API key.

## Features

- **Session Management**: Singleton `comet_ml.API()` instance accessible across all tools
- **Experiment Management**: List, search, and get details about experiments
- **Project Management**: List and manage Comet ML projects
- **Simple Configuration**: Only requires API key, workspace/project specified per tool call

## Configuration

The server uses environment variables for Comet ML configuration:

```bash
# Required: Your Comet ML API key
export COMET_API_KEY=your_comet_api_key_here
```

## Available Tools

### Basic Tools
- `add(a, b)` - Add two numbers
- `multiply(a, b)` - Multiply two numbers  
- `greet(name)` - Greet someone by name

### Comet ML Tools
- `list_experiments(limit, workspace)` - List recent experiments
- `get_experiment_details(experiment_id)` - Get detailed experiment information
- `list_projects(workspace)` - List projects in workspace
- `get_session_info()` - Get current session information
- `search_experiments(query, limit)` - Search experiments by name/description

## Usage

1. Set up your environment variables (see Configuration section)
2. Run the server: `comet-mcp`
3. Use the tools through an MCP client

## Architecture

The server uses a session context manager to provide singleton access to `comet_ml.API()` across all tools. This ensures:

- **Thread Safety**: Single API instance per server session
- **Configuration Management**: Centralized API key management
- **Error Handling**: Graceful handling of API initialization failures
- **Clean Separation**: Comet ML tools are separate from basic tools

