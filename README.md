# Comet ML MCP Server

A Model Context Protocol (MCP) server that provides tools for interacting with Comet ML API. This server allows users to manage experiments, projects, and data through Comet ML's platform using their API key.

## Features

- **Experiment Management**: Get experiments, view details, metrics, and parameters
- **Project Management**: List and explore projects within workspaces
- **Workspace Management**: Access and manage workspaces
- **Search Capabilities**: Search experiments using query strings
- **Asset Management**: View experiment assets and files
- **Logging**: Access experiment logs and debugging information

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables by copying the example file:

```bash
cp env.example .env
```

4. Edit the `.env` file with your Comet ML credentials:

```bash
# Comet ML API Configuration
COMET_API_KEY=your_comet_api_key_here
COMET_WORKSPACE=your_workspace_name
```

## Getting Your Comet ML API Key

1. Log in to your Comet ML account
2. Go to your account settings
3. Navigate to the "API Keys" section
4. Generate a new API key or copy an existing one
5. Add it to your `.env` file

## Usage

### Running the Server

To run the MCP server:

```bash
python comet_mcp_server.py
```

The server will start and listen for MCP client connections via stdio.

### Available Tools

The server provides the following tools:

#### 1. `get_experiments`
Get experiments from Comet ML workspace/project.

**Parameters:**
- `workspace` (optional): Workspace name (uses default if not provided)
- `project` (optional): Project name
- `limit` (optional): Maximum number of experiments to return (default: 100, max: 1000)
- `offset` (optional): Number of experiments to skip (default: 0)

**Example:**
```json
{
  "name": "get_experiments",
  "arguments": {
    "workspace": "my-workspace",
    "project": "my-project",
    "limit": 50
  }
}
```

#### 2. `get_experiment_details`
Get detailed information about a specific experiment.

**Parameters:**
- `experiment_key` (required): The experiment key/ID

**Example:**
```json
{
  "name": "get_experiment_details",
  "arguments": {
    "experiment_key": "abc123def456"
  }
}
```

#### 3. `get_experiment_metrics`
Get metrics for a specific experiment.

**Parameters:**
- `experiment_key` (required): The experiment key/ID

#### 4. `get_experiment_parameters`
Get parameters for a specific experiment.

**Parameters:**
- `experiment_key` (required): The experiment key/ID

#### 5. `get_projects`
Get projects from Comet ML workspace.

**Parameters:**
- `workspace` (optional): Workspace name (uses default if not provided)

#### 6. `get_workspaces`
Get available workspaces.

**Parameters:** None

#### 7. `search_experiments`
Search experiments using a query string.

**Parameters:**
- `query` (required): Search query string
- `workspace` (optional): Workspace name (uses default if not provided)
- `project` (optional): Project name
- `limit` (optional): Maximum number of experiments to return (default: 100)

**Example:**
```json
{
  "name": "search_experiments",
  "arguments": {
    "query": "accuracy > 0.9",
    "workspace": "my-workspace",
    "limit": 20
  }
}
```

#### 8. `get_experiment_assets`
Get assets (files) for a specific experiment.

**Parameters:**
- `experiment_key` (required): The experiment key/ID

#### 9. `get_experiment_logs`
Get logs for a specific experiment.

**Parameters:**
- `experiment_key` (required): The experiment key/ID

## Configuration

### Environment Variables

- `COMET_API_KEY`: Your Comet ML API key (required)
- `COMET_WORKSPACE`: Default workspace name (optional)

### Configuration File

The server uses a modular configuration system in `config.py`. You can customize:

- API base URL
- Request timeout
- Logging level
- Server name and version

## Error Handling

The server includes comprehensive error handling:

- **API Errors**: Proper handling of Comet ML API errors with status codes
- **Validation**: Input validation for all tool parameters
- **Network Errors**: Timeout and connection error handling
- **Configuration Errors**: Clear error messages for missing or invalid configuration

## Development

### Project Structure

```
comet-mcp/
├── comet_mcp_server.py    # Main MCP server
├── config.py              # Configuration management
├── comet_client.py        # Comet ML API client
├── tool_handler.py        # Tool handlers
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

### Adding New Tools

To add new tools:

1. Add the tool definition to `get_comet_tools()` in `tool_handler.py`
2. Implement the corresponding method in `CometClient` class
3. Add the handler logic in `CometToolHandler.handle_tool_call()`
4. Add formatting logic if needed

### Testing

You can test the server by running it and using an MCP client to connect. The server will log all requests and responses for debugging.

## Troubleshooting

### Common Issues

1. **"COMET_API_KEY environment variable not set"**
   - Make sure you've created a `.env` file with your API key
   - Verify the API key is correct and has proper permissions

2. **"API request failed: 401 Unauthorized"**
   - Check that your API key is valid and not expired
   - Ensure you have access to the specified workspace

3. **"Request timeout"**
   - The default timeout is 30 seconds
   - You can adjust this in the configuration if needed

4. **"No experiments found"**
   - Verify the workspace and project names are correct
   - Check that you have experiments in the specified workspace/project

### Logging

The server logs all operations at INFO level by default. You can adjust the logging level in the configuration or by setting the `LOG_LEVEL` environment variable.

## License

This project is open source. Please check the license file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues related to:
- This MCP server: Please open an issue in this repository
- Comet ML API: Please contact Comet ML support
- MCP protocol: Please refer to the MCP documentation
