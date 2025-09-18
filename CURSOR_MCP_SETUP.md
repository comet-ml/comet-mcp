# Comet ML MCP Server in Cursor

## ✅ Setup Complete!

Your Comet ML MCP server has been added to Cursor's configuration. Here's what you can now do:

### 🚀 Available Tools

The Comet ML MCP server provides these tools in Cursor:

1. **get_workspaces** - Get all available workspaces
2. **get_projects** - Get projects from a workspace
3. **get_experiments** - Get experiments from workspace/project
4. **get_experiment_details** - Get detailed info about a specific experiment
5. **get_experiment_metrics** - Get metrics for an experiment
6. **get_experiment_parameters** - Get parameters for an experiment
7. **search_experiments** - Search experiments with a query
8. **get_experiment_assets** - Get files/assets for an experiment
9. **get_experiment_logs** - Get logs for an experiment

### 💬 How to Use

1. **Restart Cursor** to load the new MCP server configuration
2. **Ask questions** like:
   - "Show me my Comet ML workspaces"
   - "Get experiments from my project"
   - "Search for experiments with 'accuracy' in the name"
   - "Show me the metrics for experiment XYZ"

### 🔧 Configuration

The server is configured in `~/.config/Cursor/User/settings.json`:

```json
"mcp.servers": {
    "comet-ml": {
        "command": "python",
        "args": ["-m", "comet_mcp"],
        "cwd": "/home/dsblank/comet/comet-mcp",
        "env": {
            "COMET_API_KEY": "BVZZk68boDqH2YyczAtT0uSCN",
            "COMET_WORKSPACE": ""
        }
    }
}
```

### 🛠️ Troubleshooting

If the MCP server doesn't work:

1. **Check the server status** in Cursor's MCP panel
2. **Verify the API key** is correct in your `.env` file
3. **Test manually**: `cd /home/dsblank/comet/comet-mcp && python -m comet_mcp`
4. **Check logs** in Cursor's developer console

### 📝 Example Queries

Try asking Cursor:

- "What workspaces do I have in Comet ML?"
- "Show me the latest experiments in my default workspace"
- "Find experiments that contain 'model' in the name"
- "Get the details for experiment [experiment-key]"

The MCP server will automatically handle the API calls and return formatted results!

