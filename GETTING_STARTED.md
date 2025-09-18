# Getting Started with Comet ML MCP Server

## Quick Setup Guide

### 1. Get Your Comet ML API Key

1. **Sign up/Login**: Go to [https://www.comet.ml/](https://www.comet.ml/) and create an account or log in
2. **Navigate to API Keys**: 
   - Click on your profile/avatar in the top right
   - Go to "Account Settings" or "Profile"
   - Look for "API Keys" section
3. **Generate API Key**:
   - Click "Generate New API Key" or "Create API Key"
   - Copy the generated key (it will look like: `abc123def456...`)
   - **Important**: Save it securely - you won't be able to see it again!

### 2. Configure the Server

1. **Edit your `.env` file**:
   ```bash
   nano .env
   ```

2. **Add your API key**:
   ```bash
   COMET_API_KEY=your_actual_api_key_here
   COMET_WORKSPACE=your_workspace_name  # optional
   ```

### 3. Test the Connection

```bash
python test_server.py
```

You should see:
```
✅ Successfully connected! Found X workspaces
```

### 4. Start the MCP Server

```bash
python comet_mcp_server.py
```

The server will start and wait for MCP client connections.

### 5. Connect with an MCP Client

#### Option A: Claude Desktop
1. Install Claude Desktop
2. Add the server to your MCP configuration
3. The server will be available as tools in Claude

#### Option B: Test with Python
```python
# Example of how to use the tools programmatically
from tool_handler import CometToolHandler
from comet_client import CometClient
from config import get_config

async def test_tools():
    comet_config, _ = get_config()
    client = CometClient(comet_config)
    handler = CometToolHandler(client)
    
    # Get workspaces
    result = await handler.handle_tool_call("get_workspaces", {})
    print(result)

# Run it
import asyncio
asyncio.run(test_tools())
```

## Troubleshooting

### Common Issues

**"COMET_API_KEY environment variable not set"**
- Make sure you've created a `.env` file
- Check that the API key is correct in the file

**"API request failed: 401 Unauthorized"**
- Your API key is invalid or expired
- Generate a new API key from Comet ML

**"API request failed: 404 Not Found"**
- The API endpoint might have changed
- Check your Comet ML account status

**"No experiments found"**
- You might not have any experiments in your workspace
- Try creating a test experiment in Comet ML first
- Check that the workspace/project names are correct

### Getting Help

1. **Check the logs**: The server logs all operations
2. **Test individual components**: Use `python test_server.py`
3. **Verify API key**: Test your key directly with Comet ML's API
4. **Check Comet ML status**: Make sure their service is running

## Example Workflows

### Explore Your Data
```bash
# 1. See what workspaces you have
# Tool: get_workspaces

# 2. See projects in a workspace  
# Tool: get_projects (with workspace parameter)

# 3. See recent experiments
# Tool: get_experiments (with workspace/project)

# 4. Get details about a specific experiment
# Tool: get_experiment_details (with experiment_key)
```

### Search and Analyze
```bash
# 1. Search for high-performing experiments
# Tool: search_experiments (query: "accuracy > 0.9")

# 2. Get metrics for analysis
# Tool: get_experiment_metrics (with experiment_key)

# 3. Compare parameters
# Tool: get_experiment_parameters (with experiment_key)
```

## Next Steps

Once you have the server running:

1. **Explore your data**: Use the tools to browse your experiments
2. **Set up monitoring**: Use the server to track experiment performance
3. **Integrate with workflows**: Connect the server to your ML pipelines
4. **Extend functionality**: Add new tools for your specific needs

Happy experimenting! 🚀
