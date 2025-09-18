# Installation Guide

## Quick Installation

### From Source (Development)

1. Clone the repository:
```bash
git clone <repository-url>
cd comet-mcp
```

2. Install in development mode:
```bash
pip install -e .
```

### From PyPI (When Published)

```bash
pip install comet-mcp
```

## Configuration

1. Copy the environment template:
```bash
cp env.example .env
```

2. Edit `.env` with your Comet ML credentials:
```bash
COMET_API_KEY=your_api_key_here
COMET_WORKSPACE=your_workspace_name  # Optional
```

## Usage

### Running the MCP Server

```bash
# Using the installed package entry point
comet-mcp-server

# Or using the module directly
python -m comet_mcp
```

### Testing the Installation

```bash
# Run basic tests
python test_server.py

# Run comprehensive test suite
pytest test_comet_mcp.py

# Run examples
python example_usage.py
```

## Development Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run linting and formatting:
```bash
black comet_mcp/
isort comet_mcp/
flake8 comet_mcp/
mypy comet_mcp/
```

3. Run tests:
```bash
pytest
```

## Package Structure

```
comet-mcp/
├── comet_mcp/              # Main package
│   ├── __init__.py         # Package exports
│   ├── __main__.py         # Module entry point (python -m comet_mcp)
│   ├── comet_client.py     # Comet ML API client
│   ├── config.py           # Configuration management
│   ├── tool_handler.py     # MCP tool handlers
│   └── server.py           # MCP server implementation
├── example_usage.py        # Usage examples
├── test_server.py          # Basic testing script
├── test_comet_mcp.py       # Comprehensive test suite
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the package with `pip install -e .`
2. **API Key Issues**: Verify your `.env` file has the correct `COMET_API_KEY`
3. **Permission Errors**: Ensure the module can be executed: `python -m comet_mcp`

### Getting Help

- Check the logs for detailed error messages
- Verify your Comet ML API key is valid
- Ensure you have the required dependencies installed
