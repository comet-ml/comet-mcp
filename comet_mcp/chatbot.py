# python client.py [config.json]
# Example: python client.py config.json
# If no config file is provided, uses config.json by default

import asyncio, sys, json, os, subprocess
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from litellm import completion  # can handle tools
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """ 
You are a helpful AI assistant that can use various tools to help
users with their tasks. You have access to MCP (Model Context
Protocol) servers that provide specialized tools. When users ask
questions, use the available tools to gather information and provide
comprehensive answers.
"""


@dataclass
class ServerConfig:
    name: str
    description: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


def _mcp_tools_to_openai_tools(tools_resp) -> List[Dict[str, Any]]:
    # Map MCP tool spec to OpenAI function tools
    tools = []
    for t in tools_resp.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    # MCP provides a proper JSON schema in inputSchema
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _mk_user_msg(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def _mk_assistant_tool_msg(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Record the assistant's tool calls so the model has the chain
    return {"role": "assistant", "tool_calls": tool_calls, "content": ""}


def _mk_tool_result_msg(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class MCPChatbot:
    def __init__(
        self,
        config_path: str,
        model: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_rounds: Optional[int] = 4,
    ):
        self.system_prompt = (
            system_prompt if system_prompt is not None else SYSTEM_PROMPT
        )
        self.servers = self.load_config(config_path)
        self.model = model
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.max_rounds = max_rounds
        self.sessions: Dict[str, ClientSession] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.exit_stack = AsyncExitStack()
        self.clear_messages()

    @staticmethod
    def load_config(config_path: str = "config.json") -> List[ServerConfig]:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            config = {
                "servers": [
                    {
                        "name": "comet-mcp",
                        "description": "Comet ML MCP server for experiment management",
                        "command": "comet-mcp",
                    }
                ]
            }
        else:
            with open(config_path, "r") as f:
                config = json.load(f)

        servers = []
        for server_data in config.get("servers", []):
            # Expand environment variables in env dict
            env = server_data.get("env", {})
            expanded_env = {}
            for key, value in env.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    expanded_env[key] = os.getenv(env_var, "")
                else:
                    expanded_env[key] = value

            servers.append(
                ServerConfig(
                    name=server_data["name"],
                    description=server_data.get("description", ""),
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=expanded_env if expanded_env else None,
                )
            )

        return servers

    async def connect_all_servers(self):
        """Connect to all configured MCP servers via subprocess."""
        for server_config in self.servers:
            try:
                await self._connect_server(server_config)
                print(
                    f"✓ Connected to {server_config.name}: {server_config.description}"
                )
            except Exception as e:
                print(f"✗ Failed to connect to {server_config.name}: {e}")

    async def _connect_server(self, server_config: ServerConfig):
        """Connect to a single MCP server via subprocess."""
        # Set up environment variables for the subprocess
        if server_config.env:
            # Update the current process environment for the subprocess
            original_env = {}
            for key, value in server_config.env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            # Create MCP client session using stdio client
            params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
            )

            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdin, write = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdin, write)
            )
            await session.initialize()

            self.sessions[server_config.name] = session
        finally:
            # Restore original environment variables
            if server_config.env:
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    async def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Aggregate tools from all connected MCP servers."""
        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                tools_resp = await session.list_tools()
                server_tools = _mcp_tools_to_openai_tools(tools_resp)
                # Prefix tool names with server name to avoid conflicts
                for tool in server_tools:
                    tool["function"][
                        "name"
                    ] = f"{server_name}_{tool['function']['name']}"
                all_tools.extend(server_tools)
            except Exception as e:
                print(f"Warning: Failed to get tools from {server_name}: {e}")
        return all_tools

    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a tool call on the appropriate MCP server."""
        fn_name = tool_call.function.name
        args_raw = tool_call.function.arguments or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}

        # Parse server name from tool name (format: server_name_tool_name)
        if "_" in fn_name:
            # Find the first underscore to split server name from tool name
            parts = fn_name.split("_", 1)
            if len(parts) == 2:
                server_name, actual_tool_name = parts
            else:
                # Fallback: treat as tool name without server prefix
                server_name = None
                actual_tool_name = fn_name
        else:
            # Fallback: try to find the tool in any server
            server_name = None
            actual_tool_name = fn_name

        if server_name and server_name in self.sessions:
            session = self.sessions[server_name]
        else:
            # Try to find the tool in any connected server
            session = None
            for srv_name, sess in self.sessions.items():
                try:
                    tools_resp = await sess.list_tools()
                    tool_names = [t.name for t in tools_resp.tools]
                    if actual_tool_name in tool_names:
                        session = sess
                        break
                except Exception:
                    continue

            if session is None:
                return f"Error: Tool '{actual_tool_name}' not found in any connected server"

        try:
            # Call the MCP tool
            result = await session.call_tool(actual_tool_name, args)

            # Best-effort stringify of MCP result content
            if hasattr(result, "content") and result.content is not None:
                try:
                    content_str = json.dumps(result.content)
                except Exception:
                    content_str = str(result.content)
            else:
                content_str = str(result)

            return content_str
        except Exception as e:
            return f"Error executing tool '{actual_tool_name}': {e}"

    async def chat_once(self, user_text: str) -> str:
        if not self.sessions:
            raise RuntimeError("Not connected to any MCP servers.")

        # 1) Fetch tool catalog from all MCP servers
        tools = await self._get_all_tools()

        # 2) Add user message to persistent history
        user_msg = _mk_user_msg(user_text)
        self.messages.append(user_msg)

        # 3) Chat loop with tool calling using persistent messages
        text_reply: str = ""

        for _ in range(self.max_rounds):
            resp = completion(
                model=self.model,
                messages=self.messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else "none",
                **self.model_kwargs,
            )

            choice = resp.choices[0].message
            # If the model just replied with text (no tool calls), return it
            tool_calls = getattr(choice, "tool_calls", None)
            if not tool_calls:
                text_reply = (choice.content or "").strip()
                # Add assistant's final response to persistent history
                self.messages.append({"role": "assistant", "content": text_reply})
                break

            # 4) Execute each requested tool via MCP
            executed_tool_msgs: List[Dict[str, Any]] = []
            assistant_tool_stub = []

            for tc in tool_calls:
                content_str = await self._execute_tool_call(tc)

                # Build messages to feed back to the model
                assistant_tool_stub.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )
                executed_tool_msgs.append(_mk_tool_result_msg(tc.id, content_str))

            # Add the assistant tool-call stub + tool results to persistent history
            self.messages.append(_mk_assistant_tool_msg(assistant_tool_stub))
            self.messages.extend(executed_tool_msgs)

        return text_reply

    def clear_messages(self):
        """Clear the message history, keeping only the system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get a copy of the current message history."""
        return self.messages.copy()

    def get_message_count(self) -> int:
        """Get the number of messages in the history (excluding system prompt)."""
        return len(self.messages) - 1  # Subtract 1 for system prompt

    async def run(self):
        """Run the complete chat session with server connections and chat loop."""
        try:
            print(f"Loaded configuration")
            print(f"Found {len(self.servers)} server(s) to connect to:")
            for server in self.servers:
                print(f"  - {server.name}: {server.description}")

            await self.connect_all_servers()

            if not self.sessions:
                print("No servers connected successfully. Exiting.")
                return

            print(f"\nConnected to {len(self.sessions)} server(s). Ready for chat!")
            print("Type 'quit' or 'exit' to stop.")
            print("Type '/clear' to clear conversation history.\n")

            while True:
                try:
                    q = input("You: ")
                except EOFError:
                    q = ""

                q = q.strip()
                if q.lower() in {"quit", "exit", ""}:
                    break
                elif q.lower() == "/clear":
                    self.clear_messages()
                    print("Conversation history cleared.")
                    continue
                a = await self.chat_once(q)
                print("Bot:", a or "(no reply)")
                print()  # Add spacing between exchanges
        finally:
            print("\nClosing...")
            await self.close()

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    model = "openai/gpt-4o-mini"
    model_kwargs = {"temperature": 0.2, "max_tokens": 700}

    system_prompt = """
You are a helpful AI system for answering questions of Comet ML's
experiment management system. You have access to many Comet tools.
"""

    bot = MCPChatbot(
        config_path, model=model, model_kwargs=model_kwargs, system_prompt=system_prompt
    )
    await bot.run()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
