#!/usr/bin/env python3
"""
agent_loop_reference.py - Claude Code Agent Loop 完整参考实现

基于 Claude Code CLI (cli.js) 的逆向分析，提取核心 agent loop 逻辑。
这是一个可读的 Python 实现，包含所有关键机制。

核心函数对照：
- UR() -> agent_loop()           # 主循环
- Ff6() -> execute_tools()       # 工具执行（支持并行）
- xH6() -> truncate_output()     # 输出截断
- W29() -> persist_large_output() # 大输出落盘
- jy() -> run_subagent()         # 子代理执行

关键机制：
1. max_turns 限制 - 防止无限循环
2. 输出截断 - 防止 context 撑爆
3. 中断处理 - 优雅退出
4. 并行工具执行 - 提高效率
5. 自动消息压缩 - 长对话处理
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from anthropic import Anthropic


# =============================================================================
# Constants (from Claude Code cli.js)
# =============================================================================

# Max turns to prevent infinite loops
MAX_TURNS_MAIN = 100
MAX_TURNS_SUBAGENT = 30

# Output truncation limits
MAX_OUTPUT_CHARS = 400_000      # ~400KB bash output limit
MAX_OUTPUT_LINES = 2000         # Max lines for file reads
MAX_LINE_CHARS = 2000           # Max chars per line
PREVIEW_SIZE = 2000             # Preview size for persisted outputs

# Concurrency
MAX_TOOL_CONCURRENCY = 10       # Max parallel tool executions


# =============================================================================
# Message Types (matching Claude Code internal types)
# =============================================================================

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    PROGRESS = "progress"
    ATTACHMENT = "attachment"
    STREAM_REQUEST_START = "stream_request_start"
    STREAM_EVENT = "stream_event"
    TOMBSTONE = "tombstone"


class AttachmentType(Enum):
    MAX_TURNS_REACHED = "max_turns_reached"
    TOOL_RESULT = "tool_result"
    HOOK_STOPPED = "hook_stopped_continuation"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Internal message representation."""
    type: MessageType
    content: Any
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    attachment: Optional[Dict] = None
    api_error: Optional[str] = None


@dataclass
class ToolUseBlock:
    """Represents a tool_use block from API response."""
    id: str
    name: str
    input: Dict


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_use_id: str
    content: str
    is_error: bool = False


# =============================================================================
# Output Truncation (from cli.js xH6, W29)
# =============================================================================

def truncate_output(
    content: str,
    max_chars: int = MAX_OUTPUT_CHARS
) -> str:
    """
    Truncate output with informative message.

    Key insight from Claude Code: Always tell the model HOW MUCH was truncated,
    so it knows information is incomplete and can request specific parts.
    """
    if len(content) <= max_chars:
        return content

    truncated = content[:max_chars]
    remaining_lines = content[max_chars:].count('\n')

    return f"{truncated}\n\n... [{remaining_lines} lines truncated] ..."


def truncate_lines(content: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Truncate by line count."""
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content

    truncated_count = len(lines) - max_lines
    return '\n'.join(lines[:max_lines]) + f"\n\n... [{truncated_count} lines truncated] ..."


def truncate_long_lines(content: str, max_chars: int = MAX_LINE_CHARS) -> str:
    """Truncate individual long lines."""
    lines = content.split('\n')
    result = []
    for line in lines:
        if len(line) > max_chars:
            result.append(line[:max_chars] + "... [line truncated]")
        else:
            result.append(line)
    return '\n'.join(result)


class OutputBuffer:
    """
    Output buffer with automatic truncation.

    Tracks total bytes received vs what's kept, so we can report
    accurate truncation information to the model.
    """

    def __init__(self, max_size: int = MAX_OUTPUT_CHARS):
        self.max_size = max_size
        self.content = ""
        self.is_truncated = False
        self.total_bytes_received = 0

    def append(self, data: str) -> None:
        self.total_bytes_received += len(data)

        if self.is_truncated and len(self.content) >= self.max_size:
            return

        if len(self.content) + len(data) > self.max_size:
            remaining = self.max_size - len(self.content)
            if remaining > 0:
                self.content += data[:remaining]
            self.is_truncated = True
        else:
            self.content += data

    def __str__(self) -> str:
        if not self.is_truncated:
            return self.content

        removed_bytes = self.total_bytes_received - self.max_size
        kb_removed = round(removed_bytes / 1024)
        return self.content + f"\n... [output truncated - {kb_removed}KB removed]"

    def clear(self) -> None:
        self.content = ""
        self.is_truncated = False
        self.total_bytes_received = 0


async def persist_large_output(
    content: str,
    tool_use_id: str,
    output_dir: Path
) -> Dict:
    """
    Persist large tool output to file system (from cli.js W29, pq1).

    Returns reference that can be included in tool result,
    allowing model to read specific parts later.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    is_json = content.strip().startswith('{') or content.strip().startswith('[')
    ext = "json" if is_json else "txt"
    filepath = output_dir / f"{tool_use_id}.{ext}"

    filepath.write_text(content)

    # Generate preview
    preview = content[:PREVIEW_SIZE]
    has_more = len(content) > PREVIEW_SIZE

    return {
        "filepath": str(filepath),
        "original_size": len(content),
        "is_json": is_json,
        "preview": preview,
        "has_more": has_more,
    }


def format_persisted_output(info: Dict) -> str:
    """Format persisted output reference for tool result."""
    result = "<persisted-output>\n"
    result += f"Output too large ({info['original_size']} bytes). "
    result += f"Full output saved to: {info['filepath']}\n\n"
    result += f"Preview (first {PREVIEW_SIZE} bytes):\n"
    result += info['preview']
    if info['has_more']:
        result += "\n..."
    result += "\n</persisted-output>"
    return result


# =============================================================================
# Tool Execution (from cli.js Ff6, clY, llY)
# =============================================================================

@dataclass
class ToolExecutionResult:
    """Result from executing a tool."""
    message: Optional[AgentMessage]
    context_modifier: Optional[Callable] = None


class Tool(ABC):
    """Base class for tools."""

    name: str
    description: str
    input_schema: Dict

    @abstractmethod
    async def execute(self, input: Dict, context: "ToolUseContext") -> str:
        """Execute the tool and return result."""
        pass

    def is_concurrency_safe(self, input: Dict) -> bool:
        """Whether this tool can be executed in parallel with others."""
        return False  # Default: sequential execution


async def execute_tools_sequential(
    tool_calls: List[ToolUseBlock],
    tools: Dict[str, Tool],
    context: "ToolUseContext"
) -> AsyncGenerator[ToolExecutionResult, None]:
    """
    Execute tools sequentially (from cli.js clY).

    Used when tools are not concurrency-safe.
    """
    for tc in tool_calls:
        tool = tools.get(tc.name)
        if not tool:
            yield ToolExecutionResult(
                message=AgentMessage(
                    type=MessageType.USER,
                    content=[{
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Unknown tool: {tc.name}",
                        "is_error": True,
                    }]
                )
            )
            continue

        try:
            output = await tool.execute(tc.input, context)

            # Apply truncation
            output = truncate_output(output)

            yield ToolExecutionResult(
                message=AgentMessage(
                    type=MessageType.USER,
                    content=[{
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": output,
                    }]
                )
            )
        except Exception as e:
            yield ToolExecutionResult(
                message=AgentMessage(
                    type=MessageType.USER,
                    content=[{
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    }]
                )
            )


async def execute_tools_parallel(
    tool_calls: List[ToolUseBlock],
    tools: Dict[str, Tool],
    context: "ToolUseContext",
    max_concurrency: int = MAX_TOOL_CONCURRENCY
) -> AsyncGenerator[ToolExecutionResult, None]:
    """
    Execute tools in parallel (from cli.js llY).

    Uses semaphore to limit concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_one(tc: ToolUseBlock) -> ToolExecutionResult:
        async with semaphore:
            tool = tools.get(tc.name)
            if not tool:
                return ToolExecutionResult(
                    message=AgentMessage(
                        type=MessageType.USER,
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": f"Unknown tool: {tc.name}",
                            "is_error": True,
                        }]
                    )
                )

            try:
                output = await tool.execute(tc.input, context)
                output = truncate_output(output)

                return ToolExecutionResult(
                    message=AgentMessage(
                        type=MessageType.USER,
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": output,
                        }]
                    )
                )
            except Exception as e:
                return ToolExecutionResult(
                    message=AgentMessage(
                        type=MessageType.USER,
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": f"Error: {e}",
                            "is_error": True,
                        }]
                    )
                )

    # Execute all in parallel
    tasks = [execute_one(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks)

    for result in results:
        yield result


def group_tools_by_concurrency(
    tool_calls: List[ToolUseBlock],
    tools: Dict[str, Tool]
) -> List[Dict]:
    """
    Group tool calls by concurrency safety (from cli.js dlY).

    Adjacent concurrency-safe tools are grouped together for parallel execution.
    """
    groups = []

    for tc in tool_calls:
        tool = tools.get(tc.name)
        is_safe = tool.is_concurrency_safe(tc.input) if tool else False

        if is_safe and groups and groups[-1]["is_concurrency_safe"]:
            groups[-1]["blocks"].append(tc)
        else:
            groups.append({
                "is_concurrency_safe": is_safe,
                "blocks": [tc],
            })

    return groups


async def execute_tools(
    tool_calls: List[ToolUseBlock],
    tools: Dict[str, Tool],
    context: "ToolUseContext"
) -> AsyncGenerator[ToolExecutionResult, None]:
    """
    Execute tools with smart concurrency (from cli.js Ff6).

    Groups tools by concurrency safety:
    - Concurrency-safe tools run in parallel
    - Others run sequentially
    """
    groups = group_tools_by_concurrency(tool_calls, tools)

    for group in groups:
        if group["is_concurrency_safe"]:
            async for result in execute_tools_parallel(
                group["blocks"], tools, context
            ):
                yield result
        else:
            async for result in execute_tools_sequential(
                group["blocks"], tools, context
            ):
                yield result


# =============================================================================
# Context and State (from cli.js toolUseContext)
# =============================================================================

@dataclass
class QueryTracking:
    """Track query chain for debugging."""
    chain_id: str
    depth: int


@dataclass
class ToolUseContext:
    """
    Context passed through the agent loop (from cli.js toolUseContext).

    Contains all state needed for tool execution and loop control.
    """
    agent_id: str
    agent_type: str
    messages: List[AgentMessage]
    tools: Dict[str, Tool]
    abort_signal: asyncio.Event
    query_tracking: Optional[QueryTracking] = None

    # Callbacks
    get_app_state: Optional[Callable] = None
    set_app_state: Optional[Callable] = None
    set_response_length: Optional[Callable] = None

    # State tracking
    in_progress_tool_ids: Set[str] = field(default_factory=set)

    def set_in_progress_tool_ids(self, ids: Set[str]) -> None:
        self.in_progress_tool_ids = ids


# =============================================================================
# Core Agent Loop (from cli.js UR function)
# =============================================================================

async def agent_loop(
    messages: List[Dict],
    system_prompt: str,
    tools: Dict[str, Tool],
    client: Anthropic,
    model: str = "claude-sonnet-4-20250514",
    max_turns: Optional[int] = None,
    abort_signal: Optional[asyncio.Event] = None,
    on_message: Optional[Callable[[AgentMessage], None]] = None,
) -> AsyncGenerator[AgentMessage, None]:
    """
    Core agent loop (from cli.js UR function).

    This is the heart of Claude Code's agentic behavior.

    Key features:
    1. max_turns limit to prevent infinite loops
    2. Output truncation for tool results
    3. Abort signal handling for interruption
    4. Parallel tool execution when safe
    5. Proper error recovery

    Args:
        messages: Initial conversation messages
        system_prompt: System prompt
        tools: Available tools
        client: Anthropic client
        model: Model to use
        max_turns: Maximum turns before stopping (None = no limit)
        abort_signal: Event to signal abort
        on_message: Callback for each message

    Yields:
        AgentMessage objects for each event
    """
    if abort_signal is None:
        abort_signal = asyncio.Event()

    # Initialize context
    context = ToolUseContext(
        agent_id=str(uuid.uuid4()),
        agent_type="main",
        messages=[],
        tools=tools,
        abort_signal=abort_signal,
    )

    # Track state across iterations (from cli.js W1 object)
    current_messages = list(messages)
    turn_count = 1
    max_output_recovery_count = 0

    # Convert tools to API format
    tools_api = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools.values()
    ]

    # Main loop (from cli.js while(true))
    while True:
        # Signal start of new request
        yield AgentMessage(type=MessageType.STREAM_REQUEST_START, content=None)

        # Check abort signal
        if abort_signal.is_set():
            yield AgentMessage(
                type=MessageType.ATTACHMENT,
                content="Aborted by user",
                attachment={"type": "aborted"}
            )
            return

        # Track query chain
        query_tracking = QueryTracking(
            chain_id=str(uuid.uuid4()),
            depth=turn_count,
        )

        # Call API
        try:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=current_messages,
                tools=tools_api if tools_api else None,
                max_tokens=8192,
            )
        except Exception as e:
            yield AgentMessage(
                type=MessageType.ATTACHMENT,
                content=f"API error: {e}",
                attachment={"type": AttachmentType.ERROR.value, "error": str(e)}
            )
            return

        # Collect assistant messages
        assistant_messages: List[AgentMessage] = []
        tool_results: List[AgentMessage] = []

        # Process response content
        assistant_msg = AgentMessage(
            type=MessageType.ASSISTANT,
            content=response.content,
        )
        assistant_messages.append(assistant_msg)
        yield assistant_msg

        if on_message:
            on_message(assistant_msg)

        # Check for abort after API call
        if abort_signal.is_set():
            # Generate error results for pending tools
            for block in response.content:
                if block.type == "tool_use":
                    yield AgentMessage(
                        type=MessageType.USER,
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Interrupted by user",
                            "is_error": True,
                        }]
                    )
            return

        # Extract tool_use blocks
        tool_calls = [
            ToolUseBlock(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]

        # No tool calls = task complete, exit loop
        if not tool_calls:
            # Check for max_output_tokens error and retry
            if (response.stop_reason == "max_tokens" and
                max_output_recovery_count < 3):

                recovery_msg = AgentMessage(
                    type=MessageType.USER,
                    content=[{
                        "type": "text",
                        "text": "Your response was cut off. Please continue."
                    }]
                )
                current_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": b.text}
                               for b in response.content if b.type == "text"]
                })
                current_messages.append({
                    "role": "user",
                    "content": recovery_msg.content
                })
                max_output_recovery_count += 1
                continue

            # Task complete
            return

        # Execute tools
        async for result in execute_tools(tool_calls, tools, context):
            if result.message:
                yield result.message
                tool_results.append(result.message)

                if on_message:
                    on_message(result.message)

        # Check abort after tool execution
        if abort_signal.is_set():
            return

        # Increment turn count
        turn_count += 1

        # Check max_turns limit (KEY FEATURE)
        if max_turns and turn_count > max_turns:
            yield AgentMessage(
                type=MessageType.ATTACHMENT,
                content=f"Reached max turns limit ({max_turns})",
                attachment={
                    "type": AttachmentType.MAX_TURNS_REACHED.value,
                    "max_turns": max_turns,
                    "turn_count": turn_count,
                }
            )
            return

        # Prepare messages for next iteration
        # Add assistant message
        current_messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, **({k: v for k, v in vars(b).items() if k != "type"})}
                for b in response.content
            ]
        })

        # Add tool results
        tool_result_content = []
        for msg in tool_results:
            if isinstance(msg.content, list):
                tool_result_content.extend(msg.content)

        if tool_result_content:
            current_messages.append({
                "role": "user",
                "content": tool_result_content
            })

        # Reset recovery count on successful tool execution
        max_output_recovery_count = 0


# =============================================================================
# Subagent Execution (from cli.js jy function)
# =============================================================================

async def run_subagent(
    prompt: str,
    agent_type: str,
    parent_context: ToolUseContext,
    tools: Dict[str, Tool],
    client: Anthropic,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = MAX_TURNS_SUBAGENT,
    system_prompt: Optional[str] = None,
) -> List[AgentMessage]:
    """
    Run a subagent task (from cli.js jy function).

    Subagents have:
    - Separate max_turns limit (usually lower)
    - Inherited abort signal
    - Own query tracking
    """
    messages = [{"role": "user", "content": prompt}]

    if system_prompt is None:
        system_prompt = f"You are a {agent_type} subagent. Complete the task and return a clear summary."

    results = []

    async for msg in agent_loop(
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        client=client,
        model=model,
        max_turns=max_turns,
        abort_signal=parent_context.abort_signal,
    ):
        results.append(msg)

    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import subprocess

    # Example tool implementation
    class BashTool(Tool):
        name = "bash"
        description = "Execute shell command"
        input_schema = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"]
        }

        async def execute(self, input: Dict, context: ToolUseContext) -> str:
            cmd = input["command"]
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=60
                )
                output = (result.stdout + result.stderr).strip() or "(no output)"

                # Apply truncation
                output = truncate_long_lines(output)
                output = truncate_output(output)

                return output
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 60 seconds"
            except Exception as e:
                return f"Error: {e}"

        def is_concurrency_safe(self, input: Dict) -> bool:
            # Read-only commands can run in parallel
            cmd = input.get("command", "")
            read_only = ["ls", "cat", "head", "tail", "grep", "find", "pwd"]
            return any(cmd.strip().startswith(c) for c in read_only)

    class ReadFileTool(Tool):
        name = "read_file"
        description = "Read file contents"
        input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"}
            },
            "required": ["path"]
        }

        async def execute(self, input: Dict, context: ToolUseContext) -> str:
            try:
                path = Path(input["path"])
                content = path.read_text()
                lines = content.splitlines()

                offset = input.get("offset", 0)
                limit = input.get("limit", MAX_OUTPUT_LINES)

                lines = lines[offset:offset + limit]
                result = "\n".join(lines)

                return truncate_long_lines(result)
            except Exception as e:
                return f"Error: {e}"

        def is_concurrency_safe(self, input: Dict) -> bool:
            return True  # File reads are always safe to parallelize

    # Demo
    async def demo():
        client = Anthropic()

        tools = {
            "bash": BashTool(),
            "read_file": ReadFileTool(),
        }

        messages = [{"role": "user", "content": "List files in current directory"}]

        print("Running agent loop with max_turns=5...\n")

        async for msg in agent_loop(
            messages=messages,
            system_prompt="You are a helpful coding assistant.",
            tools=tools,
            client=client,
            max_turns=5,
        ):
            if msg.type == MessageType.ASSISTANT:
                for block in msg.content:
                    if hasattr(block, "text"):
                        print(f"Assistant: {block.text[:200]}...")
                    elif block.type == "tool_use":
                        print(f"Tool: {block.name}({block.input})")

            elif msg.type == MessageType.USER:
                for item in msg.content:
                    if item.get("type") == "tool_result":
                        output = item.get("content", "")[:100]
                        print(f"Result: {output}...")

            elif msg.type == MessageType.ATTACHMENT:
                if msg.attachment.get("type") == "max_turns_reached":
                    print(f"\n⚠️ Max turns reached: {msg.attachment}")

        print("\nDone!")

    asyncio.run(demo())
