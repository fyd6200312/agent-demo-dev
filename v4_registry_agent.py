#!/usr/bin/env python3
"""
v4_registry_agent.py - Registry Pattern Tool Management

Evolution from v4_skills_agent.py:
==================================
v4 used simple if-else dispatch:
    if name == "bash": return run_bash(args["command"])
    if name == "read_file": return run_read(args["path"])
    ...

This works but has limitations:
- Adding tools requires modifying execute_tool()
- No dynamic tool registration/removal
- Schema definitions scattered across code

This version introduces Registry pattern with STATELESS tools:
- Tools are classes that self-describe (name, schema, execute)
- Registry manages registration and dispatch
- Tools remain pure functions - no state stored in instances
- Context passed as parameters, not stored

Key Design: Stateless Tools
---------------------------
    # WRONG: State in tool instance (nanobot's problem)
    class MessageTool:
        def set_context(self, channel, chat_id):
            self.channel = channel  # Shared mutable state!
        def execute(self, text):
            send(self.channel, text)  # Race condition!

    # RIGHT: Context as parameter (this implementation)
    class BashTool:
        def execute(self, command: str, *, workdir: Path) -> str:
            return subprocess.run(command, cwd=workdir)

Benefits:
- Open/Closed: Add tools without changing registry code
- Dynamic: Register/unregister at runtime
- Self-describing: Each tool carries its own schema
- Thread-safe: No shared mutable state

Usage:
    python v4_registry_agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# LangFuse Integration (same as v4)
# =============================================================================

try:
    from langfuse import observe, get_client as _get_langfuse_client
    LANGFUSE_ENABLED = bool(
        os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")
    )
    if not LANGFUSE_ENABLED:
        raise ImportError("LangFuse keys not configured")

    def _get_langfuse():
        return _get_langfuse_client()

except ImportError:
    LANGFUSE_ENABLED = False

    def observe(**kwargs):
        def decorator(fn):
            return fn
        return decorator

    class _FakeLangfuse:
        def update_current_span(self, **kwargs): pass
        def update_current_trace(self, **kwargs): pass
        def score_current_trace(self, **kwargs): pass

    def _get_langfuse():
        return _FakeLangfuse()


# =============================================================================
# Logging Configuration
# =============================================================================

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() == "true"


def log_api_call(caller: str, system: str, messages: list, tools: list):
    if not DEBUG_LOG:
        return
    print("\n" + "=" * 80)
    print(f"[API CALL] from: {caller}")
    print("=" * 80)
    print(json.dumps({
        "system": system,
        "messages": messages,
        "tools": tools
    }, ensure_ascii=False, indent=2, default=str))
    print("=" * 80 + "\n")


def log_api_response(caller: str, response):
    if not DEBUG_LOG:
        return
    print("\n" + "=" * 80)
    print(f"[API RESPONSE] from: {caller}")
    print("=" * 80)
    print(response)
    print("=" * 80 + "\n")


# =============================================================================
# Configuration
# =============================================================================

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# Tool Registry - The Core Pattern
# =============================================================================

@dataclass
class ToolSchema:
    """Tool schema for Claude function calling."""
    name: str
    description: str
    input_schema: dict


class Tool(ABC):
    """
    Abstract base class for all tools.

    Key principle: Tools are STATELESS.
    All context (workdir, config, etc.) is passed via execute() parameters.
    This ensures thread-safety and avoids the state-sharing bugs in nanobot.

    Each tool must implement:
    - name: Tool identifier
    - description: What the tool does
    - input_schema: JSON schema for parameters
    - execute(**kwargs, context: ToolContext) -> str
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for function calling."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the model."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, context: "ToolContext", **kwargs) -> str:
        """
        Execute the tool with given parameters.

        Args:
            context: Shared context (workdir, etc.) - passed, not stored
            **kwargs: Tool-specific parameters from the model

        Returns:
            String result to return to the model
        """
        pass

    def to_schema(self) -> dict:
        """Generate Claude function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolContext:
    """
    Context passed to tool execution - NOT stored in tool instances.

    This is the key difference from nanobot's design:
    - nanobot: tool.set_context(channel, chat_id)  # State in instance
    - here: tool.execute(context=ctx, **params)    # State as parameter

    Thread-safe because each call gets its own context.
    """
    workdir: Path
    # Add more fields as needed: user_id, session_id, permissions, etc.
    # The point is: these are PASSED, not stored on tool instances.


class ToolRegistry:
    """
    Registry for managing tools.

    Features:
    - Dynamic registration/unregistration
    - Schema generation for Claude API
    - Centralized dispatch with error handling
    - Filtering for different agent types (readonly vs full access)

    Design principles:
    - Registry holds tool CLASSES, not stateful instances
    - Context is passed at execute time, not stored
    - Thread-safe: no shared mutable state in tools
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        """
        Register a tool. Returns self for chaining.

        Example:
            registry.register(BashTool()).register(ReadFileTool())
        """
        self._tools[tool.name] = tool
        return self

    def unregister(self, name: str) -> bool:
        """Remove a tool. Returns True if found and removed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self, names: list[str] | None = None) -> list[dict]:
        """
        Get schemas for Claude API.

        Args:
            names: Optional filter - only include these tools.
                   If None, returns all tools.
        """
        if names is None:
            return [t.to_schema() for t in self._tools.values()]
        return [
            self._tools[n].to_schema()
            for n in names
            if n in self._tools
        ]

    def execute(self, name: str, context: ToolContext, **kwargs) -> str:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            context: Execution context (passed, not stored!)
            **kwargs: Tool parameters

        Returns:
            Tool result as string
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            return tool.execute(context, **kwargs)
        except Exception as e:
            return f"Error: {e}"

    def subset(self, names: list[str]) -> "ToolRegistry":
        """
        Create a new registry with only specified tools.

        Useful for creating restricted tool sets for subagents:
            readonly_registry = full_registry.subset(["bash", "read_file"])
        """
        new_registry = ToolRegistry()
        for name in names:
            if name in self._tools:
                new_registry.register(self._tools[name])
        return new_registry

    def clone(self) -> "ToolRegistry":
        """Create a copy of this registry."""
        new_registry = ToolRegistry()
        for tool in self._tools.values():
            new_registry.register(tool)
        return new_registry


# =============================================================================
# Decorator for Simple Tool Creation
# =============================================================================

def tool(name: str, description: str, schema: dict):
    """
    Decorator to create a Tool from a simple function.

    Example:
        @tool(
            name="bash",
            description="Run shell command",
            schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        )
        def bash_tool(context: ToolContext, command: str) -> str:
            return subprocess.run(command, shell=True, cwd=context.workdir, ...)

    The decorated function becomes a Tool instance that can be registered.
    """
    def decorator(func: Callable) -> Tool:
        class FunctionTool(Tool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def input_schema(self) -> dict:
                return schema

            def execute(self, context: ToolContext, **kwargs) -> str:
                return func(context, **kwargs)

        return FunctionTool()

    return decorator


# =============================================================================
# Tool Implementations - All Stateless
# =============================================================================

def safe_path(workdir: Path, p: str) -> Path:
    """Ensure path stays within workspace."""
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool(
    name="bash",
    description="Run shell command in the workspace.",
    schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
)
def bash_tool(context: ToolContext, command: str) -> str:
    """Execute shell command. Context provides workdir."""
    if any(d in command for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"
    try:
        r = subprocess.run(
            command, shell=True, cwd=context.workdir,
            capture_output=True, text=True, timeout=60
        )
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="read_file",
    description="Read file contents.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "limit": {"type": "integer", "description": "Max lines to read"}
        },
        "required": ["path"],
    }
)
def read_file_tool(context: ToolContext, path: str, limit: int = None) -> str:
    """Read file. Context provides workdir for path resolution."""
    try:
        lines = safe_path(context.workdir, path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="write_file",
    description="Write content to file.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["path", "content"],
    }
)
def write_file_tool(context: ToolContext, path: str, content: str) -> str:
    """Write file. Context provides workdir."""
    try:
        fp = safe_path(context.workdir, path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="edit_file",
    description="Replace text in file.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
    }
)
def edit_file_tool(context: ToolContext, path: str, old_text: str, new_text: str) -> str:
    """Edit file. Context provides workdir."""
    try:
        fp = safe_path(context.workdir, path)
        text = fp.read_text()
        if old_text not in text:
            return f"Error: Text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TodoManager (from v4, unchanged)
# =============================================================================

class TodoManager:
    """Task list manager with constraints."""

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated = []
        in_progress = 0

        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active = str(item.get("activeForm", "")).strip()

            if not content or not active:
                raise ValueError(f"Item {i}: content and activeForm required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status")
            if status == "in_progress":
                in_progress += 1

            validated.append({
                "content": content,
                "status": status,
                "activeForm": active
            })

        if in_progress > 1:
            raise ValueError("Only one task can be in_progress")

        self.items = validated[:20]
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for t in self.items:
            mark = "[x]" if t["status"] == "completed" else \
                   "[>]" if t["status"] == "in_progress" else "[ ]"
            lines.append(f"{mark} {t['content']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)"


# Global TODO instance (could also be passed via context if needed)
TODO = TodoManager()


# TodoWrite as a Tool class (not using decorator because it needs TODO instance)
class TodoWriteTool(Tool):
    """Todo list management tool."""

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return "Update task list."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "activeForm": {"type": "string"},
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["items"],
        }

    def execute(self, context: ToolContext, items: list) -> str:
        try:
            return TODO.update(items)
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# SkillLoader (from v4, unchanged)
# =============================================================================

class SkillLoader:
    """Loads and manages skills from SKILL.md files."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        content = path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter, body = match.groups()
        metadata = {}
        for line in frontmatter.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")

        if "name" not in metadata or "description" not in metadata:
            return None

        return {
            "name": metadata["name"],
            "description": metadata["description"],
            "body": body.strip(),
            "path": path,
            "dir": path.parent,
        }

    def load_skills(self):
        if not self.skills_dir.exists():
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self.parse_skill_md(skill_md)
            if skill:
                self.skills[skill["name"]] = skill

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.skills.items()
        )

    def get_skill_content(self, name: str) -> str:
        if name not in self.skills:
            return None

        skill = self.skills[name]
        content = f"# Skill: {skill['name']}\n\n{skill['body']}"

        resources = []
        for folder, label in [
            ("scripts", "Scripts"),
            ("references", "References"),
            ("assets", "Assets")
        ]:
            folder_path = skill["dir"] / folder
            if folder_path.exists():
                files = list(folder_path.glob("*"))
                if files:
                    resources.append(f"{label}: {', '.join(f.name for f in files)}")

        if resources:
            content += f"\n\n**Available resources in {skill['dir']}:**\n"
            content += "\n".join(f"- {r}" for r in resources)

        return content

    def list_skills(self) -> list:
        return list(self.skills.keys())


SKILLS = SkillLoader(SKILLS_DIR)


# Skill tool as a class
class SkillTool(Tool):
    """Load domain-specific knowledge on demand."""

    def __init__(self, skill_loader: SkillLoader):
        self._loader = skill_loader

    @property
    def name(self) -> str:
        return "Skill"

    @property
    def description(self) -> str:
        return f"""Load a skill for specialized knowledge.

Available skills:
{self._loader.get_descriptions()}

Use immediately when task matches a skill description."""

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Name of the skill to load"}
            },
            "required": ["skill"],
        }

    def execute(self, context: ToolContext, skill: str) -> str:
        content = self._loader.get_skill_content(skill)
        if content is None:
            available = ", ".join(self._loader.list_skills()) or "none"
            return f"Error: Unknown skill '{skill}'. Available: {available}"

        return f"""<skill-loaded name="{skill}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


# =============================================================================
# Agent Type Registry
# =============================================================================

@dataclass
class AgentType:
    """Configuration for a subagent type."""
    name: str
    description: str
    prompt: str
    tools: list[str] | str  # List of tool names or "*" for all
    include_skill: bool = False


AGENT_TYPES: dict[str, AgentType] = {
    "explore": AgentType(
        name="explore",
        description="Read-only agent for exploring code, finding files, searching",
        prompt="You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
        tools=["bash", "read_file"],
    ),
    "code": AgentType(
        name="code",
        description="Full agent for implementing features and fixing bugs",
        prompt="You are a coding agent. Implement the requested changes efficiently.",
        tools="*",
    ),
    "plan": AgentType(
        name="plan",
        description="Planning agent for designing implementation strategies",
        prompt="You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
        tools=["bash", "read_file"],
    ),
    "general": AgentType(
        name="general",
        description="General-purpose agent with full capabilities. Use when task doesn't fit other types",
        prompt="You are a general-purpose agent with full capabilities. Handle the task comprehensively. Return a clear summary when done.",
        tools="*",
        include_skill=True,
    ),
}


def get_agent_descriptions() -> str:
    """Generate agent type descriptions for system prompt."""
    return "\n".join(
        f"- {name}: {cfg.description}"
        for name, cfg in AGENT_TYPES.items()
    )


# =============================================================================
# Build the Global Registry
# =============================================================================

# Create the main registry and register all tools
REGISTRY = ToolRegistry()
REGISTRY.register(bash_tool)
REGISTRY.register(read_file_tool)
REGISTRY.register(write_file_tool)
REGISTRY.register(edit_file_tool)
REGISTRY.register(TodoWriteTool())
REGISTRY.register(SkillTool(SKILLS))

# Define tool sets for different access levels
READONLY_TOOLS = ["bash", "read_file"]
WRITE_TOOLS = ["bash", "read_file", "write_file", "edit_file", "TodoWrite"]
ALL_BASE_TOOLS = WRITE_TOOLS
ALL_TOOLS_WITH_SKILL = ALL_BASE_TOOLS + ["Skill"]


# =============================================================================
# Task Tool (Special - spawns subagents)
# =============================================================================

class TaskTool(Tool):
    """
    Spawn a subagent for focused subtasks.

    This is a meta-tool that creates subagent loops.
    NOT included in subagent tool sets to prevent infinite spawning.
    """

    def __init__(self, registry: ToolRegistry, agent_types: dict[str, AgentType]):
        self._registry = registry
        self._agent_types = agent_types

    @property
    def name(self) -> str:
        return "Task"

    @property
    def description(self) -> str:
        desc = "Spawn a subagent for a focused subtask.\n\nAgent types:\n"
        for name, cfg in self._agent_types.items():
            desc += f"- {name}: {cfg.description}\n"
        return desc

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short task description (3-5 words)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed instructions for the subagent"
                },
                "agent_type": {
                    "type": "string",
                    "enum": list(self._agent_types.keys())
                },
            },
            "required": ["description", "prompt", "agent_type"],
        }

    def _get_tools_for_agent(self, agent_type: AgentType) -> list[dict]:
        """Get tool schemas for a subagent type."""
        if agent_type.tools == "*":
            tool_names = ALL_BASE_TOOLS.copy()
        else:
            tool_names = list(agent_type.tools)

        if agent_type.include_skill:
            tool_names.append("Skill")

        return self._registry.get_schemas(tool_names)

    @observe(name="SubAgent")
    def execute(self, context: ToolContext, description: str, prompt: str, agent_type: str) -> str:
        """Execute a subagent task."""
        _get_langfuse().update_current_span(
            metadata={"agent_type": agent_type, "description": description}
        )

        if agent_type not in self._agent_types:
            return f"Error: Unknown agent type '{agent_type}'"

        config = self._agent_types[agent_type]
        sub_system = f"""You are a {agent_type} subagent at {context.workdir}.

{config.prompt}

Complete the task and return a clear, concise summary."""

        sub_tools = self._get_tools_for_agent(config)
        sub_messages = [{"role": "user", "content": prompt}]

        print(f"  [{agent_type}] {description}")
        start = time.time()
        tool_count = 0

        while True:
            log_api_call(f"subagent:{agent_type}", sub_system, sub_messages, sub_tools)

            response = client.messages.create(
                model=MODEL,
                system=sub_system,
                messages=sub_messages,
                tools=sub_tools,
                max_tokens=8000,
            )

            log_api_response(f"subagent:{agent_type}", response)

            if response.stop_reason != "tool_use":
                break

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            results = []

            for tc in tool_calls:
                tool_count += 1
                # Execute through registry with context
                output = self._registry.execute(tc.name, context, **tc.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": output
                })

                elapsed = time.time() - start
                sys.stdout.write(
                    f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s"
                )
                sys.stdout.flush()

            sub_messages.append({"role": "assistant", "content": response.content})
            sub_messages.append({"role": "user", "content": results})

        elapsed = time.time() - start
        sys.stdout.write(
            f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n"
        )

        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return "(subagent returned no text)"


# Register Task tool (only for main agent, not subagents)
TASK_TOOL = TaskTool(REGISTRY, AGENT_TYPES)
REGISTRY.register(TASK_TOOL)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents available** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

Rules:
- Use Skill tool IMMEDIATELY when a task matches a skill description
- Use Task tool for subtasks needing focused exploration or implementation
- Use 'general' agent when task doesn't clearly fit explore/code/plan types
- Use TodoWrite to track multi-step work
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Main Agent Loop
# =============================================================================

# Main agent tools (includes Task for spawning subagents)
MAIN_AGENT_TOOLS = ALL_TOOLS_WITH_SKILL + ["Task"]

# Global context for the session
SESSION_CONTEXT: ToolContext = None
SESSION_ID: str = None


@observe(name="MainAgentLoop")
def agent_loop(messages: list) -> list:
    """Main agent loop with Registry-based tool dispatch."""
    global SESSION_CONTEXT

    if SESSION_ID:
        _get_langfuse().update_current_trace(session_id=SESSION_ID)

    # Get schemas for main agent (all tools including Task)
    tools_schema = REGISTRY.get_schemas(MAIN_AGENT_TOOLS)

    while True:
        log_api_call("main_agent", SYSTEM, messages, tools_schema)

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=tools_schema,
            max_tokens=8000,
        )

        log_api_response("main_agent", response)

        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            _get_langfuse().score_current_trace(
                name="completion", value=1, comment="Agent completed successfully"
            )
            return messages

        results = []
        for tc in tool_calls:
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            # Execute through registry with context
            output = REGISTRY.execute(tc.name, SESSION_CONTEXT, **tc.input)

            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name != "Task":
                print(f"  {output}")

            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================

def main():
    global SESSION_ID, SESSION_CONTEXT

    SESSION_ID = str(uuid.uuid4())
    SESSION_CONTEXT = ToolContext(workdir=WORKDIR)

    print(f"Mini Claude Code v4-Registry - {WORKDIR}")
    print(f"Session: {SESSION_ID[:8]}...")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print(f"Registered tools: {', '.join(REGISTRY.list_names())}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            _get_langfuse().score_current_trace(
                name="completion", value=0, comment=str(e)
            )
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
