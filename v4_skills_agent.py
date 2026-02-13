#!/usr/bin/env python3
"""
v4_skills_agent.py - Mini Claude Code: Skills Mechanism (~650 lines)

Core Philosophy: "Knowledge Externalization"
============================================
v3 gave us subagents for task decomposition. But there's a deeper question:

    How does the model know HOW to handle domain-specific tasks?

- Processing PDFs? It needs to know pdftotext vs PyMuPDF
- Building MCP servers? It needs protocol specs and best practices
- Code review? It needs a systematic checklist

This knowledge isn't a tool - it's EXPERTISE. Skills solve this by letting
the model load domain knowledge on-demand.

v4 Enhancements (Claude Code Style)
===================================
1. Max Turns Limit - Prevents infinite loops in agent/subagent loops
2. Output Truncation - Handles large outputs gracefully with "[N lines truncated]"
3. Enhanced Skill Loading - Supports allowed-tools, user-invocable, disable-model-invocation

The Paradigm Shift: Knowledge Externalization
--------------------------------------------
Traditional AI: Knowledge locked in model parameters
  - To teach new skills: collect data -> train -> deploy
  - Cost: $10K-$1M+, Timeline: Weeks
  - Requires ML expertise, GPU clusters

Skills: Knowledge stored in editable files
  - To teach new skills: write a SKILL.md file
  - Cost: Free, Timeline: Minutes
  - Anyone can do it

It's like attaching a hot-swappable LoRA adapter without any training!

Tools vs Skills:
---------------
    | Concept   | What it is              | Example                    |
    |-----------|-------------------------|---------------------------|
    | **Tool**  | What model CAN do       | bash, read_file, write    |
    | **Skill** | How model KNOWS to do   | PDF processing, MCP dev   |

Tools are capabilities. Skills are knowledge.

Progressive Disclosure:
----------------------
    Layer 1: Metadata (always loaded)      ~100 tokens/skill
             name + description only

    Layer 2: SKILL.md body (on trigger)    ~2000 tokens
             Detailed instructions

    Layer 3: Resources (as needed)         Unlimited
             scripts/, references/, assets/

This keeps context lean while allowing arbitrary depth.

SKILL.md Standard (Extended):
----------------------------
    ---
    name: pdf
    description: Process PDF files
    allowed-tools: bash, read_file      # Restrict tools for this skill
    user-invocable: true                # Can user invoke directly
    disable-model-invocation: false     # Can model auto-trigger
    when_to_use: "Use when user asks to process PDF files"
    ---

    # PDF Processing Skill
    ...

Cache-Preserving Injection:
--------------------------
Critical insight: Skill content goes into tool_result (user message),
NOT system prompt. This preserves prompt cache!

    Wrong: Edit system prompt each time (cache invalidated, 20-50x cost)
    Right: Append skill as tool result (prefix unchanged, cache hit)

This is how production Claude Code works - and why it's cost-efficient.

Usage:
    python v4_skills_agent.py
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(override=True)

# =============================================================================
# Constants - Limits and Truncation (Claude Code style)
# =============================================================================

# Max turns to prevent infinite loops
MAX_TURNS_MAIN = 100      # Main agent loop
MAX_TURNS_SUBAGENT = 30   # Subagent loop

# Tool output truncation limits
MAX_OUTPUT_CHARS = 400000     # ~400KB hard limit for bash output
MAX_OUTPUT_LINES = 2000       # Max lines for file reads
MAX_LINE_CHARS = 2000         # Max chars per line
TRUNCATION_PREVIEW_CHARS = 2000  # Preview size for large outputs


def truncate_output(output: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """
    Truncate tool output and notify model about truncation.

    Key insight from Claude Code: Always tell the model how much was truncated,
    so it knows the information is incomplete and can request specific parts.
    """
    if len(output) <= max_chars:
        return output

    truncated = output[:max_chars]
    remaining_lines = output[max_chars:].count('\n')

    return f"{truncated}\n\n... [{remaining_lines} lines truncated] ..."


def truncate_lines(output: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Truncate output by line count."""
    lines = output.split('\n')
    if len(lines) <= max_lines:
        return output

    truncated_count = len(lines) - max_lines
    return '\n'.join(lines[:max_lines]) + f"\n\n... [{truncated_count} lines truncated] ..."


def truncate_long_lines(output: str, max_chars: int = MAX_LINE_CHARS) -> str:
    """Truncate individual long lines."""
    lines = output.split('\n')
    result = []
    for line in lines:
        if len(line) > max_chars:
            result.append(line[:max_chars] + "... [line truncated]")
        else:
            result.append(line)
    return '\n'.join(result)


class OutputBuffer:
    """
    Output buffer with automatic truncation (ported from Claude Code).

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


# =============================================================================
# LangFuse Integration (optional, graceful degradation)
# =============================================================================
# Set env vars to enable: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
# When not configured, @observe becomes a no-op passthrough.

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
        """No-op decorator when langfuse is not available."""
        def decorator(fn):
            return fn
        return decorator

    class _FakeLangfuse:
        def update_current_span(self, **kwargs):
            pass
        def update_current_trace(self, **kwargs):
            pass
        def score_current_trace(self, **kwargs):
            pass

    def _get_langfuse():
        return _FakeLangfuse()

# =============================================================================
# Logging Configuration
# =============================================================================

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() == "true"


def log_api_call(caller: str, system: str, messages: list, tools: list):
    """Log raw API call details for debugging."""
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
    """Log raw API response for debugging."""
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
# MCP Browser Client - connects to @playwright/mcp via stdio
# =============================================================================

class MCPBrowserClient:
    """Manages a Playwright MCP server subprocess and provides sync wrappers."""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._session = None
        self._read = None
        self._write = None
        self._cm_stdio = None
        self._cm_session = None
        self._tools_cache = None
        self._connected = False
        self._lock = threading.Lock()

    def _run_loop(self, ready_event: threading.Event):
        """Background thread: start event loop and connect to MCP server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect(ready_event))

    async def _connect(self, ready_event: threading.Event):
        """Async: launch MCP server and initialize session."""
        server_params = StdioServerParameters(
            command="npx",
            args=["@playwright/mcp@latest", "--browser", "chrome"],
        )
        self._cm_stdio = stdio_client(server_params)
        self._read, self._write = await self._cm_stdio.__aenter__()
        self._cm_session = ClientSession(self._read, self._write)
        self._session = await self._cm_session.__aenter__()
        await self._session.initialize()

        # Cache tools
        result = await self._session.list_tools()
        self._tools_cache = result.tools
        self._connected = True
        ready_event.set()

        # Keep loop alive until shutdown
        self._stop_future = self._loop.create_future()
        await self._stop_future

    def start(self):
        """Start MCP server in background thread. Blocks until connected."""
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait(timeout=30)
        if not self._connected:
            raise RuntimeError("Failed to connect to Playwright MCP server")

    def stop(self):
        """Shutdown MCP server and background thread."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_future.set_result, None)
        if self._thread:
            self._thread.join(timeout=5)

    def get_tools(self) -> list:
        """Return cached MCP tools as Anthropic tool format."""
        if not self._tools_cache:
            return []
        tools = []
        for t in self._tools_cache:
            tools.append({
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            })
        return tools

    def get_tool_names(self) -> set:
        """Return set of MCP tool names."""
        if not self._tools_cache:
            return set()
        return {t.name for t in self._tools_cache}

    def call_tool(self, name: str, args: dict) -> str:
        """Sync wrapper: call an MCP tool and return result as string."""
        if not self._connected:
            return "Error: MCP browser not connected"
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, args), self._loop
        )
        try:
            result = future.result(timeout=60)
            # MCP returns content as list of TextContent/ImageContent
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif hasattr(item, "data"):
                    parts.append(f"[image: {item.mimeType}, {len(item.data)} bytes]")
            return "\n".join(parts) if parts else "(no output)"
        except Exception as e:
            return f"Error: {e}"


# Global MCP browser client (initialized lazily in main)
MCP_BROWSER: MCPBrowserClient | None = None



# =============================================================================
# SkillLoader - The core addition in v4
# =============================================================================

class SkillLoader:
    """
    Loads and manages skills from SKILL.md files.

    A skill is a FOLDER containing:
    - SKILL.md (required): YAML frontmatter + markdown instructions
    - scripts/ (optional): Helper scripts the model can run
    - references/ (optional): Additional documentation
    - assets/ (optional): Templates, files for output

    SKILL.md Format (Extended with Claude Code fields):
    ---------------------------------------------------
        ---
        name: pdf
        description: Process PDF files. Use when reading, creating, or merging PDFs.
        allowed-tools: bash, read_file       # Optional: restrict available tools
        user-invocable: true                  # Optional: can user invoke directly (default: true)
        disable-model-invocation: false       # Optional: prevent auto-trigger (default: false)
        when_to_use: "Use when user asks to process PDF files"  # Optional: trigger hints
        ---

        # PDF Processing Skill

        ## Reading PDFs

        Use pdftotext for quick extraction:
        ```bash
        pdftotext input.pdf -
        ```
        ...

    The YAML frontmatter provides metadata and control flags.
    The markdown body provides detailed instructions.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def _parse_bool(self, value: str, default: bool = False) -> bool:
        """Parse boolean value from YAML string."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', 'yes', '1')

    def _parse_tools_list(self, value: str) -> list:
        """Parse allowed-tools field into list of tool names."""
        if not value:
            return None  # None means all tools allowed
        if isinstance(value, list):
            return value
        # Parse comma-separated or space-separated list
        tools = re.split(r'[,\s]+', str(value).strip())
        return [t.strip() for t in tools if t.strip()]

    def parse_skill_md(self, path: Path) -> dict:
        """
        Parse a SKILL.md file into metadata and body.

        Returns dict with: name, description, body, path, dir, and control fields
        Returns None if file doesn't match format.
        """
        content = path.read_text()

        # Match YAML frontmatter between --- markers
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter, body = match.groups()

        # Parse YAML-like frontmatter (simple key: value)
        metadata = {}
        for line in frontmatter.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")

        # Require name and description
        if "name" not in metadata or "description" not in metadata:
            return None

        return {
            "name": metadata["name"],
            "description": metadata["description"],
            "body": body.strip(),
            "path": path,
            "dir": path.parent,
            # New Claude Code style fields
            "allowed_tools": self._parse_tools_list(metadata.get("allowed-tools")),
            "user_invocable": self._parse_bool(metadata.get("user-invocable"), default=True),
            "disable_model_invocation": self._parse_bool(metadata.get("disable-model-invocation"), default=False),
            "when_to_use": metadata.get("when_to_use", ""),
        }

    def load_skills(self):
        """
        Scan skills directory and load all valid SKILL.md files.

        Only loads metadata at startup - body is loaded on-demand.
        This keeps the initial context lean.
        """
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
        """
        Generate skill descriptions for system prompt.

        This is Layer 1 - only name and description, ~100 tokens per skill.
        Full content (Layer 2) is loaded only when Skill tool is called.

        Includes when_to_use hints for better auto-triggering.
        Excludes skills with disable_model_invocation=True.
        """
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            # Skip skills that shouldn't be auto-triggered by model
            if skill.get("disable_model_invocation", False):
                continue

            desc = skill['description']
            when = skill.get('when_to_use', '')
            if when:
                lines.append(f"- {name}: {desc} (Trigger: {when})")
            else:
                lines.append(f"- {name}: {desc}")

        return "\n".join(lines) if lines else "(no skills available)"

    def get_user_invocable_skills(self) -> list:
        """Return list of skills that can be invoked by user directly."""
        return [
            name for name, skill in self.skills.items()
            if skill.get("user_invocable", True)
        ]

    def get_allowed_tools(self, skill_name: str) -> list:
        """Get allowed tools for a skill. Returns None if all tools allowed."""
        if skill_name not in self.skills:
            return None
        return self.skills[skill_name].get("allowed_tools")

    def get_skill_content(self, name: str) -> str:
        """
        Get full skill content for injection.

        This is Layer 2 - the complete SKILL.md body, plus any available
        resources (Layer 3 hints).

        Returns None if skill not found.
        """
        if name not in self.skills:
            return None

        skill = self.skills[name]
        content = f"# Skill: {skill['name']}\n\n{skill['body']}"

        # List available resources (Layer 3 hints)
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
        """Return list of available skill names."""
        return list(self.skills.keys())


# Global skill loader instance
SKILLS = SkillLoader(SKILLS_DIR)

# Session ID for grouping traces in Langfuse
SESSION_ID = None


# =============================================================================
# Agent Type Registry (from v3)
# =============================================================================

AGENT_TYPES = {
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
    "general": {
        "description": "General-purpose agent with full capabilities. Use when task doesn't fit other agent types",
        "tools": "*",  # All base tools + Skill tool
        "prompt": "You are a general-purpose agent with full capabilities. Handle the task comprehensively using all available tools. Return a clear summary when done.",
        "include_skill": True,  # Flag to include Skill tool
    },
}


def get_agent_descriptions() -> str:
    """Generate agent type descriptions for system prompt."""
    return "\n".join(
        f"- {name}: {cfg['description']}"
        for name, cfg in AGENT_TYPES.items()
    )


# =============================================================================
# TodoManager (from v2)
# =============================================================================

class TodoManager:
    """Task list manager with constraints. See v2 for details."""

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


TODO = TodoManager()


# =============================================================================
# System Prompt - Updated for v4
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

# How you work
Loop: plan -> act with tools -> report.
- Prefer tools over prose. Act, don't just explain.
- Use TodoWrite to track multi-step work.
- After finishing, summarize what changed.

# Tool output handling
- Output may be truncated with "[N lines truncated]" messages
- If you receive truncation warnings, use read_file with offset/limit to read specific portions
- Never assume truncated content - always verify by reading the actual data

# Tools available

**Skills** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

- browser: Control web browser (powered by Playwright MCP, tools loaded dynamically)

# Using your tools
- Use Skill tool IMMEDIATELY when a task matches a skill description.
- Use Task tool for subtasks needing focused exploration or implementation.
- Use 'general' agent when task doesn't clearly fit explore/code/plan types."""


# =============================================================================
# Tool Definitions
# =============================================================================

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents. Use offset/limit for large files when truncation occurs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "Line number to start reading from (0-based)"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "TodoWrite",
        "description": "Update task list.",
        "input_schema": {
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
        },
    },
]

# Task tool (from v3)
TASK_TOOL = {
    "name": "Task",
    "description": f"Spawn a subagent for a focused subtask.\n\nAgent types:\n{get_agent_descriptions()}",
    "input_schema": {
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
                "enum": list(AGENT_TYPES.keys())
            },
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

# NEW in v4: Skill tool
SKILL_TOOL = {
    "name": "Skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.

Available skills:
{SKILLS.get_descriptions()}

When to use:
- IMMEDIATELY when user task matches a skill description
- Before attempting domain-specific work (PDF, MCP, etc.)

The skill content will be injected into the conversation, giving you
detailed instructions and access to resources.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Name of the skill to load"
            }
        },
        "required": ["skill"],
    },
}

ALL_TOOLS = BASE_TOOLS + [TASK_TOOL, SKILL_TOOL]
# Browser tools from MCP are appended dynamically in main() after MCP_BROWSER.start()


def get_all_tools() -> list:
    """Return all tools including dynamically loaded MCP browser tools."""
    tools = ALL_TOOLS.copy()
    if MCP_BROWSER and MCP_BROWSER._connected:
        tools.extend(MCP_BROWSER.get_tools())
    return tools


def get_tools_for_agent(agent_type: str) -> list:
    """Filter tools based on agent type.

    Note: Subagents never get Task tool to prevent infinite spawning.
    The 'general' agent gets Skill tool for full capability without spawning.
    """
    config = AGENT_TYPES.get(agent_type, {})
    allowed = config.get("tools", "*")
    include_skill = config.get("include_skill", False)

    if allowed == "*":
        tools = BASE_TOOLS.copy()
    else:
        tools = [t for t in BASE_TOOLS if t["name"] in allowed]

    # Add Skill tool for agents that need it (e.g., general)
    if include_skill:
        tools = tools + [SKILL_TOOL]

    return tools


# =============================================================================
# Tool Implementations
# =============================================================================

def safe_path(p: str) -> Path:
    """Ensure path stays within workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    """Execute shell command with output truncation."""
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        output = (r.stdout + r.stderr).strip() or "(no output)"

        # Apply truncation (Claude Code style)
        output = truncate_long_lines(output)
        output = truncate_output(output)

        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, offset: int = None, limit: int = None) -> str:
    """Read file contents with offset/limit and line/char truncation."""
    try:
        content = safe_path(path).read_text()
        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset
        start_line = offset if offset else 0
        if start_line > 0:
            lines = lines[start_line:]

        # Apply line limit
        actual_limit = limit if limit else MAX_OUTPUT_LINES
        if len(lines) > actual_limit:
            truncated_count = len(lines) - actual_limit
            lines = lines[:actual_limit]
            result = "\n".join(lines)
            result += f"\n\n... [{truncated_count} lines truncated, total file: {total_lines} lines] ..."
            if offset:
                result += f"\n(showing lines {start_line + 1}-{start_line + actual_limit})"
        else:
            result = "\n".join(lines)
            if offset:
                result += f"\n\n(showing lines {start_line + 1}-{start_line + len(lines)} of {total_lines})"

        # Truncate long lines
        result = truncate_long_lines(result)

        return result
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """Write content to file."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = safe_path(path)
        text = fp.read_text()
        if old_text not in text:
            return f"Error: Text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(items: list) -> str:
    """Update the todo list."""
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    """
    Load a skill and inject it into the conversation.

    This is the key mechanism:
    1. Get skill content (SKILL.md body + resource hints)
    2. Return it wrapped in <skill-loaded> tags
    3. Model receives this as tool_result (user message)
    4. Model now "knows" how to do the task

    Why tool_result instead of system prompt?
    - System prompt changes invalidate cache (20-50x cost increase)
    - Tool results append to end (prefix unchanged, cache hit)

    This is how production systems stay cost-efficient.
    """
    content = SKILLS.get_skill_content(skill_name)

    if content is None:
        available = ", ".join(SKILLS.list_skills()) or "none"
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"

    # Wrap in tags so model knows it's skill content
    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


@observe(name="SubAgent")
def run_task(description: str, prompt: str, agent_type: str, max_turns: int = None) -> str:
    """
    Execute a subagent task with max_turns limit (from v3, enhanced).

    Key improvements from Claude Code:
    - max_turns parameter to prevent infinite loops
    - Output truncation for tool results
    - Clear error reporting when limit reached
    """
    _get_langfuse().update_current_span(
        metadata={"agent_type": agent_type, "description": description}
    )

    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]
    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    # Use provided max_turns or default
    turns_limit = max_turns if max_turns else MAX_TURNS_SUBAGENT
    current_turn = 0

    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    while current_turn < turns_limit:
        current_turn += 1
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
            output = execute_tool(tc.name, tc.input)

            # Truncate tool output before adding to results
            output = truncate_output(output)

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

    # Check if we hit max_turns limit
    if current_turn >= turns_limit:
        sys.stdout.write(
            f"\r  [{agent_type}] {description} - STOPPED (max turns {turns_limit} reached, {tool_count} tools, {elapsed:.1f}s)\n"
        )
        return f"Error: Subagent reached max turns limit ({turns_limit}). Task may be incomplete."

    sys.stdout.write(
        f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n"
    )

    for block in response.content:
        if hasattr(block, "text"):
            return block.text

    return "(subagent returned no text)"


@observe(name="ToolExecution")
def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
    _get_langfuse().update_current_span(
        metadata={"tool": name, "args": args}
    )
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("offset"), args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"])
    if name == "Skill":
        return run_skill(args["skill"])
    # MCP browser tools (browser_click, browser_snapshot, etc.)
    if MCP_BROWSER and name in MCP_BROWSER.get_tool_names():
        output = MCP_BROWSER.call_tool(name, args)
        # Truncate MCP tool output too
        return truncate_output(output)
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================

@observe(name="MainAgentLoop")
def agent_loop(messages: list, max_turns: int = None) -> list:
    """
    Main agent loop with skills support and max_turns limit.

    Key improvements from Claude Code:
    - max_turns parameter to prevent infinite loops
    - Output truncation for all tool results
    - Clear error reporting when limit reached
    """
    # Set session_id to group all traces from same conversation
    if SESSION_ID:
        _get_langfuse().update_current_trace(session_id=SESSION_ID)

    # Use provided max_turns or default
    turns_limit = max_turns if max_turns else MAX_TURNS_MAIN
    current_turn = 0

    while current_turn < turns_limit:
        current_turn += 1
        all_tools = get_all_tools()
        log_api_call("main_agent", SYSTEM, messages, all_tools)

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=all_tools,
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
            # Special display for different tool types
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            elif tc.name.startswith("browser_"):
                print(f"\n> Browser: {tc.name}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Apply truncation to tool output before storing
            truncated_output = truncate_output(output)

            # Skill tool shows summary, not full content
            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name.startswith("browser_"):
                # Browser output can be large (snapshots), show truncated
                lines = output.strip().split("\n")
                if len(lines) > 10:
                    print(f"  " + "\n  ".join(lines[:8]))
                    print(f"  ... ({len(lines)} lines total)")
                else:
                    print(f"  {output}")
            elif tc.name != "Task":
                # Show truncated preview in console
                display_output = output[:500] + "..." if len(output) > 500 else output
                print(f"  {display_output}")

            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": truncated_output
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    # Reached max_turns limit
    print(f"\n⚠️  Agent reached max turns limit ({turns_limit})")
    _get_langfuse().score_current_trace(
        name="completion", value=0, comment=f"Reached max turns limit ({turns_limit})"
    )
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": f"I've reached the maximum number of turns ({turns_limit}). The task may be incomplete."}]
    })
    return messages


# =============================================================================
# Main REPL
# =============================================================================

def main():
    global SESSION_ID, MCP_BROWSER
    SESSION_ID = str(uuid.uuid4())

    print(f"Mini Claude Code v4 (with Skills + Playwright MCP) - {WORKDIR}")
    print(f"Session: {SESSION_ID[:8]}...")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print(f"Limits: max_turns={MAX_TURNS_MAIN} (main), {MAX_TURNS_SUBAGENT} (subagent)")
    print(f"Truncation: {MAX_OUTPUT_CHARS} chars, {MAX_OUTPUT_LINES} lines")

    # Start Playwright MCP browser
    print("Starting Playwright MCP browser...")
    try:
        MCP_BROWSER = MCPBrowserClient()
        MCP_BROWSER.start()
        tool_names = sorted(MCP_BROWSER.get_tool_names())
        print(f"Browser tools loaded: {', '.join(tool_names)}")
    except Exception as e:
        print(f"Warning: Playwright MCP failed to start: {e}")
        print("Browser tools will not be available.")
        MCP_BROWSER = None

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

    # Cleanup MCP browser
    if MCP_BROWSER:
        print("Stopping Playwright MCP browser...")
        MCP_BROWSER.stop()


if __name__ == "__main__":
    main()
