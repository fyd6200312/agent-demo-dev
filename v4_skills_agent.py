#!/usr/bin/env python3
"""
v4_skills_agent.py - Mini Claude Code: Skills Mechanism (~550 lines)

Core Philosophy: "Knowledge Externalization"
============================================
v3 gave us subagents for task decomposition. But there's a deeper question:

    How does the model know HOW to handle domain-specific tasks?

- Processing PDFs? It needs to know pdftotext vs PyMuPDF
- Building MCP servers? It needs protocol specs and best practices
- Code review? It needs a systematic checklist

This knowledge isn't a tool - it's EXPERTISE. Skills solve this by letting
the model load domain knowledge on-demand.

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

SKILL.md Standard:
-----------------
    skills/
    |-- pdf/
    |   |-- SKILL.md          # Required: YAML frontmatter + Markdown body
    |-- mcp-builder/
    |   |-- SKILL.md
    |   |-- references/       # Optional: docs, specs
    |-- code-review/
        |-- SKILL.md
        |-- scripts/          # Optional: helper scripts

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

import json
import os
import re
import subprocess
import sys
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

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

# Browser control server (moltbot)
BROWSER_SERVER_URL = os.getenv("BROWSER_SERVER_URL", "http://127.0.0.1:18791")



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

    SKILL.md Format:
    ----------------
        ---
        name: pdf
        description: Process PDF files. Use when reading, creating, or merging PDFs.
        ---

        # PDF Processing Skill

        ## Reading PDFs

        Use pdftotext for quick extraction:
        ```bash
        pdftotext input.pdf -
        ```
        ...

    The YAML frontmatter provides metadata (name, description).
    The markdown body provides detailed instructions.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        """
        Parse a SKILL.md file into metadata and body.

        Returns dict with: name, description, body, path, dir
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
        """
        if not self.skills:
            return "(no skills available)"

        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.skills.items()
        )

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

# Tools available

**Skills** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

**Browser** (invoke with browser tool for web automation):
- browser: Control web browser (navigate, snapshot, click, type, screenshot, etc.)

# Using your tools
- Use Skill tool IMMEDIATELY when a task matches a skill description.
- Use Task tool for subtasks needing focused exploration or implementation.
- Use browser tool when user needs web browsing, data scraping, or UI automation.
- Use 'general' agent when task doesn't clearly fit explore/code/plan types.

# When to STOP and ask the user
Do NOT try to power through these situations. Stop and ask immediately:
- A page redirects to login/SSO or shows a login form — ask for credentials.
- A previous login attempt failed — ask the user to confirm credentials.
- You've failed the same approach twice — try a different method or ask the user.
- You need information only the user has (account details, preferences, choices).
- An action is destructive or irreversible — confirm before proceeding.

# Doing tasks
- Read before you write. Understand existing code/content before modifying.
- When blocked, do not brute-force. Consider alternative approaches or ask the user.
- Do not silently give up. If you cannot complete a task, tell the user what failed and why.
- Trust the user: if user says "already done X", verify current state instead of arguing.
- When summarizing web content, list all visited URLs under a 'Sources:' section."""


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
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
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

# Browser tool - controls moltbot's browser HTTP service
BROWSER_TOOL = {
    "name": "browser",
    "description": (
        "Control the browser via browser control server."
        "\n\n"
        "PREFER action='batch' to combine multiple steps into ONE tool call. "
        "This saves round-trips and tokens. The server executes steps sequentially "
        "and auto-stops on login detection or errors."
        "\n\n"
        "Workflow (use batch for steps 2-4):\n"
        "1. start (single call)\n"
        "2. batch: [navigate, snapshot] -> read refs from result\n"
        "3. batch: [act, act, ..., snapshot] -> verify result\n"
        "4. Repeat step 3 as needed. stop when done.\n"
        "\n"
        "Batch examples:\n"
        "- Search: [{navigate, url}, {snapshot, maxElements:30}, {act, kind:'type', ref:'e10', text:'query', submit:true}, {snapshot, maxElements:30}]\n"
        "- Read article: [{navigate, url}, {act, kind:'text'}] — no snapshot needed\n"
        "- Login: [{act, kind:'fill', fields:[{ref:'e1',value:'user'},{ref:'e2',value:'pass'}]}, {act, kind:'click', ref:'e3'}, {snapshot}]\n"
        "- New tab: [{act, kind:'click', ref:'e5'}, {focus, targetId:'NEW_TAB_ID'}] — check click result for newTabs first\n"
        "\n"
        "Login detection (server-side, automatic):\n"
        "- The server checks every navigate/click result for login keywords (login, sso, 登录, etc.)\n"
        "- If detected: batch stops early, result includes stopReason='login_required'.\n"
        "- When you see login_required: STOP and ask the user for credentials. Do NOT bypass.\n"
        "\n"
        "When NOT to use batch (use single action instead):\n"
        "- start / stop\n"
        "- When the next step depends on a previous result you haven't seen yet "
        "(e.g. you need refs from a snapshot before you can click)\n"
        "\n"
        "Snapshot rules (minimize calls):\n"
        "- DO snapshot: after navigate (to get refs), to verify critical results.\n"
        "- DO NOT snapshot: to check if click worked (read navigated/newTabs instead), "
        "between consecutive fill/type, when using act:text for content reading.\n"
        "- Use maxElements 20-30 to limit refs and reduce tokens.\n"
        "\n"
        "Element targeting:\n"
        "- Use ref (e.g. e12) from the most recent snapshot.\n"
        "- Use selector (CSS) or text (visible text) when ref is unavailable.\n"
        "\n"
        "Act kinds: click, type, fill, press, hover, select, scroll, scrollIntoView, evaluate, text, wait, close.\n"
        "- click: {ref/selector/text} — response has navigated, url, title, newTabs.\n"
        "- type: {ref/selector/text, text, submit?} — submit=true adds Enter.\n"
        "- fill: {fields: [{ref, value}]} — fill multiple fields at once.\n"
        "- text: {selector?, maxLength?} — extract page text. Preferred for reading content.\n"
        "- scroll: {direction, amount?} — amount in pixels, default 500.\n"
        "\n"
        "Error handling:\n"
        "- invalid_target: Bad ref. Run snapshot for fresh refs.\n"
        "- timeout: Element hidden. Try scrolling or dismissing popups.\n"
        "- page_closed: Tab closed. Check 'tabs' for new tabs.\n"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "navigate", "snapshot", "screenshot",
                         "act", "batch", "tabs", "open", "close", "focus", "console",
                         "cookies", "cookies_set", "cookies_clear",
                         "dialog", "requests", "response_body", "errors",
                         "download", "wait_download",
                         "storage_local", "storage_session"],
                "description": "Browser action to perform. Use 'batch' to execute multiple steps in one call."
            },
            "targetUrl": {
                "type": "string",
                "description": "URL for navigate/open actions"
            },
            "targetId": {
                "type": "string",
                "description": "Tab target ID (from snapshot/tabs response)"
            },
            "snapshotFormat": {
                "type": "string",
                "enum": ["ai", "aria"],
                "description": "Snapshot format: 'ai' (default, structured with refs) or 'aria' (raw tree)"
            },
            "maxElements": {
                "type": "integer",
                "description": "For snapshot: limit interactive elements returned (0=unlimited, default 0). Use 20-30 to reduce tokens."
            },
            "request": {
                "type": "object",
                "description": "For act action. Required fields depend on kind:\n"
                    "- click: ref or selector or text\n"
                    "- type: (ref or selector or text) + text\n"
                    "- press: key, optional ref\n"
                    "- fill: fields [{ref, value}]\n"
                    "- scroll: direction, optional amount\n"
                    "- evaluate: expression\n"
                    "- text: optional selector, optional maxLength\n"
                    "- wait: optional timeMs, waitFor, text/selector/url/state",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["click", "type", "press", "hover",
                                 "select", "fill", "scroll", "scrollIntoView",
                                 "evaluate", "text", "wait", "close"]
                    },
                    "ref": {"type": "string"},
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                    "submit": {"type": "boolean"},
                    "doubleClick": {"type": "boolean"},
                    "expression": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "amount": {"type": "number"},
                    "waitFor": {"type": "string", "enum": ["text", "textGone", "selector", "url", "loadState"]},
                    "url": {"type": "string"},
                    "state": {"type": "string"},
                    "timeMs": {"type": "number"},
                    "timeoutMs": {"type": "number"},
                },
            },
            "steps": {
                "type": "array",
                "description": "For batch action. Array of steps to execute sequentially. "
                    "Each step is an object with 'action' (navigate/snapshot/act/focus/close) "
                    "and action-specific fields. Server auto-detects login pages and stops early. "
                    "Example: [{\"action\":\"navigate\",\"url\":\"...\"}, {\"action\":\"snapshot\",\"maxElements\":30}, "
                    "{\"action\":\"act\",\"kind\":\"type\",\"ref\":\"e10\",\"text\":\"query\",\"submit\":true}]",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["navigate", "snapshot", "act", "focus", "close"]},
                        "url": {"type": "string"},
                        "targetId": {"type": "string"},
                        "maxElements": {"type": "integer"},
                        "format": {"type": "string"},
                        "kind": {"type": "string"},
                        "ref": {"type": "string"},
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "key": {"type": "string"},
                        "submit": {"type": "boolean"},
                        "fields": {"type": "array"},
                        "direction": {"type": "string"},
                        "amount": {"type": "number"},
                        "expression": {"type": "string"},
                        "maxLength": {"type": "integer"},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["action"],
    },
}

ALL_TOOLS = BASE_TOOLS + [TASK_TOOL, SKILL_TOOL, BROWSER_TOOL]


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
    """Execute shell command."""
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """Read file contents."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        return "\n".join(lines)
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


def _browser_request(method: str, path: str, params: dict = None, body: dict = None,
                     timeout: float = 15.0) -> str:
    """Send HTTP request to browser control server. Uses only stdlib."""
    url = BROWSER_SERVER_URL + path
    if params:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        if qs:
            url += "?" + qs
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return raw
    except urllib.error.HTTPError as e:
        # Read the response body for detailed error info (errorType, hint, etc.)
        try:
            error_body = e.read().decode("utf-8")
            error_data = json.loads(error_body)
            return json.dumps(error_data, ensure_ascii=False, indent=2)
        except Exception:
            return f"Error: HTTP {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "urlopen error" in str(e):
            return (
                "Error: 浏览器服务未启动。请先启动 moltbot 浏览器服务 "
                f"(确保 {BROWSER_SERVER_URL} 可访问)。"
            )
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def run_browser(args: dict) -> str:
    """Execute browser tool action via HTTP API."""
    action = args.get("action", "")
    target_url = args.get("targetUrl", "")
    target_id = args.get("targetId", "")
    snapshot_format = args.get("snapshotFormat", "ai")
    req = args.get("request") or {}

    qp = {}

    if action == "start":
        return _browser_request("POST", "/start", params=qp)

    elif action == "stop":
        return _browser_request("POST", "/stop", params=qp)

    elif action == "batch":
        steps = args.get("steps", [])
        if not steps:
            return "Error: steps array is required for batch"
        return _browser_request("POST", "/batch", body={"steps": steps}, timeout=120.0)

    elif action == "navigate":
        if not target_url:
            return "Error: targetUrl is required for navigate"
        body = {"url": target_url}
        if target_id:
            body["targetId"] = target_id
        return _browser_request("POST", "/navigate", params=qp, body=body)

    elif action == "snapshot":
        qp["format"] = snapshot_format
        if target_id:
            qp["targetId"] = target_id
        max_elements = args.get("maxElements")
        if max_elements:
            qp["maxElements"] = str(max_elements)
        return _browser_request("GET", "/snapshot", params=qp)

    elif action == "screenshot":
        body = {}
        if target_id:
            body["targetId"] = target_id
        return _browser_request("POST", "/screenshot", params=qp, body=body)

    elif action == "act":
        if not req or not isinstance(req, dict):
            return "Error: request object is required for act (e.g. {kind: 'click', ref: 'e12'})"
        if target_id:
            req["targetId"] = target_id
        return _browser_request("POST", "/act", params=qp, body=req, timeout=35.0)

    elif action == "tabs":
        return _browser_request("GET", "/tabs", params=qp)

    elif action == "open":
        if not target_url:
            return "Error: targetUrl is required for open"
        return _browser_request("POST", "/tabs/open", params=qp, body={"url": target_url})

    elif action == "focus":
        if not target_id:
            return "Error: targetId is required for focus"
        return _browser_request("POST", "/tabs/focus", params=qp, body={"targetId": target_id})

    elif action == "close":
        if target_id:
            return _browser_request("DELETE", f"/tabs/{urllib.parse.quote(target_id)}", params=qp)
        return _browser_request("POST", "/act", params=qp, body={"kind": "close"})

    elif action == "console":
        return _browser_request("GET", "/console", params=qp)

    elif action == "errors":
        return _browser_request("GET", "/errors", params=qp)

    elif action == "cookies":
        return _browser_request("GET", "/cookies", params=qp)

    elif action == "cookies_set":
        if not req.get("cookies"):
            return "Error: request.cookies array is required"
        return _browser_request("POST", "/cookies/set", params=qp, body=req)

    elif action == "cookies_clear":
        return _browser_request("POST", "/cookies/clear", params=qp)

    elif action == "dialog":
        return _browser_request("POST", "/hooks/dialog", params=qp, body=req)

    elif action == "requests":
        rqp = dict(qp)
        if req.get("url"):
            rqp["url"] = req["url"]
        if req.get("resourceType"):
            rqp["resourceType"] = req["resourceType"]
        if req.get("limit"):
            rqp["limit"] = str(req["limit"])
        return _browser_request("GET", "/requests", params=rqp)

    elif action == "response_body":
        if not req.get("url"):
            return "Error: request.url is required for response_body"
        body = {"url": req["url"]}
        if target_id:
            body["targetId"] = target_id
        return _browser_request("POST", "/response/body", params=qp, body=body)

    elif action == "download":
        return _browser_request("GET", "/download", params=qp)

    elif action == "wait_download":
        if target_id:
            req["targetId"] = target_id
        return _browser_request("POST", "/wait/download", params=qp, body=req, timeout=35.0)

    elif action == "storage_local":
        if not req or req.get("action") is None:
            sqp = dict(qp)
            if req.get("key"):
                sqp["key"] = req["key"]
            if target_id:
                sqp["targetId"] = target_id
            return _browser_request("GET", "/storage/local", params=sqp)
        else:
            if target_id:
                req["targetId"] = target_id
            return _browser_request("POST", "/storage/local", params=qp, body=req)

    elif action == "storage_session":
        if not req or req.get("action") is None:
            sqp = dict(qp)
            if req.get("key"):
                sqp["key"] = req["key"]
            if target_id:
                sqp["targetId"] = target_id
            return _browser_request("GET", "/storage/session", params=sqp)
        else:
            if target_id:
                req["targetId"] = target_id
            return _browser_request("POST", "/storage/session", params=qp, body=req)

    else:
        return f"Error: Unknown browser action '{action}'"


@observe(name="SubAgent")
def run_task(description: str, prompt: str, agent_type: str) -> str:
    """Execute a subagent task (from v3). See v3 for details."""
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
            output = execute_tool(tc.name, tc.input)
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


@observe(name="ToolExecution")
def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
    _get_langfuse().update_current_span(
        metadata={"tool": name, "args": args}
    )
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
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
    if name == "browser":
        return run_browser(args)
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================

@observe(name="MainAgentLoop")
def agent_loop(messages: list) -> list:
    """
    Main agent loop with skills support.

    Same pattern as v3, but now with Skill tool.
    When model loads a skill, it receives domain knowledge.
    """
    # Set session_id to group all traces from same conversation
    if SESSION_ID:
        _get_langfuse().update_current_trace(session_id=SESSION_ID)

    while True:
        log_api_call("main_agent", SYSTEM, messages, ALL_TOOLS)

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=ALL_TOOLS,
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
            elif tc.name == "browser":
                action = tc.input.get('action', '?')
                detail = tc.input.get('targetUrl', '')
                if not detail and isinstance(tc.input.get('request'), dict):
                    detail = tc.input['request'].get('kind', '')
                print(f"\n> Browser: {action}" + (f" ({detail})" if detail else ""))
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Skill tool shows summary, not full content
            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name == "browser":
                # Browser output can be large (snapshots), show truncated
                lines = output.strip().split("\n")
                if len(lines) > 10:
                    print(f"  " + "\n  ".join(lines[:8]))
                    print(f"  ... ({len(lines)} lines total)")
                else:
                    print(f"  {output}")
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
    global SESSION_ID
    SESSION_ID = str(uuid.uuid4())

    print(f"Mini Claude Code v4 (with Skills) - {WORKDIR}")
    print(f"Session: {SESSION_ID[:8]}...")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
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
