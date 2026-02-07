#!/usr/bin/env python3
"""
v5_sandbox_agent.py - Mini Claude Code: Permission Sandbox (~700 lines)

Core Philosophy: "Trust but Verify"
===================================
v4 gave us skills for domain knowledge. But there's a security question:

    How do we let the model be powerful WITHOUT being dangerous?

v4's security is a joke:
    if "rm -rf /" in cmd: return "Error"  # trivially bypassed

This is like a security guard who only checks for the word "bomb".
Write "b0mb" and you're through.

The Real Problem - Permission is a Spectrum:
-------------------------------------------
Binary security (allow/deny) doesn't match reality:

    | Operation              | Should we...           |
    |------------------------|------------------------|
    | `ls -la`               | Always allow           |
    | `cat file.py`          | Always allow           |
    | `rm temp.txt`          | Ask user first         |
    | `npm install pkg`      | Ask user first         |
    | `rm -rf /`             | NEVER allow            |
    | `curl | bash`          | NEVER allow            |

Three levels, not two. This is how real systems work.

The Solution - Layered Permission Model:
---------------------------------------
    +------------------------------------------------------------------+
    |                     Permission Flow                              |
    +------------------------------------------------------------------+
    |                                                                  |
    |    Tool Call                                                     |
    |        |                                                         |
    |        v                                                         |
    |    +------------------+                                          |
    |    | PermissionCheck  |                                          |
    |    +--------+---------+                                          |
    |             |                                                    |
    |    +--------+--------+--------+                                  |
    |    |        |        |        |                                  |
    |    v        v        v        v                                  |
    | +------+ +------+ +------+ +------+                              |
    | | READ | | WRITE| | EXEC | | NET  |                              |
    | |ALLOW | | ASK  | | ASK  | | DENY |                              |
    | +------+ +------+ +------+ +------+                              |
    |                                                                  |
    | Permission Levels:                                               |
    |   ALLOW  - Execute without asking                                |
    |   ASK    - Require user confirmation                             |
    |   DENY   - Block completely                                      |
    +------------------------------------------------------------------+

Key Insight - Permission is UI:
------------------------------
Permission isn't just security - it's COMMUNICATION.

When the model asks "Can I run npm install?", it:
1. Shows transparency (user sees what's happening)
2. Builds trust (user is in control)
3. Catches mistakes (user can say "wait, wrong project!")

This is why Claude Code asks before destructive operations.

Session Grants - "Allow for this session":
-----------------------------------------
Nobody wants to approve "git add" 50 times. Session grants remember:

    First time:  "Allow git add?"  -> User: "yes, always"
    Next times:  (auto-approved)

This balances security with usability.

Usage:
    python v5_sandbox_agent.py
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

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
# NEW in v5: Permission System
# =============================================================================

class Permission(Enum):
    """
    Three-level permission model.

    Why three levels instead of two?
    - ALLOW: Safe operations (reading files, listing directories)
    - ASK: Potentially dangerous but often needed (writing files, running scripts)
    - DENY: Never allowed (system destruction, privilege escalation)

    Binary allow/deny is too coarse. This matches how humans think about risk.
    """
    ALLOW = "allow"      # Execute without asking
    ASK = "ask"          # Require user confirmation
    DENY = "deny"        # Block completely


@dataclass
class PermissionRule:
    """
    A single permission rule with glob pattern matching.

    Examples:
        PermissionRule("ls *", Permission.ALLOW, "Listing is safe")
        PermissionRule("rm -rf /*", Permission.DENY, "System destruction")
        PermissionRule("npm *", Permission.ASK, "Package operations")
    """
    pattern: str         # Glob pattern (supports * and ?)
    permission: Permission
    reason: str          # Human-readable explanation


class PermissionManager:
    """
    Layered permission management for tool operations.

    Categories:
    ----------
    - read:   File reading operations (default: ALLOW)
    - write:  File writing operations (default: ASK)
    - exec:   Command execution (default: ASK for unknown)
    - net:    Network operations (default: DENY)

    Session Grants:
    --------------
    When user approves with "always", the grant is cached for the session.
    This prevents repetitive confirmations for the same operation type.

    Why glob patterns?
    -----------------
    Exact string matching is too rigid. "ls" vs "ls -la" vs "ls ." are all safe.
    Glob patterns like "ls *" catch all variants with one rule.
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.session_grants: dict[str, bool] = {}  # Cache for session approvals

        # Default rules by category
        # Rules are checked in order - first match wins
        self.rules: dict[str, list[PermissionRule]] = {
            "read": [
                PermissionRule("*", Permission.ALLOW, "Reading is safe"),
            ],
            "write": [
                # Safe writes
                PermissionRule("*.md", Permission.ALLOW, "Documentation"),
                PermissionRule("*.txt", Permission.ALLOW, "Text files"),
                # Dangerous writes
                PermissionRule("*.env*", Permission.ASK, "Environment files may contain secrets"),
                PermissionRule("*credentials*", Permission.DENY, "Credential files"),
                PermissionRule("*secret*", Permission.DENY, "Secret files"),
                PermissionRule("*.pem", Permission.DENY, "Private keys"),
                PermissionRule("*.key", Permission.DENY, "Private keys"),
                # Default: ask
                PermissionRule("*", Permission.ASK, "File modification"),
            ],
            "exec": [
                # === ALWAYS ALLOW: Safe, read-only commands ===
                PermissionRule("ls *", Permission.ALLOW, "Listing"),
                PermissionRule("ls", Permission.ALLOW, "Listing"),
                PermissionRule("pwd", Permission.ALLOW, "Current directory"),
                PermissionRule("cat *", Permission.ALLOW, "Reading files"),
                PermissionRule("head *", Permission.ALLOW, "Reading files"),
                PermissionRule("tail *", Permission.ALLOW, "Reading files"),
                PermissionRule("grep *", Permission.ALLOW, "Searching"),
                PermissionRule("find *", Permission.ALLOW, "Finding files"),
                PermissionRule("wc *", Permission.ALLOW, "Counting"),
                PermissionRule("echo *", Permission.ALLOW, "Printing"),
                PermissionRule("which *", Permission.ALLOW, "Finding commands"),
                PermissionRule("type *", Permission.ALLOW, "Command info"),
                PermissionRule("file *", Permission.ALLOW, "File type"),

                # Git read operations
                PermissionRule("git status*", Permission.ALLOW, "Git status"),
                PermissionRule("git diff*", Permission.ALLOW, "Git diff"),
                PermissionRule("git log*", Permission.ALLOW, "Git log"),
                PermissionRule("git branch*", Permission.ALLOW, "Git branches"),
                PermissionRule("git show*", Permission.ALLOW, "Git show"),
                PermissionRule("git remote -v", Permission.ALLOW, "Git remotes"),

                # === ALWAYS DENY: Dangerous commands ===
                PermissionRule("rm -rf /*", Permission.DENY, "System destruction"),
                PermissionRule("rm -rf /", Permission.DENY, "System destruction"),
                PermissionRule("sudo *", Permission.DENY, "Privilege escalation"),
                PermissionRule("su *", Permission.DENY, "User switching"),
                PermissionRule("chmod 777 *", Permission.DENY, "Dangerous permissions"),
                PermissionRule(":(){ :|:& };:", Permission.DENY, "Fork bomb"),
                PermissionRule("> /dev/*", Permission.DENY, "Device manipulation"),
                PermissionRule("mkfs*", Permission.DENY, "Filesystem destruction"),
                PermissionRule("dd if=*of=/dev/*", Permission.DENY, "Device overwrite"),
                PermissionRule("shutdown*", Permission.DENY, "System shutdown"),
                PermissionRule("reboot*", Permission.DENY, "System reboot"),
                PermissionRule("curl*|*bash*", Permission.DENY, "Remote code execution"),
                PermissionRule("wget*|*bash*", Permission.DENY, "Remote code execution"),
                PermissionRule("*| bash*", Permission.DENY, "Piped execution"),
                PermissionRule("*| sh*", Permission.DENY, "Piped execution"),

                # === DEFAULT: Ask for everything else ===
                PermissionRule("*", Permission.ASK, "Unknown command"),
            ],
            "net": [
                # Network is disabled by default
                PermissionRule("curl *", Permission.ASK, "HTTP request"),
                PermissionRule("wget *", Permission.ASK, "HTTP download"),
                PermissionRule("*", Permission.DENY, "Network access"),
            ],
        }

    def check(self, category: str, operation: str) -> tuple[Permission, str]:
        """
        Check permission for an operation.

        Args:
            category: One of "read", "write", "exec", "net"
            operation: The specific operation (command, path, etc.)

        Returns:
            (Permission, reason) tuple
        """
        # First check session grants
        cache_key = f"{category}:{self._normalize(operation)}"
        if cache_key in self.session_grants:
            return Permission.ALLOW, "Granted this session"

        # Then check rules
        for rule in self.rules.get(category, []):
            if self._match(operation, rule.pattern):
                return rule.permission, rule.reason

        # Default to ASK if no rules match
        return Permission.ASK, "No matching rule"

    def grant_session(self, category: str, operation: str):
        """
        Grant permission for this session.

        Called when user approves with "always".
        Uses normalized operation to catch similar commands.
        """
        cache_key = f"{category}:{self._normalize(operation)}"
        self.session_grants[cache_key] = True

    def _normalize(self, operation: str) -> str:
        """
        Normalize operation for caching.

        Examples:
            "git add file1.py" -> "git add *"
            "cat src/main.py" -> "cat *"

        This allows "git add" approval to cover all "git add" operations.
        """
        # For commands, normalize arguments
        parts = operation.split()
        if len(parts) > 1:
            # Keep command and first subcommand, wildcard the rest
            if parts[0] == "git" and len(parts) > 2:
                return f"{parts[0]} {parts[1]} *"
            return f"{parts[0]} *"
        return operation

    def _match(self, operation: str, pattern: str) -> bool:
        """
        Match operation against glob pattern.

        Uses fnmatch for glob-style matching:
        - * matches everything
        - ? matches single character
        """
        return fnmatch.fnmatch(operation, pattern)


def ask_user_permission(operation: str, reason: str, category: str) -> tuple[bool, bool]:
    """
    Interactive permission prompt.

    Returns:
        (allowed, remember) tuple
        - allowed: Whether to allow this operation
        - remember: Whether to remember for session

    Display format:
        ┌─ Permission Required ──────────────────────┐
        │ Category: exec                             │
        │ Operation: npm install lodash              │
        │ Reason: Package operations                 │
        │                                            │
        │ [y] Allow once                             │
        │ [n] Deny                                   │
        │ [a] Allow for this session                 │
        └────────────────────────────────────────────┘
    """
    print(f"\n┌─ Permission Required {'─' * 30}┐")
    print(f"│ Category: {category}")
    print(f"│ Operation: {operation[:50]}")
    print(f"│ Reason: {reason}")
    print(f"│")
    print(f"│ [y] Allow once")
    print(f"│ [n] Deny")
    print(f"│ [a] Allow for this session")

    while True:
        response = input("└─> ").strip().lower()
        if response in ("y", "yes"):
            return True, False
        if response in ("n", "no"):
            return False, False
        if response in ("a", "always"):
            return True, True
        print("  Please enter y/n/a")


# Global permission manager instance
PERMISSIONS = PermissionManager(WORKDIR)


# =============================================================================
# SkillLoader (from v4)
# =============================================================================

class SkillLoader:
    """Loads and manages skills from SKILL.md files. See v4 for details."""

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
}


def get_agent_descriptions() -> str:
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
# System Prompt - Updated for v5
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents available** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

**Permission System (NEW in v5)**:
Some operations require user permission. If a tool returns "Permission denied",
the user has blocked that operation. Try an alternative approach or ask the user
how they'd like to proceed.

Rules:
- Use Skill tool IMMEDIATELY when a task matches a skill description
- Use Task tool for subtasks needing focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over prose. Act, don't just explain.
- If permission denied, explain what you wanted to do and ask for guidance.
- After finishing, summarize what changed."""


# =============================================================================
# Tool Definitions
# =============================================================================

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command. Some commands may require user permission.",
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
        "description": "Write to file. May require user permission for non-documentation files.",
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
        "description": "Replace text in file. May require user permission.",
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

SKILL_TOOL = {
    "name": "Skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.

Available skills:
{SKILLS.get_descriptions()}

When to use:
- IMMEDIATELY when user task matches a skill description
- Before attempting domain-specific work (PDF, MCP, etc.)""",
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


def get_tools_for_agent(agent_type: str) -> list:
    """Filter tools based on agent type."""
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return BASE_TOOLS
    return [t for t in BASE_TOOLS if t["name"] in allowed]


# =============================================================================
# Tool Implementations - WITH PERMISSION CHECKS
# =============================================================================

def safe_path(p: str) -> Path:
    """Ensure path stays within workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    """
    Execute shell command WITH permission check.

    This is the key change from v4:
    1. Check permission BEFORE executing
    2. If ASK, prompt user
    3. If DENY, return error without executing
    4. If ALLOW or approved, execute normally
    """
    # Check permission
    permission, reason = PERMISSIONS.check("exec", cmd)

    if permission == Permission.DENY:
        return f"Permission denied: {reason}"

    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(cmd, reason, "exec")
        if not allowed:
            return "Permission denied by user"
        if remember:
            PERMISSIONS.grant_session("exec", cmd)

    # Execute command
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        return (r.stdout + r.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s limit)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """Read file contents (always allowed by default)."""
    permission, reason = PERMISSIONS.check("read", path)

    if permission == Permission.DENY:
        return f"Permission denied: {reason}"

    try:
        lines = safe_path(path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """Write content to file WITH permission check."""
    permission, reason = PERMISSIONS.check("write", path)

    if permission == Permission.DENY:
        return f"Permission denied: {reason}"

    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(
            f"write to {path} ({len(content)} bytes)",
            reason,
            "write"
        )
        if not allowed:
            return "Permission denied by user"
        if remember:
            PERMISSIONS.grant_session("write", path)

    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file WITH permission check."""
    permission, reason = PERMISSIONS.check("write", path)

    if permission == Permission.DENY:
        return f"Permission denied: {reason}"

    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(
            f"edit {path}",
            reason,
            "write"
        )
        if not allowed:
            return "Permission denied by user"
        if remember:
            PERMISSIONS.grant_session("write", path)

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
    """Update the todo list (no permission needed)."""
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    """Load a skill (no permission needed)."""
    content = SKILLS.get_skill_content(skill_name)

    if content is None:
        available = ", ".join(SKILLS.list_skills()) or "none"
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"

    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


def run_task(description: str, prompt: str, agent_type: str) -> str:
    """Execute a subagent task. See v3 for details."""
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


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
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
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================

def agent_loop(messages: list) -> list:
    """Main agent loop with permission support."""
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
            return messages

        results = []
        for tc in tool_calls:
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Show permission denials prominently
            if "Permission denied" in output:
                print(f"  {output}")
            elif tc.name == "Skill":
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
    print(f"Mini Claude Code v5 (with Permission Sandbox) - {WORKDIR}")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print("Permission system: ACTIVE (some operations require approval)")
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
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
