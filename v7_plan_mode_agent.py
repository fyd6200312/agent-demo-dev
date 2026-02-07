#!/usr/bin/env python3
"""
v7_plan_mode_agent.py - Mini Claude Code: Plan Mode (~950 lines)

Core Philosophy: "Think Before You Act"
=======================================
v6 gave us specialized tools. But there's a workflow question:

    Should the model ALWAYS be allowed to modify files?

Ask a v6 agent to "refactor the authentication system" and watch:

    Turn 1: Read auth.py... OK
    Turn 2: Edit auth.py... WRONG APPROACH
    Turn 3: Undo... Read more files...
    Turn 4: Edit auth.py again... STILL WRONG
    Turn 5: "Let me think about this differently..."

The model wastes tokens (= money) and makes messy changes because it
dives into EXECUTION before understanding the full picture.

The Problem - Ready, Fire, Aim:
-------------------------------
Without a planning phase, agents suffer from:

    1. Premature execution - Editing before understanding
    2. Wasted effort - Undoing bad changes
    3. User surprise - "Why did you change THAT file?"
    4. Context pollution - Failed attempts fill the context

Humans don't write code this way. We:
    1. Read and understand the codebase
    2. Design an approach
    3. Get approval (code review, design doc)
    4. THEN implement

The Solution - Plan Mode:
------------------------
A STATE MACHINE that separates thinking from doing:

    +------------------------------------------------------------------+
    |                      Plan Mode Flow                              |
    +------------------------------------------------------------------+
    |                                                                  |
    |  User: "Refactor auth to use JWT"                                |
    |         |                                                        |
    |    [Agent decides task is complex]                                |
    |         |                                                        |
    |         v                                                        |
    |  +-----------------------+                                       |
    |  | EnterPlanMode         |                                       |
    |  | "Let me analyze first"|                                       |
    |  +-----------+-----------+                                       |
    |              |                                                   |
    |              v                                                   |
    |  +--------------------------+                                    |
    |  |      PLAN MODE           |                                    |
    |  |  Available: Glob, Grep,  |                                    |
    |  |    Read, TodoWrite       |                                    |
    |  |  Blocked: bash, write,   |                                    |
    |  |    edit                   |                                    |
    |  |                          |                                    |
    |  |  Agent explores code...  |                                    |
    |  |  Agent creates plan...   |                                    |
    |  +-----------+--------------+                                    |
    |              |                                                   |
    |              v                                                   |
    |  +-----------------------+                                       |
    |  | ExitPlanMode          |                                       |
    |  | Submits plan for      |                                       |
    |  | user review           |                                       |
    |  +-----------+-----------+                                       |
    |              |                                                   |
    |              v                                                   |
    |  +--------------------------+                                    |
    |  |    USER REVIEWS PLAN     |                                    |
    |  |                          |                                    |
    |  |  "approve" -> Execute    |                                    |
    |  |  "revise"  -> Re-plan    |                                    |
    |  |  "cancel"  -> Abort      |                                    |
    |  +-----------+--------------+                                    |
    |              |                                                   |
    |         [approve]                                                |
    |              |                                                   |
    |              v                                                   |
    |  +--------------------------+                                    |
    |  |    EXECUTION MODE        |                                    |
    |  |  All tools available     |                                    |
    |  |  Follows approved plan   |                                    |
    |  +--------------------------+                                    |
    |                                                                  |
    +------------------------------------------------------------------+

Key Insight - Plan Mode is a State Machine:
------------------------------------------
It's NOT just "system prompt says be careful". It's ENFORCED:

    v6:  System prompt: "Please plan before acting"    (suggestion)
    v7:  Plan Mode: write_file tool REMOVED from list  (enforcement)

The model CANNOT write files in plan mode because the tool doesn't exist
in that state. This is constraint as architecture, not as suggestion.

Why This Matters:
----------------
    | Approach         | What happens                | Result       |
    |------------------|----------------------------|--------------|
    | Ask nicely       | Model sometimes plans      | Inconsistent |
    | Remove tools     | Model CAN'T modify         | Guaranteed   |

This is the same insight as v2 (constraints enable), applied to workflow.

When to Use Plan Mode:
---------------------
- Complex refactoring (multiple files)
- Architecture changes (new patterns)
- Unfamiliar codebases (need exploration first)
- User explicitly requests a plan

When to Skip:
- Simple bug fixes (one-line change)
- File creation (clear requirements)
- User says "just do it"

Usage:
    python v7_plan_mode_agent.py
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

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
        "system": system, "messages": messages, "tools": tools
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
# Permission System (from v5)
# =============================================================================

class Permission(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    pattern: str
    permission: Permission
    reason: str


class PermissionManager:
    """Permission management. See v5 for details."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.session_grants: dict[str, bool] = {}
        self.rules: dict[str, list[PermissionRule]] = {
            "read": [PermissionRule("*", Permission.ALLOW, "Reading is safe")],
            "write": [
                PermissionRule("*.md", Permission.ALLOW, "Documentation"),
                PermissionRule("*.txt", Permission.ALLOW, "Text files"),
                PermissionRule("*.env*", Permission.ASK, "Environment files"),
                PermissionRule("*credentials*", Permission.DENY, "Credentials"),
                PermissionRule("*secret*", Permission.DENY, "Secrets"),
                PermissionRule("*.pem", Permission.DENY, "Private keys"),
                PermissionRule("*.key", Permission.DENY, "Private keys"),
                PermissionRule("*", Permission.ASK, "File modification"),
            ],
            "exec": [
                PermissionRule("ls *", Permission.ALLOW, "Listing"),
                PermissionRule("ls", Permission.ALLOW, "Listing"),
                PermissionRule("pwd", Permission.ALLOW, "Current directory"),
                PermissionRule("echo *", Permission.ALLOW, "Printing"),
                PermissionRule("which *", Permission.ALLOW, "Finding commands"),
                PermissionRule("git status*", Permission.ALLOW, "Git status"),
                PermissionRule("git diff*", Permission.ALLOW, "Git diff"),
                PermissionRule("git log*", Permission.ALLOW, "Git log"),
                PermissionRule("git branch*", Permission.ALLOW, "Git branches"),
                PermissionRule("git show*", Permission.ALLOW, "Git show"),
                PermissionRule("rm -rf /*", Permission.DENY, "System destruction"),
                PermissionRule("rm -rf /", Permission.DENY, "System destruction"),
                PermissionRule("sudo *", Permission.DENY, "Privilege escalation"),
                PermissionRule("*| bash*", Permission.DENY, "Piped execution"),
                PermissionRule("*| sh*", Permission.DENY, "Piped execution"),
                PermissionRule("*", Permission.ASK, "Unknown command"),
            ],
        }

    def check(self, category: str, operation: str) -> tuple[Permission, str]:
        cache_key = f"{category}:{self._normalize(operation)}"
        if cache_key in self.session_grants:
            return Permission.ALLOW, "Granted this session"
        for rule in self.rules.get(category, []):
            if fnmatch.fnmatch(operation, rule.pattern):
                return rule.permission, rule.reason
        return Permission.ASK, "No matching rule"

    def grant_session(self, category: str, operation: str):
        self.session_grants[f"{category}:{self._normalize(operation)}"] = True

    def _normalize(self, operation: str) -> str:
        parts = operation.split()
        if len(parts) > 1:
            if parts[0] == "git" and len(parts) > 2:
                return f"{parts[0]} {parts[1]} *"
            return f"{parts[0]} *"
        return operation


def ask_user_permission(operation: str, reason: str, category: str) -> tuple[bool, bool]:
    print(f"\n┌─ Permission Required {'─' * 30}┐")
    print(f"│ Category: {category}")
    print(f"│ Operation: {operation[:50]}")
    print(f"│ Reason: {reason}")
    print(f"│ [y] Allow once  [n] Deny  [a] Allow for session")
    while True:
        response = input("└─> ").strip().lower()
        if response in ("y", "yes"):
            return True, False
        if response in ("n", "no"):
            return False, False
        if response in ("a", "always"):
            return True, True
        print("  Please enter y/n/a")


PERMISSIONS = PermissionManager(WORKDIR)


# =============================================================================
# NEW in v7: Plan Mode - State Machine
# =============================================================================

class AgentMode(Enum):
    """
    Agent operating modes.

    NORMAL:   All tools available. Default mode.
    PLANNING: Read-only tools only. For analysis and plan creation.

    State transitions:
        NORMAL --[EnterPlanMode]--> PLANNING
        PLANNING --[ExitPlanMode + user approve]--> NORMAL
        PLANNING --[user cancel]--> NORMAL (no plan)
    """
    NORMAL = "normal"
    PLANNING = "planning"


class ModeManager:
    """
    Manages agent mode transitions.

    The core mechanism: in PLANNING mode, write/edit/bash tools are REMOVED
    from the tool list. The model literally cannot call them.

    This is enforcement through architecture, not through instructions.

    Plan storage:
    - Current plan is stored in self.current_plan
    - Plan is shown to user via ExitPlanMode
    - On approval, plan is injected as context for execution
    """

    def __init__(self):
        self.mode = AgentMode.NORMAL
        self.current_plan: str | None = None

    def enter_plan_mode(self) -> str:
        """
        Transition to planning mode.

        Returns status message that becomes tool_result.
        """
        if self.mode == AgentMode.PLANNING:
            return "Already in plan mode. Use ExitPlanMode to submit your plan."

        self.mode = AgentMode.PLANNING
        self.current_plan = None

        return """Entered PLAN MODE.

You are now in read-only mode. Available tools:
  - Glob: Find files by pattern
  - Grep: Search file contents
  - Read: Read files with line numbers
  - TodoWrite: Track your analysis progress

Blocked tools: bash, write_file, edit_file

Your goal:
1. Explore the codebase to understand the current state
2. Design an implementation approach
3. Call ExitPlanMode with your detailed plan

The plan should include:
- Files to modify and why
- Specific changes for each file
- Any risks or considerations
- Testing approach"""

    def exit_plan_mode(self, plan: str) -> str:
        """
        Submit plan and request user approval.

        The plan is NOT executed yet - it's shown to the user.
        Returns formatted plan for display.
        """
        if self.mode != AgentMode.PLANNING:
            return "Error: Not in plan mode. Use EnterPlanMode first."

        if not plan.strip():
            return "Error: Plan cannot be empty."

        self.current_plan = plan

        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   IMPLEMENTATION PLAN                       ║
╠══════════════════════════════════════════════════════════════╣

{plan}

╠══════════════════════════════════════════════════════════════╣
║  Reply:                                                     ║
║    "approve"         - Execute this plan                    ║
║    "revise: <notes>" - Request changes to the plan          ║
║    "cancel"          - Abort and return to normal mode       ║
╚══════════════════════════════════════════════════════════════╝"""

    def handle_user_response(self, user_input: str) -> tuple[str, str | None]:
        """
        Handle user response to a pending plan.

        Returns:
            (action, context)
            - action: "approve", "revise", "cancel", or "not_pending"
            - context: The plan (for approve) or revision notes (for revise)
        """
        if self.current_plan is None:
            return "not_pending", None

        normalized = user_input.strip().lower()

        if normalized in ("approve", "yes", "y", "ok", "go", "lgtm"):
            plan = self.current_plan
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "approve", plan

        if normalized in ("cancel", "abort", "no", "n"):
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "cancel", None

        if normalized.startswith("revise:") or normalized.startswith("revise "):
            notes = user_input[7:].strip()
            self.current_plan = None  # Clear plan, keep mode
            return "revise", notes

        # Treat anything else as revision notes
        return "revise", user_input

    def get_available_tools(self, all_tools: list) -> list:
        """
        Filter tools based on current mode.

        In PLANNING mode, only read-only tools + TodoWrite + plan tools.
        In NORMAL mode, all tools.

        This is the ENFORCEMENT mechanism - not a suggestion, a restriction.
        """
        if self.mode == AgentMode.PLANNING:
            READ_ONLY = {"Glob", "Grep", "Read", "TodoWrite", "ExitPlanMode"}
            return [t for t in all_tools if t["name"] in READ_ONLY]
        return all_tools

    def get_mode_prompt(self) -> str:
        """Generate mode-specific system prompt addition."""
        if self.mode == AgentMode.PLANNING:
            return """

*** PLAN MODE ACTIVE ***
You are in READ-ONLY planning mode.
- Use Glob, Grep, Read to explore the codebase
- Use TodoWrite to track your analysis
- DO NOT attempt modifications (tools are disabled)
- When ready, call ExitPlanMode with your implementation plan"""
        return ""


# Global mode manager
MODE = ModeManager()


# =============================================================================
# Specialized Tools (from v6)
# =============================================================================

EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    '.idea', '.vscode', 'dist', 'build', '.next', '.nuxt',
    'coverage', '.pytest_cache', '.mypy_cache', 'egg-info'
}


def run_glob(pattern: str, path: str = None) -> str:
    """File pattern matching. See v6 for details."""
    try:
        base = (WORKDIR / path).resolve() if path else WORKDIR
        if not base.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"
        if not base.exists():
            return f"Error: Path does not exist: {path}"

        results = []
        for p in base.glob(pattern):
            if any(ex in p.parts for ex in EXCLUDE_DIRS):
                continue
            if p.is_file():
                try:
                    results.append((p.stat().st_mtime, p))
                except OSError:
                    continue

        results.sort(reverse=True)
        MAX_RESULTS = 100
        paths = [str(p.relative_to(WORKDIR)) for _, p in results[:MAX_RESULTS]]

        if not paths:
            return "No matches found."
        output = "\n".join(paths)
        if len(results) > MAX_RESULTS:
            output += f"\n... and {len(results) - MAX_RESULTS} more files"
        return output
    except Exception as e:
        return f"Error: {e}"


def run_grep(pattern: str, path: str = None, glob_pattern: str = None,
             output_mode: str = "files_with_matches", context: int = 0) -> str:
    """Content search. See v6 for details."""
    try:
        base = (WORKDIR / path).resolve() if path else WORKDIR
        if not base.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        file_pattern = glob_pattern or "**/*"
        results = []
        files_searched = 0

        for fp in base.glob(file_pattern):
            if not fp.is_file() or any(ex in fp.parts for ex in EXCLUDE_DIRS):
                continue
            try:
                content = fp.read_text(errors='ignore')
            except Exception:
                continue

            files_searched += 1
            if files_searched > 1000:
                results.append("... (stopped after 1000 files)")
                break

            lines = content.splitlines()
            matches = [(i + 1, line) for i, line in enumerate(lines) if regex.search(line)]

            if matches:
                rel_path = str(fp.relative_to(WORKDIR))
                if output_mode == "files_with_matches":
                    results.append(rel_path)
                elif output_mode == "count":
                    results.append(f"{rel_path}: {len(matches)}")
                else:
                    results.append(f"\n{rel_path}:")
                    for lineno, line in matches[:10]:
                        display = line[:200] + "..." if len(line) > 200 else line
                        results.append(f"  {lineno:>5}: {display}")
                    if len(matches) > 10:
                        results.append(f"  ... and {len(matches) - 10} more")

            if len(results) >= 100:
                results.append("... (limited to 100 results)")
                break

        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error: {e}"


def run_read_file(file_path: str, offset: int = 1, limit: int = 2000) -> str:
    """Smart file reading. See v6 for details."""
    try:
        fp = (WORKDIR / file_path).resolve()
        if not fp.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"
        if not fp.exists():
            return f"Error: File not found: {file_path}"
        if not fp.is_file():
            return f"Error: Not a file: {file_path}"

        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            try:
                r = subprocess.run(["pdftotext", str(fp), "-"],
                                   capture_output=True, text=True, timeout=30)
                return r.stdout[:50000] or "(PDF is empty or image-only)"
            except FileNotFoundError:
                return "Error: pdftotext not installed"

        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            return f"[Image file: {file_path}]"

        try:
            content = fp.read_text()
        except UnicodeDecodeError:
            return f"Error: Binary file: {file_path}"

        lines = content.splitlines()
        total = len(lines)
        start = max(0, offset - 1)
        end = min(total, start + limit)

        output_lines = []
        for i, line in enumerate(lines[start:end]):
            lineno = start + i + 1
            if len(line) > 500:
                line = line[:500] + "..."
            output_lines.append(f"{lineno:>6}| {line}")

        result = "\n".join(output_lines)
        if end < total:
            result += f"\n\n... {total - end} more lines. Use offset={end + 1} to continue."
        return result
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# SkillLoader (from v4)
# =============================================================================

class SkillLoader:
    """See v4 for details."""

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
        return {"name": metadata["name"], "description": metadata["description"],
                "body": body.strip(), "path": path, "dir": path.parent}

    def load_skills(self):
        if not self.skills_dir.exists():
            return
        for d in self.skills_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                skill = self.parse_skill_md(d / "SKILL.md")
                if skill:
                    self.skills[skill["name"]] = skill

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        return "\n".join(f"- {n}: {s['description']}" for n, s in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        if name not in self.skills:
            return None
        skill = self.skills[name]
        content = f"# Skill: {skill['name']}\n\n{skill['body']}"
        for folder, label in [("scripts", "Scripts"), ("references", "References"), ("assets", "Assets")]:
            fp = skill["dir"] / folder
            if fp.exists():
                files = list(fp.glob("*"))
                if files:
                    content += f"\n\n**{label}:** {', '.join(f.name for f in files)}"
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
        "tools": ["Glob", "Grep", "Read", "bash"],
        "prompt": "You are an exploration agent. Use Glob, Grep, Read to analyze. Return a concise summary.",
    },
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["Glob", "Grep", "Read"],
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
}


def get_agent_descriptions() -> str:
    return "\n".join(f"- {n}: {c['description']}" for n, c in AGENT_TYPES.items())


# =============================================================================
# TodoManager (from v2)
# =============================================================================

class TodoManager:
    """See v2 for details."""

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
            validated.append({"content": content, "status": status, "activeForm": active})
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
# Tool Definitions - Updated for v7
# =============================================================================

GLOB_TOOL = {
    "name": "Glob",
    "description": "Find files matching a glob pattern. Returns paths sorted by modification time.\nExamples: '**/*.py', 'src/**/*.ts', '**/test_*'",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Base directory (default: workspace root)"}
        },
        "required": ["pattern"],
    },
}

GREP_TOOL = {
    "name": "Grep",
    "description": "Search file contents with regex.\nOutput modes: files_with_matches (default), content, count.\nExample: Grep(pattern='TODO', glob='*.py', output_mode='content')",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Directory (default: workspace root)"},
            "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
            "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
            "context": {"type": "integer", "description": "Context lines (content mode)"}
        },
        "required": ["pattern"],
    },
}

READ_TOOL = {
    "name": "Read",
    "description": "Read file with line numbers. Supports offset/limit for large files.\nExample: Read(file_path='src/main.py', offset=50, limit=100)",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "offset": {"type": "integer", "description": "Start line (1-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to read"}
        },
        "required": ["file_path"],
    },
}

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command. Use for builds, git, scripts. NOT for file search/read (use Glob/Grep/Read).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file. Use Read first to see line numbers.",
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
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
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
    "description": f"Spawn a subagent.\n\nAgent types:\n{get_agent_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

SKILL_TOOL = {
    "name": "Skill",
    "description": f"Load a skill.\n\nAvailable:\n{SKILLS.get_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {"skill": {"type": "string"}},
        "required": ["skill"],
    },
}

# NEW in v7: Plan Mode tools
ENTER_PLAN_TOOL = {
    "name": "EnterPlanMode",
    "description": """Enter planning mode for complex tasks.

In plan mode, ONLY read-only tools are available (Glob, Grep, Read, TodoWrite).
Write/edit/bash are disabled until the plan is approved.

Use when:
- Task is complex (multiple files, refactoring)
- You need to understand the codebase before acting
- User explicitly asks for a plan
- You're unsure about the right approach

Do NOT use for:
- Simple one-line fixes
- Clear, specific instructions""",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

EXIT_PLAN_TOOL = {
    "name": "ExitPlanMode",
    "description": """Submit your implementation plan for user review.

Include:
- Files to modify and specific changes
- New files to create (if any)
- Risks or considerations
- Testing approach

The user will approve, request revisions, or cancel.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "Detailed implementation plan in markdown"
            }
        },
        "required": ["plan"],
    },
}

SPECIALIZED_TOOLS = [GLOB_TOOL, GREP_TOOL, READ_TOOL]
PLAN_TOOLS = [ENTER_PLAN_TOOL, EXIT_PLAN_TOOL]
ALL_TOOLS = SPECIALIZED_TOOLS + BASE_TOOLS + [TASK_TOOL, SKILL_TOOL] + PLAN_TOOLS


def get_tools_for_agent(agent_type: str) -> list:
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return SPECIALIZED_TOOLS + BASE_TOOLS
    return [t for t in (SPECIALIZED_TOOLS + BASE_TOOLS) if t["name"] in allowed]


# =============================================================================
# System Prompt - Dynamic based on mode
# =============================================================================

def get_system_prompt() -> str:
    """
    Generate system prompt dynamically based on current mode.

    In NORMAL mode: Full instructions with all capabilities.
    In PLANNING mode: Append planning-specific instructions.
    """
    base = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Specialized Tools** (prefer these for file operations):
- Glob: Find files by pattern
- Grep: Search file contents
- Read: Read files with line numbers

**General Tools**: bash (builds/git/scripts), write_file, edit_file
**Planning**: EnterPlanMode / ExitPlanMode (for complex tasks)

**Skills**: {SKILLS.get_descriptions()}
**Subagents**: {get_agent_descriptions()}

Rules:
- For complex tasks (refactoring, multi-file changes): use EnterPlanMode first
- Use Glob/Grep/Read instead of bash for file operations
- Use Skill tool when task matches a skill
- Use Task tool for focused subtasks
- Use TodoWrite to track multi-step work
- After finishing, summarize what changed."""

    base += MODE.get_mode_prompt()
    return base


# =============================================================================
# Tool Implementations
# =============================================================================

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    permission, reason = PERMISSIONS.check("exec", cmd)
    if permission == Permission.DENY:
        return f"Permission denied: {reason}"
    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(cmd, reason, "exec")
        if not allowed:
            return "Permission denied by user"
        if remember:
            PERMISSIONS.grant_session("exec", cmd)
    try:
        r = subprocess.run(cmd, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    permission, reason = PERMISSIONS.check("write", path)
    if permission == Permission.DENY:
        return f"Permission denied: {reason}"
    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(f"write to {path}", reason, "write")
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
    permission, reason = PERMISSIONS.check("write", path)
    if permission == Permission.DENY:
        return f"Permission denied: {reason}"
    if permission == Permission.ASK:
        allowed, remember = ask_user_permission(f"edit {path}", reason, "write")
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
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    content = SKILLS.get_skill_content(skill_name)
    if content is None:
        available = ", ".join(SKILLS.list_skills()) or "none"
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"
    return f'<skill-loaded name="{skill_name}">\n{content}\n</skill-loaded>\n\nFollow the instructions above.'


def run_task(description: str, prompt: str, agent_type: str) -> str:
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]
    sub_system = f"You are a {agent_type} subagent at {WORKDIR}.\n\n{config['prompt']}\n\nReturn a clear, concise summary."
    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    while True:
        log_api_call(f"subagent:{agent_type}", sub_system, sub_messages, sub_tools)
        response = client.messages.create(model=MODEL, system=sub_system,
                                          messages=sub_messages, tools=sub_tools, max_tokens=8000)
        log_api_response(f"subagent:{agent_type}", response)

        if response.stop_reason != "tool_use":
            break

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        results = []
        for tc in tool_calls:
            tool_count += 1
            output = execute_tool(tc.name, tc.input)
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})
            sys.stdout.write(f"\r  [{agent_type}] {description} ... {tool_count} tools, {time.time() - start:.1f}s")
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    sys.stdout.write(f"\r  [{agent_type}] {description} - done ({tool_count} tools, {time.time() - start:.1f}s)\n")

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "(subagent returned no text)"


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
    # Specialized tools
    if name == "Glob":
        return run_glob(args["pattern"], args.get("path"))
    if name == "Grep":
        return run_grep(args["pattern"], args.get("path"), args.get("glob"),
                        args.get("output_mode", "files_with_matches"), args.get("context", 0))
    if name == "Read":
        return run_read_file(args["file_path"], args.get("offset", 1), args.get("limit", 2000))

    # Plan mode tools (v7)
    if name == "EnterPlanMode":
        return MODE.enter_plan_mode()
    if name == "ExitPlanMode":
        return MODE.exit_plan_mode(args["plan"])

    # Base tools
    if name == "bash":
        return run_bash(args["command"])
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
# Main Agent Loop - With Plan Mode Support
# =============================================================================

def agent_loop(messages: list) -> list:
    """
    Main agent loop with plan mode support.

    Key difference from v6:
    - Tool list changes dynamically based on MODE
    - System prompt changes dynamically based on MODE
    """
    while True:
        # Dynamic tool list based on mode
        available_tools = MODE.get_available_tools(ALL_TOOLS)
        system = get_system_prompt()

        log_api_call("main_agent", system, messages, available_tools)

        response = client.messages.create(
            model=MODEL,
            system=system,
            messages=messages,
            tools=available_tools,
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
            # Display
            if tc.name == "EnterPlanMode":
                print("\n> Entering Plan Mode...")
            elif tc.name == "ExitPlanMode":
                print("\n> Submitting Plan...")
            elif tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Display output
            if tc.name in ("EnterPlanMode", "ExitPlanMode"):
                print(output)
            elif tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name == "Glob":
                lines = output.split("\n")
                print(f"  Found {len(lines)} files")
            elif tc.name == "Grep":
                lines = output.split("\n")
                print(f"  {len(lines)} results")
            elif tc.name == "Read":
                lines = output.split("\n")
                print(f"  Read {len(lines)} lines")
            elif tc.name != "Task":
                if len(output) > 500:
                    print(f"  {output[:500]}...")
                else:
                    print(f"  {output}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL - With Plan Approval Flow
# =============================================================================

def main():
    print(f"Mini Claude Code v7 (with Plan Mode) - {WORKDIR}")
    print(f"Tools: Glob, Grep, Read, bash, write_file, edit_file")
    print(f"Planning: EnterPlanMode / ExitPlanMode")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            # Show mode indicator in prompt
            mode_indicator = " [PLAN]" if MODE.mode == AgentMode.PLANNING else ""
            user_input = input(f"You{mode_indicator}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Handle plan approval flow
        if MODE.current_plan is not None:
            action, context = MODE.handle_user_response(user_input)

            if action == "approve":
                print("\n Plan approved. Executing...\n")
                history.append({
                    "role": "user",
                    "content": f"Plan approved. Execute the following plan:\n\n{context}"
                })
                try:
                    agent_loop(history)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue

            elif action == "cancel":
                print("\n Plan cancelled. Back to normal mode.\n")
                continue

            elif action == "revise":
                print(f"\n Revising plan with feedback: {context}\n")
                history.append({
                    "role": "user",
                    "content": f"Please revise the plan based on this feedback:\n{context}"
                })
                try:
                    agent_loop(history)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue

        # Normal flow
        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
