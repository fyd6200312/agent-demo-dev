#!/usr/bin/env python3
"""
v8_advanced_subagent.py - Mini Claude Code: Advanced Subagents (~1050 lines)

Core Philosophy: "Agents as Processes"
=====================================
v7 gave us plan mode for thoughtful execution. But there's a parallelism question:

    Why does everything have to happen SYNCHRONOUSLY?

Watch v7 handle "analyze auth, tests, and API modules":

    Turn 1: Task(explore auth)... 30 seconds... done
    Turn 2: Task(explore tests)... 25 seconds... done
    Turn 3: Task(explore API)... 35 seconds... done
    Total: 90 seconds (sequential)

But these tasks are INDEPENDENT. They could run in parallel:

    Turn 1: Task(auth, background=True)  -> "Started task-abc"
            Task(tests, background=True) -> "Started task-def"
            Task(API, background=True)   -> "Started task-ghi"
    Turn 2: TaskOutput(task-abc) -> auth results
            TaskOutput(task-def) -> tests results
            TaskOutput(task-ghi) -> API results
    Total: ~35 seconds (parallel)

The Problem - Subagents are Blocking:
------------------------------------
v7 subagents work like function calls - synchronous and blocking:

    main_agent()
        |
        |-- run_task("explore auth", ...)  # Blocks until complete
        |       |
        |       |-- [explore agent runs]
        |       |-- [10+ tool calls]
        |       |-- [returns result]
        |       |
        |-- run_task("explore tests", ...) # Can't start until above finishes
        |
        v

This is WASTEFUL when tasks are independent.

The Solution - Background Tasks:
-------------------------------
Treat subagents like processes - they can run in background:

    +------------------------------------------------------------------+
    |                  Task Manager Architecture                       |
    +------------------------------------------------------------------+
    |                                                                  |
    |  Main Agent                                                      |
    |      |                                                           |
    |      |-- Task(background=True) ──┐                               |
    |      |       Returns immediately  |                              |
    |      |       with task_id        |                               |
    |      |                           v                               |
    |      |                    +------------------+                   |
    |      |                    | Background Task  |                   |
    |      |                    | status: RUNNING  |                   |
    |      |                    | thread: active   |                   |
    |      |                    +------------------+                   |
    |      |                                                           |
    |      |-- TaskList() ────────> Shows all tasks and status         |
    |      |                                                           |
    |      |-- TaskOutput(id) ────> Waits for and returns result      |
    |      |                                                           |
    |      |-- Continue working on other things...                     |
    |                                                                  |
    +------------------------------------------------------------------+

Key Insight - State Enables Control:
-----------------------------------
Once tasks have STATE (pending/running/completed), we can:

    | Capability     | How                           |
    |----------------|-------------------------------|
    | Parallel       | Start multiple, collect later |
    | Monitor        | TaskList() shows status       |
    | Wait           | TaskOutput(id, wait=True)     |
    | Check          | TaskOutput(id, wait=False)    |

State is the foundation of process management.

New Tools in v8:
---------------
    | Tool        | Purpose                                     |
    |-------------|---------------------------------------------|
    | Task        | Enhanced with `background` parameter        |
    | TaskOutput  | Get result from background task             |
    | TaskList    | List all tasks and their status             |

Usage Pattern:
-------------
    # Start tasks in parallel
    Task(description="explore auth", ..., background=True)  -> task-abc
    Task(description="explore tests", ..., background=True) -> task-def

    # Do other work while they run...

    # Collect results
    TaskOutput(task_id="task-abc")  -> auth analysis
    TaskOutput(task_id="task-def")  -> tests analysis

Threading Note:
--------------
We use Python threads for simplicity. In production:
- Consider asyncio for better resource usage
- Use proper task queues for distributed execution
- Add timeout handling and cancellation

Usage:
    python v8_advanced_subagent.py
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
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
    if not DEBUG_LOG:
        return
    print("\n" + "=" * 80)
    print(f"[API CALL] from: {caller}")
    print("=" * 80)
    print(json.dumps({"system": system, "messages": messages, "tools": tools},
                     ensure_ascii=False, indent=2, default=str))
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
    """See v5 for details."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.session_grants: dict[str, bool] = {}
        self.rules = {
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
                PermissionRule("git status*", Permission.ALLOW, "Git status"),
                PermissionRule("git diff*", Permission.ALLOW, "Git diff"),
                PermissionRule("git log*", Permission.ALLOW, "Git log"),
                PermissionRule("git branch*", Permission.ALLOW, "Git branches"),
                PermissionRule("rm -rf /*", Permission.DENY, "System destruction"),
                PermissionRule("sudo *", Permission.DENY, "Privilege escalation"),
                PermissionRule("*| bash*", Permission.DENY, "Piped execution"),
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
# Plan Mode (from v7)
# =============================================================================

class AgentMode(Enum):
    NORMAL = "normal"
    PLANNING = "planning"


class ModeManager:
    """See v7 for details."""

    def __init__(self):
        self.mode = AgentMode.NORMAL
        self.current_plan: str | None = None

    def enter_plan_mode(self) -> str:
        if self.mode == AgentMode.PLANNING:
            return "Already in plan mode."
        self.mode = AgentMode.PLANNING
        self.current_plan = None
        return """Entered PLAN MODE.
Read-only tools available: Glob, Grep, Read, TodoWrite
When ready, call ExitPlanMode with your implementation plan."""

    def exit_plan_mode(self, plan: str) -> str:
        if self.mode != AgentMode.PLANNING:
            return "Error: Not in plan mode."
        if not plan.strip():
            return "Error: Plan cannot be empty."
        self.current_plan = plan
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   IMPLEMENTATION PLAN                       ║
╠══════════════════════════════════════════════════════════════╣

{plan}

╠══════════════════════════════════════════════════════════════╣
║  "approve" / "revise: <notes>" / "cancel"                   ║
╚══════════════════════════════════════════════════════════════╝"""

    def handle_user_response(self, user_input: str) -> tuple[str, str | None]:
        if self.current_plan is None:
            return "not_pending", None
        normalized = user_input.strip().lower()
        if normalized in ("approve", "yes", "y", "ok", "go"):
            plan = self.current_plan
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "approve", plan
        if normalized in ("cancel", "abort", "no", "n"):
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "cancel", None
        if normalized.startswith("revise"):
            notes = user_input[6:].strip().lstrip(":").strip()
            self.current_plan = None
            return "revise", notes or user_input
        return "revise", user_input

    def get_available_tools(self, all_tools: list) -> list:
        if self.mode == AgentMode.PLANNING:
            READ_ONLY = {"Glob", "Grep", "Read", "TodoWrite", "ExitPlanMode"}
            return [t for t in all_tools if t["name"] in READ_ONLY]
        return all_tools

    def get_mode_prompt(self) -> str:
        if self.mode == AgentMode.PLANNING:
            return "\n\n*** PLAN MODE ACTIVE - Read-only tools only ***"
        return ""


MODE = ModeManager()


# =============================================================================
# NEW in v8: Advanced Task Manager
# =============================================================================

class TaskStatus(Enum):
    """
    Task lifecycle states.

    PENDING:    Task created but not started
    RUNNING:    Task is executing
    COMPLETED:  Task finished successfully
    FAILED:     Task encountered an error
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubagentTask:
    """
    Represents a subagent task with full state.

    Key fields:
    - id: Unique identifier for referencing the task
    - description: Human-readable summary
    - status: Current lifecycle state
    - result: Final output (when completed)
    - thread: Background thread (if running async)
    """
    id: str
    description: str
    agent_type: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class TaskManager:
    """
    Manages subagent tasks with background execution support.

    Capabilities:
    1. Synchronous execution (v7 behavior) - blocks until complete
    2. Background execution (new in v8) - returns immediately with task_id
    3. Status monitoring - list all tasks
    4. Result retrieval - wait for or check background tasks

    Thread safety:
    - Uses a lock for task dictionary access
    - Each background task runs in its own thread

    Example flow:
        task_id = manager.start_task(..., background=True)  # Returns immediately
        # ... do other work ...
        result = manager.get_output(task_id)  # Waits for completion
    """

    def __init__(self):
        self.tasks: dict[str, SubagentTask] = {}
        self._lock = threading.Lock()

    def _generate_id(self) -> str:
        """Generate a short, memorable task ID."""
        return f"task-{uuid.uuid4().hex[:6]}"

    def start_task(
        self,
        description: str,
        prompt: str,
        agent_type: str,
        background: bool = False,
        task_id: str = None
    ) -> tuple[str, Optional[str]]:
        """
        Start a new subagent task.

        Args:
            description: Short description (3-5 words)
            prompt: Detailed instructions
            agent_type: One of "explore", "code", "plan"
            background: If True, run async and return immediately
            task_id: Optional custom ID

        Returns:
            (task_id, result_or_none)
            - Synchronous: (task_id, result_string)
            - Background: (task_id, None)
        """
        task_id = task_id or self._generate_id()

        task = SubagentTask(
            id=task_id,
            description=description,
            agent_type=agent_type,
            prompt=prompt,
        )

        with self._lock:
            self.tasks[task_id] = task

        if background:
            # Start in background thread
            thread = threading.Thread(
                target=self._execute_task,
                args=(task_id,),
                daemon=True,
                name=f"subagent-{task_id}"
            )
            task.thread = thread
            task.status = TaskStatus.RUNNING
            thread.start()

            return task_id, None
        else:
            # Synchronous execution (v7 behavior)
            return task_id, self._execute_task(task_id)

    def _execute_task(self, task_id: str) -> str:
        """
        Execute a subagent task.

        This is the actual agent loop for the subagent.
        Runs in the calling thread (sync) or a background thread (async).
        """
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: Task {task_id} not found"

        task.status = TaskStatus.RUNNING

        # Get agent configuration
        config = AGENT_TYPES.get(task.agent_type)
        if not config:
            task.status = TaskStatus.FAILED
            task.error = f"Unknown agent type: {task.agent_type}"
            return task.error

        sub_system = f"""You are a {task.agent_type} subagent at {WORKDIR}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

        sub_tools = get_tools_for_agent(task.agent_type)
        sub_messages = [{"role": "user", "content": task.prompt}]

        # Progress indicator for background tasks
        is_background = task.thread is not None

        try:
            tool_count = 0
            start_time = time.time()

            while True:
                log_api_call(f"subagent:{task.agent_type}:{task_id}", sub_system, sub_messages, sub_tools)

                response = client.messages.create(
                    model=MODEL,
                    system=sub_system,
                    messages=sub_messages,
                    tools=sub_tools,
                    max_tokens=8000,
                )

                log_api_response(f"subagent:{task.agent_type}:{task_id}", response)

                if response.stop_reason != "tool_use":
                    # Task completed
                    result = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            result += block.text

                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()

                    if not is_background:
                        elapsed = time.time() - start_time
                        print(f"  [{task.agent_type}] {task.description} - done ({tool_count} tools, {elapsed:.1f}s)")

                    return result

                # Execute tool calls
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                results = []

                for tc in tool_calls:
                    tool_count += 1
                    output = execute_tool_for_subagent(tc.name, tc.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": output
                    })

                sub_messages.append({"role": "assistant", "content": response.content})
                sub_messages.append({"role": "user", "content": results})

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            return f"Error: {e}"

    def get_output(self, task_id: str, wait: bool = True, timeout: float = 300) -> str:
        """
        Get output from a task.

        Args:
            task_id: Task identifier
            wait: If True, block until task completes
            timeout: Max seconds to wait (default 5 minutes)

        Returns:
            Task result or status message
        """
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found. Use TaskList to see available tasks."

        # Already completed?
        if task.status == TaskStatus.COMPLETED:
            return task.result

        if task.status == TaskStatus.FAILED:
            return f"Task failed: {task.error}"

        if task.status == TaskStatus.PENDING:
            return f"Task '{task_id}' is pending (not started yet)"

        # Task is running
        if not wait:
            elapsed = time.time() - task.created_at
            return f"Task '{task_id}' is still running ({elapsed:.1f}s elapsed)"

        # Wait for completion
        if task.thread and task.thread.is_alive():
            task.thread.join(timeout=timeout)

        # Check result after waiting
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            return f"Task failed: {task.error}"
        else:
            return f"Task '{task_id}' timed out after {timeout}s"

    def list_tasks(self) -> str:
        """
        List all tasks with their status.

        Returns formatted table of tasks.
        """
        if not self.tasks:
            return "No tasks."

        lines = ["Task ID       | Status     | Type    | Description"]
        lines.append("-" * 60)

        for task_id, task in sorted(self.tasks.items(), key=lambda x: x[1].created_at):
            elapsed = ""
            if task.status == TaskStatus.RUNNING:
                elapsed = f" ({time.time() - task.created_at:.1f}s)"
            elif task.completed_at:
                elapsed = f" ({task.completed_at - task.created_at:.1f}s)"

            lines.append(
                f"{task_id:13} | {task.status.value:10} | {task.agent_type:7} | "
                f"{task.description[:25]}{elapsed}"
            )

        return "\n".join(lines)

    def clear_completed(self):
        """Remove completed and failed tasks from the list."""
        with self._lock:
            to_remove = [
                tid for tid, task in self.tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            for tid in to_remove:
                del self.tasks[tid]


# Global task manager
TASKS = TaskManager()


# =============================================================================
# Specialized Tools (from v6)
# =============================================================================

EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    '.idea', '.vscode', 'dist', 'build', 'coverage', '.pytest_cache'
}


def run_glob(pattern: str, path: str = None) -> str:
    """See v6 for details."""
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
        paths = [str(p.relative_to(WORKDIR)) for _, p in results[:100]]
        if not paths:
            return "No matches found."
        output = "\n".join(paths)
        if len(results) > 100:
            output += f"\n... and {len(results) - 100} more"
        return output
    except Exception as e:
        return f"Error: {e}"


def run_grep(pattern: str, path: str = None, glob_pattern: str = None,
             output_mode: str = "files_with_matches", context: int = 0) -> str:
    """See v6 for details."""
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

            if len(results) >= 100:
                break

        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error: {e}"


def run_read_file(file_path: str, offset: int = 1, limit: int = 2000) -> str:
    """See v6 for details."""
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
                return r.stdout[:50000] or "(PDF empty)"
            except FileNotFoundError:
                return "Error: pdftotext not installed"

        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
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
            result += f"\n\n... {total - end} more lines."
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
        for folder, label in [("scripts", "Scripts"), ("references", "References")]:
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
# Agent Type Registry
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
        "prompt": "You are a planning agent. Analyze and output a numbered implementation plan. Do NOT make changes.",
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
# Tool Definitions - Updated for v8
# =============================================================================

GLOB_TOOL = {
    "name": "Glob",
    "description": "Find files matching a glob pattern. Returns paths sorted by modification time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Base directory"}
        },
        "required": ["pattern"],
    },
}

GREP_TOOL = {
    "name": "Grep",
    "description": "Search file contents with regex. Output modes: files_with_matches, content, count.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
        },
        "required": ["pattern"],
    },
}

READ_TOOL = {
    "name": "Read",
    "description": "Read file with line numbers. Supports offset/limit for large files.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"}
        },
        "required": ["file_path"],
    },
}

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command. Use for builds, git, scripts.",
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
        "description": "Replace exact text in file.",
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

# Enhanced Task tool (v8)
TASK_TOOL = {
    "name": "Task",
    "description": f"""Spawn a subagent for a focused subtask.

Agent types:
{get_agent_descriptions()}

NEW in v8 - Background execution:
  background=false (default): Blocks until complete, returns result
  background=true: Returns immediately with task_id

Example parallel execution:
  Task(description="explore auth", ..., background=true)   -> task-abc
  Task(description="explore tests", ..., background=true)  -> task-def
  # Later:
  TaskOutput(task_id="task-abc")  -> auth results
  TaskOutput(task_id="task-def")  -> tests results""",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short description (3-5 words)"},
            "prompt": {"type": "string", "description": "Detailed instructions"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
            "background": {
                "type": "boolean",
                "description": "Run in background (returns task_id immediately)"
            },
            "task_id": {"type": "string", "description": "Optional custom task ID"},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

# NEW in v8: TaskOutput tool
TASK_OUTPUT_TOOL = {
    "name": "TaskOutput",
    "description": """Get output from a background task.

Args:
  task_id: The task ID returned by Task(..., background=true)
  wait: If true (default), block until task completes
        If false, return current status immediately

Example:
  TaskOutput(task_id="task-abc")            -> Waits and returns result
  TaskOutput(task_id="task-abc", wait=false) -> Returns status if still running""",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task ID to get output from"},
            "wait": {"type": "boolean", "description": "Wait for completion (default: true)"},
            "timeout": {"type": "number", "description": "Max seconds to wait (default: 300)"},
        },
        "required": ["task_id"],
    },
}

# NEW in v8: TaskList tool
TASK_LIST_TOOL = {
    "name": "TaskList",
    "description": "List all subagent tasks and their status (pending, running, completed, failed).",
    "input_schema": {
        "type": "object",
        "properties": {},
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

ENTER_PLAN_TOOL = {
    "name": "EnterPlanMode",
    "description": "Enter planning mode (read-only) for complex tasks.",
    "input_schema": {"type": "object", "properties": {}},
}

EXIT_PLAN_TOOL = {
    "name": "ExitPlanMode",
    "description": "Submit implementation plan for user review.",
    "input_schema": {
        "type": "object",
        "properties": {"plan": {"type": "string", "description": "Implementation plan"}},
        "required": ["plan"],
    },
}

SPECIALIZED_TOOLS = [GLOB_TOOL, GREP_TOOL, READ_TOOL]
PLAN_TOOLS = [ENTER_PLAN_TOOL, EXIT_PLAN_TOOL]
TASK_TOOLS = [TASK_TOOL, TASK_OUTPUT_TOOL, TASK_LIST_TOOL]
ALL_TOOLS = SPECIALIZED_TOOLS + BASE_TOOLS + TASK_TOOLS + [SKILL_TOOL] + PLAN_TOOLS


def get_tools_for_agent(agent_type: str) -> list:
    """Get tools available for a subagent type."""
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return SPECIALIZED_TOOLS + BASE_TOOLS
    return [t for t in (SPECIALIZED_TOOLS + BASE_TOOLS) if t["name"] in allowed]


# =============================================================================
# System Prompt
# =============================================================================

def get_system_prompt() -> str:
    base = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Specialized Tools**: Glob, Grep, Read (prefer for file operations)
**General Tools**: bash, write_file, edit_file
**Task Management** (v8):
  - Task: Spawn subagent (supports background=true for parallel execution)
  - TaskOutput: Get result from background task
  - TaskList: List all tasks
**Planning**: EnterPlanMode / ExitPlanMode

**Skills**: {SKILLS.get_descriptions()}

Rules:
- For parallel analysis: use Task(..., background=true) then TaskOutput()
- For complex tasks: use EnterPlanMode first
- Use Glob/Grep/Read instead of bash for file operations
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
        return "Error: Command timed out"
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
    return f'<skill-loaded name="{skill_name}">\n{content}\n</skill-loaded>'


def run_task(description: str, prompt: str, agent_type: str,
             background: bool = False, task_id: str = None) -> str:
    """Execute or start a subagent task."""
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    tid, result = TASKS.start_task(
        description=description,
        prompt=prompt,
        agent_type=agent_type,
        background=background,
        task_id=task_id
    )

    if background:
        return f"Started background task: {tid}\n\nUse TaskOutput(task_id=\"{tid}\") to get results."
    else:
        return result


def run_task_output(task_id: str, wait: bool = True, timeout: float = 300) -> str:
    """Get output from a task."""
    return TASKS.get_output(task_id, wait=wait, timeout=timeout)


def run_task_list() -> str:
    """List all tasks."""
    return TASKS.list_tasks()


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation (main agent)."""
    # Specialized tools
    if name == "Glob":
        return run_glob(args["pattern"], args.get("path"))
    if name == "Grep":
        return run_grep(args["pattern"], args.get("path"), args.get("glob"),
                        args.get("output_mode", "files_with_matches"), args.get("context", 0))
    if name == "Read":
        return run_read_file(args["file_path"], args.get("offset", 1), args.get("limit", 2000))

    # Plan tools
    if name == "EnterPlanMode":
        return MODE.enter_plan_mode()
    if name == "ExitPlanMode":
        return MODE.exit_plan_mode(args["plan"])

    # Task tools (v8)
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"],
                        args.get("background", False), args.get("task_id"))
    if name == "TaskOutput":
        return run_task_output(args["task_id"], args.get("wait", True), args.get("timeout", 300))
    if name == "TaskList":
        return run_task_list()

    # Base tools
    if name == "bash":
        return run_bash(args["command"])
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])
    if name == "Skill":
        return run_skill(args["skill"])

    return f"Unknown tool: {name}"


def execute_tool_for_subagent(name: str, args: dict) -> str:
    """
    Tool execution for subagents.

    Subagents have limited tools and no access to Task/TaskOutput
    to prevent infinite spawning.
    """
    # Specialized tools
    if name == "Glob":
        return run_glob(args["pattern"], args.get("path"))
    if name == "Grep":
        return run_grep(args["pattern"], args.get("path"), args.get("glob"),
                        args.get("output_mode", "files_with_matches"), 0)
    if name == "Read":
        return run_read_file(args["file_path"], args.get("offset", 1), args.get("limit", 2000))

    # Base tools (limited for subagents)
    if name == "bash":
        return run_bash(args["command"])
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])

    return f"Tool not available for subagent: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================

def agent_loop(messages: list) -> list:
    """Main agent loop with background task support."""
    while True:
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
                bg = tc.input.get("background", False)
                mode = " (background)" if bg else ""
                print(f"\n> Task{mode}: {tc.input.get('description', 'subtask')}")
            elif tc.name == "TaskOutput":
                print(f"\n> Getting output: {tc.input.get('task_id', '?')}")
            elif tc.name == "TaskList":
                print("\n> Listing tasks...")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Display output
            if tc.name in ("EnterPlanMode", "ExitPlanMode", "TaskList"):
                print(output)
            elif tc.name == "Task" and tc.input.get("background"):
                print(f"  {output}")
            elif tc.name == "TaskOutput":
                if len(output) > 500:
                    print(f"  {output[:500]}...")
                else:
                    print(f"  {output}")
            elif tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name == "Glob":
                print(f"  Found {len(output.split(chr(10)))} files")
            elif tc.name == "Grep":
                print(f"  {len(output.split(chr(10)))} results")
            elif tc.name == "Read":
                print(f"  Read {len(output.split(chr(10)))} lines")
            elif tc.name not in ("Task",):
                if len(output) > 500:
                    print(f"  {output[:500]}...")
                else:
                    print(f"  {output}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================

def main():
    print(f"Mini Claude Code v8 (with Advanced Subagents) - {WORKDIR}")
    print(f"Task tools: Task (with background), TaskOutput, TaskList")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            mode_indicator = " [PLAN]" if MODE.mode == AgentMode.PLANNING else ""
            user_input = input(f"You{mode_indicator}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Plan approval flow
        if MODE.current_plan is not None:
            action, context = MODE.handle_user_response(user_input)
            if action == "approve":
                print("\n Plan approved. Executing...\n")
                history.append({"role": "user", "content": f"Plan approved. Execute:\n\n{context}"})
                try:
                    agent_loop(history)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue
            elif action == "cancel":
                print("\n Plan cancelled.\n")
                continue
            elif action == "revise":
                print(f"\n Revising: {context}\n")
                history.append({"role": "user", "content": f"Revise plan: {context}"})
                try:
                    agent_loop(history)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
