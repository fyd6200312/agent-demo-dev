#!/usr/bin/env python3
"""
v9_context_management.py - Mini Claude Code: Context Management (~1150 lines)

Core Philosophy: "Forgetting is a Feature"
==========================================
v8 gave us parallel subagents. But there's a resource question:

    What happens when the conversation gets LONG?

Watch a v8 session after 30 minutes:

    Turn 1:  User query                        500 tokens
    Turn 5:  + 4 rounds of tools             5,000 tokens
    Turn 10: + exploration results          15,000 tokens
    Turn 20: + code edits and outputs       40,000 tokens
    Turn 30: + more exploration             80,000 tokens
    Turn 40: CONTEXT OVERFLOW! Model forgets early decisions.

The conversation history grows MONOTONICALLY. Every tool result,
every model response, every user message stays forever. Eventually
the context window fills up and critical early information is lost.

The Problem - Unbounded Context Growth:
--------------------------------------
    Token Usage Over Time:

    80K |                                         X OVERFLOW
        |                                    X
    60K |                               X
        |                          X
    40K |                     X
        |                X
    20K |           X
        |      X
     0K | X
        +-----+-----+-----+-----+-----+-----+
          0     5    10    15    20    25    30  turns

Every turn adds tokens, nothing is removed.

    | Content Type     | Tokens | Useful?                    |
    |------------------|--------|----------------------------|
    | Early exploration| 5000   | Partially (some facts)     |
    | Failed attempts  | 3000   | No (already abandoned)     |
    | Tool outputs     | 15000  | Partially (raw data)       |
    | Final code       | 2000   | Yes (current state)        |

Most tokens are HISTORICAL DETAIL, not current relevance.

The Solution - Three-Layer Context:
----------------------------------
    +------------------------------------------------------------------+
    |               Context Management Architecture                    |
    +------------------------------------------------------------------+
    |                                                                  |
    |  Layer 1: Working Context (current conversation)                 |
    |  ┌─────────────────────────────────────────────────────────────┐ |
    |  │  Recent messages (full detail)                              │ |
    |  │  Last 6-8 turns of conversation                             │ |
    |  │  Token budget: ~30K                                         │ |
    |  └─────────────────────────────────────────────────────────────┘ |
    |                              |                                   |
    |                  [auto-compress when > threshold]                 |
    |                              v                                   |
    |  Layer 2: Condensed History (summaries)                          |
    |  ┌─────────────────────────────────────────────────────────────┐ |
    |  │  <summary turns="1-10">                                     │ |
    |  │    User asked to refactor auth. Explored 5 files.           │ |
    |  │    Found: JWT used in login.py, session.py.                 │ |
    |  │    Decision: migrate to refresh tokens.                     │ |
    |  │  </summary>                                                 │ |
    |  │                                                             │ |
    |  │  ~500 tokens (compressed from ~15000)                       │ |
    |  └─────────────────────────────────────────────────────────────┘ |
    |                              |                                   |
    |                  [extract important facts]                       |
    |                              v                                   |
    |  Layer 3: Persistent Memory (MEMORY.md)                          |
    |  ┌─────────────────────────────────────────────────────────────┐ |
    |  │  # Project Memory                                          │ |
    |  │  ## Architecture                                            │ |
    |  │  - Auth uses JWT with refresh tokens                        │ |
    |  │  ## Patterns                                                │ |
    |  │  - Tests in tests/ follow pytest conventions                │ |
    |  │  ## User Preferences                                       │ |
    |  │  - Prefers functional style                                 │ |
    |  │                                                             │ |
    |  │  Survives across sessions!                                  │ |
    |  └─────────────────────────────────────────────────────────────┘ |
    |                                                                  |
    +------------------------------------------------------------------+

Key Insight - Lossy Compression Preserves Decisions:
---------------------------------------------------
Summarization is LOSSY - we lose raw tool output.
But we keep DECISIONS and FACTS, which is what matters.

    Raw (15000 tokens):
        cat auth/login.py -> [200 lines of code]
        cat auth/session.py -> [150 lines of code]
        grep "jwt" -> [30 matches in 5 files]
        "I see that JWT is used in login.py and session.py..."

    Summary (300 tokens):
        "Explored auth system. JWT used in login.py (token creation)
         and session.py (validation). 5 files reference jwt."

The summary preserves the CONCLUSION while discarding the RAW DATA.
The model can always re-read files if it needs details.

MEMORY.md - Cross-Session Knowledge:
------------------------------------
The real insight: some knowledge transcends sessions.

    Session 1: "Auth uses JWT" (discovered through exploration)
    Session 2: "Auth uses JWT" (loaded from MEMORY.md instantly)

Without MEMORY.md, the model re-discovers the same facts every session.
With MEMORY.md, knowledge accumulates.

Usage:
    python v9_context_management.py
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
MEMORY_DIR = WORKDIR / ".claude"

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# NEW in v9: Context Compressor
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (~4 chars per token for English).

    Production systems use tiktoken or the API's token counter.
    This is a simple heuristic that's good enough for compression decisions.
    """
    return len(text) // 4


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens in message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += estimate_tokens(json.dumps(item, default=str))
    return total


class ContextCompressor:
    """
    Automatic context compression via summarization.

    How it works:
    1. Monitor message count and token usage
    2. When threshold is exceeded, compress old messages
    3. Keep recent messages (last N turns) intact
    4. Replace old messages with a summary

    The summary becomes a special first message that provides context
    without consuming excessive tokens.

    Why not just truncate?
    - Truncation loses information randomly
    - Summarization preserves DECISIONS and FACTS
    - The model can re-read files if it needs raw data

    Configuration:
        compress_threshold: Token count that triggers compression
        keep_recent: Number of recent messages to keep intact
    """

    def __init__(self, compress_threshold: int = 25000, keep_recent: int = 8):
        self.compress_threshold = compress_threshold
        self.keep_recent = keep_recent
        self.compression_count = 0

    def should_compress(self, messages: list) -> bool:
        """Check if compression is needed."""
        return estimate_messages_tokens(messages) > self.compress_threshold

    def compress(self, messages: list) -> list:
        """
        Compress message history.

        Steps:
        1. Split messages into [old | recent]
        2. Summarize old messages using a lightweight API call
        3. Return [summary_message | recent]

        The summary preserves:
        - User requests and decisions
        - Key findings and facts
        - Current state of the task
        """
        if len(messages) <= self.keep_recent:
            return messages

        to_compress = messages[:-self.keep_recent]
        to_keep = messages[-self.keep_recent:]

        # Generate summary
        summary = self._generate_summary(to_compress)
        self.compression_count += 1

        # Build compressed message list
        # Summary goes as a user message at the start
        compressed = [
            {
                "role": "user",
                "content": f"""<conversation-summary turns="1-{len(to_compress)}">
{summary}
</conversation-summary>

[Previous conversation compressed. Key context preserved above.]"""
            },
            {
                "role": "assistant",
                "content": "Understood. I have the context from our previous conversation. Continuing..."
            }
        ]

        compressed.extend(to_keep)
        return compressed

    def _generate_summary(self, messages: list) -> str:
        """
        Summarize old messages using a fast, cheap model.

        Uses claude-3-haiku for cost efficiency.
        Falls back to simple extraction if API fails.
        """
        # Simplify messages for summarization
        simplified = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, list):
                # Extract text from tool results
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            parts.append(str(item["text"])[:300])
                        elif "content" in item:
                            parts.append(str(item["content"])[:300])
                content = " | ".join(parts)

            if isinstance(content, str):
                simplified.append(f"{role}: {content[:500]}")

        conversation_text = "\n".join(simplified[-30:])  # Last 30 entries max

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": f"""Summarize this conversation concisely. Preserve:
1. What the user requested
2. Key decisions made
3. Important findings and facts
4. Current state of the task
5. Files that were modified

Conversation:
{conversation_text}

Output ONLY the summary, no preamble."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            # Fallback: simple extraction
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: list) -> str:
        """Extract key information without API call."""
        user_messages = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), str):
                user_messages.append(msg["content"][:200])

        return "Previous conversation topics:\n" + "\n".join(
            f"- {m}" for m in user_messages[:10]
        )


# =============================================================================
# NEW in v9: Memory Manager (MEMORY.md)
# =============================================================================

class MemoryManager:
    """
    Persistent project memory via MEMORY.md.

    MEMORY.md is a structured markdown file that persists across sessions.
    The model can write facts here, and they'll be loaded in future sessions.

    File format:
        # Project Memory

        ## Architecture
        - Auth uses JWT with refresh tokens
        - Database is PostgreSQL via SQLAlchemy

        ## Patterns
        - Tests follow pytest conventions
        - Components use factory pattern

        ## User Preferences
        - Prefers functional style
        - Uses 4-space indentation

        ## Known Issues
        - Login timeout under high load (#123)

    Why structured sections?
    - Model can write to specific sections
    - Easy to read and edit manually
    - Prevents duplicate entries
    """

    VALID_SECTIONS = [
        "Architecture",
        "Patterns",
        "User Preferences",
        "Known Issues",
        "Dependencies",
    ]

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_file = memory_dir / "MEMORY.md"
        self.sections: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        """Load existing memory from file."""
        if not self.memory_file.exists():
            self.sections = {s: [] for s in self.VALID_SECTIONS}
            return

        content = self.memory_file.read_text()
        current_section = None

        for line in content.splitlines():
            if line.startswith("## "):
                current_section = line[3:].strip()
                if current_section not in self.sections:
                    self.sections[current_section] = []
            elif current_section and line.startswith("- "):
                fact = line[2:].strip()
                if fact:
                    self.sections.setdefault(current_section, []).append(fact)

        # Ensure all valid sections exist
        for s in self.VALID_SECTIONS:
            if s not in self.sections:
                self.sections[s] = []

    def add(self, section: str, fact: str) -> str:
        """
        Add a fact to memory.

        Returns status message.
        """
        if section not in self.sections:
            available = ", ".join(self.sections.keys())
            return f"Error: Unknown section '{section}'. Available: {available}"

        fact = fact.strip()
        if not fact:
            return "Error: Fact cannot be empty"

        # Check for duplicates
        if fact in self.sections[section]:
            return f"Already exists in {section}: {fact}"

        self.sections[section].append(fact)
        self._save()
        return f"Saved to {section}: {fact}"

    def remove(self, section: str, fact: str) -> str:
        """Remove a fact from memory."""
        if section not in self.sections:
            return f"Error: Unknown section '{section}'"

        if fact in self.sections[section]:
            self.sections[section].remove(fact)
            self._save()
            return f"Removed from {section}: {fact}"

        return f"Not found in {section}: {fact}"

    def get_context(self) -> str:
        """
        Generate context string for injection into system prompt.

        Only includes sections with content.
        Returns empty string if no memories.
        """
        populated = {s: facts for s, facts in self.sections.items() if facts}

        if not populated:
            return ""

        lines = ["<project-memory>"]
        for section, facts in populated.items():
            lines.append(f"\n## {section}")
            for fact in facts:
                lines.append(f"- {fact}")
        lines.append("\n</project-memory>")

        return "\n".join(lines)

    def read_all(self) -> str:
        """Read all memory contents."""
        if not any(self.sections.values()):
            return "No memories stored yet."

        lines = ["# Project Memory", ""]
        for section, facts in self.sections.items():
            lines.append(f"## {section}")
            if facts:
                for fact in facts:
                    lines.append(f"- {fact}")
            else:
                lines.append("(empty)")
            lines.append("")

        return "\n".join(lines)

    def _save(self):
        """Write memory to file."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        lines = ["# Project Memory", ""]
        for section in self.VALID_SECTIONS:
            lines.append(f"## {section}")
            for fact in self.sections.get(section, []):
                lines.append(f"- {fact}")
            lines.append("")

        # Include any custom sections
        for section, facts in self.sections.items():
            if section not in self.VALID_SECTIONS and facts:
                lines.append(f"## {section}")
                for fact in facts:
                    lines.append(f"- {fact}")
                lines.append("")

        self.memory_file.write_text("\n".join(lines))


# Global instances
COMPRESSOR = ContextCompressor()
MEMORY = MemoryManager(MEMORY_DIR)


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
    """See v5."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.session_grants: dict[str, bool] = {}
        self.rules = {
            "read": [PermissionRule("*", Permission.ALLOW, "Safe")],
            "write": [
                PermissionRule("*.md", Permission.ALLOW, "Docs"),
                PermissionRule("*.env*", Permission.ASK, "Env"),
                PermissionRule("*secret*", Permission.DENY, "Secrets"),
                PermissionRule("*.key", Permission.DENY, "Keys"),
                PermissionRule("*", Permission.ASK, "Write"),
            ],
            "exec": [
                PermissionRule("ls*", Permission.ALLOW, "List"),
                PermissionRule("pwd", Permission.ALLOW, "Cwd"),
                PermissionRule("echo *", Permission.ALLOW, "Print"),
                PermissionRule("git status*", Permission.ALLOW, "Git"),
                PermissionRule("git diff*", Permission.ALLOW, "Git"),
                PermissionRule("git log*", Permission.ALLOW, "Git"),
                PermissionRule("rm -rf /*", Permission.DENY, "Danger"),
                PermissionRule("sudo *", Permission.DENY, "Sudo"),
                PermissionRule("*| bash*", Permission.DENY, "Pipe"),
                PermissionRule("*", Permission.ASK, "Unknown"),
            ],
        }

    def check(self, category: str, operation: str) -> tuple[Permission, str]:
        cache_key = f"{category}:{self._normalize(operation)}"
        if cache_key in self.session_grants:
            return Permission.ALLOW, "Session grant"
        for rule in self.rules.get(category, []):
            if fnmatch.fnmatch(operation, rule.pattern):
                return rule.permission, rule.reason
        return Permission.ASK, "Default"

    def grant_session(self, category: str, operation: str):
        self.session_grants[f"{category}:{self._normalize(operation)}"] = True

    def _normalize(self, op: str) -> str:
        parts = op.split()
        if len(parts) > 1:
            return f"{parts[0]} *"
        return op


def ask_user_permission(operation: str, reason: str, category: str) -> tuple[bool, bool]:
    print(f"\n┌─ Permission Required {'─' * 30}┐")
    print(f"│ {category}: {operation[:50]}")
    print(f"│ [y] Allow  [n] Deny  [a] Always")
    while True:
        r = input("└─> ").strip().lower()
        if r in ("y", "yes"):
            return True, False
        if r in ("n", "no"):
            return False, False
        if r in ("a", "always"):
            return True, True


PERMISSIONS = PermissionManager(WORKDIR)


# =============================================================================
# Plan Mode (from v7)
# =============================================================================

class AgentMode(Enum):
    NORMAL = "normal"
    PLANNING = "planning"


class ModeManager:
    """See v7."""

    def __init__(self):
        self.mode = AgentMode.NORMAL
        self.current_plan: str | None = None

    def enter_plan_mode(self) -> str:
        if self.mode == AgentMode.PLANNING:
            return "Already in plan mode."
        self.mode = AgentMode.PLANNING
        self.current_plan = None
        return "Entered PLAN MODE. Read-only tools only. Call ExitPlanMode when ready."

    def exit_plan_mode(self, plan: str) -> str:
        if self.mode != AgentMode.PLANNING:
            return "Error: Not in plan mode."
        self.current_plan = plan
        return f"\n{'='*60}\nIMPLEMENTATION PLAN\n{'='*60}\n\n{plan}\n\n{'='*60}\n\"approve\" / \"revise: ...\" / \"cancel\""

    def handle_user_response(self, user_input: str) -> tuple[str, str | None]:
        if self.current_plan is None:
            return "not_pending", None
        n = user_input.strip().lower()
        if n in ("approve", "yes", "y", "ok"):
            plan = self.current_plan
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "approve", plan
        if n in ("cancel", "no", "n"):
            self.mode = AgentMode.NORMAL
            self.current_plan = None
            return "cancel", None
        notes = user_input[7:].strip() if user_input.lower().startswith("revise") else user_input
        self.current_plan = None
        return "revise", notes

    def get_available_tools(self, all_tools: list) -> list:
        if self.mode == AgentMode.PLANNING:
            ok = {"Glob", "Grep", "Read", "TodoWrite", "ExitPlanMode"}
            return [t for t in all_tools if t["name"] in ok]
        return all_tools

    def get_mode_prompt(self) -> str:
        if self.mode == AgentMode.PLANNING:
            return "\n\n*** PLAN MODE - Read-only tools only ***"
        return ""


MODE = ModeManager()


# =============================================================================
# Task Manager (from v8)
# =============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubagentTask:
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
    """See v8."""

    def __init__(self):
        self.tasks: dict[str, SubagentTask] = {}
        self._lock = threading.Lock()

    def start_task(self, description: str, prompt: str, agent_type: str,
                   background: bool = False, task_id: str = None) -> tuple[str, Optional[str]]:
        task_id = task_id or f"task-{uuid.uuid4().hex[:6]}"
        task = SubagentTask(id=task_id, description=description,
                            agent_type=agent_type, prompt=prompt)
        with self._lock:
            self.tasks[task_id] = task

        if background:
            thread = threading.Thread(target=self._execute, args=(task_id,), daemon=True)
            task.thread = thread
            task.status = TaskStatus.RUNNING
            thread.start()
            return task_id, None
        return task_id, self._execute(task_id)

    def _execute(self, task_id: str) -> str:
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        config = AGENT_TYPES.get(task.agent_type)
        if not config:
            task.status = TaskStatus.FAILED
            task.error = f"Unknown type: {task.agent_type}"
            return task.error

        sub_system = f"You are a {task.agent_type} subagent at {WORKDIR}.\n{config['prompt']}\nReturn a concise summary."
        sub_tools = get_tools_for_agent(task.agent_type)
        sub_messages = [{"role": "user", "content": task.prompt}]

        try:
            while True:
                response = client.messages.create(model=MODEL, system=sub_system,
                                                  messages=sub_messages, tools=sub_tools, max_tokens=8000)
                if response.stop_reason != "tool_use":
                    result = "".join(b.text for b in response.content if hasattr(b, "text"))
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    return result

                results = []
                for tc in [b for b in response.content if b.type == "tool_use"]:
                    output = execute_tool_for_subagent(tc.name, tc.input)
                    results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})
                sub_messages.append({"role": "assistant", "content": response.content})
                sub_messages.append({"role": "user", "content": results})
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return f"Error: {e}"

    def get_output(self, task_id: str, wait: bool = True, timeout: float = 300) -> str:
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found"
        if task.status == TaskStatus.COMPLETED:
            return task.result
        if task.status == TaskStatus.FAILED:
            return f"Failed: {task.error}"
        if not wait:
            return f"Still running ({time.time() - task.created_at:.1f}s)"
        if task.thread and task.thread.is_alive():
            task.thread.join(timeout=timeout)
        if task.status == TaskStatus.COMPLETED:
            return task.result
        return f"Timed out after {timeout}s"

    def list_tasks(self) -> str:
        if not self.tasks:
            return "No tasks."
        lines = ["ID            | Status     | Type    | Description"]
        lines.append("-" * 60)
        for t in sorted(self.tasks.values(), key=lambda x: x.created_at):
            lines.append(f"{t.id:13} | {t.status.value:10} | {t.agent_type:7} | {t.description[:25]}")
        return "\n".join(lines)


TASKS = TaskManager()


# =============================================================================
# Specialized Tools (from v6)
# =============================================================================

EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.idea', 'dist', 'build'}


def run_glob(pattern: str, path: str = None) -> str:
    try:
        base = (WORKDIR / path).resolve() if path else WORKDIR
        if not base.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"
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
        return "\n".join(paths) if paths else "No matches."
    except Exception as e:
        return f"Error: {e}"


def run_grep(pattern: str, path: str = None, glob_pattern: str = None,
             output_mode: str = "files_with_matches", context: int = 0) -> str:
    try:
        base = (WORKDIR / path).resolve() if path else WORKDIR
        if not base.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"
        regex = re.compile(pattern)
        results = []
        for fp in base.glob(glob_pattern or "**/*"):
            if not fp.is_file() or any(ex in fp.parts for ex in EXCLUDE_DIRS):
                continue
            try:
                content = fp.read_text(errors='ignore')
            except Exception:
                continue
            matches = [(i+1, l) for i, l in enumerate(content.splitlines()) if regex.search(l)]
            if matches:
                rel = str(fp.relative_to(WORKDIR))
                if output_mode == "files_with_matches":
                    results.append(rel)
                elif output_mode == "count":
                    results.append(f"{rel}: {len(matches)}")
                else:
                    results.append(f"\n{rel}:")
                    for n, l in matches[:10]:
                        results.append(f"  {n:>5}: {l[:200]}")
            if len(results) >= 100:
                break
        return "\n".join(results) if results else "No matches."
    except re.error as e:
        return f"Invalid regex: {e}"
    except Exception as e:
        return f"Error: {e}"


def run_read_file(file_path: str, offset: int = 1, limit: int = 2000) -> str:
    try:
        fp = (WORKDIR / file_path).resolve()
        if not fp.is_relative_to(WORKDIR) or not fp.exists():
            return f"Error: Cannot read {file_path}"
        if fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
            return f"[Image: {file_path}]"
        content = fp.read_text()
        lines = content.splitlines()
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        out = [f"{start+i+1:>6}| {l[:500]}" for i, l in enumerate(lines[start:end])]
        result = "\n".join(out)
        if end < len(lines):
            result += f"\n\n... {len(lines)-end} more lines."
        return result
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# SkillLoader (from v4)
# =============================================================================

class SkillLoader:
    """See v4."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        content = path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None
        fm, body = match.groups()
        meta = {}
        for line in fm.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("\"'")
        if "name" not in meta or "description" not in meta:
            return None
        return {"name": meta["name"], "description": meta["description"],
                "body": body.strip(), "path": path, "dir": path.parent}

    def load_skills(self):
        if not self.skills_dir.exists():
            return
        for d in self.skills_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                s = self.parse_skill_md(d / "SKILL.md")
                if s:
                    self.skills[s["name"]] = s

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(none)"
        return "\n".join(f"- {n}: {s['description']}" for n, s in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        if name not in self.skills:
            return None
        s = self.skills[name]
        return f"# Skill: {s['name']}\n\n{s['body']}"

    def list_skills(self) -> list:
        return list(self.skills.keys())


SKILLS = SkillLoader(SKILLS_DIR)

# =============================================================================
# Agent Types, TodoManager
# =============================================================================

AGENT_TYPES = {
    "explore": {
        "description": "Read-only exploration",
        "tools": ["Glob", "Grep", "Read", "bash"],
        "prompt": "Explore and analyze. Return a concise summary.",
    },
    "code": {
        "description": "Full implementation",
        "tools": "*",
        "prompt": "Implement changes efficiently.",
    },
    "plan": {
        "description": "Planning and design",
        "tools": ["Glob", "Grep", "Read"],
        "prompt": "Analyze and output a numbered plan. Do NOT modify files.",
    },
}


def get_agent_descriptions() -> str:
    return "\n".join(f"- {n}: {c['description']}" for n, c in AGENT_TYPES.items())


class TodoManager:
    """See v2."""
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated, ip = [], 0
        for i, item in enumerate(items):
            c = str(item.get("content", "")).strip()
            s = str(item.get("status", "pending")).lower()
            a = str(item.get("activeForm", "")).strip()
            if not c or not a:
                raise ValueError(f"Item {i}: fields required")
            if s not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status")
            if s == "in_progress":
                ip += 1
            validated.append({"content": c, "status": s, "activeForm": a})
        if ip > 1:
            raise ValueError("Only one in_progress")
        self.items = validated[:20]
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for t in self.items:
            m = "[x]" if t["status"] == "completed" else "[>]" if t["status"] == "in_progress" else "[ ]"
            lines.append(f"{m} {t['content']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)"


TODO = TodoManager()

# =============================================================================
# Tool Definitions
# =============================================================================

GLOB_TOOL = {"name": "Glob", "description": "Find files by glob pattern.",
             "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}
GREP_TOOL = {"name": "Grep", "description": "Search file contents with regex. Modes: files_with_matches, content, count.",
             "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]}}, "required": ["pattern"]}}
READ_TOOL = {"name": "Read", "description": "Read file with line numbers.",
             "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["file_path"]}}

BASE_TOOLS = [
    {"name": "bash", "description": "Run shell command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "write_file", "description": "Write to file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace text in file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "TodoWrite", "description": "Update task list.", "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["content", "status", "activeForm"]}}}, "required": ["items"]}},
]

TASK_TOOL = {"name": "Task", "description": f"Spawn subagent. Types: {', '.join(AGENT_TYPES)}. Use background=true for parallel.",
             "input_schema": {"type": "object", "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": list(AGENT_TYPES)}, "background": {"type": "boolean"}, "task_id": {"type": "string"}}, "required": ["description", "prompt", "agent_type"]}}
TASK_OUTPUT_TOOL = {"name": "TaskOutput", "description": "Get background task result.",
                    "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}, "wait": {"type": "boolean"}, "timeout": {"type": "number"}}, "required": ["task_id"]}}
TASK_LIST_TOOL = {"name": "TaskList", "description": "List all tasks.", "input_schema": {"type": "object", "properties": {}}}
SKILL_TOOL = {"name": "Skill", "description": f"Load skill. Available: {SKILLS.get_descriptions()}",
              "input_schema": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}}
ENTER_PLAN_TOOL = {"name": "EnterPlanMode", "description": "Enter read-only planning mode.",
                   "input_schema": {"type": "object", "properties": {}}}
EXIT_PLAN_TOOL = {"name": "ExitPlanMode", "description": "Submit plan for approval.",
                  "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}

# NEW in v9: Memory tools
MEMORY_WRITE_TOOL = {
    "name": "MemoryWrite",
    "description": f"""Save a fact to persistent project memory (MEMORY.md).

Facts survive across sessions. Use for important discoveries.

Sections: {', '.join(MemoryManager.VALID_SECTIONS)}

Examples:
  MemoryWrite(section="Architecture", fact="Auth uses JWT")
  MemoryWrite(section="Patterns", fact="Tests follow pytest")""",
    "input_schema": {
        "type": "object",
        "properties": {
            "section": {"type": "string", "enum": MemoryManager.VALID_SECTIONS,
                        "description": "Memory section"},
            "fact": {"type": "string", "description": "Fact to remember"},
        },
        "required": ["section", "fact"],
    },
}

MEMORY_READ_TOOL = {
    "name": "MemoryRead",
    "description": "Read all project memory (MEMORY.md).",
    "input_schema": {"type": "object", "properties": {}},
}

SPECIALIZED_TOOLS = [GLOB_TOOL, GREP_TOOL, READ_TOOL]
PLAN_TOOLS = [ENTER_PLAN_TOOL, EXIT_PLAN_TOOL]
TASK_MGMT_TOOLS = [TASK_TOOL, TASK_OUTPUT_TOOL, TASK_LIST_TOOL]
MEMORY_TOOLS = [MEMORY_WRITE_TOOL, MEMORY_READ_TOOL]
ALL_TOOLS = SPECIALIZED_TOOLS + BASE_TOOLS + TASK_MGMT_TOOLS + [SKILL_TOOL] + PLAN_TOOLS + MEMORY_TOOLS


def get_tools_for_agent(agent_type: str) -> list:
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return SPECIALIZED_TOOLS + BASE_TOOLS
    return [t for t in (SPECIALIZED_TOOLS + BASE_TOOLS) if t["name"] in allowed]


# =============================================================================
# System Prompt - With Memory injection
# =============================================================================

def get_system_prompt() -> str:
    memory_context = MEMORY.get_context()
    memory_section = f"\n\n**Project Memory** (from MEMORY.md):\n{memory_context}" if memory_context else ""

    base = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Tools**: Glob, Grep, Read, bash, write_file, edit_file
**Task Management**: Task (background), TaskOutput, TaskList
**Planning**: EnterPlanMode / ExitPlanMode
**Memory** (v9): MemoryWrite (save facts), MemoryRead (recall)
**Skills**: {SKILLS.get_descriptions()}{memory_section}

Rules:
- Save important discoveries with MemoryWrite (architecture, patterns, preferences)
- For parallel work: Task(..., background=true) then TaskOutput()
- For complex tasks: EnterPlanMode first
- Use Glob/Grep/Read for file operations
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
    perm, reason = PERMISSIONS.check("exec", cmd)
    if perm == Permission.DENY:
        return f"Permission denied: {reason}"
    if perm == Permission.ASK:
        ok, remember = ask_user_permission(cmd, reason, "exec")
        if not ok:
            return "Denied by user"
        if remember:
            PERMISSIONS.grant_session("exec", cmd)
    try:
        r = subprocess.run(cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    perm, reason = PERMISSIONS.check("write", path)
    if perm == Permission.DENY:
        return f"Permission denied: {reason}"
    if perm == Permission.ASK:
        ok, remember = ask_user_permission(f"write {path}", reason, "write")
        if not ok:
            return "Denied by user"
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
    perm, reason = PERMISSIONS.check("write", path)
    if perm == Permission.DENY:
        return f"Permission denied: {reason}"
    if perm == Permission.ASK:
        ok, remember = ask_user_permission(f"edit {path}", reason, "write")
        if not ok:
            return "Denied by user"
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


def execute_tool(name: str, args: dict) -> str:
    if name == "Glob": return run_glob(args["pattern"], args.get("path"))
    if name == "Grep": return run_grep(args["pattern"], args.get("path"), args.get("glob"), args.get("output_mode", "files_with_matches"), 0)
    if name == "Read": return run_read_file(args["file_path"], args.get("offset", 1), args.get("limit", 2000))
    if name == "EnterPlanMode": return MODE.enter_plan_mode()
    if name == "ExitPlanMode": return MODE.exit_plan_mode(args["plan"])
    if name == "Task":
        if args["agent_type"] not in AGENT_TYPES:
            return f"Error: Unknown type '{args['agent_type']}'"
        tid, result = TASKS.start_task(args["description"], args["prompt"], args["agent_type"], args.get("background", False), args.get("task_id"))
        if args.get("background"):
            return f"Started: {tid}. Use TaskOutput(\"{tid}\") for results."
        return result
    if name == "TaskOutput": return TASKS.get_output(args["task_id"], args.get("wait", True), args.get("timeout", 300))
    if name == "TaskList": return TASKS.list_tasks()
    if name == "bash": return run_bash(args["command"])
    if name == "write_file": return run_write(args["path"], args["content"])
    if name == "edit_file": return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        try: return TODO.update(args["items"])
        except Exception as e: return f"Error: {e}"
    if name == "Skill":
        c = SKILLS.get_skill_content(args["skill"])
        return f'<skill-loaded>\n{c}\n</skill-loaded>' if c else f"Unknown skill: {args['skill']}"
    # v9 memory tools
    if name == "MemoryWrite": return MEMORY.add(args["section"], args["fact"])
    if name == "MemoryRead": return MEMORY.read_all()
    return f"Unknown tool: {name}"


def execute_tool_for_subagent(name: str, args: dict) -> str:
    """Subagent tool execution (no Task/Memory/Plan tools)."""
    if name == "Glob": return run_glob(args["pattern"], args.get("path"))
    if name == "Grep": return run_grep(args["pattern"], args.get("path"), args.get("glob"), args.get("output_mode", "files_with_matches"), 0)
    if name == "Read": return run_read_file(args["file_path"], args.get("offset", 1), args.get("limit", 2000))
    if name == "bash": return run_bash(args["command"])
    if name == "write_file": return run_write(args["path"], args["content"])
    if name == "edit_file": return run_edit(args["path"], args["old_text"], args["new_text"])
    return f"Tool not available for subagent: {name}"


# =============================================================================
# Main Agent Loop - With Context Compression
# =============================================================================

def agent_loop(messages: list) -> list:
    """
    Main loop with automatic context compression.

    Key addition from v8:
    - Before each API call, check if compression is needed
    - If yes, summarize old messages and replace
    """
    while True:
        # v9: Check for context compression
        if COMPRESSOR.should_compress(messages):
            old_count = len(messages)
            old_tokens = estimate_messages_tokens(messages)
            messages[:] = COMPRESSOR.compress(messages)
            new_tokens = estimate_messages_tokens(messages)
            print(f"\n[Context compressed: {old_count} msgs -> {len(messages)} msgs, ~{old_tokens} -> ~{new_tokens} tokens]")

        available_tools = MODE.get_available_tools(ALL_TOOLS)
        system = get_system_prompt()

        log_api_call("main_agent", system, messages, available_tools)

        response = client.messages.create(
            model=MODEL, system=system,
            messages=messages, tools=available_tools, max_tokens=8000,
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
            display_name = tc.name
            if tc.name == "Task" and tc.input.get("background"):
                display_name = "Task (background)"
            print(f"\n> {display_name}")

            output = execute_tool(tc.name, tc.input)

            # Abbreviated display
            if tc.name in ("EnterPlanMode", "ExitPlanMode", "TaskList", "MemoryRead"):
                print(output)
            elif tc.name == "MemoryWrite":
                print(f"  {output}")
            elif tc.name in ("Glob", "Grep", "Read"):
                lines = output.split("\n")
                print(f"  {len(lines)} lines")
            elif tc.name == "Skill":
                print(f"  Loaded ({len(output)} chars)")
            elif tc.name not in ("Task",):
                print(f"  {output[:300]}{'...' if len(output) > 300 else ''}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================

def main():
    print(f"Mini Claude Code v9 (with Context Management) - {WORKDIR}")
    print(f"Context: auto-compression at ~{COMPRESSOR.compress_threshold} tokens")
    print(f"Memory: {MEMORY.memory_file} ({'exists' if MEMORY.memory_file.exists() else 'will be created'})")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            mode_ind = " [PLAN]" if MODE.mode == AgentMode.PLANNING else ""
            user_input = input(f"You{mode_ind}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Plan approval
        if MODE.current_plan is not None:
            action, ctx = MODE.handle_user_response(user_input)
            if action == "approve":
                print("\n Plan approved.\n")
                history.append({"role": "user", "content": f"Execute plan:\n\n{ctx}"})
            elif action == "cancel":
                print("\n Cancelled.\n")
                continue
            elif action == "revise":
                history.append({"role": "user", "content": f"Revise: {ctx}"})
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
