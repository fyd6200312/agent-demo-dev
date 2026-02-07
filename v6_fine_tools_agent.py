#!/usr/bin/env python3
"""
v6_fine_tools_agent.py - Mini Claude Code: Fine-Grained Tools (~850 lines)

Core Philosophy: "Specialized Tools Beat General Tools"
======================================================
v5 gave us permission control. But there's an efficiency question:

    Why does the model use bash for EVERYTHING?

Watch v5 explore a codebase:
    bash("find . -name '*.py'")
    bash("grep -r 'def main' .")
    bash("cat src/main.py | head -50")
    bash("wc -l src/*.py")

Each bash call has overhead:
    - Model must think of the right command flags
    - Shell spawns a process
    - Output is unstructured text
    - No built-in limits (can dump 10MB to context)

The Problem - Bash is a Swiss Army Knife:
----------------------------------------
Swiss Army Knives are great for camping. But a chef uses specialized knives.

    | Task             | Bash (v5)                      | Specialized (v6)        |
    |------------------|--------------------------------|-------------------------|
    | Find files       | `find . -name "*.py" | head`   | Glob(pattern="**/*.py") |
    | Search content   | `grep -rn "TODO" --include=*`  | Grep(pattern="TODO")    |
    | Read file        | `cat file.py | head -100`      | Read(path, limit=100)   |

Specialized tools are:
1. **Easier** - No flag memorization
2. **Safer** - Built-in limits and sandboxing
3. **Faster** - Direct Python, no shell spawn
4. **Structured** - Predictable output format

The Solution - Tool Specialization:
----------------------------------
    +------------------------------------------------------------------+
    |                    Tool Specialization                           |
    +------------------------------------------------------------------+
    |                                                                  |
    |  Before (v5):                     After (v6):                    |
    |                                                                  |
    |  +----------+                     +---------+                    |
    |  |   bash   |                     |  Glob   | Pattern matching   |
    |  | (does    |                     +---------+                    |
    |  |  all)    |     ─────────>      +---------+                    |
    |  +----------+                     |  Grep   | Content search     |
    |                                   +---------+                    |
    |                                   +---------+                    |
    |                                   |  Read   | Smart file read    |
    |                                   +---------+                    |
    |                                   +---------+                    |
    |                                   |  Bash   | (real shell only)  |
    |                                   +---------+                    |
    |                                                                  |
    +------------------------------------------------------------------+

Key Insight - More Tools = Less Thinking:
----------------------------------------
Counterintuitively, MORE specialized tools means LESS cognitive load for the model.

With bash only:
    "I need to find Python files... what's the find syntax? -name? -iname?
     Should I use -type f? How do I exclude __pycache__?"

With Glob tool:
    "I need to find Python files. Glob(pattern='**/*.py')"

Tools encode EXPERTISE. The model doesn't need to know ripgrep flags
because the Grep tool handles that.

Claude Code has 20+ tools for a reason.

The Three New Tools:
-------------------
    | Tool   | Purpose              | Key Features                     |
    |--------|----------------------|----------------------------------|
    | Glob   | Find files by path   | Excludes noise dirs, sorted      |
    | Grep   | Search file contents | Regex, output modes, limits      |
    | Read   | Read file contents   | Line numbers, ranges, PDF/image  |

Bash remains for:
- Running builds (npm, make, cargo)
- Git operations (commit, push, pull)
- Custom scripts
- Anything that truly needs a shell

Usage:
    python v6_fine_tools_agent.py
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
            "read": [
                PermissionRule("*", Permission.ALLOW, "Reading is safe"),
            ],
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
                # Safe commands
                PermissionRule("ls *", Permission.ALLOW, "Listing"),
                PermissionRule("ls", Permission.ALLOW, "Listing"),
                PermissionRule("pwd", Permission.ALLOW, "Current directory"),
                PermissionRule("echo *", Permission.ALLOW, "Printing"),
                PermissionRule("which *", Permission.ALLOW, "Finding commands"),
                # Git read
                PermissionRule("git status*", Permission.ALLOW, "Git status"),
                PermissionRule("git diff*", Permission.ALLOW, "Git diff"),
                PermissionRule("git log*", Permission.ALLOW, "Git log"),
                PermissionRule("git branch*", Permission.ALLOW, "Git branches"),
                PermissionRule("git show*", Permission.ALLOW, "Git show"),
                # Dangerous
                PermissionRule("rm -rf /*", Permission.DENY, "System destruction"),
                PermissionRule("rm -rf /", Permission.DENY, "System destruction"),
                PermissionRule("sudo *", Permission.DENY, "Privilege escalation"),
                PermissionRule("*| bash*", Permission.DENY, "Piped execution"),
                PermissionRule("*| sh*", Permission.DENY, "Piped execution"),
                # Default
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
        cache_key = f"{category}:{self._normalize(operation)}"
        self.session_grants[cache_key] = True

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
# NEW in v6: Specialized Tools (Glob, Grep, Read)
# =============================================================================

# Directories to exclude from searches (noise reduction)
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    '.idea', '.vscode', 'dist', 'build', '.next', '.nuxt',
    'coverage', '.pytest_cache', '.mypy_cache', 'egg-info'
}


def run_glob(pattern: str, path: str = None) -> str:
    """
    Fast file pattern matching.

    Why a dedicated Glob tool instead of bash find?
    1. Automatically excludes noise directories (.git, node_modules, etc.)
    2. Sorts by modification time (recent files are more relevant)
    3. Built-in limit (prevents context overflow)
    4. Simpler syntax than find

    Examples:
        Glob(pattern="**/*.py")           -> All Python files
        Glob(pattern="src/**/*.ts")       -> TypeScript in src/
        Glob(pattern="**/test_*.py")      -> Test files
        Glob(pattern="*.md", path="docs") -> Markdown in docs/
    """
    try:
        base = (WORKDIR / path).resolve() if path else WORKDIR

        if not base.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"

        if not base.exists():
            return f"Error: Path does not exist: {path}"

        results = []
        for p in base.glob(pattern):
            # Skip excluded directories
            if any(ex in p.parts for ex in EXCLUDE_DIRS):
                continue
            if p.is_file():
                try:
                    mtime = p.stat().st_mtime
                    results.append((mtime, p))
                except OSError:
                    continue

        # Sort by modification time (newest first)
        results.sort(reverse=True)

        # Limit results
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


def run_grep(
    pattern: str,
    path: str = None,
    glob_pattern: str = None,
    output_mode: str = "files_with_matches",
    context: int = 0
) -> str:
    """
    Search file contents with regex.

    Why a dedicated Grep tool instead of bash grep?
    1. Structured output modes (files only, content, counts)
    2. Built-in limits (won't dump 10MB to context)
    3. Automatic noise directory exclusion
    4. Simpler than remembering grep flags

    Output modes:
        files_with_matches - Just file paths (default, most efficient)
        content           - Show matching lines with context
        count             - Show match counts per file

    Examples:
        Grep(pattern="TODO")                          -> Find TODOs
        Grep(pattern="def test_", glob="*.py")        -> Find test functions
        Grep(pattern="import", output_mode="content") -> Show imports with context
    """
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
        MAX_FILES = 1000
        MAX_RESULTS = 100

        for fp in base.glob(file_pattern):
            if not fp.is_file():
                continue

            # Skip excluded directories
            if any(ex in fp.parts for ex in EXCLUDE_DIRS):
                continue

            # Skip binary files
            try:
                content = fp.read_text(errors='ignore')
            except Exception:
                continue

            files_searched += 1
            if files_searched > MAX_FILES:
                results.append(f"... (stopped after {MAX_FILES} files)")
                break

            lines = content.splitlines()
            matches = []

            for i, line in enumerate(lines):
                if regex.search(line):
                    matches.append((i + 1, line))

            if matches:
                rel_path = str(fp.relative_to(WORKDIR))

                if output_mode == "files_with_matches":
                    results.append(rel_path)

                elif output_mode == "count":
                    results.append(f"{rel_path}: {len(matches)}")

                else:  # content
                    results.append(f"\n{rel_path}:")
                    for lineno, line in matches[:10]:  # Limit per file
                        # Truncate long lines
                        display_line = line[:200] + "..." if len(line) > 200 else line
                        results.append(f"  {lineno:>5}: {display_line}")

                    if len(matches) > 10:
                        results.append(f"  ... and {len(matches) - 10} more matches")

            if len(results) >= MAX_RESULTS:
                results.append(f"... (limited to {MAX_RESULTS} results)")
                break

        if not results:
            return "No matches found."

        return "\n".join(results)

    except Exception as e:
        return f"Error: {e}"


def run_read_v6(file_path: str, offset: int = 1, limit: int = 2000) -> str:
    """
    Smart file reading with line numbers and ranges.

    Why a dedicated Read tool instead of bash cat?
    1. Automatic line numbers (essential for edit_file)
    2. Range support (read specific portions of large files)
    3. File type detection (PDF, images)
    4. Consistent output format

    Features:
        - Line numbers in output (for easy reference in edits)
        - Range reading (offset + limit)
        - Large file handling with truncation notice
        - PDF text extraction (if pdftotext available)
        - Image detection (prompts for vision use)

    Examples:
        Read(file_path="src/main.py")              -> Full file with line numbers
        Read(file_path="big.log", offset=100, limit=50) -> Lines 100-150
    """
    try:
        fp = (WORKDIR / file_path).resolve()

        if not fp.is_relative_to(WORKDIR):
            return "Error: Path escapes workspace"

        if not fp.exists():
            return f"Error: File not found: {file_path}"

        if not fp.is_file():
            return f"Error: Not a file: {file_path}"

        suffix = fp.suffix.lower()

        # PDF handling
        if suffix == ".pdf":
            return _read_pdf(fp)

        # Image handling
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            return f"[Image file: {file_path}]\nUse vision capabilities to analyze images."

        # Binary file detection
        try:
            content = fp.read_text()
        except UnicodeDecodeError:
            return f"Error: Binary file cannot be read as text: {file_path}"

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        start_idx = max(0, offset - 1)  # 1-indexed to 0-indexed
        end_idx = min(len(lines), start_idx + limit)
        selected = lines[start_idx:end_idx]

        # Format with line numbers
        output_lines = []
        for i, line in enumerate(selected):
            lineno = start_idx + i + 1
            # Truncate very long lines
            if len(line) > 500:
                line = line[:500] + "..."
            output_lines.append(f"{lineno:>6}| {line}")

        result = "\n".join(output_lines)

        # Add truncation notice
        if end_idx < total_lines:
            remaining = total_lines - end_idx
            result += f"\n\n... {remaining} more lines. Use offset={end_idx + 1} to continue."

        return result

    except Exception as e:
        return f"Error: {e}"


def _read_pdf(fp: Path) -> str:
    """Extract text from PDF using pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", str(fp), "-"],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout[:50000]  # Limit PDF content
        if not text.strip():
            return "(PDF is empty or contains only images)"
        return f"[PDF Content: {fp.name}]\n\n{text}"
    except FileNotFoundError:
        return "Error: pdftotext not installed. Install with: brew install poppler (macOS) or apt install poppler-utils (Linux)"
    except subprocess.TimeoutExpired:
        return "Error: PDF extraction timed out"
    except Exception as e:
        return f"Error reading PDF: {e}"


# =============================================================================
# Tool Definitions - Updated for v6
# =============================================================================

# NEW specialized tools
GLOB_TOOL = {
    "name": "Glob",
    "description": """Fast file pattern matching.

Finds files matching a glob pattern. Returns paths sorted by modification time (newest first).
Automatically excludes noise directories (.git, node_modules, __pycache__, etc.)

Examples:
  - "**/*.py"        -> All Python files
  - "src/**/*.ts"    -> TypeScript files in src/
  - "**/test_*.py"   -> All test files
  - "*.md"           -> Markdown files in current dir

Use this instead of `bash find` for file discovery.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: workspace root)"
            }
        },
        "required": ["pattern"],
    },
}

GREP_TOOL = {
    "name": "Grep",
    "description": """Search file contents with regex.

Searches for a pattern across files. More powerful than bash grep with structured output.

Output modes:
  - files_with_matches (default): Just file paths containing matches
  - content: Show matching lines with line numbers
  - count: Show number of matches per file

Examples:
  - pattern="TODO" -> Find all TODOs
  - pattern="def test_", glob="*.py" -> Find test functions
  - pattern="import", output_mode="content" -> Show all imports

Use this instead of `bash grep` for content search.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: workspace root)"
            },
            "glob": {
                "type": "string",
                "description": "File pattern filter (e.g., '*.py', '**/*.ts')"
            },
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "description": "Output format (default: files_with_matches)"
            },
            "context": {
                "type": "integer",
                "description": "Lines of context around matches (for content mode)"
            }
        },
        "required": ["pattern"],
    },
}

READ_TOOL = {
    "name": "Read",
    "description": """Read file contents with line numbers.

Reads a file with automatic line numbering (essential for edit_file references).
Supports reading specific ranges of large files.

Features:
  - Line numbers in output
  - Range reading with offset/limit
  - PDF text extraction
  - Image file detection

Examples:
  - Read(file_path="src/main.py") -> Full file with line numbers
  - Read(file_path="big.log", offset=100, limit=50) -> Lines 100-150

Use this instead of `bash cat` for reading files.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read"
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (1-indexed, default: 1)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (default: 2000)"
            }
        },
        "required": ["file_path"],
    },
}

# Base tools (write operations kept, read moved to specialized tools)
BASE_TOOLS = [
    {
        "name": "bash",
        "description": """Run shell command.

Use for:
  - Build commands (npm, make, cargo, pip)
  - Git operations (commit, push, pull)
  - Running scripts
  - System commands

Do NOT use for:
  - Finding files (use Glob instead)
  - Searching content (use Grep instead)
  - Reading files (use Read instead)""",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file. Creates parent directories if needed.",
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

# Combine all tools
SPECIALIZED_TOOLS = [GLOB_TOOL, GREP_TOOL, READ_TOOL]


# =============================================================================
# SkillLoader (from v4)
# =============================================================================

class SkillLoader:
    """Loads and manages skills. See v4 for details."""

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
        for folder, label in [("scripts", "Scripts"), ("references", "References"), ("assets", "Assets")]:
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
# Agent Type Registry (from v3) - Updated for v6
# =============================================================================

AGENT_TYPES = {
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["Glob", "Grep", "Read", "bash"],  # Now uses specialized tools
        "prompt": "You are an exploration agent. Use Glob to find files, Grep to search content, and Read to examine files. Return a concise summary.",
    },
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["Glob", "Grep", "Read", "bash"],  # Read-only tools
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
    """Task list manager. See v2 for details."""

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
# Task and Skill tools
# =============================================================================

TASK_TOOL = {
    "name": "Task",
    "description": f"Spawn a subagent for a focused subtask.\n\nAgent types:\n{get_agent_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task description (3-5 words)"},
            "prompt": {"type": "string", "description": "Detailed instructions for the subagent"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
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
            "skill": {"type": "string", "description": "Name of the skill to load"}
        },
        "required": ["skill"],
    },
}


ALL_TOOLS = SPECIALIZED_TOOLS + BASE_TOOLS + [TASK_TOOL, SKILL_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
    """Filter tools based on agent type."""
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return SPECIALIZED_TOOLS + BASE_TOOLS
    return [t for t in (SPECIALIZED_TOOLS + BASE_TOOLS) if t["name"] in allowed]


# =============================================================================
# System Prompt - Updated for v6
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available**:
{SKILLS.get_descriptions()}

**Subagents available**:
{get_agent_descriptions()}

Rules:
- Use Skill tool when task matches a skill description
- Use Task tool for focused subtasks
- Use TodoWrite to track multi-step work
- After finishing, summarize what changed."""


# =============================================================================
# Tool Implementations
# =============================================================================

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    """Execute shell command with permission check."""
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
        r = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        return (r.stdout + r.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s limit)"
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """Write content to file with permission check."""
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
    """Replace text in file with permission check."""
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
    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above."""


def run_task(description: str, prompt: str, agent_type: str) -> str:
    """Execute a subagent task."""
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
            sys.stdout.write(f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s")
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    elapsed = time.time() - start
    sys.stdout.write(f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n")

    for block in response.content:
        if hasattr(block, "text"):
            return block.text

    return "(subagent returned no text)"


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
    # Specialized tools (v6)
    if name == "Glob":
        return run_glob(args["pattern"], args.get("path"))
    if name == "Grep":
        return run_grep(
            args["pattern"],
            args.get("path"),
            args.get("glob"),
            args.get("output_mode", "files_with_matches"),
            args.get("context", 0)
        )
    if name == "Read":
        return run_read_v6(
            args["file_path"],
            args.get("offset", 1),
            args.get("limit", 2000)
        )

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
# Main Agent Loop
# =============================================================================

def agent_loop(messages: list) -> list:
    """Main agent loop."""
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

            # Show output (abbreviated for some tools)
            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name == "Glob":
                lines = output.split("\n")
                print(f"  Found {len(lines)} files")
                if len(lines) <= 5:
                    print(f"  {output}")
            elif tc.name == "Grep":
                lines = output.split("\n")
                print(f"  {len(lines)} results")
            elif tc.name == "Read":
                lines = output.split("\n")
                print(f"  Read {len(lines)} lines")
            elif tc.name != "Task":
                # Truncate long output
                if len(output) > 500:
                    print(f"  {output[:500]}...")
                else:
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
    print(f"Mini Claude Code v6 (with Fine-Grained Tools) - {WORKDIR}")
    print(f"Specialized tools: Glob, Grep, Read")
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
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
