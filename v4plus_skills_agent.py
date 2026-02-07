#!/usr/bin/env python3
"""
v4plus_skills_agent.py - Mini Claude Code: Specialist Team Pattern (~550 lines)

Core Philosophy: "Skills Belong to Specialists, Not Managers"
=============================================================
v4 introduced skills - domain knowledge loaded on-demand. But there's a
design tension:

    In v4, the MAIN agent loads skills into its OWN context.
    Then it either acts on the knowledge itself, or delegates to a
    skill-less subagent. This is like a manager reading the manual
    then telling a worker what to do step-by-step.

The Specialist Team Pattern:
---------------------------
    v4:  Manager reads manual -> tells worker what to do
    v4+: Manager identifies need -> spawns specialist WITH the manual

    +------------------------------------------------------------------+
    |                  v4: Manager Does Everything                      |
    +------------------------------------------------------------------+
    |  Main Agent                                                       |
    |    |-- Skill("pdf")  -> knowledge injected into MAIN context      |
    |    |-- (now main agent knows PDF processing)                      |
    |    |-- Task(code, "process the PDF")  -> subagent has NO skill    |
    |    |       subagent just follows main agent's instructions        |
    +------------------------------------------------------------------+

    +------------------------------------------------------------------+
    |                  v4+: Specialist Team                             |
    +------------------------------------------------------------------+
    |  Main Agent (Project Manager)                                     |
    |    |-- sees: "pdf skill available" (Layer 1 metadata only)        |
    |    |-- Task(code, "process the PDF", load_skills=["pdf"])         |
    |    |       |                                                      |
    |    |       v                                                      |
    |    |   Subagent born with:                                        |
    |    |     - PDF skill knowledge in system prompt (Layer 2)         |
    |    |     - All coding tools (bash, read, write, edit)             |
    |    |     - Fresh context dedicated to this task                   |
    +------------------------------------------------------------------+

Why This Is Better:
------------------
1. Main agent context stays LEAN (no skill content bloating it)
2. Subagent has BOTH knowledge AND tools (closed-loop, no telephone game)
3. Skill content in subagent system prompt is fine (fresh context, no cache concern)
4. Natural mapping: one specialist per domain task

The Key Insight:
---------------
    v4's Skill tool injects into tool_result to preserve prompt cache.
    But subagents have FRESH context - no cache to preserve!
    So injecting skills into the subagent's SYSTEM PROMPT is optimal.

What Changes from v4:
--------------------
    | Component      | v4                    | v4+                          |
    |----------------|-----------------------|------------------------------|
    | Skill tool     | Main agent loads      | REMOVED from main agent      |
    | Task tool      | No skill awareness    | Has load_skills parameter    |
    | Main prompt    | "invoke Skill tool"   | "delegate with load_skills"  |
    | run_task()     | No skill injection    | Injects skills into subagent |
    | SkillLoader    | Unchanged             | Unchanged                    |

Usage:
    python v4plus_skills_agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# Logging Configuration
# =============================================================================

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() == "true"

# Event callback for web frontend (None = CLI mode, no change in behavior)
event_callback = None


def emit_event(event_type: str, **kwargs):
    """Emit an event to the web frontend if callback is set."""
    if event_callback:
        # Truncate large outputs to prevent frontend freeze
        if event_type == "tool_result" and "output" in kwargs:
            out = kwargs["output"]
            if isinstance(out, str) and len(out) > 8000:
                kwargs["output"] = out[:8000] + f"\n\n... (truncated, {len(out)} chars total)"
        if event_type == "subagent_complete" and "result" in kwargs:
            out = kwargs["result"]
            if isinstance(out, str) and len(out) > 8000:
                kwargs["result"] = out[:8000] + f"\n\n... (truncated, {len(out)} chars total)"
        event_callback({"type": event_type, **kwargs})


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
# SkillLoader - Unchanged from v4
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
        Full content (Layer 2) is loaded only when a subagent needs it.
        """
        if not self.skills:
            return "(no skills available)"

        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.skills.items()
        )

    def get_skill_content(self, name: str) -> str:
        """
        Get full skill content for injection into subagent system prompt.

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
    "general": {
        "description": "Full agent for domain tasks like PDF processing, data analysis, code review, etc.",
        "tools": "*",
        "prompt": "You are a general-purpose agent. Use your loaded skills and tools to complete the task. Focus on accuracy and follow skill instructions precisely.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
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
# System Prompt - Updated for v4+ (Specialist Team Pattern)
# =============================================================================

SYSTEM = f"""You are a project manager agent at {WORKDIR}.

Loop: plan -> act or delegate -> synthesize results -> report.

**Skills available:**
{SKILLS.get_descriptions()}

**Subagent types** (use with Task tool):
{get_agent_descriptions()}

Strategy:
- Simple domain task (e.g. parse a PDF): use Skill tool to load knowledge, then act directly.
- Complex domain task (e.g. build a report from PDF + write code): use Task with load_skills to spawn a specialist subagent.
- Non-domain task (file reads, quick bash): act directly with base tools.
- Use TodoWrite to track multi-step work.
- Prefer tools over prose. Act, don't just explain.
- After all subtasks complete, synthesize and summarize results."""


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

# Task tool - Enhanced with load_skills parameter (core change in v4+)
TASK_TOOL = {
    "name": "Task",
    "description": (
        f"Spawn a specialist subagent for a focused subtask.\n\n"
        f"Agent types:\n{get_agent_descriptions()}\n\n"
        f"Available skills (pass via load_skills):\n{SKILLS.get_descriptions()}"
    ),
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
            "load_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Skills to equip the subagent with. "
                    f"Available: {', '.join(SKILLS.list_skills())}"
                ),
            },
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

# Skill tool - kept from v4 for simple domain tasks handled by main agent
SKILL_TOOL = {
    "name": "Skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.

Available skills:
{SKILLS.get_descriptions()}

When to use:
- For SIMPLE domain tasks you can handle directly (e.g. parse a PDF, quick code review)
- For COMPLEX tasks needing focused implementation, prefer Task with load_skills instead

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

# v4+: Both Skill (main agent) and Task with load_skills (subagent) available
ALL_TOOLS = BASE_TOOLS + [TASK_TOOL, SKILL_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
    """Filter tools based on agent type."""
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return BASE_TOOLS
    return [t for t in BASE_TOOLS if t["name"] in allowed]


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
        result = TODO.update(items)
        emit_event("todo_update", todos=TODO.items)
        return result
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    """
    Load a skill and inject it into the main agent's conversation.

    Used for SIMPLE domain tasks where the main agent can handle it directly.
    For complex tasks, use run_task() with load_skills instead.

    Returns skill content as tool_result (preserves prompt cache).
    """
    content = SKILLS.get_skill_content(skill_name)

    if content is None:
        available = ", ".join(SKILLS.list_skills()) or "none"
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"

    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


def run_task(description: str, prompt: str, agent_type: str,
             load_skills: list = None) -> str:
    """
    Execute a subagent task with optional skill injection.

    This is the core evolution from v4:
    - v4: run_task() spawns a bare subagent (no skill knowledge)
    - v4+: run_task() injects skill content into subagent's system prompt

    Why system prompt (not tool_result) for subagents?
    - Subagents have FRESH context - no prompt cache to preserve
    - System prompt is the natural place for persistent instructions
    - Skill content guides the subagent throughout its entire execution

    The subagent gets: base prompt + skill knowledge + user task
    It can then use its tools (bash, read, write, edit) with full
    domain expertise - a true closed-loop specialist.
    """
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]

    # Build skill section for subagent's system prompt
    skill_section = ""
    if load_skills:
        loaded = []
        for skill_name in load_skills:
            content = SKILLS.get_skill_content(skill_name)
            if content:
                loaded.append(content)
            else:
                available = ", ".join(SKILLS.list_skills()) or "none"
                return f"Error: Unknown skill '{skill_name}'. Available: {available}"
        if loaded:
            skill_section = (
                "\n\n=== SPECIALIZED KNOWLEDGE ===\n"
                "Follow the instructions below for domain-specific work.\n\n"
                + "\n\n---\n\n".join(loaded)
            )

    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.

{config["prompt"]}
{skill_section}

Complete the task and return a clear, concise summary."""

    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    skills_label = f" +skills:{','.join(load_skills)}" if load_skills else ""
    print(f"  [{agent_type}{skills_label}] {description}")
    emit_event("subagent_start", agent_type=agent_type,
               description=description, skills=load_skills or [])
    start = time.time()
    tool_count = 0

    while True:
        log_api_call(f"subagent:{agent_type}", sub_system, sub_messages, sub_tools)

        with client.messages.stream(
            model=MODEL,
            system=sub_system,
            messages=sub_messages,
            tools=sub_tools,
            max_tokens=64000,
        ) as stream:
            response = stream.get_final_message()

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
                f"\r  [{agent_type}{skills_label}] {description} ... {tool_count} tools, {elapsed:.1f}s"
            )
            sys.stdout.flush()
            emit_event("subagent_progress", description=description,
                       tool_count=tool_count, elapsed=round(elapsed, 1))

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    elapsed = time.time() - start
    sys.stdout.write(
        f"\r  [{agent_type}{skills_label}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n"
    )

    result_text = "(subagent returned no text)"
    for block in response.content:
        if hasattr(block, "text"):
            result_text = block.text
            break

    emit_event("subagent_complete", description=description, result=result_text)
    return result_text


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation."""
    emit_event("tool_call", tool=name, input=args)
    if name == "bash":
        result = run_bash(args["command"])
    elif name == "read_file":
        result = run_read(args["path"], args.get("limit"))
    elif name == "write_file":
        result = run_write(args["path"], args["content"])
    elif name == "edit_file":
        result = run_edit(args["path"], args["old_text"], args["new_text"])
    elif name == "TodoWrite":
        result = run_todo(args["items"])
    elif name == "Task":
        result = run_task(
            args["description"], args["prompt"],
            args["agent_type"], args.get("load_skills")
        )
    elif name == "Skill":
        result = run_skill(args["skill"])
    else:
        result = f"Unknown tool: {name}"
    emit_event("tool_result", tool=name, output=result)
    return result


# =============================================================================
# Main Agent Loop
# =============================================================================

def agent_loop(messages: list) -> list:
    """
    Main agent loop - Project Manager pattern.

    The main agent coordinates but doesn't do domain work itself.
    When a task matches a skill, it spawns a specialist subagent
    with load_skills to handle it autonomously.
    """
    while True:
        log_api_call("main_agent", SYSTEM, messages, ALL_TOOLS)

        with client.messages.stream(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=ALL_TOOLS,
            max_tokens=64000,
        ) as stream:
            response = stream.get_final_message()

        log_api_response("main_agent", response)

        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
                emit_event("message", content=block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            emit_event("done")
            return messages

        results = []
        for tc in tool_calls:
            if tc.name == "Task":
                skills = tc.input.get("load_skills", [])
                skills_info = f" [skills: {', '.join(skills)}]" if skills else ""
                print(f"\n> Task: {tc.input.get('description', 'subtask')}{skills_info}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

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
    print(f"Mini Claude Code v4+ (Specialist Team) - {WORKDIR}")
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
