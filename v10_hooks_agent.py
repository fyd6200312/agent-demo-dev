#!/usr/bin/env python3
"""
v10_hooks_agent.py - Mini Claude Code: Lifecycle Hooks (~1250 lines)

Core Philosophy: "Extensibility Through Open Points"
====================================================
v9 gave us context management and memory. But there's a customization question:

    How can different projects customize agent behavior WITHOUT modifying code?

Project A wants: "Run ESLint before every file write"
Project B wants: "Log all bash commands to audit.log"
Project C wants: "Load .env.local on session start"

Without hooks, you'd need to fork the code for each project.

The Problem - One Size Doesn't Fit All:
--------------------------------------
Every project has unique requirements:

    | Project     | Need                                |
    |-------------|-------------------------------------|
    | Frontend    | Lint + format on write              |
    | Backend     | Security scan on bash commands       |
    | Data        | Validate schema on file changes      |
    | DevOps      | Notify Slack on task completion      |

Hardcoding these into the agent is wrong:
- Too many special cases
- Maintenance nightmare
- Can't anticipate every need

The Solution - Lifecycle Hooks:
------------------------------
Let users inject custom logic at KEY POINTS in the agent lifecycle:

    +------------------------------------------------------------------+
    |                      Hook Points                                 |
    +------------------------------------------------------------------+
    |                                                                  |
    |  on_session_start ─── Load env, check prerequisites              |
    |         |                                                        |
    |         v                                                        |
    |  ┌──────────────┐                                                |
    |  │  User Input  │─── on_user_input ─── Log, transform           |
    |  └──────┬───────┘                                                |
    |         v                                                        |
    |  ┌──────────────┐                                                |
    |  │  Model Call  │─── on_model_response ─── Monitor, log         |
    |  └──────┬───────┘                                                |
    |         v                                                        |
    |  ┌──────────────┐                                                |
    |  │  Tool Call   │                                                |
    |  │              │─── pre_tool_call ─── Validate, block, modify  |
    |  │  [execute]   │                                                |
    |  │              │─── post_tool_call ── Transform output, log    |
    |  └──────┬───────┘                                                |
    |         v                                                        |
    |  ┌──────────────┐                                                |
    |  │  Continue?   │─── on_turn_end ─── Summarize, checkpoint      |
    |  └──────┬───────┘                                                |
    |         v                                                        |
    |  on_session_end ──── Cleanup, save state                         |
    |                                                                  |
    +------------------------------------------------------------------+

Key Insight - Hooks are AOP for Agents:
--------------------------------------
Aspect-Oriented Programming (AOP) separates cross-cutting concerns.

    Core logic:    tool_call -> execute -> return_result
    Cross-cutting: logging, validation, transformation

Hooks let you ADD concerns without CHANGING the core:

    # Before hooks:
    def run_write(path, content):
        # validation? logging? linting? ALL HERE = messy
        write_file(path, content)

    # After hooks:
    def run_write(path, content):
        write_file(path, content)  # Clean core logic

    # hooks.yaml:
    pre_tool_call:
      - tool: write_file
        command: "eslint --fix {path}"     # Lint
      - tool: "*"
        command: "echo '{tool}' >> log"    # Audit

Hook Configuration (.claude/hooks.yaml):
---------------------------------------
    hooks:
      on_session_start:
        - command: "source .env.local"
          description: "Load local env"

      pre_tool_call:
        - tool: write_file
          command: "echo 'Writing {path}' >> audit.log"
        - tool: bash
          command: "python .claude/validate_cmd.py '{command}'"

      post_tool_call:
        - tool: write_file
          command: "npx prettier --write {path}"
          description: "Auto-format written files"

      on_session_end:
        - command: "python .claude/save_session.py"

Hook Execution Model:
--------------------
    pre_tool_call hooks CAN BLOCK execution:
        - Exit code 0: Continue
        - Exit code non-0: Block tool, return hook's stderr as error

    post_tool_call hooks CAN MODIFY output:
        - Hook's stdout replaces tool output (if non-empty)

    Other hooks are fire-and-forget (errors logged but don't block).

Usage:
    python v10_hooks_agent.py
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
CLAUDE_DIR = WORKDIR / ".claude"
MEMORY_DIR = CLAUDE_DIR

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# NEW in v10: Lifecycle Hook System
# =============================================================================

class HookPhase(Enum):
    """
    Points in the agent lifecycle where hooks can be injected.

    SESSION_START:   Once when agent starts
    SESSION_END:     Once when agent exits
    USER_INPUT:      After each user message
    PRE_TOOL_CALL:   Before each tool execution (CAN BLOCK)
    POST_TOOL_CALL:  After each tool execution (CAN MODIFY output)
    TURN_END:        After each complete turn (model response + tools)
    """
    SESSION_START = "on_session_start"
    SESSION_END = "on_session_end"
    USER_INPUT = "on_user_input"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    TURN_END = "on_turn_end"


@dataclass
class HookConfig:
    """
    Configuration for a single hook.

    Fields:
        phase: When to trigger
        command: Shell command to execute
        tool: Tool name filter (* = all, only for pre/post_tool_call)
        description: Human-readable description
        timeout: Max execution time in seconds
    """
    phase: HookPhase
    command: str
    tool: str = "*"
    description: str = ""
    timeout: int = 30


class HookManager:
    """
    Manages lifecycle hooks loaded from .claude/hooks.yaml.

    Hook execution model:
    - pre_tool_call: Can BLOCK (non-zero exit = block)
    - post_tool_call: Can MODIFY output (stdout replaces result)
    - Others: Fire-and-forget (errors logged)

    Template variables in commands:
    - {tool}: Tool name
    - {path}: File path (for file tools)
    - {command}: Command string (for bash tool)
    - {workdir}: Workspace directory
    """

    def __init__(self, config_dir: Path, workdir: Path):
        self.config_dir = config_dir
        self.workdir = workdir
        self.hooks: dict[HookPhase, list[HookConfig]] = {
            phase: [] for phase in HookPhase
        }
        self._load_config()

    def _load_config(self):
        """Load hooks from .claude/hooks.yaml."""
        config_file = self.config_dir / "hooks.yaml"
        if not config_file.exists():
            return

        try:
            # Simple YAML parsing (avoid PyYAML dependency)
            content = config_file.read_text()
            self._parse_yaml(content)
        except Exception as e:
            print(f"Warning: Failed to load hooks: {e}")

    def _parse_yaml(self, content: str):
        """
        Minimal YAML parser for hooks config.

        Supports the specific structure needed:
            hooks:
              phase_name:
                - command: "..."
                  tool: "..."
                  description: "..."
                  timeout: 30
        """
        current_phase = None
        current_hook = {}

        for line in content.splitlines():
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if stripped == "hooks:":
                continue

            # Phase header (2-space indent)
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                phase_name = stripped[:-1]
                try:
                    current_phase = HookPhase(phase_name)
                except ValueError:
                    current_phase = None
                continue

            # Hook item start
            if stripped.startswith("- "):
                # Save previous hook if exists
                if current_hook and current_phase:
                    self._add_hook(current_phase, current_hook)
                current_hook = {}

                # Parse inline key-value
                rest = stripped[2:]
                if ":" in rest:
                    key, value = rest.split(":", 1)
                    current_hook[key.strip()] = value.strip().strip("\"'")
                continue

            # Key-value in hook
            if stripped and ":" in stripped and current_phase:
                key, value = stripped.split(":", 1)
                current_hook[key.strip()] = value.strip().strip("\"'")

        # Save last hook
        if current_hook and current_phase:
            self._add_hook(current_phase, current_hook)

    def _add_hook(self, phase: HookPhase, config: dict):
        """Add a parsed hook configuration."""
        if "command" not in config:
            return

        hook = HookConfig(
            phase=phase,
            command=config["command"],
            tool=config.get("tool", "*"),
            description=config.get("description", ""),
            timeout=int(config.get("timeout", 30)),
        )
        self.hooks[phase].append(hook)

    def trigger(
        self,
        phase: HookPhase,
        tool_name: str = None,
        tool_args: dict = None,
        tool_result: str = None,
        user_input: str = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Trigger hooks for a given phase.

        Args:
            phase: Which lifecycle phase
            tool_name: Current tool (for pre/post_tool_call)
            tool_args: Tool arguments (for template variables)
            tool_result: Tool output (for post_tool_call)
            user_input: User message (for on_user_input)

        Returns:
            (should_continue, modified_output)
            - should_continue: False if pre_tool_call blocks execution
            - modified_output: Replaced output from post_tool_call
        """
        hooks = self.hooks.get(phase, [])
        if not hooks:
            return True, None

        for hook in hooks:
            # Filter by tool name
            if phase in (HookPhase.PRE_TOOL_CALL, HookPhase.POST_TOOL_CALL):
                if hook.tool != "*" and hook.tool != tool_name:
                    continue

            # Build template variables
            variables = {
                "tool": tool_name or "",
                "workdir": str(self.workdir),
            }
            if tool_args:
                variables.update({
                    "path": tool_args.get("path", tool_args.get("file_path", "")),
                    "command": tool_args.get("command", ""),
                    "pattern": tool_args.get("pattern", ""),
                })
            if tool_result:
                # Limit result size in templates
                variables["result"] = tool_result[:1000]
            if user_input:
                variables["input"] = user_input[:500]

            # Expand template
            try:
                cmd = hook.command.format(**variables)
            except (KeyError, ValueError):
                cmd = hook.command

            # Execute hook
            success, output = self._execute(cmd, hook.timeout)

            if hook.description and DEBUG_LOG:
                print(f"  [hook:{phase.value}] {hook.description}: {'OK' if success else 'FAIL'}")

            # Handle results based on phase
            if phase == HookPhase.PRE_TOOL_CALL and not success:
                # Pre-hooks can BLOCK execution
                return False, output or f"Blocked by hook: {hook.description}"

            if phase == HookPhase.POST_TOOL_CALL and success and output:
                # Post-hooks can MODIFY output
                return True, output

        return True, None

    def _execute(self, command: str, timeout: int) -> tuple[bool, str]:
        """
        Execute a hook command.

        Returns:
            (success, output)
            - success: True if exit code == 0
            - output: stdout (success) or stderr (failure)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "CLAUDE_WORKDIR": str(self.workdir)},
            )

            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, (result.stderr or result.stdout).strip()

        except subprocess.TimeoutExpired:
            return False, f"Hook timed out ({timeout}s)"
        except Exception as e:
            return False, str(e)

    def has_hooks(self, phase: HookPhase) -> bool:
        """Check if any hooks exist for a phase."""
        return bool(self.hooks.get(phase))

    def list_hooks(self) -> str:
        """List all configured hooks."""
        if not any(self.hooks.values()):
            return "No hooks configured.\nCreate .claude/hooks.yaml to add hooks."

        lines = ["Configured hooks:"]
        for phase, hooks in self.hooks.items():
            if hooks:
                lines.append(f"\n  {phase.value}:")
                for h in hooks:
                    tool_filter = f" [tool={h.tool}]" if h.tool != "*" else ""
                    desc = f" - {h.description}" if h.description else ""
                    lines.append(f"    {h.command}{tool_filter}{desc}")

        return "\n".join(lines)


# Global hook manager
HOOKS = HookManager(CLAUDE_DIR, WORKDIR)


# =============================================================================
# Permission, Plan Mode, Task Manager, Context, Memory (from v5-v9)
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
    def __init__(self, workdir):
        self.workdir = workdir
        self.session_grants = {}
        self.rules = {
            "read": [PermissionRule("*", Permission.ALLOW, "Safe")],
            "write": [PermissionRule("*.md", Permission.ALLOW, "Docs"), PermissionRule("*secret*", Permission.DENY, "Secret"), PermissionRule("*", Permission.ASK, "Write")],
            "exec": [PermissionRule("ls*", Permission.ALLOW, "List"), PermissionRule("pwd", Permission.ALLOW, "Cwd"), PermissionRule("git status*", Permission.ALLOW, "Git"), PermissionRule("git diff*", Permission.ALLOW, "Git"), PermissionRule("git log*", Permission.ALLOW, "Git"), PermissionRule("rm -rf /*", Permission.DENY, "Danger"), PermissionRule("sudo *", Permission.DENY, "Sudo"), PermissionRule("*", Permission.ASK, "Unknown")],
        }
    def check(self, cat, op):
        key = f"{cat}:{op.split()[0]}*" if " " in op else f"{cat}:{op}"
        if key in self.session_grants: return Permission.ALLOW, "Session"
        for r in self.rules.get(cat, []):
            if fnmatch.fnmatch(op, r.pattern): return r.permission, r.reason
        return Permission.ASK, "Default"
    def grant_session(self, cat, op):
        key = f"{cat}:{op.split()[0]}*" if " " in op else f"{cat}:{op}"
        self.session_grants[key] = True

def ask_user_permission(op, reason, cat):
    print(f"\n┌─ Permission: {cat} {'─'*30}┐\n│ {op[:50]}\n│ [y/n/a]")
    while True:
        r = input("└─> ").strip().lower()
        if r in ("y","yes"): return True, False
        if r in ("n","no"): return False, False
        if r in ("a","always"): return True, True

PERMISSIONS = PermissionManager(WORKDIR)

class AgentMode(Enum):
    NORMAL = "normal"
    PLANNING = "planning"

class ModeManager:
    """See v7."""
    def __init__(self):
        self.mode = AgentMode.NORMAL
        self.current_plan = None
    def enter_plan_mode(self):
        if self.mode == AgentMode.PLANNING: return "Already planning."
        self.mode = AgentMode.PLANNING
        self.current_plan = None
        return "PLAN MODE. Read-only tools only."
    def exit_plan_mode(self, plan):
        if self.mode != AgentMode.PLANNING: return "Error: Not planning."
        self.current_plan = plan
        return f"\n{'='*50}\nPLAN:\n{plan}\n{'='*50}\n\"approve\" / \"revise: ...\" / \"cancel\""
    def handle_user_response(self, inp):
        if not self.current_plan: return "not_pending", None
        n = inp.strip().lower()
        if n in ("approve","yes","y","ok"):
            p = self.current_plan; self.mode = AgentMode.NORMAL; self.current_plan = None; return "approve", p
        if n in ("cancel","no","n"):
            self.mode = AgentMode.NORMAL; self.current_plan = None; return "cancel", None
        notes = inp[7:].strip() if inp.lower().startswith("revise") else inp
        self.current_plan = None; return "revise", notes
    def get_available_tools(self, tools):
        if self.mode == AgentMode.PLANNING:
            ok = {"Glob","Grep","Read","TodoWrite","ExitPlanMode"}
            return [t for t in tools if t["name"] in ok]
        return tools
    def get_mode_prompt(self):
        return "\n\n*** PLAN MODE ***" if self.mode == AgentMode.PLANNING else ""

MODE = ModeManager()

class TaskStatus(Enum):
    PENDING="pending"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"

@dataclass
class SubagentTask:
    id: str; description: str; agent_type: str; prompt: str
    status: TaskStatus = TaskStatus.PENDING; result: Optional[str] = None
    error: Optional[str] = None; thread: Optional[threading.Thread] = None
    created_at: float = field(default_factory=time.time); completed_at: Optional[float] = None

class TaskManager:
    """See v8."""
    def __init__(self): self.tasks = {}; self._lock = threading.Lock()
    def start_task(self, desc, prompt, atype, bg=False, tid=None):
        tid = tid or f"task-{uuid.uuid4().hex[:6]}"
        task = SubagentTask(id=tid, description=desc, agent_type=atype, prompt=prompt)
        with self._lock: self.tasks[tid] = task
        if bg:
            t = threading.Thread(target=self._exec, args=(tid,), daemon=True)
            task.thread = t; task.status = TaskStatus.RUNNING; t.start(); return tid, None
        return tid, self._exec(tid)
    def _exec(self, tid):
        task = self.tasks[tid]; task.status = TaskStatus.RUNNING
        cfg = AGENT_TYPES.get(task.agent_type)
        if not cfg: task.status = TaskStatus.FAILED; task.error = "Bad type"; return task.error
        sys_p = f"You are a {task.agent_type} subagent at {WORKDIR}.\n{cfg['prompt']}\nReturn concise summary."
        tools = get_tools_for_agent(task.agent_type)
        msgs = [{"role": "user", "content": task.prompt}]
        try:
            while True:
                r = client.messages.create(model=MODEL, system=sys_p, messages=msgs, tools=tools, max_tokens=8000)
                if r.stop_reason != "tool_use":
                    res = "".join(b.text for b in r.content if hasattr(b, "text"))
                    task.status = TaskStatus.COMPLETED; task.result = res; task.completed_at = time.time(); return res
                results = []
                for tc in [b for b in r.content if b.type == "tool_use"]:
                    out = execute_tool_for_subagent(tc.name, tc.input)
                    results.append({"type": "tool_result", "tool_use_id": tc.id, "content": out})
                msgs.append({"role": "assistant", "content": r.content})
                msgs.append({"role": "user", "content": results})
        except Exception as e:
            task.status = TaskStatus.FAILED; task.error = str(e); return f"Error: {e}"
    def get_output(self, tid, wait=True, timeout=300):
        t = self.tasks.get(tid)
        if not t: return f"Not found: {tid}"
        if t.status == TaskStatus.COMPLETED: return t.result
        if t.status == TaskStatus.FAILED: return f"Failed: {t.error}"
        if not wait: return f"Running ({time.time()-t.created_at:.0f}s)"
        if t.thread and t.thread.is_alive(): t.thread.join(timeout=timeout)
        return t.result if t.status == TaskStatus.COMPLETED else f"Timeout"
    def list_tasks(self):
        if not self.tasks: return "No tasks."
        lines = ["ID            | Status     | Description"]
        for t in sorted(self.tasks.values(), key=lambda x: x.created_at):
            lines.append(f"{t.id:13} | {t.status.value:10} | {t.description[:30]}")
        return "\n".join(lines)

TASKS = TaskManager()

def estimate_tokens(text): return len(text) // 4
def estimate_messages_tokens(msgs):
    total = 0
    for m in msgs:
        c = m.get("content", "")
        total += estimate_tokens(json.dumps(c, default=str) if not isinstance(c, str) else c)
    return total

class ContextCompressor:
    """See v9."""
    def __init__(self, threshold=25000, keep=8):
        self.threshold = threshold; self.keep = keep
    def should_compress(self, msgs): return estimate_messages_tokens(msgs) > self.threshold
    def compress(self, msgs):
        if len(msgs) <= self.keep: return msgs
        old, recent = msgs[:-self.keep], msgs[-self.keep:]
        summary = self._summarize(old)
        return [{"role":"user","content":f"<summary>{summary}</summary>"},
                {"role":"assistant","content":"Context noted."}] + recent
    def _summarize(self, msgs):
        parts = []
        for m in msgs:
            c = m.get("content","")
            if isinstance(c, str): parts.append(f"{m['role']}: {c[:300]}")
        try:
            r = client.messages.create(model=MODEL, max_tokens=500,
                messages=[{"role":"user","content":f"Summarize concisely:\n{chr(10).join(parts[-20:])}"}])
            return r.content[0].text
        except: return "\n".join(parts[-5:])

COMPRESSOR = ContextCompressor()

class MemoryManager:
    """See v9."""
    VALID_SECTIONS = ["Architecture","Patterns","User Preferences","Known Issues","Dependencies"]
    def __init__(self, mdir):
        self.mfile = mdir / "MEMORY.md"; self.sections = {}; self._load()
    def _load(self):
        self.sections = {s: [] for s in self.VALID_SECTIONS}
        if not self.mfile.exists(): return
        cur = None
        for line in self.mfile.read_text().splitlines():
            if line.startswith("## "): cur = line[3:].strip(); self.sections.setdefault(cur, [])
            elif cur and line.startswith("- "): self.sections[cur].append(line[2:].strip())
    def add(self, section, fact):
        if section not in self.sections: return f"Unknown section: {section}"
        if fact in self.sections[section]: return "Already exists"
        self.sections[section].append(fact); self._save(); return f"Saved: {fact}"
    def get_context(self):
        pop = {s:f for s,f in self.sections.items() if f}
        if not pop: return ""
        lines = ["<memory>"]
        for s, facts in pop.items():
            lines.append(f"## {s}")
            lines.extend(f"- {f}" for f in facts)
        lines.append("</memory>"); return "\n".join(lines)
    def read_all(self):
        if not any(self.sections.values()): return "No memories."
        lines = []
        for s, facts in self.sections.items():
            lines.append(f"## {s}")
            lines.extend(f"- {f}" for f in facts) if facts else lines.append("(empty)")
        return "\n".join(lines)
    def _save(self):
        self.mfile.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Project Memory", ""]
        for s in self.VALID_SECTIONS:
            lines.append(f"## {s}")
            lines.extend(f"- {f}" for f in self.sections.get(s, [])); lines.append("")
        self.mfile.write_text("\n".join(lines))

MEMORY = MemoryManager(MEMORY_DIR)


# =============================================================================
# Specialized tools, Skills, Agent types, Todo (compact, see v4-v6)
# =============================================================================

EXCLUDE_DIRS = {'.git','node_modules','__pycache__','.venv','venv','.idea','dist','build'}

def run_glob(pattern, path=None):
    try:
        base = (WORKDIR/path).resolve() if path else WORKDIR
        if not base.is_relative_to(WORKDIR): return "Error: escape"
        res = [(p.stat().st_mtime, p) for p in base.glob(pattern)
               if p.is_file() and not any(ex in p.parts for ex in EXCLUDE_DIRS)]
        res.sort(reverse=True)
        return "\n".join(str(p.relative_to(WORKDIR)) for _,p in res[:100]) or "No matches."
    except Exception as e: return f"Error: {e}"

def run_grep(pattern, path=None, glob_pat=None, mode="files_with_matches", ctx=0):
    try:
        base = (WORKDIR/path).resolve() if path else WORKDIR
        regex = re.compile(pattern); results = []
        for fp in base.glob(glob_pat or "**/*"):
            if not fp.is_file() or any(ex in fp.parts for ex in EXCLUDE_DIRS): continue
            try: content = fp.read_text(errors='ignore')
            except: continue
            matches = [(i+1,l) for i,l in enumerate(content.splitlines()) if regex.search(l)]
            if matches:
                rel = str(fp.relative_to(WORKDIR))
                if mode == "files_with_matches": results.append(rel)
                elif mode == "count": results.append(f"{rel}: {len(matches)}")
                else:
                    results.append(f"\n{rel}:")
                    results.extend(f"  {n:>5}: {l[:200]}" for n,l in matches[:10])
            if len(results) >= 100: break
        return "\n".join(results) or "No matches."
    except Exception as e: return f"Error: {e}"

def run_read_file(fpath, offset=1, limit=2000):
    try:
        fp = (WORKDIR/fpath).resolve()
        if not fp.is_relative_to(WORKDIR) or not fp.exists(): return f"Error: {fpath}"
        if fp.suffix.lower() in (".png",".jpg",".jpeg"): return f"[Image: {fpath}]"
        lines = fp.read_text().splitlines(); total = len(lines)
        s, e = max(0,offset-1), min(total, max(0,offset-1)+limit)
        out = [f"{s+i+1:>6}| {l[:500]}" for i,l in enumerate(lines[s:e])]
        r = "\n".join(out)
        if e < total: r += f"\n... {total-e} more lines."
        return r
    except Exception as e: return f"Error: {e}"

class SkillLoader:
    """See v4."""
    def __init__(self, sd):
        self.sd = sd; self.skills = {}; self.load()
    def load(self):
        if not self.sd.exists(): return
        for d in self.sd.iterdir():
            if d.is_dir() and (d/"SKILL.md").exists():
                c = (d/"SKILL.md").read_text()
                m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", c, re.DOTALL)
                if not m: continue
                meta = {}
                for l in m.group(1).strip().split("\n"):
                    if ":" in l: k,v = l.split(":",1); meta[k.strip()] = v.strip().strip("\"'")
                if "name" in meta and "description" in meta:
                    self.skills[meta["name"]] = {"name":meta["name"],"description":meta["description"],"body":m.group(2).strip(),"dir":d}
    def get_descriptions(self):
        return "\n".join(f"- {n}: {s['description']}" for n,s in self.skills.items()) or "(none)"
    def get_content(self, name):
        s = self.skills.get(name)
        return f"# Skill: {s['name']}\n\n{s['body']}" if s else None
    def list(self): return list(self.skills.keys())

SKILLS = SkillLoader(SKILLS_DIR)

AGENT_TYPES = {
    "explore": {"description": "Read-only exploration", "tools": ["Glob","Grep","Read","bash"], "prompt": "Explore and summarize."},
    "code": {"description": "Full implementation", "tools": "*", "prompt": "Implement efficiently."},
    "plan": {"description": "Planning", "tools": ["Glob","Grep","Read"], "prompt": "Plan, don't modify."},
}
def get_agent_descriptions():
    return "\n".join(f"- {n}: {c['description']}" for n,c in AGENT_TYPES.items())

class TodoManager:
    """See v2."""
    def __init__(self): self.items = []
    def update(self, items):
        val, ip = [], 0
        for i, it in enumerate(items):
            c,s,a = str(it.get("content","")).strip(), str(it.get("status","pending")).lower(), str(it.get("activeForm","")).strip()
            if not c or not a: raise ValueError(f"Item {i}: required")
            if s not in ("pending","in_progress","completed"): raise ValueError(f"Item {i}: bad status")
            if s == "in_progress": ip += 1
            val.append({"content":c,"status":s,"activeForm":a})
        if ip > 1: raise ValueError("One in_progress max")
        self.items = val[:20]; return self.render()
    def render(self):
        if not self.items: return "No todos."
        lines = [f"{'[x]' if t['status']=='completed' else '[>]' if t['status']=='in_progress' else '[ ]'} {t['content']}" for t in self.items]
        return "\n".join(lines) + f"\n({sum(1 for t in self.items if t['status']=='completed')}/{len(self.items)} done)"

TODO = TodoManager()

# =============================================================================
# Tool Definitions
# =============================================================================

GLOB_TOOL = {"name":"Glob","description":"Find files by glob pattern.","input_schema":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}
GREP_TOOL = {"name":"Grep","description":"Search contents. Modes: files_with_matches, content, count.","input_schema":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"glob":{"type":"string"},"output_mode":{"type":"string","enum":["files_with_matches","content","count"]}},"required":["pattern"]}}
READ_TOOL = {"name":"Read","description":"Read file with line numbers.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["file_path"]}}

BASE_TOOLS = [
    {"name":"bash","description":"Run shell command.","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}},
    {"name":"write_file","description":"Write to file.","input_schema":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}},
    {"name":"edit_file","description":"Replace text.","input_schema":{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]}},
    {"name":"TodoWrite","description":"Update tasks.","input_schema":{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]},"activeForm":{"type":"string"}},"required":["content","status","activeForm"]}}},"required":["items"]}},
]

TASK_TOOL = {"name":"Task","description":f"Spawn subagent. Types: {', '.join(AGENT_TYPES)}. background=true for parallel.","input_schema":{"type":"object","properties":{"description":{"type":"string"},"prompt":{"type":"string"},"agent_type":{"type":"string","enum":list(AGENT_TYPES)},"background":{"type":"boolean"},"task_id":{"type":"string"}},"required":["description","prompt","agent_type"]}}
TASK_OUTPUT_TOOL = {"name":"TaskOutput","description":"Get background task result.","input_schema":{"type":"object","properties":{"task_id":{"type":"string"},"wait":{"type":"boolean"},"timeout":{"type":"number"}},"required":["task_id"]}}
TASK_LIST_TOOL = {"name":"TaskList","description":"List tasks.","input_schema":{"type":"object","properties":{}}}
SKILL_TOOL = {"name":"Skill","description":f"Load skill. Available: {SKILLS.get_descriptions()}","input_schema":{"type":"object","properties":{"skill":{"type":"string"}},"required":["skill"]}}
ENTER_PLAN = {"name":"EnterPlanMode","description":"Enter read-only plan mode.","input_schema":{"type":"object","properties":{}}}
EXIT_PLAN = {"name":"ExitPlanMode","description":"Submit plan.","input_schema":{"type":"object","properties":{"plan":{"type":"string"}},"required":["plan"]}}
MEM_WRITE = {"name":"MemoryWrite","description":f"Save fact. Sections: {', '.join(MemoryManager.VALID_SECTIONS)}","input_schema":{"type":"object","properties":{"section":{"type":"string","enum":MemoryManager.VALID_SECTIONS},"fact":{"type":"string"}},"required":["section","fact"]}}
MEM_READ = {"name":"MemoryRead","description":"Read all memory.","input_schema":{"type":"object","properties":{}}}

ALL_TOOLS = [GLOB_TOOL, GREP_TOOL, READ_TOOL] + BASE_TOOLS + [TASK_TOOL, TASK_OUTPUT_TOOL, TASK_LIST_TOOL, SKILL_TOOL, ENTER_PLAN, EXIT_PLAN, MEM_WRITE, MEM_READ]

def get_tools_for_agent(atype):
    allowed = AGENT_TYPES.get(atype,{}).get("tools","*")
    if allowed == "*": return [GLOB_TOOL,GREP_TOOL,READ_TOOL] + BASE_TOOLS
    return [t for t in [GLOB_TOOL,GREP_TOOL,READ_TOOL]+BASE_TOOLS if t["name"] in allowed]

# =============================================================================
# System Prompt
# =============================================================================

def get_system_prompt():
    mem = MEMORY.get_context()
    mem_section = f"\n\n**Memory:**\n{mem}" if mem else ""
    hooks_note = f"\n\nHooks: {len([h for hs in HOOKS.hooks.values() for h in hs])} configured" if any(HOOKS.hooks.values()) else ""

    return f"""You are a coding agent at {WORKDIR}.

**Tools**: Glob, Grep, Read, bash, write_file, edit_file
**Tasks**: Task (background), TaskOutput, TaskList
**Plan**: EnterPlanMode / ExitPlanMode
**Memory**: MemoryWrite, MemoryRead
**Skills**: {SKILLS.get_descriptions()}{mem_section}{hooks_note}

Rules:
- Save discoveries with MemoryWrite
- Parallel work: Task(background=true) + TaskOutput
- Complex tasks: EnterPlanMode first
- Glob/Grep/Read for file ops{MODE.get_mode_prompt()}"""

# =============================================================================
# Tool Implementations - With Hook Integration
# =============================================================================

def safe_path(p):
    path = (WORKDIR/p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Escape: {p}")
    return path

def run_bash(cmd):
    perm, reason = PERMISSIONS.check("exec", cmd)
    if perm == Permission.DENY: return f"Denied: {reason}"
    if perm == Permission.ASK:
        ok, rem = ask_user_permission(cmd, reason, "exec")
        if not ok: return "Denied"
        if rem: PERMISSIONS.grant_session("exec", cmd)
    try:
        r = subprocess.run(cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
        return (r.stdout+r.stderr).strip() or "(no output)"
    except Exception as e: return f"Error: {e}"

def run_write(path, content):
    perm, reason = PERMISSIONS.check("write", path)
    if perm == Permission.DENY: return f"Denied: {reason}"
    if perm == Permission.ASK:
        ok, rem = ask_user_permission(f"write {path}", reason, "write")
        if not ok: return "Denied"
        if rem: PERMISSIONS.grant_session("write", path)
    try:
        fp = safe_path(path); fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content); return f"Wrote {len(content)} bytes to {path}"
    except Exception as e: return f"Error: {e}"

def run_edit(path, old_text, new_text):
    perm, reason = PERMISSIONS.check("write", path)
    if perm == Permission.DENY: return f"Denied: {reason}"
    if perm == Permission.ASK:
        ok, rem = ask_user_permission(f"edit {path}", reason, "write")
        if not ok: return "Denied"
        if rem: PERMISSIONS.grant_session("write", path)
    try:
        fp = safe_path(path); text = fp.read_text()
        if old_text not in text: return f"Text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1)); return f"Edited {path}"
    except Exception as e: return f"Error: {e}"


def execute_tool_with_hooks(name: str, args: dict) -> str:
    """
    Execute a tool with pre/post hook support.

    This is THE key addition in v10:
    1. Run pre_tool_call hooks (can block execution)
    2. Execute the tool
    3. Run post_tool_call hooks (can modify output)
    """
    # PRE hooks
    should_continue, blocked_msg = HOOKS.trigger(
        HookPhase.PRE_TOOL_CALL,
        tool_name=name,
        tool_args=args,
    )
    if not should_continue:
        return blocked_msg or f"Blocked by pre-hook for {name}"

    # Execute tool
    result = _execute_tool_raw(name, args)

    # POST hooks
    _, modified = HOOKS.trigger(
        HookPhase.POST_TOOL_CALL,
        tool_name=name,
        tool_args=args,
        tool_result=result,
    )

    return modified if modified else result


def _execute_tool_raw(name: str, args: dict) -> str:
    """Raw tool execution without hooks."""
    if name == "Glob": return run_glob(args["pattern"], args.get("path"))
    if name == "Grep": return run_grep(args["pattern"], args.get("path"), args.get("glob"), args.get("output_mode","files_with_matches"), 0)
    if name == "Read": return run_read_file(args["file_path"], args.get("offset",1), args.get("limit",2000))
    if name == "EnterPlanMode": return MODE.enter_plan_mode()
    if name == "ExitPlanMode": return MODE.exit_plan_mode(args["plan"])
    if name == "Task":
        if args["agent_type"] not in AGENT_TYPES: return f"Bad type: {args['agent_type']}"
        tid, res = TASKS.start_task(args["description"], args["prompt"], args["agent_type"], args.get("background",False), args.get("task_id"))
        return f"Started: {tid}" if args.get("background") else res
    if name == "TaskOutput": return TASKS.get_output(args["task_id"], args.get("wait",True), args.get("timeout",300))
    if name == "TaskList": return TASKS.list_tasks()
    if name == "bash": return run_bash(args["command"])
    if name == "write_file": return run_write(args["path"], args["content"])
    if name == "edit_file": return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        try: return TODO.update(args["items"])
        except Exception as e: return f"Error: {e}"
    if name == "Skill":
        c = SKILLS.get_content(args["skill"])
        return f'<skill>\n{c}\n</skill>' if c else f"Unknown: {args['skill']}"
    if name == "MemoryWrite": return MEMORY.add(args["section"], args["fact"])
    if name == "MemoryRead": return MEMORY.read_all()
    return f"Unknown: {name}"


def execute_tool_for_subagent(name, args):
    """Subagent tools (no hooks, no Task/Memory/Plan)."""
    if name == "Glob": return run_glob(args["pattern"], args.get("path"))
    if name == "Grep": return run_grep(args["pattern"], args.get("path"), args.get("glob"), args.get("output_mode","files_with_matches"), 0)
    if name == "Read": return run_read_file(args["file_path"], args.get("offset",1), args.get("limit",2000))
    if name == "bash": return run_bash(args["command"])
    if name == "write_file": return run_write(args["path"], args["content"])
    if name == "edit_file": return run_edit(args["path"], args["old_text"], args["new_text"])
    return f"Not available: {name}"


# =============================================================================
# Main Agent Loop - With Full Hook Integration
# =============================================================================

def agent_loop(messages: list) -> list:
    while True:
        # Context compression (v9)
        if COMPRESSOR.should_compress(messages):
            old_t = estimate_messages_tokens(messages)
            messages[:] = COMPRESSOR.compress(messages)
            print(f"\n[Compressed: ~{old_t} -> ~{estimate_messages_tokens(messages)} tokens]")

        available = MODE.get_available_tools(ALL_TOOLS)
        system = get_system_prompt()

        log_api_call("main", system, messages, available)
        response = client.messages.create(model=MODEL, system=system, messages=messages, tools=available, max_tokens=8000)
        log_api_response("main", response)

        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"): print(block.text)
            if block.type == "tool_use": tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role":"assistant","content":response.content})
            # TURN_END hook
            HOOKS.trigger(HookPhase.TURN_END)
            return messages

        results = []
        for tc in tool_calls:
            print(f"\n> {tc.name}")

            # v10: Execute with hooks
            output = execute_tool_with_hooks(tc.name, tc.input)

            # Display (abbreviated)
            if tc.name in ("EnterPlanMode","ExitPlanMode","TaskList","MemoryRead"):
                print(output)
            elif len(output) > 300:
                print(f"  {output[:300]}...")
            else:
                print(f"  {output}")

            results.append({"type":"tool_result","tool_use_id":tc.id,"content":output})

        messages.append({"role":"assistant","content":response.content})
        messages.append({"role":"user","content":results})

        # TURN_END hook
        HOOKS.trigger(HookPhase.TURN_END)


# =============================================================================
# Main REPL - With Session Hooks
# =============================================================================

def main():
    print(f"Mini Claude Code v10 (with Lifecycle Hooks) - {WORKDIR}")
    print(f"Skills: {', '.join(SKILLS.list()) or 'none'}")
    hooks_count = sum(len(hs) for hs in HOOKS.hooks.values())
    print(f"Hooks: {hooks_count} configured" if hooks_count else "Hooks: none (create .claude/hooks.yaml)")
    print("Type 'exit' to quit.\n")

    # SESSION_START hook
    HOOKS.trigger(HookPhase.SESSION_START)

    history = []

    try:
        while True:
            try:
                mode_ind = " [PLAN]" if MODE.mode == AgentMode.PLANNING else ""
                user_input = input(f"You{mode_ind}: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in ("exit","quit","q"):
                break

            # USER_INPUT hook
            HOOKS.trigger(HookPhase.USER_INPUT, user_input=user_input)

            # Plan approval
            if MODE.current_plan is not None:
                action, ctx = MODE.handle_user_response(user_input)
                if action == "approve":
                    print("\n Approved.\n")
                    history.append({"role":"user","content":f"Execute:\n\n{ctx}"})
                elif action == "cancel":
                    print("\n Cancelled.\n"); continue
                elif action == "revise":
                    history.append({"role":"user","content":f"Revise: {ctx}"})
                try: agent_loop(history)
                except Exception as e: print(f"Error: {e}")
                print(); continue

            history.append({"role":"user","content":user_input})
            try: agent_loop(history)
            except Exception as e: print(f"Error: {e}")
            print()

    finally:
        # SESSION_END hook
        HOOKS.trigger(HookPhase.SESSION_END)
        print("\nSession ended.")


if __name__ == "__main__":
    main()
