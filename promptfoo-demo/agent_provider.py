#!/usr/bin/env python3
"""
Promptfoo Provider - 真正调用 v4_skills_agent

直接 import 并调用你的 agent，捕获真实的工具调用和输出
"""
import json
import sys
import os
import io
from contextlib import redirect_stdout

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# 直接 import 你的 agent 模块
import v4_skills_agent as agent

# 记录工具调用
_tool_calls = []
_original_execute_tool = agent.execute_tool


def patched_execute_tool(name: str, args: dict) -> str:
    """
    包装原有的 execute_tool，记录所有工具调用
    """
    _tool_calls.append({"name": name, "input": args})
    return _original_execute_tool(name, args)


# 替换为包装版本
agent.execute_tool = patched_execute_tool


def call_agent(prompt: str) -> dict:
    """
    调用真实的 agent 并返回结果
    """
    global _tool_calls
    _tool_calls = []  # 重置

    messages = [{"role": "user", "content": prompt}]

    # 捕获 stdout 输出
    output_buffer = io.StringIO()

    try:
        with redirect_stdout(output_buffer):
            result_messages = agent.agent_loop(messages)

        # 提取 assistant 的最终回复
        final_output = ""
        for msg in reversed(result_messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            final_output = block.text
                            break
                        elif isinstance(block, dict) and block.get("type") == "text":
                            final_output = block.get("text", "")
                            break
                break

        return {
            "output": final_output or output_buffer.getvalue(),
            "metadata": {
                "tool_calls": _tool_calls,
                "tool_count": len(_tool_calls),
                "console_output": output_buffer.getvalue()
            }
        }

    except Exception as e:
        return {
            "output": f"Error: {e}",
            "error": str(e),
            "metadata": {
                "tool_calls": _tool_calls,
                "console_output": output_buffer.getvalue()
            }
        }


def main():
    """
    Promptfoo 入口点
    从 stdin 读取 JSON，调用 agent，输出 JSON 到 stdout
    """
    input_data = json.loads(sys.stdin.read())
    prompt = input_data.get("prompt", "")

    result = call_agent(prompt)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
