"""
Integration tests for learn-claude-code agents.

Real agent loop tests that run on GitHub Actions (Linux).
"""
import os
import sys
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_client():
    """Get OpenAI-compatible client for testing."""
    from openai import OpenAI
    api_key = os.getenv("TEST_API_KEY")
    base_url = os.getenv("TEST_BASE_URL", "https://api.openai-next.com/v1")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


MODEL = os.getenv("TEST_MODEL", "claude-3-5-sonnet-20241022")

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    }
}


def run_agent_loop(client, task, tools, max_turns=10):
    """
    Run a complete agent loop until done or max_turns.
    Returns (final_response, tool_calls_made)
    """
    import subprocess

    messages = [
        {"role": "system", "content": "You are a coding agent. Use tools to complete tasks. Be concise."},
        {"role": "user", "content": task}
    ]

    tool_calls_made = []

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            max_tokens=1000
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # No tool calls, we're done
        if finish_reason == "stop" or not message.tool_calls:
            return message.content, tool_calls_made

        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_calls_made.append((func_name, args))

            if func_name == "bash":
                cmd = args.get("command", "")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    output = result.stdout + result.stderr
                except Exception as e:
                    output = f"Error: {e}"
            else:
                output = f"Unknown tool: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output or "(empty)"
            })

    return None, tool_calls_made


# =============================================================================
# Test Cases
# =============================================================================

def test_bash_echo():
    """Test: Agent can run simple bash command."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    response, calls = run_agent_loop(
        client,
        "Run 'echo hello world' and tell me what it outputs.",
        [BASH_TOOL]
    )

    assert len(calls) >= 1, "Should have made at least 1 tool call"
    assert any("echo" in str(c) for c in calls), "Should have run echo command"
    assert response and "hello" in response.lower(), f"Response should mention hello: {response}"

    print(f"Tool calls: {calls}")
    print(f"Response: {response}")
    print("PASS: test_bash_echo")
    return True


def test_file_creation():
    """Test: Agent can create and verify a file."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")

        response, calls = run_agent_loop(
            client,
            f"Create a file at {filepath} with content 'agent test' using echo, then verify it exists with cat.",
            [BASH_TOOL]
        )

        assert len(calls) >= 2, f"Should have made at least 2 tool calls: {calls}"
        assert os.path.exists(filepath), f"File should exist: {filepath}"

        with open(filepath) as f:
            content = f.read()
        assert "agent test" in content, f"File content wrong: {content}"

        print(f"Tool calls: {calls}")
        print(f"File content: {content}")
        print("PASS: test_file_creation")
        return True


def test_directory_listing():
    """Test: Agent can list directory contents."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        for name in ["foo.txt", "bar.py", "baz.md"]:
            open(os.path.join(tmpdir, name), "w").close()

        response, calls = run_agent_loop(
            client,
            f"List all files in {tmpdir} and tell me how many there are.",
            [BASH_TOOL]
        )

        assert len(calls) >= 1, "Should have made at least 1 tool call"
        assert response and "3" in response, f"Should find 3 files: {response}"

        print(f"Tool calls: {calls}")
        print(f"Response: {response}")
        print("PASS: test_directory_listing")
        return True


def test_file_search():
    """Test: Agent can search file contents with grep."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with different content
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("hello world\nfoo bar\n")
        with open(os.path.join(tmpdir, "b.txt"), "w") as f:
            f.write("goodbye world\nbaz qux\n")

        response, calls = run_agent_loop(
            client,
            f"Search for the word 'hello' in all .txt files in {tmpdir}. Which file contains it?",
            [BASH_TOOL]
        )

        assert len(calls) >= 1, "Should have made at least 1 tool call"
        assert response and "a.txt" in response, f"Should find a.txt: {response}"

        print(f"Tool calls: {calls}")
        print(f"Response: {response}")
        print("PASS: test_file_search")
        return True


def test_multi_step_task():
    """Test: Agent can complete multi-step file manipulation."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source.txt")
        with open(src, "w") as f:
            f.write("original content")

        response, calls = run_agent_loop(
            client,
            f"1. Read {src}, 2. Append ' - modified' to it, 3. Show the final content.",
            [BASH_TOOL]
        )

        assert len(calls) >= 2, f"Should have made multiple tool calls: {calls}"

        with open(src) as f:
            content = f.read()
        assert "modified" in content, f"File should be modified: {content}"

        print(f"Tool calls: {calls}")
        print(f"Final content: {content}")
        print("PASS: test_multi_step_task")
        return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_bash_echo,
        test_file_creation,
        test_directory_listing,
        test_file_search,
        test_multi_step_task,
    ]

    failed = []
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print('='*50)
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    print('='*50)

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
