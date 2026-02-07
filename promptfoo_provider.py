"""
Promptfoo Provider for v4_skills_agent.

Zero-intrusion adapter - wraps agent_loop() as a black-box callable.

Usage:
    npx promptfoo eval
    npx promptfoo eval -o results.json        # JSON output
    npx promptfoo eval -o results.html        # HTML report

See promptfooconfig.yaml for test case definitions.
"""

import v4_skills_agent as agent


def call_agent(prompt, options, context):
    """
    Promptfoo calls this function for each test case.

    Args:
        prompt: User input string (from vars or prompts)
        options: Test config variables
        context: Promptfoo context

    Returns:
        dict with "output" (success) or "error" (failure)
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        result_messages = agent.agent_loop(messages)

        # Extract the last assistant message text
        for msg in reversed(result_messages):
            if msg.get("role") == "assistant":
                content = msg["content"]
                # content is a list of Anthropic ContentBlock objects
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                    if texts:
                        return {"output": "\n".join(texts)}
                # Fallback: plain string
                if isinstance(content, str):
                    return {"output": content}

        return {"output": "(no response)"}

    except Exception as e:
        return {"error": str(e)}
