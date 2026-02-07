#!/usr/bin/env python3
"""
v4plus_web.py - Web frontend for v4plus_skills_agent.py

Wraps the existing CLI agent with a FastAPI server + SSE streaming.

Usage:
    pip install fastapi uvicorn
    python v4plus_web.py
    # Open http://localhost:8000
"""

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import v4plus_skills_agent as agent

app = FastAPI(title="Agent v4 Web UI")

# Thread-safe event queue for SSE
event_queue = queue.Queue()

# Conversation history (shared with agent)
history = []

# Lock to prevent concurrent agent runs
agent_lock = threading.Lock()


def event_callback(event: dict):
    """Receive events from the agent and push to SSE queue."""
    try:
        # Pre-validate JSON serialization
        json.dumps(event, ensure_ascii=False, default=str)
        event_queue.put(event)
    except Exception:
        event_queue.put({"type": "error", "message": "Event serialization failed"})


# Wire up the callback
agent.event_callback = event_callback


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML."""
    html_path = Path(__file__).parent / "static" / "v4plus_index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(request: Request):
    """Accept user message and start agent processing in background thread."""
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"error": "Empty message"}

    if agent_lock.locked():
        return {"error": "Agent is busy"}

    # Drain any leftover events from previous run
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except queue.Empty:
            break

    history.append({"role": "user", "content": message})

    def run():
        with agent_lock:
            try:
                agent.agent_loop(history)
            except Exception as e:
                event_queue.put({"type": "error", "message": str(e)})
                event_queue.put({"type": "done"})

    threading.Thread(target=run, daemon=True).start()
    return {"status": "ok"}


@app.get("/api/chat/stream")
async def stream():
    """SSE endpoint - streams agent events to the browser."""
    import asyncio

    async def generate():
        while True:
            try:
                event = event_queue.get_nowait()
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                if event.get("type") == "done":
                    break
            except queue.Empty:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.3)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/skills")
async def skills():
    """Return available skills."""
    return [
        {"name": name, "description": skill["description"]}
        for name, skill in agent.SKILLS.skills.items()
    ]


@app.get("/api/agents")
async def agents():
    """Return available agent types."""
    return [
        {"name": name, "description": cfg["description"]}
        for name, cfg in agent.AGENT_TYPES.items()
    ]


if __name__ == "__main__":
    import uvicorn

    print(f"Starting Agent v4 Web UI")
    print(f"Skills: {', '.join(agent.SKILLS.list_skills()) or 'none'}")
    print(f"Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
