#!/usr/bin/env python3
"""
browser_server.py - Standalone browser control HTTP server.

Implements the same REST API as moltbot's browser control server,
powered by Playwright. Compatible with v4_skills_agent.py's browser tool.

Usage:
    python browser_server.py                    # isolated profile (default)
    python browser_server.py --system-profile   # reuse system Chrome login state
    python browser_server.py --port 18791       # custom port

API:
    POST /start          - Launch browser
    POST /stop           - Stop browser
    GET  /tabs           - List tabs
    POST /tabs/open      - Open new tab
    POST /tabs/focus     - Focus/switch to a tab
    DELETE /tabs/<id>     - Close tab
    POST /navigate       - Navigate to URL
    GET  /snapshot       - Get page snapshot (AI format with element refs)
    POST /screenshot     - Take screenshot
    POST /act            - Perform action (click/type/press/hover/scroll/scrollIntoView/evaluate/wait/upload/close)
    GET  /console        - Get console messages
    GET  /errors         - Get console errors
    GET  /cookies        - Get cookies
    POST /cookies/set    - Set cookies
    POST /cookies/clear  - Clear cookies
    POST /hooks/dialog   - Dialog handling (mode/accept/dismiss)
    GET  /requests       - Get captured network requests
    POST /response/body  - Get response body for a URL
    POST /upload         - Upload files (file chooser or direct input)
    GET  /download       - List downloads
    POST /wait/download  - Wait for and capture a download
    GET  /storage/local  - Get localStorage
    POST /storage/local  - Set/remove/clear localStorage
    GET  /storage/session - Get sessionStorage
    POST /storage/session - Set/remove/clear sessionStorage
    GET  /               - Server status
"""

import argparse
import json
import os
import platform
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

try:
    from playwright_stealth import stealth_sync
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Cookie consent / popup auto-dismiss patterns
# ---------------------------------------------------------------------------

_CONSENT_PATTERNS = [
    # Common cookie consent button texts (case-insensitive matching)
    "accept all", "accept cookies", "accept", "agree", "allow all",
    "allow cookies", "i agree", "i accept", "got it", "ok", "okay",
    "continue", "confirm", "consent", "close",
    # Chinese
    "同意", "接受", "我同意", "确定", "关闭", "知道了", "好的",
    # Japanese
    "同意する", "承諾",
]


def _try_dismiss_popups(page):
    """Best-effort auto-dismiss cookie consent and overlay popups."""
    try:
        # 1. Try ARIA button role matching
        for pattern in _CONSENT_PATTERNS:
            try:
                btn = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
                if btn.count() > 0 and btn.first.is_visible():
                    btn.first.click(timeout=2000)
                    return True
            except Exception:
                continue

        # 2. Try common CSS selectors for consent overlays
        for selector in [
            "[class*='cookie'] button",
            "[class*='consent'] button",
            "[class*='privacy'] button",
            "[id*='cookie'] button",
            "[id*='consent'] button",
            "#onetrust-accept-btn-handler",
            ".cc-accept",
            ".cc-dismiss",
            "[class*='modal'] button",
            "[class*='overlay'] button",
            "[class*='popup'] button",
            "[class*='dialog'] button",
        ]:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=500):
                    el.click(timeout=2000)
                    return True
            except Exception:
                continue

        # 3. Try clicking visible text directly (handles non-button elements
        #    like <div>同意</div> or <span>Accept</span>)
        for pattern in _CONSENT_PATTERNS:
            try:
                el = page.get_by_text(re.compile(f"^{re.escape(pattern)}$", re.IGNORECASE))
                if el.count() > 0 and el.first.is_visible():
                    el.first.click(timeout=2000)
                    return True
            except Exception:
                continue

    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# Global browser state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state = {
    "playwright": None,
    "browser": None,
    "context": None,
    "pages": {},          # targetId -> page
    "page_order": [],     # ordered targetIds
    "last_target": None,
    "console_msgs": [],   # [{level, text, url, timestamp}]
    "network_requests": [],  # [{method, url, status, resourceType, timestamp}]
    "network_responses": {},  # url -> {status, headers, body_path}
    "dialog_queue": [],   # [{type, message, default_value, handled}]
    "dialog_mode": "auto",  # "auto" = auto-accept, "manual" = queue for user
    "downloads": [],      # [{path, suggestedFilename, url}]
    "started": False,
    "system_profile": False,
    "headless": False,
}

# ---------------------------------------------------------------------------
# Snapshot: ARIA role/ref constants
# ---------------------------------------------------------------------------

INTERACTIVE_ROLES = {
    "link", "button", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "tab", "switch", "slider", "spinbutton", "searchbox",
    "option", "menuitemcheckbox", "menuitemradio", "treeitem", "listbox",
}

CONTENT_ROLES = {
    "heading", "cell", "gridcell", "columnheader", "rowheader",
    "listitem", "article", "region", "main", "navigation",
}


def _resolve_system_chrome_path():
    """Auto-detect system Chrome user data directory."""
    s = platform.system()
    home = Path.home()
    if s == "Darwin":
        return str(home / "Library" / "Application Support" / "Google" / "Chrome")
    elif s == "Linux":
        return str(home / ".config" / "google-chrome")
    elif s == "Windows":
        local = os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))
        return str(Path(local) / "Google" / "Chrome" / "User Data")
    return None


def _get_page(target_id=None):
    """Resolve a page by targetId, or return the last used / first available."""
    if target_id and target_id in _state["pages"]:
        _state["last_target"] = target_id
        return _state["pages"][target_id]["page"], target_id
    if _state["last_target"] and _state["last_target"] in _state["pages"]:
        return _state["pages"][_state["last_target"]]["page"], _state["last_target"]
    if _state["page_order"]:
        tid = _state["page_order"][0]
        _state["last_target"] = tid
        return _state["pages"][tid]["page"], tid
    return None, None


def _get_page_entry(target_id=None):
    """Resolve a page entry dict by targetId."""
    if target_id and target_id in _state["pages"]:
        _state["last_target"] = target_id
        return _state["pages"][target_id], target_id
    if _state["last_target"] and _state["last_target"] in _state["pages"]:
        return _state["pages"][_state["last_target"]], _state["last_target"]
    if _state["page_order"]:
        tid = _state["page_order"][0]
        _state["last_target"] = tid
        return _state["pages"][tid], tid
    return None, None


def _make_target_id(page):
    """Generate a short target ID for a page."""
    import hashlib
    raw = f"{id(page)}-{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12].upper()


def _register_page(page):
    """Register a page and set up console listener, dialog handler, stealth, etc."""
    tid = _make_target_id(page)
    _state["pages"][tid] = {"page": page, "refs": {}, "frame_selector": None}
    _state["page_order"].append(tid)
    _state["last_target"] = tid

    # Apply stealth to avoid bot detection
    if HAS_STEALTH:
        try:
            stealth_sync(page)
        except Exception:
            pass

    def on_console(msg):
        _state["console_msgs"].append({
            "level": msg.type,
            "text": msg.text,
            "url": page.url,
            "timestamp": time.time(),
        })
        if len(_state["console_msgs"]) > 200:
            _state["console_msgs"] = _state["console_msgs"][-200:]

    def on_request(req):
        _state["network_requests"].append({
            "method": req.method,
            "url": req.url,
            "resourceType": req.resource_type,
            "timestamp": time.time(),
        })
        if len(_state["network_requests"]) > 500:
            _state["network_requests"] = _state["network_requests"][-500:]

    def on_response(resp):
        # Store response metadata (body fetched on demand)
        _state["network_responses"][resp.url] = {
            "status": resp.status,
            "headers": dict(resp.headers),
            "url": resp.url,
        }

    def on_dialog(dialog):
        if _state["dialog_mode"] == "manual":
            _state["dialog_queue"].append({
                "type": dialog.type,
                "message": dialog.message,
                "default_value": dialog.default_value,
                "handled": False,
                "_dialog": dialog,  # keep ref for later accept/dismiss
            })
        else:
            try:
                dialog.accept()
            except Exception:
                pass

    def on_download(download):
        try:
            path = download.path()
            _state["downloads"].append({
                "path": str(path) if path else None,
                "suggestedFilename": download.suggested_filename,
                "url": download.url,
                "timestamp": time.time(),
            })
        except Exception:
            pass

    page.on("console", on_console)
    page.on("request", on_request)
    page.on("response", on_response)
    page.on("dialog", on_dialog)
    page.on("download", on_download)
    return tid


def _unregister_page(target_id):
    """Remove a page from tracking."""
    _state["pages"].pop(target_id, None)
    if target_id in _state["page_order"]:
        _state["page_order"].remove(target_id)
    if _state["last_target"] == target_id:
        _state["last_target"] = _state["page_order"][0] if _state["page_order"] else None


# ---------------------------------------------------------------------------
# Snapshot: build AI-friendly page structure with element refs
# ---------------------------------------------------------------------------

_ARIA_LINE_RE = re.compile(r'^(\s*-\s*)(\w+)(?:\s+"([^"]*)")?(.*)$')


def _build_snapshot(page, target_id=None, selector=None, frame=None, max_chars=0):
    """Build an AI-friendly snapshot using Playwright's aria_snapshot().

    Args:
        page: Playwright page object
        target_id: targetId for per-tab ref storage
        selector: CSS selector to scope the snapshot (e.g. "main")
        frame: CSS selector for iframe (e.g. "iframe#content")
        max_chars: Truncate snapshot text if exceeds this (0 = unlimited)

    Returns:
        (snapshot_text, refs_dict)
    """
    try:
        # Build scoped locator
        if frame:
            scope = page.frame_locator(frame)
            loc = scope.locator(selector) if selector else scope.locator(":root")
        else:
            loc = page.locator(selector) if selector else page.locator(":root")

        raw = loc.aria_snapshot()
    except Exception:
        # Fallback
        title = page.title()
        url = page.url
        return f"Page: {title}\nURL: {url}\n(aria snapshot unavailable)", {}

    if not raw or not raw.strip():
        title = page.title()
        url = page.url
        return f"Page: {title}\nURL: {url}\n(empty page)", {}

    # Parse aria_snapshot output and assign refs
    lines = raw.split("\n")
    ref_counter = [0]
    refs = {}
    seen_keys = {}  # "role:name" -> count
    out_lines = []

    for line in lines:
        m = _ARIA_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        prefix, role_raw, name, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        role = role_raw.lower()

        # Determine if this element should get a ref
        is_interactive = role in INTERACTIVE_ROLES
        is_content = role in CONTENT_ROLES
        should_have_ref = is_interactive or (is_content and name)

        if should_have_ref:
            ref_counter[0] += 1
            ref_id = f"e{ref_counter[0]}"

            key = f"{role}:{name or ''}"
            nth = seen_keys.get(key, 0)
            seen_keys[key] = nth + 1

            refs[ref_id] = {"role": role, "name": name or "", "nth": nth}

            # Rebuild line with ref tag
            enhanced = f"{prefix}{role_raw}"
            if name:
                enhanced += f' "{name}"'
            enhanced += f" [ref={ref_id}]"
            if nth > 0:
                enhanced += f" [nth={nth}]"
            if suffix and suffix.strip():
                enhanced += suffix
            out_lines.append(enhanced)
        else:
            out_lines.append(line)

    # Remove nth from non-duplicate refs
    dup_keys = {k for k, v in seen_keys.items() if v > 1}
    for ref_id, info in refs.items():
        key = f"{info['role']}:{info['name']}"
        if key not in dup_keys:
            info.pop("nth", None)

    title = page.title()
    url = page.url
    header = f"Page: {title}\nURL: {url}\n---\n"
    body = "\n".join(out_lines) if out_lines else "(empty page)"
    snapshot_text = header + body

    if max_chars and len(snapshot_text) > max_chars:
        snapshot_text = snapshot_text[:max_chars] + "\n\n[...TRUNCATED - page too large]"

    # Store refs in per-target entry
    if target_id and target_id in _state["pages"]:
        _state["pages"][target_id]["refs"] = refs
        _state["pages"][target_id]["frame_selector"] = frame or None

    return snapshot_text, refs


def _resolve_ref(page, ref_id, target_id=None):
    """Resolve a ref ID to a Playwright locator using per-target refs."""
    refs = {}
    frame_selector = None

    if target_id and target_id in _state["pages"]:
        entry = _state["pages"][target_id]
        refs = entry.get("refs", {})
        frame_selector = entry.get("frame_selector")

    if ref_id not in refs:
        raise ValueError(f"Unknown ref '{ref_id}'. Run snapshot first to get current refs.")

    info = refs[ref_id]
    role = info["role"]
    name = info.get("name", "")
    nth = info.get("nth")

    # Scope to iframe if snapshot was taken inside one
    if frame_selector:
        scope = page.frame_locator(frame_selector)
    else:
        scope = page

    if name:
        loc = scope.get_by_role(role, name=name)
    else:
        loc = scope.get_by_role(role)

    return loc.nth(nth) if nth is not None else loc


def _resolve_target(page, data, target_id=None):
    """Resolve a click/interaction target from ref, selector, or text.

    Priority: ref > selector > text
    This allows clicking elements that don't have ARIA roles (e.g. <div>同意</div>).
    """
    ref = data.get("ref")
    selector = data.get("selector")
    text = data.get("text")

    if ref:
        return _resolve_ref(page, ref, target_id=target_id)
    elif selector:
        return page.locator(selector).first
    elif text:
        # Try exact text match first, then partial
        loc = page.get_by_text(text, exact=True)
        if loc.count() > 0:
            return loc.first
        loc = page.get_by_text(text)
        if loc.count() > 0:
            return loc.first
        raise ValueError(f"No element found with text '{text}'")
    else:
        raise ValueError("One of 'ref', 'selector', or 'text' is required")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def status():
    profile = request.args.get("profile", "default")
    return jsonify({
        "status": "running" if _state["started"] else "stopped",
        "profile": profile,
        "tabs": len(_state["pages"]),
        "systemProfile": _state["system_profile"],
    })


@app.route("/start", methods=["POST"])
def start_browser():
    with _lock:
        if _state["started"]:
            return jsonify({"ok": True, "message": "already running"})

        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        _state["playwright"] = pw

        if _state["system_profile"]:
            user_data = _resolve_system_chrome_path()
            if not user_data or not Path(user_data).exists():
                return jsonify({"error": f"System Chrome profile not found at {user_data}"}), 400

            context = pw.chromium.launch_persistent_context(
                user_data,
                headless=False,
                channel="chrome",
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-notifications",
                ],
                no_viewport=True,
                permissions=[],  # deny all permission prompts
            )
            _state["context"] = context
            _state["browser"] = None

            # Auto-register new tabs opened via target="_blank" or window.open
            context.on("page", lambda p: _register_page(p))

            for p in context.pages:
                _register_page(p)
            if not context.pages:
                page = context.new_page()
                _register_page(page)
        else:
            browser = pw.chromium.launch(
                headless=_state["headless"],
                channel="chrome",
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-notifications",
                ],
            )
            _state["browser"] = browser
            context = browser.new_context(
                no_viewport=True,
                permissions=[],  # deny all permission prompts
            )
            _state["context"] = context

            # Auto-register new tabs opened via target="_blank" or window.open
            context.on("page", lambda p: _register_page(p))

            page = context.new_page()
            _register_page(page)

        _state["started"] = True
        return jsonify({"ok": True, "tabs": len(_state["pages"])})


@app.route("/stop", methods=["POST"])
def stop_browser():
    with _lock:
        if not _state["started"]:
            return jsonify({"ok": True, "message": "not running"})

        try:
            if _state["context"]:
                _state["context"].close()
        except Exception:
            pass
        try:
            if _state["browser"]:
                _state["browser"].close()
        except Exception:
            pass
        try:
            if _state["playwright"]:
                _state["playwright"].stop()
        except Exception:
            pass

        _state.update({
            "playwright": None,
            "browser": None,
            "context": None,
            "pages": {},
            "page_order": [],
            "last_target": None,
            "console_msgs": [],
            "network_requests": [],
            "network_responses": {},
            "dialog_queue": [],
            "dialog_mode": "auto",
            "downloads": [],
            "started": False,
        })
        return jsonify({"ok": True})


@app.route("/tabs", methods=["GET"])
def list_tabs():
    tabs = []
    for tid in _state["page_order"]:
        entry = _state["pages"].get(tid)
        if entry:
            page = entry["page"]
            try:
                tabs.append({
                    "targetId": tid,
                    "title": page.title(),
                    "url": page.url,
                    "type": "page",
                })
            except Exception:
                tabs.append({"targetId": tid, "title": "(error)", "url": "", "type": "page"})
    return jsonify({"tabs": tabs})


@app.route("/tabs/open", methods=["POST"])
def open_tab():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "about:blank")

    if not _state["started"] or not _state["context"]:
        return jsonify({"error": "Browser not started"}), 400

    page = _state["context"].new_page()
    tid = _register_page(page)
    if url and url != "about:blank":
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

    return jsonify({
        "targetId": tid,
        "title": page.title(),
        "url": page.url,
    })


@app.route("/tabs/<target_id>", methods=["DELETE"])
def close_tab(target_id):
    entry = _state["pages"].get(target_id)
    if not entry:
        return jsonify({"error": "tab not found"}), 404
    try:
        entry["page"].close()
    except Exception:
        pass
    _unregister_page(target_id)
    return jsonify({"ok": True})


@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    target_id = data.get("targetId") or request.args.get("targetId")

    if not url:
        return jsonify({"error": "url required"}), 400

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available. Start browser first."}), 400

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Auto-dismiss cookie consent / popups after navigation
    page.wait_for_timeout(500)
    _try_dismiss_popups(page)

    return jsonify({
        "ok": True,
        "targetId": tid,
        "title": page.title(),
        "url": page.url,
    })


@app.route("/snapshot", methods=["GET"])
def snapshot():
    target_id = request.args.get("targetId")
    fmt = request.args.get("format", "ai")
    selector = request.args.get("selector")
    frame = request.args.get("frame")
    max_chars = int(request.args.get("maxChars", 0))

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available. Start browser first."}), 400

    text, refs = _build_snapshot(
        page, target_id=tid, selector=selector, frame=frame, max_chars=max_chars,
    )

    return jsonify({
        "snapshot": text,
        "targetId": tid,
        "title": page.title(),
        "url": page.url,
        "format": fmt,
        "refs": list(refs.keys()),
    })


@app.route("/screenshot", methods=["POST"])
def screenshot():
    data = request.get_json(silent=True) or {}
    target_id = data.get("targetId") or request.args.get("targetId")

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    tmp = tempfile.mktemp(suffix=".png", prefix="browser_screenshot_")
    page.screenshot(path=tmp, full_page=data.get("fullPage", False))

    return jsonify({
        "ok": True,
        "path": tmp,
        "targetId": tid,
    })


@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind", "")
    target_id = data.get("targetId") or request.args.get("targetId")

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    try:
        if kind == "click":
            # Record state before click to detect navigation / new tabs
            url_before = page.url
            context = _state["context"]

            locator = _resolve_target(page, data, target_id=tid)
            new_page_obj = None
            try:
                with context.expect_page(timeout=3000) as new_page_info:
                    if data.get("doubleClick"):
                        locator.dblclick(timeout=data.get("timeoutMs", 5000))
                    else:
                        locator.click(timeout=data.get("timeoutMs", 5000))
                new_page_obj = new_page_info.value
            except Exception:
                # No new page opened - that's fine, it may be same-page navigation
                pass

            # If expect_page didn't trigger (no new tab), do the click normally
            if new_page_obj is None and not data.get("_clicked"):
                try:
                    # Click may have already happened inside expect_page even if it timed out
                    pass
                except Exception:
                    pass

            # Wait for current page load
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass

            url_after = page.url

            # Auto-dismiss popups if we navigated to a new page
            if url_after != url_before:
                page.wait_for_timeout(300)
                _try_dismiss_popups(page)

            # Build enriched response
            result = {"ok": True, "kind": kind, "targetId": tid}
            if url_after != url_before:
                result["navigated"] = True
                result["url"] = url_after
                result["title"] = page.title()

            if new_page_obj:
                try:
                    new_page_obj.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                # Find the targetId for this new page
                new_tid = None
                for t, p in _state["pages"].items():
                    if p["page"] is new_page_obj:
                        new_tid = t
                        break
                if not new_tid:
                    # Register it if context.on("page") hasn't fired yet
                    new_tid = _register_page(new_page_obj)
                try:
                    new_page_obj.wait_for_timeout(300)
                    _try_dismiss_popups(new_page_obj)
                except Exception:
                    pass
                result["newTabs"] = [{
                    "targetId": new_tid,
                    "url": new_page_obj.url,
                    "title": new_page_obj.title(),
                }]

            return jsonify(result)

        elif kind == "type":
            locator = _resolve_target(page, data, target_id=tid)
            text = data.get("text", "")
            locator.click(timeout=data.get("timeoutMs", 5000))
            locator.fill(text)
            if data.get("submit"):
                locator.press("Enter")

        elif kind == "press":
            key = data.get("key", "")
            if not key:
                return jsonify({"error": "key required for press"}), 400
            ref = data.get("ref")
            if ref:
                locator = _resolve_ref(page, ref, target_id=tid)
                locator.press(key)
            else:
                page.keyboard.press(key)

        elif kind == "hover":
            locator = _resolve_target(page, data, target_id=tid)
            locator.hover(timeout=data.get("timeoutMs", 5000))

        elif kind == "select":
            locator = _resolve_target(page, data, target_id=tid)
            values = data.get("values", [])
            locator.select_option(values, timeout=data.get("timeoutMs", 5000))

        elif kind == "fill":
            fields = data.get("fields", [])
            for field in fields:
                fref = field.get("ref")
                fval = field.get("value", "")
                if fref:
                    loc = _resolve_ref(page, fref, target_id=tid)
                    loc.fill(fval)

        elif kind == "scroll":
            direction = data.get("direction", "down")
            amount = data.get("amount", 500)
            if direction == "down":
                page.mouse.wheel(0, amount)
            elif direction == "up":
                page.mouse.wheel(0, -amount)
            elif direction == "right":
                page.mouse.wheel(amount, 0)
            elif direction == "left":
                page.mouse.wheel(-amount, 0)

        elif kind == "evaluate":
            expression = data.get("expression", "")
            if not expression:
                return jsonify({"error": "expression required for evaluate"}), 400
            result = page.evaluate(expression)
            return jsonify({"ok": True, "kind": kind, "targetId": tid, "result": result})

        elif kind == "scrollIntoView":
            locator = _resolve_target(page, data, target_id=tid)
            locator.scroll_into_view_if_needed(timeout=data.get("timeoutMs", 5000))

        elif kind == "wait":
            time_ms = data.get("timeMs", 1000)
            timeout_ms = data.get("timeoutMs", 30000)
            wait_for = data.get("waitFor")  # text, textGone, selector, url, loadState

            if wait_for == "text":
                wait_text = data.get("text", "")
                page.wait_for_function(
                    f"() => document.body.innerText.includes({json.dumps(wait_text)})",
                    timeout=timeout_ms,
                )
            elif wait_for == "textGone":
                wait_text = data.get("text", "")
                page.wait_for_function(
                    f"() => !document.body.innerText.includes({json.dumps(wait_text)})",
                    timeout=timeout_ms,
                )
            elif wait_for == "selector":
                wait_selector = data.get("selector", "")
                page.wait_for_selector(wait_selector, timeout=timeout_ms)
            elif wait_for == "url":
                wait_url = data.get("url", "")
                page.wait_for_url(f"**{wait_url}**", timeout=timeout_ms)
            elif wait_for == "loadState":
                state = data.get("state", "load")
                page.wait_for_load_state(state, timeout=timeout_ms)
            else:
                # Simple time-based wait
                page.wait_for_timeout(time_ms)

        elif kind == "text":
            # Extract page text content for summarization / content reading
            selector = data.get("selector")
            max_length = data.get("maxLength", 50000)
            if selector:
                try:
                    el = page.locator(selector).first
                    text_content = el.inner_text(timeout=data.get("timeoutMs", 5000))
                except Exception:
                    text_content = page.evaluate("() => document.body.innerText")
            else:
                text_content = page.evaluate("() => document.body.innerText")
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + f"\n...(truncated, total {len(text_content)} chars)"
            return jsonify({
                "ok": True,
                "kind": kind,
                "targetId": tid,
                "text": text_content,
                "url": page.url,
                "title": page.title(),
            })

        elif kind == "upload":
            # Direct set_input_files on a file input element
            paths = data.get("paths", [])
            if not paths:
                return jsonify({"error": "paths required for upload"}), 400
            locator = _resolve_target(page, data, target_id=tid)
            locator.set_input_files(paths)
            # Trigger input/change events for compatibility
            try:
                locator.evaluate("""el => {
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }""")
            except Exception:
                pass

        elif kind == "close":
            if tid:
                try:
                    page.close()
                except Exception:
                    pass
                _unregister_page(tid)
            return jsonify({"ok": True, "closed": tid})

        else:
            return jsonify({"error": f"Unknown act kind: {kind}"}), 400

        return jsonify({"ok": True, "kind": kind, "targetId": tid})

    except ValueError as e:
        # Element not found, invalid ref, missing params
        return jsonify({"error": str(e), "errorType": "invalid_target"}), 400
    except TimeoutError as e:
        # Playwright timeout - element not visible, not clickable, etc.
        return jsonify({
            "error": f"Timeout: {e}",
            "errorType": "timeout",
            "hint": "Element may be hidden, covered by an overlay, or not yet loaded. "
                    "Try dismissing popups, scrolling to the element, or waiting for page load.",
        }), 408
    except Exception as e:
        err_str = str(e)
        # Detect common Playwright errors and give actionable hints
        if "Target closed" in err_str or "target page" in err_str.lower():
            return jsonify({
                "error": err_str,
                "errorType": "page_closed",
                "hint": "The page or tab was closed. A click may have opened a new tab. "
                        "Use 'tabs' action to check for new tabs.",
            }), 410
        if "navigation" in err_str.lower():
            return jsonify({
                "error": err_str,
                "errorType": "navigation",
                "hint": "A navigation occurred during the action. "
                        "Use snapshot to check the current page state.",
            }), 200
        return jsonify({"error": err_str, "errorType": "unknown"}), 500


# ---------------------------------------------------------------------------
# File upload (file chooser mode)
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload files via file chooser or direct input.

    Two modes:
      1. inputRef mode: directly set files on an <input type="file"> element
      2. ref mode: arm a file chooser listener, then click the trigger element

    Request body:
        paths: list of file paths to upload (required)
        inputRef: ref of the <input type="file"> element (mode 1)
        ref: ref of the button/element that triggers file chooser (mode 2)
        targetId: tab to operate on
    """
    data = request.get_json(silent=True) or {}
    target_id = data.get("targetId") or request.args.get("targetId")
    paths = data.get("paths", [])
    input_ref = data.get("inputRef")
    ref = data.get("ref")

    if not paths:
        return jsonify({"error": "paths required"}), 400

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    try:
        if input_ref:
            # Mode 1: direct set_input_files on file input
            locator = _resolve_ref(page, input_ref, target_id=tid)
            locator.set_input_files(paths)
            try:
                locator.evaluate("""el => {
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }""")
            except Exception:
                pass
        elif ref:
            # Mode 2: arm file chooser then click trigger
            with page.expect_file_chooser(timeout=data.get("timeoutMs", 30000)) as fc_info:
                locator = _resolve_ref(page, ref, target_id=tid)
                locator.click(timeout=data.get("timeoutMs", 5000))
            file_chooser = fc_info.value
            file_chooser.set_files(paths)
        else:
            return jsonify({"error": "inputRef or ref required"}), 400

        return jsonify({"ok": True, "targetId": tid, "files": len(paths)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Login detection helper
# ---------------------------------------------------------------------------

_LOGIN_KEYWORDS = ["login", "signin", "sign-in", "sign_in", "sso", "auth",
                   "登录", "登陆", "注册"]


def _is_login_page(url: str = "", title: str = "") -> bool:
    """Check if a URL or title indicates a login/auth page."""
    text = (url + " " + title).lower()
    return any(kw in text for kw in _LOGIN_KEYWORDS)


# ---------------------------------------------------------------------------
# Batch endpoint - execute multiple steps in one request
# ---------------------------------------------------------------------------

@app.route("/batch", methods=["POST"])
def batch():
    """Execute a sequence of browser steps. Stops early on login detection or error.

    Request body:
        {"steps": [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "snapshot", "maxElements": 30},
            {"action": "act", "kind": "type", "ref": "e10", "text": "query", "submit": true},
            {"action": "snapshot", "maxElements": 30}
        ]}

    Response:
        {"results": [...], "stoppedAt": null|int, "stopReason": null|"login_required"|"error"}
    """
    data = request.get_json(silent=True) or {}
    steps = data.get("steps", [])
    if not steps:
        return jsonify({"error": "steps array required"}), 400

    results = []
    stop_reason = None
    stopped_at = None

    for i, step in enumerate(steps):
        action = step.get("action", "")
        target_id = step.get("targetId") or None

        try:
            if action == "navigate":
                url = step.get("url", "")
                if not url:
                    results.append({"error": "url required", "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break
                page, tid = _get_page(target_id)
                if not page:
                    results.append({"error": "No page available", "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception as e:
                    results.append({"error": str(e), "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break
                page.wait_for_timeout(500)
                _try_dismiss_popups(page)
                result = {
                    "ok": True, "action": "navigate", "targetId": tid,
                    "title": page.title(), "url": page.url,
                }
                # Login detection
                if _is_login_page(page.url, page.title()):
                    result["login_required"] = True
                    results.append(result)
                    stop_reason = "login_required"
                    stopped_at = i
                    break
                results.append(result)

            elif action == "snapshot":
                fmt = step.get("format", "ai")
                selector = step.get("selector")
                frame = step.get("frame")
                max_chars = int(step.get("maxChars", 0))
                page, tid = _get_page(target_id)
                if not page:
                    results.append({"error": "No page available", "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break
                text, refs = _build_snapshot(
                    page, target_id=tid, selector=selector, frame=frame, max_chars=max_chars,
                )
                results.append({
                    "snapshot": text, "targetId": tid,
                    "title": page.title(), "url": page.url,
                    "format": fmt, "refs": list(refs.keys()),
                })

            elif action == "act":
                kind = step.get("kind", "")
                page, tid = _get_page(target_id)
                if not page:
                    results.append({"error": "No page available", "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break

                # Delegate to the existing /act handler logic inline
                act_result = _execute_act(page, tid, step)
                results.append(act_result)

                # Login detection on click results
                if kind == "click":
                    new_tabs = act_result.get("newTabs", [])
                    for tab in new_tabs:
                        if _is_login_page(tab.get("url", ""), tab.get("title", "")):
                            act_result["login_required"] = True
                            stop_reason = "login_required"
                            stopped_at = i
                            break
                    if act_result.get("navigated"):
                        if _is_login_page(act_result.get("url", ""), act_result.get("title", "")):
                            act_result["login_required"] = True
                            stop_reason = "login_required"
                            stopped_at = i
                    if stop_reason:
                        break

                # Check for errors
                if "error" in act_result:
                    stop_reason = "error"
                    stopped_at = i
                    break

            elif action == "focus":
                ftid = step.get("targetId", "")
                if not ftid or ftid not in _state["pages"]:
                    results.append({"error": "Invalid targetId", "step": i})
                    stop_reason = "error"
                    stopped_at = i
                    break
                fp = _state["pages"][ftid]["page"]
                try:
                    fp.bring_to_front()
                except Exception:
                    pass
                _state["last_target"] = ftid
                results.append({"ok": True, "action": "focus", "targetId": ftid})

            elif action == "close":
                ctid = step.get("targetId", "")
                if ctid and ctid in _state["pages"]:
                    try:
                        _state["pages"][ctid]["page"].close()
                    except Exception:
                        pass
                    _unregister_page(ctid)
                results.append({"ok": True, "action": "close", "targetId": ctid})

            else:
                results.append({"error": f"Unsupported batch action: {action}", "step": i})
                stop_reason = "error"
                stopped_at = i
                break

        except Exception as e:
            results.append({"error": str(e), "step": i})
            stop_reason = "error"
            stopped_at = i
            break

    return jsonify({
        "results": results,
        "stoppedAt": stopped_at,
        "stopReason": stop_reason,
    })


def _execute_act(page, tid, data):
    """Execute a single act operation. Returns result dict (not a Flask response)."""
    kind = data.get("kind", "")

    try:
        if kind == "click":
            url_before = page.url
            context = _state["context"]
            locator = _resolve_target(page, data, target_id=tid)

            new_page_obj = None
            try:
                with context.expect_page(timeout=3000) as new_page_info:
                    if data.get("doubleClick"):
                        locator.dblclick(timeout=data.get("timeoutMs", 5000))
                    else:
                        locator.click(timeout=data.get("timeoutMs", 5000))
                new_page_obj = new_page_info.value
            except Exception:
                pass

            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass

            url_after = page.url
            if url_after != url_before:
                page.wait_for_timeout(300)
                _try_dismiss_popups(page)

            result = {"ok": True, "kind": kind, "targetId": tid}
            if url_after != url_before:
                result["navigated"] = True
                result["url"] = url_after
                result["title"] = page.title()

            if new_page_obj:
                try:
                    new_page_obj.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                new_tid = None
                for t, p in _state["pages"].items():
                    if p["page"] is new_page_obj:
                        new_tid = t
                        break
                if not new_tid:
                    new_tid = _register_page(new_page_obj)
                try:
                    new_page_obj.wait_for_timeout(300)
                    _try_dismiss_popups(new_page_obj)
                except Exception:
                    pass
                result["newTabs"] = [{
                    "targetId": new_tid,
                    "url": new_page_obj.url,
                    "title": new_page_obj.title(),
                }]
            return result

        elif kind == "type":
            locator = _resolve_target(page, data, target_id=tid)
            text = data.get("text", "")
            locator.click(timeout=data.get("timeoutMs", 5000))
            locator.fill(text)
            if data.get("submit"):
                locator.press("Enter")
            return {"ok": True, "kind": kind, "targetId": tid}

        elif kind == "press":
            key = data.get("key", "")
            if not key:
                return {"error": "key required for press"}
            ref = data.get("ref")
            if ref:
                locator = _resolve_ref(page, ref, target_id=tid)
                locator.press(key)
            else:
                page.keyboard.press(key)
            return {"ok": True, "kind": kind, "targetId": tid}

        elif kind == "fill":
            fields = data.get("fields", [])
            for field in fields:
                fref = field.get("ref")
                fval = field.get("value", "")
                if fref:
                    loc = _resolve_ref(page, fref, target_id=tid)
                    loc.fill(fval)
            return {"ok": True, "kind": kind, "targetId": tid}

        elif kind == "scroll":
            direction = data.get("direction", "down")
            amount = data.get("amount", 500)
            if direction == "down":
                page.mouse.wheel(0, amount)
            elif direction == "up":
                page.mouse.wheel(0, -amount)
            elif direction == "right":
                page.mouse.wheel(amount, 0)
            elif direction == "left":
                page.mouse.wheel(-amount, 0)
            return {"ok": True, "kind": kind, "targetId": tid}

        elif kind == "evaluate":
            expression = data.get("expression", "")
            if not expression:
                return {"error": "expression required for evaluate"}
            result = page.evaluate(expression)
            return {"ok": True, "kind": kind, "targetId": tid, "result": result}

        elif kind == "text":
            selector = data.get("selector")
            max_length = data.get("maxLength", 50000)
            if selector:
                try:
                    el = page.locator(selector).first
                    text_content = el.inner_text(timeout=data.get("timeoutMs", 5000))
                except Exception:
                    text_content = page.evaluate("() => document.body.innerText")
            else:
                text_content = page.evaluate("() => document.body.innerText")
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + f"\n...(truncated, total {len(text_content)} chars)"
            return {
                "ok": True, "kind": kind, "targetId": tid,
                "text": text_content, "url": page.url, "title": page.title(),
            }

        elif kind == "wait":
            time_ms = data.get("timeMs", 1000)
            timeout_ms = data.get("timeoutMs", 30000)
            wait_for = data.get("waitFor")
            if wait_for == "text":
                page.wait_for_function(
                    f"() => document.body.innerText.includes({json.dumps(data.get('text', ''))})",
                    timeout=timeout_ms)
            elif wait_for == "textGone":
                page.wait_for_function(
                    f"() => !document.body.innerText.includes({json.dumps(data.get('text', ''))})",
                    timeout=timeout_ms)
            elif wait_for == "selector":
                page.wait_for_selector(data.get("selector", ""), timeout=timeout_ms)
            elif wait_for == "url":
                page.wait_for_url(f"**{data.get('url', '')}**", timeout=timeout_ms)
            elif wait_for == "loadState":
                page.wait_for_load_state(data.get("state", "load"), timeout=timeout_ms)
            else:
                page.wait_for_timeout(time_ms)
            return {"ok": True, "kind": kind, "targetId": tid}

        else:
            return {"error": f"Unknown act kind: {kind}"}

    except ValueError as e:
        return {"error": str(e), "errorType": "invalid_target"}
    except TimeoutError as e:
        return {"error": f"Timeout: {e}", "errorType": "timeout"}
    except Exception as e:
        err_str = str(e)
        if "Target closed" in err_str or "target page" in err_str.lower():
            return {"error": err_str, "errorType": "page_closed"}
        if "navigation" in err_str.lower():
            return {"error": err_str, "errorType": "navigation"}
        return {"error": err_str, "errorType": "unknown"}


@app.route("/console", methods=["GET"])
def console_messages():
    level = request.args.get("level")
    msgs = _state["console_msgs"]
    if level:
        msgs = [m for m in msgs if m["level"] == level]
    return jsonify({"messages": msgs[-50:]})


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------

@app.route("/cookies", methods=["GET"])
def get_cookies():
    if not _state["context"]:
        return jsonify({"error": "Browser not started"}), 400
    cookies = _state["context"].cookies()
    # Filter by url if provided
    url = request.args.get("url")
    if url:
        cookies = [c for c in cookies if url in c.get("domain", "")]
    return jsonify({"cookies": cookies})


@app.route("/cookies/set", methods=["POST"])
def set_cookies():
    if not _state["context"]:
        return jsonify({"error": "Browser not started"}), 400
    data = request.get_json(silent=True) or {}
    cookies = data.get("cookies", [])
    if not cookies:
        return jsonify({"error": "cookies array required"}), 400
    _state["context"].add_cookies(cookies)
    return jsonify({"ok": True, "count": len(cookies)})


@app.route("/cookies/clear", methods=["POST"])
def clear_cookies():
    if not _state["context"]:
        return jsonify({"error": "Browser not started"}), 400
    _state["context"].clear_cookies()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Dialog (alert/confirm/prompt) control
# ---------------------------------------------------------------------------

@app.route("/hooks/dialog", methods=["POST"])
def dialog_hook():
    """Switch dialog handling mode and manage queued dialogs."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")  # "auto" or "manual"
    action = data.get("action")  # "accept" or "dismiss" (for manual mode)
    prompt_text = data.get("promptText")  # text for prompt dialogs

    if mode:
        _state["dialog_mode"] = mode
        return jsonify({"ok": True, "mode": mode})

    if action:
        # Handle the oldest unhandled dialog
        for d in _state["dialog_queue"]:
            if not d["handled"]:
                dialog_obj = d.get("_dialog")
                if dialog_obj:
                    try:
                        if action == "accept":
                            if prompt_text is not None:
                                dialog_obj.accept(prompt_text)
                            else:
                                dialog_obj.accept()
                        else:
                            dialog_obj.dismiss()
                    except Exception:
                        pass
                d["handled"] = True
                return jsonify({"ok": True, "dialog": {
                    "type": d["type"],
                    "message": d["message"],
                    "action": action,
                }})
        return jsonify({"ok": False, "message": "No pending dialogs"})

    # GET-like: return pending dialogs
    pending = [
        {"type": d["type"], "message": d["message"], "default_value": d["default_value"]}
        for d in _state["dialog_queue"] if not d["handled"]
    ]
    return jsonify({"dialogs": pending, "mode": _state["dialog_mode"]})


# ---------------------------------------------------------------------------
# Tab management (focus / actions)
# ---------------------------------------------------------------------------

@app.route("/tabs/focus", methods=["POST"])
def focus_tab():
    data = request.get_json(silent=True) or {}
    target_id = data.get("targetId")
    if not target_id or target_id not in _state["pages"]:
        return jsonify({"error": "Invalid targetId"}), 400
    page = _state["pages"][target_id]["page"]
    try:
        page.bring_to_front()
    except Exception:
        pass
    _state["last_target"] = target_id
    return jsonify({"ok": True, "targetId": target_id})


# ---------------------------------------------------------------------------
# Network requests / responses
# ---------------------------------------------------------------------------

@app.route("/requests", methods=["GET"])
def get_requests():
    url_filter = request.args.get("url")
    resource_type = request.args.get("resourceType")
    reqs = _state["network_requests"]
    if url_filter:
        reqs = [r for r in reqs if url_filter in r["url"]]
    if resource_type:
        reqs = [r for r in reqs if r["resourceType"] == resource_type]
    limit = int(request.args.get("limit", 50))
    return jsonify({"requests": reqs[-limit:]})


@app.route("/response/body", methods=["POST"])
def get_response_body():
    """Fetch the response body for a given URL (re-fetches via page context)."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    target_id = data.get("targetId")

    if not url:
        return jsonify({"error": "url required"}), 400

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    # Use evaluate to fetch the URL from the page context (same cookies/session)
    try:
        body = page.evaluate("""
            async (url) => {
                const resp = await fetch(url, {credentials: 'include'});
                const text = await resp.text();
                return text;
            }
        """, url)
        resp_meta = _state["network_responses"].get(url, {})
        return jsonify({
            "ok": True,
            "url": url,
            "status": resp_meta.get("status"),
            "body": body,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/errors", methods=["GET"])
def get_errors():
    """Return console errors and page errors."""
    errors = [m for m in _state["console_msgs"] if m["level"] in ("error", "warning")]
    limit = int(request.args.get("limit", 50))
    return jsonify({"errors": errors[-limit:]})


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

@app.route("/download", methods=["GET"])
def get_downloads():
    """List captured downloads."""
    return jsonify({"downloads": _state["downloads"]})


@app.route("/wait/download", methods=["POST"])
def wait_download():
    """Wait for a download to start, then return its info."""
    data = request.get_json(silent=True) or {}
    target_id = data.get("targetId")
    timeout_ms = data.get("timeoutMs", 30000)

    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    try:
        with page.expect_download(timeout=timeout_ms) as download_info:
            # If an action is provided, execute it to trigger the download
            action_kind = data.get("triggerAction")
            if action_kind == "click":
                locator = _resolve_target(page, data, target_id=tid)
                locator.click(timeout=data.get("clickTimeoutMs", 5000))
        download = download_info.value
        path = download.path()
        result = {
            "ok": True,
            "path": str(path) if path else None,
            "suggestedFilename": download.suggested_filename,
            "url": download.url,
        }
        _state["downloads"].append({**result, "timestamp": time.time()})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Storage (localStorage / sessionStorage)
# ---------------------------------------------------------------------------

@app.route("/storage/local", methods=["GET", "POST"])
def local_storage():
    target_id = request.args.get("targetId")
    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    if request.method == "GET":
        key = request.args.get("key")
        if key:
            val = page.evaluate(f"() => localStorage.getItem({json.dumps(key)})")
            return jsonify({"key": key, "value": val})
        data = page.evaluate("() => { const o = {}; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); o[k] = localStorage.getItem(k); } return o; }")
        return jsonify({"storage": data})

    # POST: set or delete
    data = request.get_json(silent=True) or {}
    action = data.get("action", "set")  # set, remove, clear
    if action == "clear":
        page.evaluate("() => localStorage.clear()")
    elif action == "remove":
        key = data.get("key", "")
        page.evaluate(f"() => localStorage.removeItem({json.dumps(key)})")
    else:
        key = data.get("key", "")
        value = data.get("value", "")
        page.evaluate(f"() => localStorage.setItem({json.dumps(key)}, {json.dumps(value)})")
    return jsonify({"ok": True})


@app.route("/storage/session", methods=["GET", "POST"])
def session_storage():
    target_id = request.args.get("targetId")
    page, tid = _get_page(target_id)
    if not page:
        return jsonify({"error": "No page available"}), 400

    if request.method == "GET":
        key = request.args.get("key")
        if key:
            val = page.evaluate(f"() => sessionStorage.getItem({json.dumps(key)})")
            return jsonify({"key": key, "value": val})
        data = page.evaluate("() => { const o = {}; for (let i = 0; i < sessionStorage.length; i++) { const k = sessionStorage.key(i); o[k] = sessionStorage.getItem(k); } return o; }")
        return jsonify({"storage": data})

    data = request.get_json(silent=True) or {}
    action = data.get("action", "set")
    if action == "clear":
        page.evaluate("() => sessionStorage.clear()")
    elif action == "remove":
        key = data.get("key", "")
        page.evaluate(f"() => sessionStorage.removeItem({json.dumps(key)})")
    else:
        key = data.get("key", "")
        value = data.get("value", "")
        page.evaluate(f"() => sessionStorage.setItem({json.dumps(key)}, {json.dumps(value)})")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone browser control server")
    parser.add_argument("--port", type=int, default=18791, help="Server port (default: 18791)")
    parser.add_argument("--system-profile", action="store_true",
                        help="Use system Chrome profile (inherits login state)")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode (ignored for system profile)")
    args = parser.parse_args()

    _state["system_profile"] = args.system_profile
    _state["headless"] = args.headless

    if args.system_profile:
        chrome_path = _resolve_system_chrome_path()
        print(f"System Chrome profile: {chrome_path}")
        print("NOTE: Close Chrome manually before starting with --system-profile")

    print(f"Browser control server starting on http://127.0.0.1:{args.port}/")
    print("Endpoints: /start /stop /tabs /navigate /snapshot /screenshot /act /console")
    print("           /cookies /hooks/dialog /requests /response/body /errors")
    print("           /download /wait/download /storage/local /storage/session")
    # threaded=False: Playwright sync API must run on the same thread
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
