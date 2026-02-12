# Claude Code 模型路由机制深度分析

> 本文档基于 Claude Code CLI v2.1.39 逆向分析整理

## 目录

- [概述](#概述)
- [核心架构](#核心架构)
- [Haiku 使用场景详解](#haiku-使用场景详解)
  - [场景一：Explore Agent（代码库探索）](#场景一explore-agent代码库探索)
  - [场景二：claude-code-guide Agent（文档查询）](#场景二claude-code-guide-agent文档查询)
  - [场景三：WebFetch 工具（网页内容处理）](#场景三webfetch-工具网页内容处理)
  - [场景四：Hook Prompt（条件评估）](#场景四hook-prompt条件评估)
  - [场景五：Hook Agent（验证任务）](#场景五hook-agent验证任务)
  - [场景六：Token 计数 Fallback](#场景六token-计数-fallback)
- [完整代码实现](#完整代码实现)
- [设计原则总结](#设计原则总结)

---

## 概述

Claude Code 采用**静态规则配置**的模型路由策略，而非动态智能路由。核心思想是：

- **主模型**（Sonnet/Opus）：处理需要深度推理的复杂任务
- **Haiku**：处理不需要深度推理的辅助任务（"廉价劳动力"）

### Haiku 在整体架构中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户请求                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   主模型 (Sonnet/Opus)                          │
│                                                                  │
│  - 理解用户意图                                                  │
│  - 复杂推理和决策                                                │
│  - 代码生成和修改                                                │
│  - 调度子任务                                                    │
└─────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
    │  Explore      │    │  WebFetch     │    │  Hook         │
    │  Agent        │    │  处理         │    │  评估         │
    │  (Haiku)      │    │  (Haiku)      │    │  (Haiku)      │
    │               │    │               │    │               │
    │  快速搜索     │    │  内容摘要     │    │  条件判断     │
    │  代码探索     │    │  信息提取     │    │  true/false   │
    └───────────────┘    └───────────────┘    └───────────────┘
            │                    │                    │
            └────────────────────┴────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   结果返回主模型                                 │
│                   主模型继续处理                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 使用场景总览

| 场景 | 模型 | Haiku 操作工具 | 多轮对话 | 核心作用 |
|------|------|---------------|---------|---------|
| Explore Agent | Haiku | ✅ 是 | ✅ 是 | 快速代码搜索 |
| claude-code-guide | Haiku | ✅ 是 | ✅ 是 | 文档查询 |
| WebFetch | Haiku | ❌ 否 | ❌ 否 | 内容摘要提取 |
| Hook Prompt | Haiku | ❌ 否 | ❌ 否 | 条件判断 |
| Hook Agent | Haiku | ✅ 是 | ✅ 是 | 验证任务 |
| Token 计数 | Haiku | ❌ 否 | ❌ 否 | API 降级重试 |

---

## 核心架构

### 模型获取函数

```javascript
// Small Fast Model - 所有辅助任务的入口
function getSmallFastModel() {
  return process.env.ANTHROPIC_SMALL_FAST_MODEL || getHaikuModel();
}

// Haiku 模型
function getHaikuModel() {
  if (process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL) {
    return process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL;
  }
  return getModelConfig().haiku45;
}
```

### 模型 ID 映射（多云支持）

```javascript
const MODEL_IDS = {
  haiku45: {
    firstParty: "claude-haiku-4-5-20251001",
    bedrock: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    vertex: "claude-haiku-4-5@20251001",
    foundry: "claude-haiku-4-5"
  },
  sonnet45: {
    firstParty: "claude-sonnet-4-5-20250929",
    bedrock: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    vertex: "claude-sonnet-4-5@20250929",
    foundry: "claude-sonnet-4-5"
  },
  opus46: {
    firstParty: "claude-opus-4-6",
    bedrock: "us.anthropic.claude-opus-4-6-v1",
    vertex: "claude-opus-4-6",
    foundry: "claude-opus-4-6"
  }
};
```

---

## Haiku 使用场景详解

---

### 场景一：Explore Agent（代码库探索）

#### 在整体任务中的位置

```
用户: "帮我找到所有处理用户认证的代码"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  判断: 这是一个代码搜索任务，需要探索代码库                       │
│  决策: 调用 Task 工具，subagent_type: "Explore"                  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Explore Agent (Haiku) ← 在这里使用 Haiku                       │
│                                                                  │
│  Haiku 自主执行多轮搜索:                                         │
│  1. Glob("**/auth*.ts") → 找到 5 个文件                         │
│  2. Grep("authenticate|authorization") → 找到 20 个匹配         │
│  3. Read("src/auth/login.ts") → 读取关键文件                    │
│  4. Read("src/middleware/auth.ts") → 读取更多文件               │
│  5. 生成探索报告                                                 │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  接收 Explore 结果，整合并回复用户                               │
└─────────────────────────────────────────────────────────────────┘
```

#### 详细流程

1. **触发条件**: 主模型识别到需要探索代码库的任务
2. **启动方式**: 主模型调用 `Task` 工具，指定 `subagent_type: "Explore"`
3. **模型选择**: 硬编码 `model: "haiku"`
4. **可用工具**: `Glob`, `Grep`, `Read`, `Bash`（只读命令）
5. **禁用工具**: `Edit`, `Write`, `NotebookEdit`, `Task`, `ExitPlanMode`
6. **执行模式**: Haiku 自主进行多轮工具调用
7. **输出**: 探索报告返回给主模型

#### Haiku 的角色

- **执行者**: Haiku 直接操作搜索和读取工具
- **自主决策**: 决定搜索什么、读取哪些文件
- **效率优先**: System Prompt 强调并行调用工具
- **只读限制**: 无法修改任何文件

#### 完整代码

```javascript
// ==========================================
// Explore Agent 完整实现
// ==========================================

const EXPLORE_SYSTEM_PROMPT = `You are a fast exploration agent for Claude Code. Your role is to efficiently explore codebases and find relevant information.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and find relevant information. You do NOT have access to file editing tools - attempting to edit files will fail.

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly.`;

const ExploreAgent = {
  agentType: "Explore",

  whenToUse: 'Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions.',

  // 禁用的工具 - 确保只读
  disallowedTools: ["Task", "NotebookEdit", "Write", "Edit", "ExitPlanMode"],

  source: "built-in",
  baseDir: "built-in",

  // ★ 关键配置：硬编码使用 haiku
  model: "haiku",

  getSystemPrompt: () => EXPLORE_SYSTEM_PROMPT,

  criticalSystemReminder_EXPERIMENTAL: "CRITICAL: This is a READ-ONLY task. You CANNOT edit, write, or create files."
};

// Agent Spawn 时的模型解析
function spawnExploreAgent(params, parentModel) {
  const agent = ExploreAgent;

  // 解析模型: "haiku" → 实际模型 ID
  const resolvedModel = resolveModelAlias(agent.model);
  // resolvedModel = "claude-haiku-4-5-20251001"

  return createAgent({
    ...params,
    model: resolvedModel,
    systemPrompt: agent.getSystemPrompt(),
    disallowedTools: agent.disallowedTools
  });
}
```

---

### 场景二：claude-code-guide Agent（文档查询）

#### 在整体任务中的位置

```
用户: "How do I configure hooks in Claude Code?"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  判断: 这是关于 Claude Code 功能的问题                           │
│  决策: 调用 Task 工具，subagent_type: "claude-code-guide"        │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  claude-code-guide Agent (Haiku) ← 在这里使用 Haiku             │
│                                                                  │
│  Haiku 执行文档搜索:                                             │
│  1. WebFetch(docs_map_url) → 获取文档目录                       │
│  2. WebFetch(hooks_doc_url) → 获取 hooks 文档                   │
│  3. Glob("**/.claude/**") → 搜索本地配置                        │
│  4. Read(".claude/settings.json") → 读取用户配置                │
│  5. 整合信息，生成指南                                          │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  接收指南，格式化后回复用户                                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 详细流程

1. **触发条件**: 用户询问 Claude Code/Agent SDK/API 相关问题
2. **启动方式**: 主模型调用 `Task` 工具
3. **模型选择**: 硬编码 `model: "haiku"`
4. **可用工具**: `Glob`, `Grep`, `Read`, `WebFetch`, `WebSearch`
5. **权限模式**: `permissionMode: "dontAsk"` - 自动授权
6. **执行模式**: Haiku 自主搜索本地文件和在线文档
7. **输出**: 基于官方文档的准确指南

#### Haiku 的角色

- **信息检索者**: 搜索本地配置和在线文档
- **知识整合者**: 将多个来源的信息整合
- **不需要深度推理**: 主要是信息查找和组织

#### 完整代码

```javascript
// ==========================================
// claude-code-guide Agent 完整实现
// ==========================================

const DOCS_MAP_URL = "https://code.claude.com/docs/en/claude_code_docs_map.md";
const API_DOCS_URL = "https://platform.claude.com/llms.txt";

const GUIDE_SYSTEM_PROMPT = `You are the Claude guide agent. Your primary responsibility is helping users understand and use Claude Code, the Claude Agent SDK, and the Claude API effectively.

**Your expertise spans three domains:**

1. **Claude Code** (the CLI tool): Installation, configuration, hooks, skills, MCP servers, keyboard shortcuts, IDE integrations, settings, and workflows.

2. **Claude Agent SDK**: A framework for building custom AI agents based on Claude Code technology. Available for Node.js/TypeScript and Python.

3. **Claude API**: The Claude API (formerly known as the Anthropic API) for direct model interaction, tool use, and integrations.

**Documentation sources:**

- **Claude Code docs** (${DOCS_MAP_URL}): Fetch this for questions about the Claude Code CLI tool, including:
  - Installation, setup, and getting started
  - Hooks (pre/post command execution)
  - Custom skills
  - MCP server configuration
  - IDE integrations (VS Code, JetBrains)
  - Settings files and configuration
  - Keyboard shortcuts and hotkeys
  - Subagents and plugins
  - Sandboxing and security

- **Claude Agent SDK docs** (${API_DOCS_URL}): Fetch this for questions about building agents with the SDK

- **Claude API docs** (${API_DOCS_URL}): Fetch this for questions about the Claude API

**Approach:**
1. Determine which domain the user's question falls into
2. Use WebFetch to fetch the appropriate docs map
3. Identify the most relevant documentation URLs from the map
4. Fetch the specific documentation pages
5. Provide clear, actionable guidance based on official documentation
6. Use WebSearch if docs don't cover the topic
7. Reference local project files (CLAUDE.md, .claude/ directory) when relevant using Read, Glob, and Grep

**Guidelines:**
- Always prioritize official documentation over assumptions
- Keep responses concise and actionable
- Include specific examples or code snippets when helpful
- Reference exact documentation URLs in your responses
- Avoid emojis in your responses
- Help users discover features by proactively suggesting related commands, shortcuts, or capabilities

Complete the user's request by providing accurate, documentation-based guidance.`;

const ClaudeCodeGuideAgent = {
  agentType: "claude-code-guide",

  whenToUse: 'Use this agent when the user asks questions ("Can Claude...", "Does Claude...", "How do I...") about: (1) Claude Code (the CLI tool) - features, hooks, slash commands, MCP servers, settings, IDE integrations, keyboard shortcuts; (2) Claude Agent SDK - building custom agents; (3) Claude API (formerly Anthropic API) - API usage, tool use, Anthropic SDK usage. **IMPORTANT:** Before spawning a new agent, check if there is already a running or recently completed claude-code-guide agent that you can resume using the "resume" parameter.',

  // 只有搜索和读取相关的工具
  tools: ["Glob", "Grep", "Read", "WebFetch", "WebSearch"],

  source: "built-in",
  baseDir: "built-in",

  // ★ 关键配置：硬编码使用 haiku
  model: "haiku",

  // ★ 关键配置：不询问权限，自动允许
  permissionMode: "dontAsk",

  getSystemPrompt: ({ toolUseContext }) => {
    const commands = toolUseContext.options.commands;
    const contextParts = [];

    // 添加项目中的自定义 skills
    const skills = commands.filter(c => c.type === "prompt");
    if (skills.length > 0) {
      const skillsList = skills.map(s => `- /${s.name}: ${s.description}`).join("\n");
      contextParts.push(`**Available custom skills in this project:**\n${skillsList}`);
    }

    // 添加自定义 agents
    const customAgents = toolUseContext.options.agentDefinitions.activeAgents
      .filter(a => a.source !== "built-in");
    if (customAgents.length > 0) {
      const agentsList = customAgents.map(a => `- ${a.agentType}: ${a.whenToUse}`).join("\n");
      contextParts.push(`**Available custom agents configured:**\n${agentsList}`);
    }

    // 添加 MCP servers
    const mcpClients = toolUseContext.options.mcpClients;
    if (mcpClients?.length > 0) {
      const mcpList = mcpClients.map(c => `- ${c.name}`).join("\n");
      contextParts.push(`**Configured MCP servers:**\n${mcpList}`);
    }

    // 添加用户设置
    const settings = getUserSettings();
    if (Object.keys(settings).length > 0) {
      const settingsJson = JSON.stringify(settings, null, 2);
      contextParts.push(`**User's settings.json:**\n\`\`\`json\n${settingsJson}\n\`\`\``);
    }

    let prompt = GUIDE_SYSTEM_PROMPT;
    if (contextParts.length > 0) {
      prompt += `\n\n---\n\n# User's Current Configuration\n\n${contextParts.join("\n\n")}`;
    }

    return prompt;
  }
};
```

---

### 场景三：WebFetch 工具（网页内容处理）

#### 在整体任务中的位置

```
用户: "帮我看下这个网页说了什么 https://example.com/docs"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  判断: 需要获取网页内容                                          │
│  决策: 调用 WebFetch 工具                                        │
│  参数: { url: "...", prompt: "总结这个页面的主要内容" }          │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  WebFetch 工具内部处理                                          │
│                                                                  │
│  步骤1: 系统层面 (不用模型)                                      │
│  ├─ HTTP GET 请求获取 HTML                                      │
│  ├─ Turndown 库将 HTML 转换为 Markdown                          │
│  └─ 内容截断到 100,000 字符                                     │
│                                                                  │
│  步骤2: Haiku 处理 ← 在这里使用 Haiku                           │
│  ├─ 输入: 用户 prompt + markdown 内容                           │
│  ├─ 模型: getSmallFastModel() = Haiku                           │
│  ├─ 单次 API 调用，无工具                                       │
│  └─ 输出: 处理后的摘要/提取结果                                  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  主模型 (Sonnet/Opus)                                           │
│  接收处理结果，继续与用户对话                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 详细流程

1. **触发条件**: 主模型调用 `WebFetch` 工具
2. **输入参数**: `url` (网页地址) + `prompt` (处理指令)
3. **HTTP 获取**: 系统层面获取 HTML，不用模型
4. **格式转换**: Turndown 库将 HTML 转 Markdown
5. **模型处理**: 调用 Haiku 处理内容
6. **模型选择**: `model: getSmallFastModel()` = Haiku
7. **执行模式**: 单次 API 调用，无工具使用
8. **输出**: 处理后的文本返回给主模型

#### Haiku 的角色

- **内容处理器**: 根据用户 prompt 提取/摘要信息
- **不操作工具**: 只做文本分析
- **单轮调用**: 一次调用返回结果
- **大量内容处理**: 适合处理大段网页内容

#### 完整代码

```javascript
// ==========================================
// WebFetch 工具完整实现
// ==========================================

import TurndownService from "turndown";
import axios from "axios";

const turndown = new TurndownService();
const MAX_CONTENT_LENGTH = 100000;
const MAX_FETCH_SIZE = 10 * 1024 * 1024; // 10MB

// ★ 核心: 获取 small fast model
function getSmallFastModel() {
  return process.env.ANTHROPIC_SMALL_FAST_MODEL || getHaikuModel();
}

// 获取网页内容 (系统层面，不用模型)
async function fetchUrlContent(url, signal) {
  const response = await axios.get(url, {
    signal,
    maxRedirects: 5,
    responseType: "arraybuffer",
    maxContentLength: MAX_FETCH_SIZE,
    headers: { Accept: "text/markdown, text/html, */*" }
  });

  const html = Buffer.from(response.data).toString("utf-8");
  const contentType = response.headers["content-type"] ?? "";

  // HTML 转 Markdown
  let content;
  if (contentType.includes("text/html")) {
    content = turndown.turndown(html);
  } else {
    content = html;
  }

  return {
    code: response.status,
    codeText: response.statusText,
    content,
    contentType,
    bytes: Buffer.byteLength(html)
  };
}

// ★ 核心: 使用 Haiku 处理内容
async function processWebContent(userPrompt, markdownContent, signal, options) {
  // 截断过长内容
  const truncatedContent = markdownContent.length > MAX_CONTENT_LENGTH
    ? markdownContent.slice(0, MAX_CONTENT_LENGTH) + "\n[Content truncated due to length...]"
    : markdownContent;

  // 构建完整 prompt
  const fullPrompt = `${userPrompt}\n\n${truncatedContent}`;

  // 调用 API
  const response = await queryModel({
    systemPrompt: [],
    userPrompt: fullPrompt,
    signal,
    options: {
      // ★ 关键: 使用 Haiku
      model: getSmallFastModel(),
      querySource: "web_fetch_apply",
      agents: [],
      isNonInteractiveSession: options.isNonInteractiveSession,
      hasAppendSystemPrompt: false,
      mcpTools: []
    }
  });

  // 提取文本响应
  const { content } = response.message;
  if (content.length > 0) {
    const firstBlock = content[0];
    if ("text" in firstBlock) {
      return firstBlock.text;
    }
  }
  return "No response from model";
}

// WebFetch 工具定义
const WebFetchTool = {
  name: "WebFetch",
  maxResultSizeChars: 100000,

  description: `- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content`,

  inputSchema: {
    type: "object",
    properties: {
      url: { type: "string", format: "uri", description: "The URL to fetch content from" },
      prompt: { type: "string", description: "The prompt to run on the fetched content" }
    },
    required: ["url", "prompt"]
  },

  async call(input, context) {
    const { url, prompt } = input;
    const startTime = performance.now();

    // 1. 获取网页内容 (系统层面)
    const fetchResult = await fetchUrlContent(url, context.abortController.signal);

    // 处理重定向
    if (fetchResult.type === "redirect") {
      return {
        type: "redirect",
        originalUrl: url,
        redirectUrl: fetchResult.redirectUrl,
        statusCode: fetchResult.statusCode
      };
    }

    // 2. 使用 Haiku 处理内容
    const result = await processWebContent(
      prompt,
      fetchResult.content,
      context.abortController.signal,
      context.options
    );

    const durationMs = performance.now() - startTime;

    return {
      data: {
        bytes: fetchResult.bytes,
        code: fetchResult.code,
        codeText: fetchResult.codeText,
        result,
        durationMs,
        url
      }
    };
  },

  isReadOnly: () => true,
  isConcurrencySafe: () => true
};
```

---

### 场景四：Hook Prompt（条件评估）

#### 在整体任务中的位置

```
用户配置了 Hook:
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write",
      "hooks": [{
        "type": "prompt",
        "prompt": "Verify the code follows our style guide"
      }]
    }]
  }
}

主模型要调用 Write 工具写文件
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hook 系统拦截                                                   │
│  检测到 PreToolUse 事件匹配 "Write"                              │
│  发现有 prompt 类型的 hook                                       │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hook Prompt 评估 (Haiku) ← 在这里使用 Haiku                    │
│                                                                  │
│  输入:                                                           │
│  - System: "You are evaluating a hook..."                       │
│  - User: hook.prompt + 当前上下文                                │
│                                                                  │
│  模型: hook.model ?? getSmallFastModel() = Haiku                │
│                                                                  │
│  输出格式 (JSON Schema 强制):                                    │
│  { "ok": true } 或 { "ok": false, "reason": "..." }             │
└─────────────────────────────────────────────────────────────────┘
                    │
          ┌────────┴────────┐
          ▼                 ▼
    ok = true          ok = false
          │                 │
          ▼                 ▼
    继续执行            阻止执行
    Write 工具          显示原因
```

#### 详细流程

1. **触发条件**: 某个 Hook 事件触发（如 PreToolUse）
2. **Hook 类型**: `type: "prompt"` - 使用模型评估条件
3. **模型选择**: `hook.model ?? getSmallFastModel()` - 默认 Haiku
4. **输出格式**: JSON Schema 强制 `{ ok: boolean, reason?: string }`
5. **执行模式**: 单次 API 调用，无工具
6. **结果处理**: `ok=true` 继续，`ok=false` 阻止

#### Haiku 的角色

- **条件判断者**: 评估是否满足条件
- **简单输出**: 只需要返回 true/false
- **不操作工具**: 纯粹的判断任务
- **快速响应**: 不需要复杂推理

#### 完整代码

```javascript
// ==========================================
// Hook Prompt 完整实现
// ==========================================

import { randomUUID } from "crypto";

// Hook Prompt 输出 Schema
const HookPromptOutputSchema = {
  type: "object",
  properties: {
    ok: { type: "boolean", description: "Whether the condition was met" },
    reason: { type: "string", description: "Reason, if the condition was not met" }
  },
  required: ["ok"],
  additionalProperties: false
};

async function executePromptHook(
  hook,           // hook 配置 { type: "prompt", prompt: "...", model?: "...", timeout?: number }
  hookName,       // hook 名称
  hookEvent,      // hook 事件类型 (PreToolUse, PostToolUse, etc.)
  hookInput,      // hook 输入数据
  abortSignal,    // 中止信号
  toolUseContext, // 工具上下文
  priorMessages,  // 之前的消息 (可选)
  toolUseID       // 工具使用 ID (可选)
) {
  const id = toolUseID || `hook-${randomUUID()}`;

  try {
    // 替换 prompt 中的变量 ($ARGUMENTS 等)
    const prompt = replaceVariables(hook.prompt, hookInput);
    console.log(`Hooks: Processing prompt hook with prompt: ${prompt}`);

    // 构建消息
    const userMessage = createUserMessage({ content: prompt });
    const messages = priorMessages?.length > 0
      ? [...priorMessages, userMessage]
      : [userMessage];

    console.log(`Hooks: Querying model with ${messages.length} messages`);

    // 设置超时 (默认 30 秒)
    const timeout = hook.timeout ? hook.timeout * 1000 : 30000;
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeout);

    const { signal, cleanup } = combineSignals(abortSignal, timeoutController.signal);

    try {
      // 调用模型
      const response = await queryModel({
        messages,
        systemPrompt: [
          `You are evaluating a hook in Claude Code.

Your response must be a JSON object matching one of the following schemas:
1. If the condition is met, return: {"ok": true}
2. If the condition is not met, return: {"ok": false, "reason": "Reason for why it is not met"}`
        ],
        maxThinkingTokens: 0,
        tools: toolUseContext.options.tools,
        signal,
        options: {
          async getToolPermissionContext() {
            return (await toolUseContext.getAppState()).toolPermissionContext;
          },
          // ★ 关键: 使用 hook 指定的模型，或默认 Haiku
          model: hook.model ?? getSmallFastModel(),
          toolChoice: undefined,
          isNonInteractiveSession: true,
          hasAppendSystemPrompt: false,
          agents: [],
          querySource: "hook_prompt",
          mcpTools: [],
          agentId: toolUseContext.agentId,
          // ★ 关键: 强制 JSON 输出格式
          outputFormat: {
            type: "json_schema",
            schema: HookPromptOutputSchema
          }
        }
      });

      clearTimeout(timeoutId);
      cleanup();

      // 更新响应长度
      const responseText = response.message.content
        .filter(block => block.type === "text")
        .map(block => block.text)
        .join("");
      toolUseContext.setResponseLength(len => len + responseText.length);

      const trimmedResponse = responseText.trim();
      console.log(`Hooks: Model response: ${trimmedResponse}`);

      // 解析 JSON
      const parsed = parseJSON(trimmedResponse);
      if (!parsed) {
        console.log(`Hooks: error parsing response as JSON: ${trimmedResponse}`);
        return {
          hook,
          outcome: "non_blocking_error",
          message: createAttachment({
            type: "hook_non_blocking_error",
            hookName,
            toolUseID: id,
            hookEvent,
            stderr: "JSON validation failed",
            stdout: trimmedResponse,
            exitCode: 1
          })
        };
      }

      // 验证 Schema
      const validated = HookPromptOutputSchema.safeParse(parsed);
      if (!validated.success) {
        console.log(`Hooks: model response does not conform to expected schema: ${validated.error.message}`);
        return {
          hook,
          outcome: "non_blocking_error",
          message: createAttachment({
            type: "hook_non_blocking_error",
            hookName,
            toolUseID: id,
            hookEvent,
            stderr: `Schema validation failed: ${validated.error.message}`,
            stdout: trimmedResponse,
            exitCode: 1
          })
        };
      }

      // 条件未满足
      if (!validated.data.ok) {
        console.log(`Hooks: Prompt hook condition was not met: ${validated.data.reason}`);
        return {
          hook,
          outcome: "blocking",
          blockingError: {
            blockingError: `Prompt hook condition was not met: ${validated.data.reason}`,
            command: hook.prompt
          },
          preventContinuation: true,
          stopReason: validated.data.reason
        };
      }

      // 条件满足
      console.log("Hooks: Prompt hook condition was met");
      return {
        hook,
        outcome: "success",
        message: createAttachment({
          type: "hook_success",
          hookName,
          toolUseID: id,
          hookEvent,
          content: "Condition met"
        })
      };

    } catch (error) {
      clearTimeout(timeoutId);
      cleanup();
      if (signal.aborted) {
        return { hook, outcome: "cancelled" };
      }
      throw error;
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.log(`Hooks: Prompt hook error: ${errorMessage}`);
    return {
      hook,
      outcome: "non_blocking_error",
      message: createAttachment({
        type: "hook_non_blocking_error",
        hookName,
        toolUseID: id,
        hookEvent,
        stderr: `Error executing prompt hook: ${errorMessage}`,
        stdout: "",
        exitCode: 1
      })
    };
  }
}
```

---

### 场景五：Hook Agent（验证任务）

#### 在整体任务中的位置

```
用户配置了 Agent Hook:
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "agent",
        "prompt": "Verify that unit tests ran and passed"
      }]
    }]
  }
}

主模型完成任务，触发 Stop 事件
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hook 系统拦截                                                   │
│  检测到 Stop 事件                                                │
│  发现有 agent 类型的 hook                                        │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hook Agent (Haiku) ← 在这里使用 Haiku                          │
│                                                                  │
│  Haiku 执行验证任务 (可以多轮):                                  │
│  1. Read("test-results.json") → 读取测试结果                    │
│  2. 分析结果                                                     │
│  3. 调用 Stop 工具返回验证结果                                   │
│                                                                  │
│  模型: hook.model ?? getSmallFastModel() = Haiku                │
│  可用工具: Read, Stop                                            │
│  最大轮数: 50                                                    │
└─────────────────────────────────────────────────────────────────┘
                    │
          ┌────────┴────────┐
          ▼                 ▼
    ok = true          ok = false
          │                 │
          ▼                 ▼
    任务完成            阻止完成
                        显示原因
```

#### 详细流程

1. **触发条件**: Hook 事件触发，且配置了 `type: "agent"` 的 hook
2. **模型选择**: `hook.model ?? getSmallFastModel()` - 默认 Haiku
3. **可用工具**: `Read` (受限) + `Stop` (返回结果)
4. **执行模式**: 可以多轮对话，最多 50 轮
5. **结束方式**: 必须调用 `Stop` 工具返回结果
6. **输出格式**: `{ ok: boolean, reason?: string }`

#### Haiku 的角色

- **验证执行者**: 可以读取文件进行验证
- **多轮对话**: 可以进行多轮工具调用
- **受限工具**: 只能用只读工具
- **轻量级任务**: 不需要复杂推理

#### 完整代码

```javascript
// ==========================================
// Hook Agent 完整实现
// ==========================================

import { randomUUID } from "crypto";

// Stop 工具 Schema
const StopToolSchema = {
  type: "object",
  properties: {
    ok: { type: "boolean", description: "Whether the condition was met" },
    reason: { type: "string", description: "Reason, if the condition was not met" }
  },
  required: ["ok"],
  additionalProperties: false
};

// Stop 工具定义
function createStopTool() {
  return {
    name: "Stop",
    inputSchema: StopToolSchema,
    inputJSONSchema: StopToolSchema,
    async prompt() {
      return "Use this tool to return your verification result. You MUST call this tool exactly once at the end of your response.";
    }
  };
}

// 注册停止提示 (提醒 agent 调用 Stop 工具)
function registerStopReminder(setAppState, agentId) {
  // 在一定时间后提醒 agent 调用 Stop 工具
  scheduleReminder(setAppState, agentId, "Stop", "",
    (content) => hasToolUse(content, "Stop"),
    `You MUST call the Stop tool to complete this request. Call this tool now.`,
    { timeout: 5000 }
  );
}

async function executeAgentHook(
  hook,           // hook 配置 { type: "agent", prompt: "...", model?: "...", timeout?: number }
  hookName,       // hook 名称
  hookEvent,      // hook 事件类型
  hookInput,      // hook 输入数据
  abortSignal,    // 中止信号
  toolUseContext, // 工具上下文
  priorMessages,  // 之前的消息
  toolUseID       // 工具使用 ID
) {
  const id = toolUseID || `hook-${randomUUID()}`;
  const startTime = Date.now();

  try {
    // 替换变量
    const promptFunction = hook.prompt;
    const prompt = typeof promptFunction === "function"
      ? promptFunction(hookInput)
      : replaceVariables(promptFunction, hookInput);

    console.log(`Hooks: Processing agent hook with prompt: ${prompt}`);

    // 获取 app state
    const appState = await toolUseContext.getAppState();

    // 构建消息
    const userMessage = createUserMessage({ content: prompt });
    const messages = priorMessages?.length > 0
      ? [...priorMessages, userMessage]
      : [userMessage];

    // 系统提示
    const systemPrompt = [
      `You are verifying a condition for a hook in Claude Code.

Your task is to verify whether the following condition is met:
${prompt}

After investigating, you MUST call the Stop tool with your conclusion:
- ok: true if the condition is met
- ok: false with reason if the condition is not met`
    ];

    // 设置超时 (默认 60 秒)
    const timeout = hook.timeout ? hook.timeout * 1000 : 60000;
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeout);

    const { signal, cleanup } = combineSignals(abortSignal, timeoutController.signal);

    // 可用工具
    const tools = [ReadTool, createStopTool()];

    // ★ 关键: 使用 hook 指定的模型，或默认 Haiku
    const model = hook.model ?? getSmallFastModel();
    const maxTurns = 50;

    // 创建 agent 上下文
    const agentId = `hook-agent-${randomUUID()}`;
    const agentContext = {
      ...toolUseContext,
      agentId,
      abortController: timeoutController,
      options: {
        ...toolUseContext.options,
        tools,
        mainLoopModel: model,  // ★ Haiku
        isNonInteractiveSession: true,
        maxThinkingTokens: 0   // 禁用 thinking
      },
      setInProgressToolUseIDs: () => {},
      async getAppState() {
        const state = await toolUseContext.getAppState();
        // 设置权限为自动允许
        const allowRules = state.toolPermissionContext.alwaysAllowRules.session ?? [];
        return {
          ...state,
          toolPermissionContext: {
            ...state.toolPermissionContext,
            mode: "dontAsk",
            alwaysAllowRules: {
              ...state.toolPermissionContext.alwaysAllowRules,
              session: [...allowRules, `Read(/...)`]
            }
          }
        };
      }
    };

    // 注册停止提示
    registerStopReminder(toolUseContext.setAppState, agentId);

    let result = null;
    let turnCount = 0;
    let maxTurnsReached = false;

    // 监听中止信号
    const handleAbort = () => {
      cleanup();
      clearTimeout(timeoutId);
    };
    abortSignal.addEventListener("abort", handleAbort);

    try {
      // 执行 agent 循环
      for await (const event of runAgentLoop({
        messages,
        systemPrompt,
        userContext: {},
        systemContext: {},
        canUseTool: defaultCanUseTool,
        toolUseContext: agentContext,
        querySource: "hook_agent"
      })) {
        // 处理流事件
        handleStreamEvent(
          event,
          () => {},
          (text) => toolUseContext.setResponseLength(len => len + text.length),
          toolUseContext.setStreamMode ?? (() => {}),
          () => {}
        );

        if (event.type === "stream_event" || event.type === "stream_request_start") {
          continue;
        }

        // 计数轮次
        if (event.type === "assistant") {
          turnCount++;
          if (turnCount >= maxTurns) {
            maxTurnsReached = true;
            console.log(`Hooks: Agent turn ${turnCount} hit max turns, aborting`);
            timeoutController.abort();
            break;
          }
        }

        // 检查是否有结构化输出 (Stop 工具调用)
        if (event.type === "attachment" && event.attachment.type === "structured_output") {
          const parsed = StopToolSchema.safeParse(event.attachment.data);
          if (parsed.success) {
            result = parsed.data;
            console.log(`Hooks: Got structured output: ${JSON.stringify(result)}`);
            timeoutController.abort();
            break;
          }
        }
      }
    } finally {
      abortSignal.removeEventListener("abort", handleAbort);
      cleanup();
      clearTimeout(timeoutId);
      removeStopReminder(toolUseContext.setAppState, agentId);
    }

    // 处理结果
    if (!result) {
      if (maxTurnsReached) {
        console.log("Hooks: Agent hook did not complete within 50 turns");
        telemetry("tengu_agent_stop_hook_max_turns", {
          durationMs: Date.now() - startTime,
          turnCount
        });
        return { hook, outcome: "cancelled" };
      }
      console.log("Hooks: Agent hook did not return structured output");
      telemetry("tengu_agent_stop_hook_error", {
        durationMs: Date.now() - startTime,
        turnCount,
        errorType: 1
      });
      return { hook, outcome: "cancelled" };
    }

    if (!result.ok) {
      console.log(`Hooks: Agent hook condition was not met: ${result.reason}`);
      return {
        hook,
        outcome: "blocking",
        blockingError: {
          blockingError: `Agent hook condition was not met: ${result.reason}`,
          command: prompt
        }
      };
    }

    console.log("Hooks: Agent hook condition was met");
    telemetry("tengu_agent_stop_hook_success", {
      durationMs: Date.now() - startTime,
      turnCount
    });

    return {
      hook,
      outcome: "success",
      message: createAttachment({
        type: "hook_success",
        hookName,
        toolUseID: id,
        hookEvent,
        content: "Condition met"
      })
    };

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.log(`Hooks: Agent hook error: ${errorMessage}`);
    telemetry("tengu_agent_stop_hook_error", {
      durationMs: Date.now() - startTime,
      errorType: 2
    });
    return {
      hook,
      outcome: "non_blocking_error",
      message: createAttachment({
        type: "hook_non_blocking_error",
        hookName,
        toolUseID: id,
        hookEvent,
        stderr: `Error executing agent hook: ${errorMessage}`,
        stdout: "",
        exitCode: 1
      })
    };
  }
}
```

---

### 场景六：Token 计数 Fallback

#### 在整体任务中的位置

```
系统需要计算消息的 token 数量
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  尝试使用主模型计数                                              │
│  调用 API: messages.countTokens(...)                            │
└─────────────────────────────────────────────────────────────────┘
                    │
          ┌────────┴────────┐
          ▼                 ▼
      成功返回            失败/超时
          │                 │
          ▼                 ▼
      使用结果         Haiku Fallback ← 在这里使用 Haiku
                            │
                            ▼
                    ┌───────────────────┐
                    │  使用 Haiku 重试   │
                    │  countTokens API  │
                    └───────────────────┘
```

#### 详细流程

1. **触发条件**: 需要计算 token 数量
2. **首次尝试**: 使用主模型的 countTokens API
3. **失败处理**: 如果失败，使用 Haiku 重试
4. **模型选择**: `getSmallFastModel()` = Haiku
5. **原因**: Haiku 的 API 更稳定/便宜

#### Haiku 的角色

- **降级方案**: 主模型失败时的备选
- **稳定性**: Haiku API 更稳定
- **成本**: 更便宜

#### 完整代码

```javascript
// ==========================================
// Token 计数 Fallback 完整实现
// ==========================================

// 使用 Haiku 计数 token
async function countTokensWithHaiku(content, tools) {
  const client = await createClient({
    maxRetries: 1,
    model: getHaikuModel()
  });

  const messages = content.length > 0
    ? content
    : [{ role: "user", content: "count" }];

  const betas = getBetas(getHaikuModel());

  const response = await client.beta.messages.create({
    model: normalizeModelId(getHaikuModel()),
    max_tokens: 1,
    messages,
    tools: tools.length > 0 ? tools : undefined,
    ...(betas.length > 0 ? { betas } : {}),
    metadata: getMetadata(),
    ...getCacheConfig()
  });

  return {
    input_tokens: response.usage.input_tokens,
    cache_creation_input_tokens: response.usage.cache_creation_input_tokens || 0,
    cache_read_input_tokens: response.usage.cache_read_input_tokens || 0
  };
}

// 带 fallback 的 token 计数
async function countTokensWithFallback(content, tools) {
  // 先尝试主模型
  try {
    return await countTokens(content, tools);
  } catch (error) {
    console.log(`countTokensWithFallback: API failed: ${error.message}`);
    logError(error instanceof Error ? error : Error(String(error)));
  }

  // ★ Haiku fallback
  try {
    console.log(`countTokensWithFallback: haiku fallback (${tools.length} tools)`);
    const result = await countTokensWithHaiku(content, tools);
    if (result === null) {
      console.log(`countTokensWithFallback: haiku fallback also returned null (${tools.length} tools)`);
    }
    return result;
  } catch (error) {
    console.log(`countTokensWithFallback: haiku fallback failed: ${error.message}`);
    logError(error instanceof Error ? error : Error(String(error)));
    return null;
  }
}
```

---

## 完整代码实现

### 模型路由核心模块

```javascript
// ==========================================
// model-router.js - 模型路由核心实现
// ==========================================

// 模型 ID 配置 (支持多云)
const MODEL_IDS = {
  haiku35: {
    firstParty: "claude-3-5-haiku-20241022",
    bedrock: "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    vertex: "claude-3-5-haiku@20241022",
    foundry: "claude-3-5-haiku"
  },
  haiku45: {
    firstParty: "claude-haiku-4-5-20251001",
    bedrock: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    vertex: "claude-haiku-4-5@20251001",
    foundry: "claude-haiku-4-5"
  },
  sonnet45: {
    firstParty: "claude-sonnet-4-5-20250929",
    bedrock: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    vertex: "claude-sonnet-4-5@20250929",
    foundry: "claude-sonnet-4-5"
  },
  opus41: {
    firstParty: "claude-opus-4-1-20250805",
    bedrock: "us.anthropic.claude-opus-4-1-20250805-v1:0",
    vertex: "claude-opus-4-1@20250805",
    foundry: "claude-opus-4-1"
  },
  opus45: {
    firstParty: "claude-opus-4-5-20251101",
    bedrock: "us.anthropic.claude-opus-4-5-20251101-v1:0",
    vertex: "claude-opus-4-5@20251101",
    foundry: "claude-opus-4-5"
  },
  opus46: {
    firstParty: "claude-opus-4-6",
    bedrock: "us.anthropic.claude-opus-4-6-v1",
    vertex: "claude-opus-4-6",
    foundry: "claude-opus-4-6"
  }
};

// 获取当前 provider
function getProvider() {
  if (process.env.CLAUDE_CODE_USE_BEDROCK) return "bedrock";
  if (process.env.CLAUDE_CODE_USE_VERTEX) return "vertex";
  if (process.env.CLAUDE_CODE_USE_FOUNDRY) return "foundry";
  return "firstParty";
}

// 获取模型配置
function getModelConfig() {
  const provider = getProvider();
  return {
    haiku35: MODEL_IDS.haiku35[provider],
    haiku45: MODEL_IDS.haiku45[provider],
    sonnet45: MODEL_IDS.sonnet45[provider],
    opus41: MODEL_IDS.opus41[provider],
    opus45: MODEL_IDS.opus45[provider],
    opus46: MODEL_IDS.opus46[provider]
  };
}

// ==========================================
// 核心模型获取函数
// ==========================================

// 获取 Haiku 模型
function getHaikuModel() {
  if (process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL) {
    return process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL;
  }
  return getModelConfig().haiku45;
}

// 获取 Sonnet 模型
function getSonnetModel() {
  if (process.env.ANTHROPIC_DEFAULT_SONNET_MODEL) {
    return process.env.ANTHROPIC_DEFAULT_SONNET_MODEL;
  }
  return getModelConfig().sonnet45;
}

// 获取 Opus 模型
function getOpusModel() {
  if (process.env.ANTHROPIC_DEFAULT_OPUS_MODEL) {
    return process.env.ANTHROPIC_DEFAULT_OPUS_MODEL;
  }
  // firstParty 用最新的 opus46，其他云用 opus41
  if (getProvider() === "firstParty") {
    return getModelConfig().opus46;
  }
  return getModelConfig().opus41;
}

// ★ 核心函数: 获取 Small Fast Model (默认 Haiku)
function getSmallFastModel() {
  return process.env.ANTHROPIC_SMALL_FAST_MODEL || getHaikuModel();
}

// 获取 Best Model (= Opus)
function getBestModel() {
  return getOpusModel();
}

// ==========================================
// 模型别名解析
// ==========================================

const MODEL_ALIASES = ["sonnet", "opus", "haiku", "best", "sonnet[1m]", "opus[1m]", "opusplan"];

function isValidModelAlias(alias) {
  return MODEL_ALIASES.includes(alias.toLowerCase());
}

function resolveModelAlias(modelInput) {
  const trimmed = modelInput.trim();
  const lowered = trimmed.toLowerCase();
  const is1MContext = lowered.endsWith("[1m]");
  const baseAlias = is1MContext ? lowered.replace(/\[1m]$/i, "").trim() : lowered;

  if (isValidModelAlias(baseAlias)) {
    switch (baseAlias) {
      case "opusplan":
        // opusplan 模式下主模型用 sonnet
        return getSonnetModel() + (is1MContext ? "[1m]" : "");
      case "sonnet":
        return getSonnetModel() + (is1MContext ? "[1m]" : "");
      case "haiku":
        return getHaikuModel() + (is1MContext ? "[1m]" : "");
      case "opus":
        return getOpusModel() + (is1MContext ? "[1m]" : "");
      case "best":
        return getBestModel();
      default:
        break;
    }
  }

  // 不是别名，返回原始输入
  if (is1MContext) {
    return trimmed.replace(/\[1m\]$/i, "").trim() + "[1m]";
  }
  return trimmed;
}

// ==========================================
// Agent 模型解析
// ==========================================

function resolveAgentModel(agentConfig, explicitModel, parentModel) {
  // 1. 显式指定的 model 优先
  if (explicitModel && explicitModel !== "inherit") {
    return resolveModelAlias(explicitModel);
  }

  // 2. 检查 agent 配置的默认 model
  const agentModel = agentConfig.model;

  if (agentModel === "inherit" || agentModel === undefined) {
    // 继承父级模型
    return parentModel;
  }

  // 3. 解析 agent 的默认模型别名
  return resolveModelAlias(agentModel);
}

// ==========================================
// 内置 Agent 配置
// ==========================================

const BUILT_IN_AGENTS = {
  "Explore": {
    model: "haiku",           // ★ 硬编码 Haiku
    disallowedTools: ["Edit", "Write", "NotebookEdit", "Task", "ExitPlanMode"]
  },

  "Plan": {
    model: "inherit",         // 继承父级模型
    disallowedTools: ["Edit", "Write", "NotebookEdit", "Task", "ExitPlanMode"]
  },

  "claude-code-guide": {
    model: "haiku",           // ★ 硬编码 Haiku
    permissionMode: "dontAsk",
    tools: ["Glob", "Grep", "Read", "WebFetch", "WebSearch"]
  },

  "statusline-setup": {
    model: "sonnet",          // 配置任务用 Sonnet
    tools: ["Read", "Edit"]
  },

  "general-purpose": {
    model: undefined,         // 使用主会话模型
    tools: ["*"]
  },

  "Bash": {
    model: "inherit",         // 继承父级模型
    tools: ["Bash"]
  },

  "magic-docs": {
    model: "sonnet",
    tools: ["Write"]
  }
};

// ==========================================
// 导出
// ==========================================

module.exports = {
  // Provider
  getProvider,

  // Model config
  getModelConfig,
  MODEL_IDS,

  // Model getters
  getHaikuModel,
  getSonnetModel,
  getOpusModel,
  getSmallFastModel,
  getBestModel,

  // Resolution
  resolveModelAlias,
  resolveAgentModel,
  isValidModelAlias,

  // Agents
  BUILT_IN_AGENTS
};
```

---

## 设计原则总结

### Haiku 使用原则

```
┌─────────────────────────────────────────────────────────────────┐
│                    任务分类决策树                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    需要深度推理吗？
                    ┌─────┴─────┐
                   是           否
                    │           │
                    ▼           ▼
            用主模型       用 Haiku
         (Sonnet/Opus)   (smallFastModel)
                    │           │
                    ▼           ▼
            ┌───────────┐ ┌───────────────────────────┐
            │ 复杂推理   │ │ • 代码搜索 (Explore)       │
            │ 代码生成   │ │ • 文档查询 (guide)         │
            │ 架构设计   │ │ • 内容摘要 (WebFetch)      │
            │ 问题分析   │ │ • 条件判断 (Hook Prompt)   │
            └───────────┘ │ • 验证任务 (Hook Agent)    │
                          │ • Token 计数 (Fallback)    │
                          └───────────────────────────┘
```

### 环境变量覆盖

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `ANTHROPIC_SMALL_FAST_MODEL` | 覆盖 smallFastModel | `getHaikuModel()` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | 覆盖 Haiku 模型 | `claude-haiku-4-5-20251001` |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | 覆盖 Sonnet 模型 | `claude-sonnet-4-5-20250929` |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | 覆盖 Opus 模型 | `claude-opus-4-6` |

### 关键设计决策

1. **静态配置 > 动态路由**: 模型选择写死在 agent 定义里，不在运行时判断
2. **环境变量 > 默认值**: 允许用户通过环境变量覆盖所有模型
3. **inherit 机制**: 某些 agent 继承父级模型，保持一致性
4. **成本优化**: 所有不需要深度推理的任务都用 Haiku

### 你的项目可以借鉴的模式

```python
# Python 实现示例
class ModelRouter:
    # Small Fast Model - 所有辅助任务的入口
    def get_small_fast_model(self):
        return os.environ.get("SMALL_FAST_MODEL") or self.get_haiku_model()

    # Agent 默认模型配置
    AGENT_MODEL_DEFAULTS = {
        "explore": "haiku",      # 快速搜索
        "docs-guide": "haiku",   # 文档查询
        "plan": "inherit",       # 继承父级
        "general": None,         # 用主模型
    }

    def resolve_model(self, agent_type: str, explicit_model: str = None, parent_model: str = None):
        # 1. 显式指定优先
        if explicit_model and explicit_model != "inherit":
            return self._resolve_alias(explicit_model)

        # 2. 查 agent 默认配置
        default = self.AGENT_MODEL_DEFAULTS.get(agent_type)
        if default == "inherit" or default is None:
            return parent_model

        return self._resolve_alias(default)
```

---

## 参考资料

- 基于 Claude Code CLI v2.1.39 逆向分析
- 安装路径: `/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js`
- 分析日期: 2026-02-12
