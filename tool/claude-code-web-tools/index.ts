/**
 * Claude Code CLI - Web Tools
 *
 * 从 Claude Code CLI 逆向提取的 WebFetch 和 WebSearch 工具实现
 *
 * 目录结构:
 * - WebFetchTool.ts  - 网页内容获取工具
 * - WebSearchTool.ts - 网络搜索工具
 * - index.ts         - 导出和使用示例
 */

export { WebFetchTool, DomainBlockedError, DomainCheckFailedError } from './WebFetchTool';
export type { WebFetchInput, WebFetchOutput, LLMProvider } from './WebFetchTool';

export {
  WebSearchTool,
  BraveSearchProvider,
  GoogleSearchProvider
} from './WebSearchTool';
export type {
  WebSearchInput,
  WebSearchOutput,
  WebSearchResult,
  SearchProvider
} from './WebSearchTool';

// ==================== 使用示例 ====================

/*
import { WebFetchTool, WebSearchTool, BraveSearchProvider } from './index';

// ========== WebFetch 使用示例 ==========

// 1. 创建 LLM Provider (可选)
const llmProvider = {
  async chat(params: { messages: Array<{ role: string; content: string }>; maxTokens?: number; temperature?: number }) {
    // 调用你的 LLM API
    const response = await anthropic.messages.create({
      model: "claude-3-haiku-20240307",
      max_tokens: params.maxTokens || 4096,
      messages: params.messages.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))
    });
    return { content: response.content[0].type === 'text' ? response.content[0].text : '' };
  }
};

// 2. 创建 WebFetch 工具
const webFetch = new WebFetchTool({
  llmProvider,
  skipPreflight: false  // 是否跳过域名安全检查
});

// 3. 执行获取
const abortController = new AbortController();

try {
  const result = await webFetch.call(
    {
      url: "https://react.dev/learn",
      prompt: "Summarize the main concepts covered in this React tutorial"
    },
    {
      abortController,
      isNonInteractiveSession: false
    }
  );

  console.log("Fetched bytes:", result.bytes);
  console.log("HTTP status:", result.code, result.codeText);
  console.log("Duration:", result.durationMs, "ms");
  console.log("Result:", result.result);
} catch (error) {
  if (error.name === "DomainBlockedError") {
    console.error("Domain is blocked:", error.message);
  } else if (error.name === "DomainCheckFailedError") {
    console.error("Domain check failed:", error.message);
  } else {
    throw error;
  }
}

// ========== WebSearch 使用示例 ==========

// 方法 1: 使用 Anthropic API 的 server tool (推荐)
const anthropicResponse = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 4096,
  tools: [
    {
      type: "web_search_20250305",
      name: "web_search",
      // allowed_domains: ["react.dev", "github.com"],
      // blocked_domains: ["spam.com"]
    }
  ],
  messages: [
    { role: "user", content: "What are the latest features in React 19?" }
  ]
});

// 方法 2: 使用自定义搜索提供者
const braveProvider = new BraveSearchProvider(process.env.BRAVE_API_KEY!);
const webSearch = new WebSearchTool({ searchProvider: braveProvider });

const searchResult = await webSearch.call({
  query: "React 19 new features 2024",
  allowed_domains: ["react.dev", "github.com", "dev.to"]
});

console.log("Search results:");
console.log(webSearch.formatResultsAsMarkdown(searchResult));
console.log(webSearch.formatSources(searchResult));

// ========== 权限检查示例 ==========

// WebFetch 权限检查
const fetchPermission = await webFetch.checkPermissions({
  url: "https://react.dev/learn",
  prompt: "Summarize this"
});

if (fetchPermission.behavior === "allow") {
  console.log("Allowed:", fetchPermission.decisionReason);
} else if (fetchPermission.behavior === "ask") {
  console.log("Need to ask user:", fetchPermission.message);
  console.log("Suggestions:", fetchPermission.suggestions);
}

// ========== 输入验证示例 ==========

const validation = webFetch.validateInput({
  url: "not-a-valid-url",
  prompt: "test"
});

if (!validation.result) {
  console.error("Validation failed:", validation.message);
}
*/

// ==================== 常量导出 ====================

export const CONSTANTS = {
  // WebFetch 常量
  CACHE_TTL_MS: 900_000,              // 15分钟
  MAX_CACHE_SIZE_BYTES: 52_428_800,   // 50MB
  MAX_URL_LENGTH: 2000,
  MAX_CONTENT_LENGTH: 10_485_760,     // 10MB
  MAX_LLM_INPUT_CHARS: 100_000,       // 100k

  // WebSearch 常量
  DEFAULT_MAX_RESULTS: 10,
  MAX_QUERY_LENGTH: 500
};

// ==================== 工具 Schema 导出 (用于 Function Calling) ====================

export const TOOL_SCHEMAS = {
  webFetch: {
    type: "function",
    function: {
      name: "WebFetch",
      description: "Fetches content from a URL and processes it using an AI model",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            format: "uri",
            description: "The URL to fetch content from"
          },
          prompt: {
            type: "string",
            description: "The prompt to run on the fetched content"
          }
        },
        required: ["url", "prompt"]
      }
    }
  },

  webSearch: {
    type: "function",
    function: {
      name: "WebSearch",
      description: "Searches the web and returns results with links",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            minLength: 2,
            description: "The search query to use"
          },
          allowed_domains: {
            type: "array",
            items: { type: "string" },
            description: "Only include search results from these domains"
          },
          blocked_domains: {
            type: "array",
            items: { type: "string" },
            description: "Never include search results from these domains"
          }
        },
        required: ["query"]
      }
    }
  }
};
