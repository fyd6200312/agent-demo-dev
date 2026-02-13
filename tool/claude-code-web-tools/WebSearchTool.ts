/**
 * Claude Code CLI - WebSearch Tool 完整实现
 *
 * 从 Claude Code CLI 逆向提取的实现
 * 用于搜索网络并用结果为响应提供信息
 *
 * 注意：WebSearch 在 Claude Code 中是通过 Anthropic API 的 server_tool_use 实现的，
 * 即服务端工具，在 API 调用时自动执行。此实现提供了接口定义和客户端包装。
 */

// ==================== 类型定义 ====================

export interface WebSearchInput {
  query: string;
  allowed_domains?: string[];
  blocked_domains?: string[];
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  publishedDate?: string;
}

export interface WebSearchOutput {
  results: WebSearchResult[];
  query: string;
  totalResults?: number;
}

/**
 * 搜索提供者接口
 * 可以实现不同的搜索后端（Brave、Google、Bing 等）
 */
export interface SearchProvider {
  search(params: {
    query: string;
    allowedDomains?: string[];
    blockedDomains?: string[];
    maxResults?: number;
  }): Promise<WebSearchResult[]>;
}

// ==================== 常量定义 ====================

const DEFAULT_MAX_RESULTS = 10;
const MAX_QUERY_LENGTH = 500;

// ==================== WebSearch 工具类 ====================

export class WebSearchTool {
  public readonly name = "WebSearch";

  private readonly searchProvider?: SearchProvider;

  public readonly description: string;

  public readonly inputSchema = {
    type: "object" as const,
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
  };

  constructor(options: {
    searchProvider?: SearchProvider;
  } = {}) {
    this.searchProvider = options.searchProvider;

    // 动态生成描述，包含当前日期
    const currentDate = this.getCurrentDate();
    const currentYear = new Date().getFullYear();

    this.description = `
- Allows Claude to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond Claude's knowledge cutoff
- Searches are performed automatically within a single API call

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported to include or block specific websites
  - Web search is only available in the US

IMPORTANT - Use the correct year in search queries:
  - Today's date is ${currentDate}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation ${currentYear}", NOT "React documentation ${currentYear - 1}"
`;
  }

  // ==================== 公共方法 ====================

  /**
   * 检查工具是否启用
   */
  public isEnabled(): boolean {
    return true;
  }

  /**
   * 工具是否只读
   */
  public isReadOnly(): boolean {
    return true;
  }

  /**
   * 是否支持并发
   */
  public isConcurrencySafe(): boolean {
    return true;
  }

  /**
   * 获取用户友好的工具名称
   */
  public userFacingName(): string {
    return "Search";
  }

  /**
   * 获取活动描述
   */
  public getActivityDescription(input: WebSearchInput): string {
    const summary = this.getToolUseSummary(input);
    return summary ? `Searching for "${summary}"` : "Searching the web";
  }

  /**
   * 获取工具使用摘要
   */
  public getToolUseSummary(input: WebSearchInput): string | null {
    if (!input?.query) return null;
    return this.truncateQuery(input.query, 50);
  }

  /**
   * 验证输入
   */
  public validateInput(input: WebSearchInput): {
    result: boolean;
    message?: string;
    meta?: { reason: string };
    errorCode?: number;
  } {
    const { query } = input;

    if (!query || query.trim().length < 2) {
      return {
        result: false,
        message: "Error: Search query must be at least 2 characters.",
        meta: { reason: "query_too_short" },
        errorCode: 1
      };
    }

    if (query.length > MAX_QUERY_LENGTH) {
      return {
        result: false,
        message: `Error: Search query exceeds maximum length of ${MAX_QUERY_LENGTH} characters.`,
        meta: { reason: "query_too_long" },
        errorCode: 1
      };
    }

    return { result: true };
  }

  /**
   * 检查权限 - WebSearch 通常不需要额外权限
   */
  public async checkPermissions(input: WebSearchInput): Promise<{
    behavior: "allow" | "deny" | "ask";
    message?: string;
    updatedInput?: WebSearchInput;
    decisionReason?: { type: string; reason?: string };
  }> {
    return {
      behavior: "allow",
      updatedInput: input,
      decisionReason: { type: "other", reason: "Web search is allowed by default" }
    };
  }

  /**
   * 执行搜索
   *
   * 注意：在 Claude Code 中，WebSearch 是通过 Anthropic API 的
   * server_tool_use (web_search_20250305) 实现的。
   * 这里提供一个客户端实现，可以接入不同的搜索提供者。
   */
  public async call(
    input: WebSearchInput,
    options: {
      abortController?: AbortController;
      maxResults?: number;
    } = {}
  ): Promise<WebSearchOutput> {
    const { query, allowed_domains, blocked_domains } = input;
    const maxResults = options.maxResults ?? DEFAULT_MAX_RESULTS;

    // 如果没有搜索提供者，返回提示信息
    if (!this.searchProvider) {
      return {
        results: [],
        query,
        totalResults: 0
      };
    }

    // 执行搜索
    const results = await this.searchProvider.search({
      query,
      allowedDomains: allowed_domains,
      blockedDomains: blocked_domains,
      maxResults
    });

    return {
      results,
      query,
      totalResults: results.length
    };
  }

  /**
   * 格式化搜索结果为 Markdown
   */
  public formatResultsAsMarkdown(output: WebSearchOutput): string {
    if (output.results.length === 0) {
      return `No results found for: "${output.query}"`;
    }

    const lines: string[] = [];

    for (let i = 0; i < output.results.length; i++) {
      const result = output.results[i];
      lines.push(`${i + 1}. [${result.title}](${result.url})`);
      if (result.snippet) {
        lines.push(`   ${result.snippet}`);
      }
      lines.push("");
    }

    return lines.join("\n");
  }

  /**
   * 生成 Sources 部分
   */
  public formatSources(output: WebSearchOutput): string {
    if (output.results.length === 0) {
      return "";
    }

    const lines: string[] = ["", "Sources:"];

    for (const result of output.results) {
      lines.push(`- [${result.title}](${result.url})`);
    }

    return lines.join("\n");
  }

  /**
   * 获取 Anthropic API 的 server tool 定义
   * 用于在 API 调用时启用 web_search
   */
  public getAnthropicServerToolDefinition(options: {
    allowedDomains?: string[];
    blockedDomains?: string[];
  } = {}): object {
    const tool: any = {
      type: "web_search_20250305",
      name: "web_search"
    };

    if (options.allowedDomains && options.allowedDomains.length > 0) {
      tool.allowed_domains = options.allowedDomains;
    }

    if (options.blockedDomains && options.blockedDomains.length > 0) {
      tool.blocked_domains = options.blockedDomains;
    }

    return tool;
  }

  // ==================== 私有方法 ====================

  /**
   * 获取当前日期 (YYYY-MM-DD)
   */
  private getCurrentDate(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  /**
   * 截断查询
   */
  private truncateQuery(query: string, maxLength: number): string {
    if (query.length <= maxLength) return query;
    return query.slice(0, maxLength - 1) + "…";
  }
}

// ==================== Brave Search Provider 实现 ====================

/**
 * Brave Search API 实现
 */
export class BraveSearchProvider implements SearchProvider {
  private readonly apiKey: string;
  private readonly baseUrl = "https://api.search.brave.com/res/v1/web/search";

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async search(params: {
    query: string;
    allowedDomains?: string[];
    blockedDomains?: string[];
    maxResults?: number;
  }): Promise<WebSearchResult[]> {
    const { query, allowedDomains, blockedDomains, maxResults = 10 } = params;

    // 构建查询（添加域名过滤）
    let searchQuery = query;

    if (allowedDomains && allowedDomains.length > 0) {
      const siteFilter = allowedDomains.map(d => `site:${d}`).join(" OR ");
      searchQuery = `(${siteFilter}) ${query}`;
    }

    if (blockedDomains && blockedDomains.length > 0) {
      const siteFilter = blockedDomains.map(d => `-site:${d}`).join(" ");
      searchQuery = `${searchQuery} ${siteFilter}`;
    }

    try {
      const response = await fetch(
        `${this.baseUrl}?q=${encodeURIComponent(searchQuery)}&count=${maxResults}`,
        {
          headers: {
            Accept: "application/json",
            "X-Subscription-Token": this.apiKey
          }
        }
      );

      if (!response.ok) {
        throw new Error(`Brave Search API error: ${response.status}`);
      }

      const data = await response.json() as any;
      const results = data.web?.results || [];

      return results.map((r: any) => ({
        title: r.title || "",
        url: r.url || "",
        snippet: r.description || "",
        publishedDate: r.published_date
      }));
    } catch (error) {
      console.error("Brave Search error:", error);
      return [];
    }
  }
}

// ==================== Google Custom Search Provider 实现 ====================

/**
 * Google Custom Search API 实现
 */
export class GoogleSearchProvider implements SearchProvider {
  private readonly apiKey: string;
  private readonly searchEngineId: string;
  private readonly baseUrl = "https://www.googleapis.com/customsearch/v1";

  constructor(apiKey: string, searchEngineId: string) {
    this.apiKey = apiKey;
    this.searchEngineId = searchEngineId;
  }

  async search(params: {
    query: string;
    allowedDomains?: string[];
    blockedDomains?: string[];
    maxResults?: number;
  }): Promise<WebSearchResult[]> {
    const { query, allowedDomains, blockedDomains, maxResults = 10 } = params;

    // 构建查询
    let searchQuery = query;

    if (allowedDomains && allowedDomains.length > 0) {
      const siteFilter = allowedDomains.map(d => `site:${d}`).join(" OR ");
      searchQuery = `(${siteFilter}) ${query}`;
    }

    if (blockedDomains && blockedDomains.length > 0) {
      const siteFilter = blockedDomains.map(d => `-site:${d}`).join(" ");
      searchQuery = `${searchQuery} ${siteFilter}`;
    }

    try {
      const url = new URL(this.baseUrl);
      url.searchParams.set("key", this.apiKey);
      url.searchParams.set("cx", this.searchEngineId);
      url.searchParams.set("q", searchQuery);
      url.searchParams.set("num", Math.min(maxResults, 10).toString());

      const response = await fetch(url.toString());

      if (!response.ok) {
        throw new Error(`Google Search API error: ${response.status}`);
      }

      const data = await response.json() as any;
      const items = data.items || [];

      return items.map((item: any) => ({
        title: item.title || "",
        url: item.link || "",
        snippet: item.snippet || ""
      }));
    } catch (error) {
      console.error("Google Search error:", error);
      return [];
    }
  }
}

// ==================== 使用示例 ====================

/*
// 使用 Anthropic API 的 server tool (推荐)
const anthropic = new Anthropic();
const response = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 4096,
  tools: [
    {
      type: "web_search_20250305",
      name: "web_search",
      // allowed_domains: ["example.com"],
      // blocked_domains: ["spam.com"]
    }
  ],
  messages: [
    { role: "user", content: "What are the latest features in React 19?" }
  ]
});

// 使用自定义搜索提供者
const braveProvider = new BraveSearchProvider(process.env.BRAVE_API_KEY!);
const searchTool = new WebSearchTool({ searchProvider: braveProvider });

const result = await searchTool.call({
  query: "React 19 new features",
  allowed_domains: ["react.dev", "github.com"]
});

console.log(searchTool.formatResultsAsMarkdown(result));
console.log(searchTool.formatSources(result));
*/

export default WebSearchTool;
