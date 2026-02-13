/**
 * Claude Code CLI - WebFetch Tool 完整实现
 *
 * 从 Claude Code CLI 逆向提取的实现
 * 用于从 URL 获取内容并通过 AI 模型处理
 */

import TurndownService from 'turndown';
import axios, { AxiosResponse } from 'axios';
import { LRUCache } from 'lru-cache';

// ==================== 常量定义 ====================

const CACHE_TTL_MS = 900_000;              // 15分钟
const MAX_CACHE_SIZE_BYTES = 52_428_800;   // 50MB
const MAX_URL_LENGTH = 2000;               // URL 最大长度
const MAX_CONTENT_LENGTH = 10_485_760;     // 10MB 最大响应内容
const MAX_LLM_INPUT_CHARS = 100_000;       // 100k 字符 LLM 输入限制
const MAX_RESULT_SIZE_CHARS = 100_000;     // 工具结果最大字符数

// ==================== 白名单域名 ====================

const ALLOWED_DOMAINS = new Set([
  // Claude/Anthropic
  "platform.claude.com",
  "code.claude.com",
  "modelcontextprotocol.io",
  "github.com/anthropics",
  "agentskills.io",

  // 语言文档
  "docs.python.org",
  "en.cppreference.com",
  "docs.oracle.com",
  "learn.microsoft.com",
  "developer.mozilla.org",
  "go.dev",
  "pkg.go.dev",
  "www.php.net",
  "docs.swift.org",
  "kotlinlang.org",
  "ruby-doc.org",
  "doc.rust-lang.org",
  "www.typescriptlang.org",

  // 前端框架
  "react.dev",
  "angular.io",
  "vuejs.org",
  "nextjs.org",
  "expressjs.com",
  "nodejs.org",
  "bun.sh",
  "jquery.com",
  "getbootstrap.com",
  "tailwindcss.com",
  "d3js.org",
  "threejs.org",
  "redux.js.org",
  "webpack.js.org",
  "jestjs.io",
  "reactrouter.com",

  // 后端框架
  "docs.djangoproject.com",
  "flask.palletsprojects.com",
  "fastapi.tiangolo.com",
  "laravel.com",
  "symfony.com",
  "wordpress.org",
  "docs.spring.io",
  "hibernate.org",
  "tomcat.apache.org",
  "gradle.org",
  "maven.apache.org",
  "asp.net",
  "dotnet.microsoft.com",
  "nuget.org",
  "blazor.net",

  // 数据科学/ML
  "pandas.pydata.org",
  "numpy.org",
  "www.tensorflow.org",
  "pytorch.org",
  "scikit-learn.org",
  "matplotlib.org",
  "requests.readthedocs.io",
  "jupyter.org",
  "keras.io",
  "spark.apache.org",
  "huggingface.co",
  "www.kaggle.com",

  // 移动开发
  "reactnative.dev",
  "docs.flutter.dev",
  "developer.apple.com",
  "developer.android.com",

  // 数据库
  "www.mongodb.com",
  "redis.io",
  "www.postgresql.org",
  "dev.mysql.com",
  "www.sqlite.org",
  "graphql.org",
  "prisma.io",

  // 云/DevOps
  "docs.aws.amazon.com",
  "cloud.google.com",
  "kubernetes.io",
  "www.docker.com",
  "www.terraform.io",
  "www.ansible.com",
  "vercel.com/docs",
  "docs.netlify.com",
  "devcenter.heroku.com",

  // 测试/游戏/其他
  "cypress.io",
  "selenium.dev",
  "docs.unity.com",
  "docs.unrealengine.com",
  "git-scm.com",
  "nginx.org",
  "httpd.apache.org"
]);

// ==================== 类型定义 ====================

export interface WebFetchInput {
  url: string;
  prompt: string;
}

export interface WebFetchOutput {
  bytes: number;
  code: number;
  codeText: string;
  result: string;
  durationMs: number;
  url: string;
}

interface CacheEntry {
  bytes: number;
  code: number;
  codeText: string;
  content: string;
  contentType: string;
}

interface RedirectResult {
  type: "redirect";
  originalUrl: string;
  redirectUrl: string;
  statusCode: number;
}

interface FetchResult {
  bytes: number;
  code: number;
  codeText: string;
  content: string;
  contentType: string;
}

type FetchResponse = FetchResult | RedirectResult;

// ==================== 错误类 ====================

export class DomainBlockedError extends Error {
  constructor(domain: string) {
    super(`Claude Code is unable to fetch from ${domain}`);
    this.name = "DomainBlockedError";
  }
}

export class DomainCheckFailedError extends Error {
  constructor(domain: string) {
    super(
      `Unable to verify if domain ${domain} is safe to fetch. ` +
      `This may be due to network restrictions or enterprise security policies blocking claude.ai.`
    );
    this.name = "DomainCheckFailedError";
  }
}

// ==================== LLM Provider 接口 ====================

export interface LLMProvider {
  chat(params: {
    messages: Array<{ role: string; content: string }>;
    maxTokens?: number;
    temperature?: number;
  }): Promise<{ content: string }>;
}

// ==================== WebFetch 工具类 ====================

export class WebFetchTool {
  public readonly name = "WebFetch";
  public readonly maxResultSizeChars = MAX_RESULT_SIZE_CHARS;

  private readonly turndown: TurndownService;
  private readonly cache: LRUCache<string, CacheEntry>;
  private readonly llmProvider?: LLMProvider;
  private readonly skipPreflight: boolean;

  public readonly description = `
- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL
  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.
  - For GitHub URLs, prefer using the gh CLI via Bash instead (e.g., gh pr view, gh issue view, gh api).
`;

  public readonly inputSchema = {
    type: "object" as const,
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
  };

  constructor(options: {
    llmProvider?: LLMProvider;
    skipPreflight?: boolean;
  } = {}) {
    this.llmProvider = options.llmProvider;
    this.skipPreflight = options.skipPreflight ?? false;

    this.turndown = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced"
    });

    this.cache = new LRUCache<string, CacheEntry>({
      maxSize: MAX_CACHE_SIZE_BYTES,
      sizeCalculation: (entry) => Buffer.byteLength(entry.content),
      ttl: CACHE_TTL_MS
    });
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
    return "Fetch";
  }

  /**
   * 获取活动描述
   */
  public getActivityDescription(input: WebFetchInput): string {
    const summary = this.getToolUseSummary(input);
    return summary ? `Fetching ${summary}` : "Fetching web page";
  }

  /**
   * 获取工具使用摘要
   */
  public getToolUseSummary(input: WebFetchInput): string | null {
    if (!input?.url) return null;
    return this.truncateUrl(input.url, 80);
  }

  /**
   * 验证输入
   */
  public validateInput(input: WebFetchInput): {
    result: boolean;
    message?: string;
    meta?: { reason: string };
    errorCode?: number;
  } {
    const { url } = input;

    try {
      new URL(url);
    } catch {
      return {
        result: false,
        message: `Error: Invalid URL "${url}". The URL provided could not be parsed.`,
        meta: { reason: "invalid_url" },
        errorCode: 1
      };
    }

    if (!this.isValidUrl(url)) {
      return {
        result: false,
        message: `Error: Invalid URL "${url}".`,
        meta: { reason: "invalid_url" },
        errorCode: 1
      };
    }

    return { result: true };
  }

  /**
   * 检查权限
   */
  public async checkPermissions(input: WebFetchInput): Promise<{
    behavior: "allow" | "deny" | "ask";
    message?: string;
    updatedInput?: WebFetchInput;
    decisionReason?: { type: string; reason?: string };
    suggestions?: any[];
  }> {
    try {
      const { url } = input;
      const parsed = new URL(url);
      const hostname = parsed.hostname;
      const pathname = parsed.pathname;

      // 检查白名单
      for (const domain of ALLOWED_DOMAINS) {
        if (domain.includes("/")) {
          const [host, ...pathParts] = domain.split("/");
          const path = "/" + pathParts.join("/");
          if (hostname === host && pathname.startsWith(path)) {
            return {
              behavior: "allow",
              updatedInput: input,
              decisionReason: { type: "other", reason: "Preapproved host and path" }
            };
          }
        } else if (hostname === domain) {
          return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: { type: "other", reason: "Preapproved host" }
          };
        }
      }
    } catch {
      // URL 解析失败，继续请求权限
    }

    // 非白名单域名需要询问
    return {
      behavior: "ask",
      message: `Claude requested permissions to use ${this.name}, but you haven't granted it yet.`,
      suggestions: this.generatePermissionSuggestions(input)
    };
  }

  /**
   * 执行工具调用
   */
  public async call(
    input: WebFetchInput,
    options: {
      abortController: AbortController;
      isNonInteractiveSession?: boolean;
    }
  ): Promise<WebFetchOutput> {
    const startTime = Date.now();
    const { url, prompt } = input;

    // 1. 获取内容
    const fetchResult = await this.fetchUrl(url, {
      signal: options.abortController.signal
    });

    // 2. 处理跨域重定向
    if (this.isRedirectResult(fetchResult)) {
      const statusText = this.getRedirectStatusText(fetchResult.statusCode);

      const result = `REDIRECT DETECTED: The URL redirects to a different host.

Original URL: ${fetchResult.originalUrl}
Redirect URL: ${fetchResult.redirectUrl}
Status: ${fetchResult.statusCode} ${statusText}

To complete your request, I need to fetch content from the redirected URL. Please use WebFetch again with these parameters:
- url: "${fetchResult.redirectUrl}"
- prompt: "${prompt}"`;

      return {
        bytes: 0,
        code: fetchResult.statusCode,
        codeText: statusText,
        result,
        durationMs: Date.now() - startTime,
        url
      };
    }

    // 3. 使用 LLM 处理内容
    const result = await this.processWithLLM(
      fetchResult.content,
      prompt,
      url,
      options.abortController.signal,
      options.isNonInteractiveSession ?? false
    );

    return {
      bytes: fetchResult.bytes,
      code: fetchResult.code,
      codeText: fetchResult.codeText,
      result,
      durationMs: Date.now() - startTime,
      url
    };
  }

  // ==================== 私有方法 ====================

  /**
   * 检查 URL 是否在白名单中
   */
  private isAllowedDomain(url: string): boolean {
    try {
      const parsed = new URL(url);
      const hostname = parsed.hostname;
      const pathname = parsed.pathname;

      for (const domain of ALLOWED_DOMAINS) {
        if (domain.includes("/")) {
          const [host, ...pathParts] = domain.split("/");
          const path = "/" + pathParts.join("/");
          if (hostname === host && pathname.startsWith(path)) {
            return true;
          }
        } else if (hostname === domain) {
          return true;
        }
      }
      return false;
    } catch {
      return false;
    }
  }

  /**
   * URL 验证
   */
  private isValidUrl(url: string): boolean {
    if (url.length > MAX_URL_LENGTH) return false;

    try {
      const parsed = new URL(url);
      // 禁止 URL 中有用户名密码
      if (parsed.username || parsed.password) return false;
      // 必须有有效域名
      if (parsed.hostname.split(".").length < 2) return false;
      return true;
    } catch {
      return false;
    }
  }

  /**
   * 域名安全检查 API
   */
  private async checkDomainSafety(domain: string): Promise<{
    status: "allowed" | "blocked" | "check_failed";
    error?: Error;
  }> {
    try {
      const response = await axios.get(
        `https://api.anthropic.com/api/web/domain_info?domain=${encodeURIComponent(domain)}`,
        { timeout: 5000 }
      );
      if (response.status === 200) {
        return response.data.can_fetch === true
          ? { status: "allowed" }
          : { status: "blocked" };
      }
      return { status: "check_failed", error: new Error(`Status ${response.status}`) };
    } catch (error) {
      return { status: "check_failed", error: error as Error };
    }
  }

  /**
   * 检查同源重定向
   */
  private isSameOriginRedirect(originalUrl: string, redirectUrl: string): boolean {
    try {
      const orig = new URL(originalUrl);
      const redirect = new URL(redirectUrl);

      if (redirect.protocol !== orig.protocol) return false;
      if (redirect.port !== orig.port) return false;
      if (redirect.username || redirect.password) return false;

      const normalize = (h: string) => h.replace(/^www\./, "");
      return normalize(orig.hostname) === normalize(redirect.hostname);
    } catch {
      return false;
    }
  }

  /**
   * 带重定向处理的 fetch
   */
  private async fetchWithRedirect(
    url: string,
    signal: AbortSignal
  ): Promise<AxiosResponse | RedirectResult> {
    try {
      return await axios.get(url, {
        signal,
        maxRedirects: 0,
        responseType: "arraybuffer",
        maxContentLength: MAX_CONTENT_LENGTH,
        headers: {
          Accept: "text/markdown, text/html, */*",
          "User-Agent": "Mozilla/5.0 (compatible; ClaudeCode/1.0)"
        },
        timeout: 30000
      });
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        const status = error.response.status;
        if ([301, 302, 307, 308].includes(status)) {
          const location = error.response.headers.location;
          if (!location) throw new Error("Redirect missing Location header");

          const redirectUrl = new URL(location, url).toString();

          // 同源重定向 - 继续 fetch
          if (this.isSameOriginRedirect(url, redirectUrl)) {
            return this.fetchWithRedirect(redirectUrl, signal);
          }

          // 跨域重定向 - 返回提示
          return {
            type: "redirect",
            originalUrl: url,
            redirectUrl,
            statusCode: status
          };
        }
      }
      throw error;
    }
  }

  /**
   * 主 fetch 函数
   */
  private async fetchUrl(
    url: string,
    options: { signal: AbortSignal }
  ): Promise<FetchResponse> {
    // 1. URL 验证
    if (!this.isValidUrl(url)) {
      throw new Error("Invalid URL");
    }

    // 2. 检查缓存
    const cached = this.cache.get(url);
    if (cached) {
      return {
        bytes: cached.bytes,
        code: cached.code,
        codeText: cached.codeText,
        content: cached.content,
        contentType: cached.contentType
      };
    }

    // 3. HTTP → HTTPS 升级
    let fetchUrl = url;
    try {
      const parsed = new URL(url);
      if (parsed.protocol === "http:") {
        parsed.protocol = "https:";
        fetchUrl = parsed.toString();
      }

      // 4. 域名安全检查 (可配置跳过)
      if (!this.skipPreflight) {
        const hostname = parsed.hostname;
        const domainCheck = await this.checkDomainSafety(hostname);

        switch (domainCheck.status) {
          case "allowed":
            break;
          case "blocked":
            throw new DomainBlockedError(hostname);
          case "check_failed":
            throw new DomainCheckFailedError(hostname);
        }
      }
    } catch (error) {
      if (error instanceof DomainBlockedError ||
          error instanceof DomainCheckFailedError) {
        throw error;
      }
    }

    // 5. 执行 fetch
    const response = await this.fetchWithRedirect(fetchUrl, options.signal);

    // 6. 处理跨域重定向
    if (this.isRedirectResult(response)) {
      return response;
    }

    // 7. 处理响应
    const rawContent = Buffer.from(response.data).toString("utf-8");
    const contentType = (response.headers["content-type"] as string) ?? "";
    const bytes = Buffer.byteLength(rawContent);

    // 8. HTML → Markdown
    let content: string;
    if (contentType.includes("text/html")) {
      content = this.turndown.turndown(rawContent);
    } else {
      content = rawContent;
    }

    // 9. 缓存结果
    const result: CacheEntry = {
      bytes,
      code: response.status,
      codeText: response.statusText,
      content,
      contentType
    };
    this.cache.set(url, result);

    return {
      bytes,
      code: response.status,
      codeText: response.statusText,
      content,
      contentType
    };
  }

  /**
   * 使用 LLM 处理内容
   */
  private async processWithLLM(
    content: string,
    userPrompt: string,
    url: string,
    signal: AbortSignal,
    isNonInteractive: boolean
  ): Promise<string> {
    // 截断过长内容
    const truncatedContent = content.length > MAX_LLM_INPUT_CHARS
      ? content.slice(0, MAX_LLM_INPUT_CHARS) + "\n\n[Content truncated due to length...]"
      : content;

    // 生成 LLM 提示
    const llmPrompt = this.generateLLMPrompt(truncatedContent, userPrompt, url);

    // 如果没有 LLM provider，直接返回内容
    if (!this.llmProvider) {
      return truncatedContent;
    }

    // 调用 LLM
    if (signal.aborted) {
      throw new Error("Request aborted");
    }

    const response = await this.llmProvider.chat({
      messages: [{ role: "user", content: llmPrompt }],
      maxTokens: 4096,
      temperature: 0.2
    });

    return response.content || "No response from model";
  }

  /**
   * 生成 LLM 提示
   */
  private generateLLMPrompt(content: string, userPrompt: string, url: string): string {
    const isAllowedSite = this.isAllowedDomain(url);

    const systemInstruction = isAllowedSite
      ? `Provide a concise response based on the content above. Include relevant details, code examples, and documentation excerpts as needed.`
      : `Provide a concise response based only on the content above. In your response:
 - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is ok as long as we respect the license.
 - Use quotation marks for exact language from articles; any language outside of the quotation should never be word-for-word the same.
 - You are not a lawyer and never comment on the legality of your own prompts and responses.
 - Never produce or reproduce exact song lyrics.`;

    return `${content}\n\n${systemInstruction}\n\nUser request: ${userPrompt}`;
  }

  /**
   * 检查是否为重定向结果
   */
  private isRedirectResult(result: any): result is RedirectResult {
    return result && "type" in result && result.type === "redirect";
  }

  /**
   * 获取重定向状态文本
   */
  private getRedirectStatusText(statusCode: number): string {
    switch (statusCode) {
      case 301: return "Moved Permanently";
      case 308: return "Permanent Redirect";
      case 307: return "Temporary Redirect";
      default: return "Found";
    }
  }

  /**
   * 截断 URL
   */
  private truncateUrl(url: string, maxLength: number): string {
    if (url.length <= maxLength) return url;
    return url.slice(0, maxLength - 1) + "…";
  }

  /**
   * 生成权限建议
   */
  private generatePermissionSuggestions(input: WebFetchInput): any[] {
    try {
      const hostname = new URL(input.url).hostname;
      return [{
        type: "addRules",
        destination: "localSettings",
        rules: [{ toolName: this.name, ruleContent: `domain:${hostname}` }],
        behavior: "allow"
      }];
    } catch {
      return [];
    }
  }
}

export default WebFetchTool;
