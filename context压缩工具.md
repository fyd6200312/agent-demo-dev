Claude Code 上下文压缩工具完整实现

一、核心常量定义

// ==================== 上下文窗口常量 ====================

const OUTPUT_BUFFER_RESERVED = 20000; // EFY - 预留给输出的 token const AUTOCOMPACT_BUFFER = 13000; // XSA - 自动压缩缓冲区 const WARNING_THRESHOLD = 20000; // kFY - 警告阈值 const ERROR_THRESHOLD = 20000; // LFY - 错误阈值 const BLOCKING_LIMIT = 3000; // DSA - 阻塞限制

// ==================== Session Memory 常量 ====================

const MAX_SECTION_TOKENS = 2000; // SZ6 - 每个 section 最大 token const MAX_TOTAL_MEMORY_TOKENS = 12000; // Rs4 - session memory 总上限

// ==================== LLM 摘要常量 ====================

const MAX_LLM_INPUT_CHARS = 100000; // 发给摘要 LLM 的最大字符数 const MAX_SUMMARY_OUTPUT_CHARS = 100000; // 摘要输出最大字符数

// ==================== 缓存常量 ====================

const CACHE_TTL_MS = 900000; // 15 分钟 const MAX_CACHE_SIZE_BYTES = 52428800; // 50MB

二、Token 估算函数

// ==================== 简单 Token 估算（字符数 / 4）====================

function w2(text) { // 简单估算：约 4 个字符 = 1 token if (!text) return 0; return Math.ceil(text.length / 4); }

// ==================== JSON 序列化后估算 ====================

function F1(obj) { return JSON.stringify(obj); }

function estimateTokensJson(obj) { return w2(F1(obj)); }

// ==================== 消息列表总 Token 数 ====================

function vv(messages) { // countMessageTokens - 计算消息列表总 token let total = 0; for (const msg of messages) { total += estimateMessageTokens(msg); } return total; }

function estimateMessageTokens(msg) { const content = msg.message?.content || msg.content;

```
  if (typeof content === 'string') {
      return w2(content);
  }

  if (Array.isArray(content)) {
      let tokens = 0;
      for (const block of content) {
          if (block.type === 'text') {
              tokens += w2(block.text || '');
          } else if (block.type === 'tool_use') {
              tokens += w2(block.name || '');
              tokens += estimateTokensJson(block.input || {});
          } else if (block.type === 'tool_result') {
              if (typeof block.content === 'string') {
                  tokens += w2(block.content);
              } else {
                  tokens += estimateTokensJson(block.content);
              }
          } else if (block.type === 'image') {
              tokens += 1000; // 图片粗略估计
          } else {
              tokens += estimateTokensJson(block);
          }
      }
      return tokens;
  }

  return 0;
```

}

// ==================== 压缩后消息 Token 数 ====================

function SU1(summaryMessages) { return vv(summaryMessages); }

// ==================== API Token 计数（精确）====================

async function lx1(messages, tools) { // 调用 Anthropic API 的 token counting 端点 // POST /v1/messages/count_tokens?beta=true // headers: { "anthropic-beta": "token-counting-2024-11-01" }

```
  const response = await client.post("/v1/messages/count_tokens?beta=true", {
      body: { messages, tools },
      headers: { "anthropic-beta": "token-counting-2024-11-01" }
  });

  return response.input_tokens;
```

}

// 带回退的 token 计数 async function pU1(messages, tools) { try { const count = await lx1(messages, tools); if (count !== null) return count; } catch (error) { console.error('API token count failed:', error); }

```
  // 回退到 Haiku 模型尝试
  try {
      return await RL7(messages, tools); // haiku fallback
  } catch (error) {
      return null;
  }
```

}

三、上下文窗口状态检查

// ==================== 获取有效上下文窗口 ====================

function a51(model) { // getEffectiveContextWindow const outputBuffer = Math.min(getMaxOutputTokens(model), OUTPUT_BUFFER_RESERVED); return getContextWindowSize(model) - outputBuffer; }

// ==================== 获取自动压缩阈值 ====================

function lg1(model) { // getAutoCompactThreshold const effectiveWindow = a51(model); const threshold = effectiveWindow - AUTOCOMPACT_BUFFER;

```
  // 支持环境变量覆盖
  const override = process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
  if (override) {
      const pct = parseFloat(override);
      if (!isNaN(pct) && pct > 0 && pct <= 100) {
          const customThreshold = Math.floor(effectiveWindow * (pct / 100));
          return Math.min(customThreshold, threshold);
      }
  }

  return threshold;
```

}

// ==================== 检查上下文状态 ====================

function qc(usedTokens, model) { // checkContextStatus const autoCompactThreshold = lg1(model); const effectiveWindow = um() ? autoCompactThreshold : a51(model);

```
  const percentLeft = Math.max(0, Math.round((effectiveWindow - usedTokens) / effectiveWindow * 100));
  const warningThreshold = effectiveWindow - WARNING_THRESHOLD;
  const errorThreshold = effectiveWindow - ERROR_THRESHOLD;
  const blockingLimit = effectiveWindow - BLOCKING_LIMIT;

  return {
      percentLeft,
      isAboveWarningThreshold: usedTokens >= warningThreshold,
      isAboveErrorThreshold: usedTokens >= errorThreshold,
      isAboveAutoCompactThreshold: um() && usedTokens >= autoCompactThreshold,
      isAtBlockingLimit: usedTokens >= blockingLimit
  };
```

}

// ==================== 是否启用自动压缩 ====================

function um() { // isAutoCompactEnabled if (process.env.DISABLE_COMPACT) return false; if (process.env.DISABLE_AUTO_COMPACT) return false; return getConfig().autoCompactEnabled; // 默认 true }

四、自动压缩触发逻辑

// ==================== 是否应该触发自动压缩 ====================

async function RFY(messages, model, source) { // shouldTriggerAutoCompact

```
  // 跳过压缩自身触发的调用
  if (source === "session_memory" || source === "compact") {
      return false;
  }

  if (!um()) return false;

  const usedTokens = vv(messages);
  const threshold = lg1(model);
  const effectiveWindow = a51(model);

  console.log(`autocompact: tokens=${usedTokens} threshold=${threshold} effectiveWindow=${effectiveWindow}`);

  const { isAboveAutoCompactThreshold } = qc(usedTokens, model);
  return isAboveAutoCompactThreshold;
```

}

// ==================== 执行自动压缩 ====================

async function Qs4(messages, toolUseContext, systemContext, source) { // performAutoCompact

```
  if (process.env.DISABLE_COMPACT) {
      return { wasCompacted: false };
  }

  const model = toolUseContext.options.mainLoopModel;

  if (!await RFY(messages, model, source)) {
      return { wasCompacted: false };
  }

  // 优先尝试 Session Memory 压缩（轻量级）
  const sessionMemoryResult = await mZ6(
      messages,
      toolUseContext.agentId,
      lg1(model)
  );

  if (sessionMemoryResult) {
      clearCompactProgress();
      return { wasCompacted: true, compactionResult: sessionMemoryResult };
  }

  // 回退到完整 LLM 摘要压缩
  try {
      const fullCompactResult = await MW1(
          messages,
          toolUseContext,
          systemContext,
          true,      // isAutoCompact
          undefined, // customInstructions
          true       // skipHooks
      );

      clearCompactProgress();
      return { wasCompacted: true, compactionResult: fullCompactResult };

  } catch (error) {
      if (!isKnownError(error, "COMPACT_ABORTED")) {
          console.error(error);
      }
      return { wasCompacted: false };
  }
```

}

五、完整压缩主函数（MW1）

async function MW1(messages, toolUseContext, systemContext, isAutoCompact, customInstructions, skipHooks = false) { // performFullCompact - 完整压缩主函数

```
  try {
      if (messages.length === 0) {
          throw Error("No messages to compact");
      }

      // 1. 计算压缩前 token 数
      const preCompactTokens = vv(messages);

      // 2. 分析 token 使用情况
      const tokenBreakdown = ea4(messages);
      const breakdownStats = As4(tokenBreakdown);

      // 3. 获取应用状态
      const appState = await toolUseContext.getAppState();

      // 4. 通知压缩开始
      toolUseContext.onCompactProgress?.({ type: "hooks_start", hookType: "pre_compact" });
      toolUseContext.setSDKStatus?.("compacting");

      // 5. 执行 PreCompact hooks
      const hookResult = await sW6(
          {
              trigger: isAutoCompact ? "auto" : "manual",
              customInstructions: customInstructions ?? null
          },
          toolUseContext.abortController.signal
      );

      // 合并 hook 返回的自定义指令
      if (hookResult.newCustomInstructions) {
          customInstructions = customInstructions
              ? `${customInstructions}n${hookResult.newCustomInstructions}`
              : hookResult.newCustomInstructions;
      }

      const userDisplayMessage = hookResult.userDisplayMessage;

      // 6. 设置流模式
      toolUseContext.setStreamMode?.("requesting");
      toolUseContext.setResponseLength?.(() => 0);
      toolUseContext.onCompactProgress?.({ type: "compact_start" });

      // 7. 构建摘要 Prompt
      const summaryPrompt = UOA(customInstructions); // buildFullSummaryPrompt
      const summaryRequestMessage = createUserMessage({ content: summaryPrompt });

      // 8. 调用 LLM 生成摘要
      const summaryResponse = await $s4({
          messages,
          summaryRequest: summaryRequestMessage,
          appState,
          context: toolUseContext,
          preCompactTokenCount: preCompactTokens,
          cacheSafeParams: systemContext
      });

      // 9. 提取摘要文本
      const summaryText = o51(summaryResponse); // extractSummaryText

      if (!summaryText) {
          console.error("Compact failed: no summary text in response");
          throw Error("Failed to generate conversation summary");
      }

      if (summaryText.startsWith("[API_ERROR]")) {
          throw Error(summaryText);
      }

      if (summaryText.startsWith("[PROMPT_TOO_LONG]")) {
          throw Error("Prompt too long for summarization");
      }

      // 10. 清理文件读取状态缓存
      const previousReadState = copyReadFileState(toolUseContext.readFileState);
      toolUseContext.readFileState.clear();
      clearAllCaches();

      // 11. 收集附件
      const [readStateAttachments, otherAttachments] = await Promise.all([
          Os4(previousReadState, toolUseContext, MAX_ATTACHMENTS),
          Xs4(toolUseContext)
      ]);

      const attachments = [...readStateAttachments, ...otherAttachments];

      // 12. 添加 Session Memory 附件
      const sessionMemoryAttachment = _s4(toolUseContext.agentId ?? getSessionId());
      if (sessionMemoryAttachment) {
          attachments.push(sessionMemoryAttachment);
      }

      // 13. 添加 Memory Attachment
      const memoryAttachment = RZ6(toolUseContext.agentId);
      if (memoryAttachment) {
          attachments.push(memoryAttachment);
      }

      // 14. 添加 Tool Capabilities
      const toolCapabilities = Js4();
      if (toolCapabilities) {
          attachments.push(toolCapabilities);
      }

      // 15. 通知 hooks 开始
      toolUseContext.onCompactProgress?.({ type: "hooks_start", hookType: "session_start" });

      // 16. 获取压缩模型配置
      const compactModel = await getCompactModel("compact", { model: toolUseContext.options.mainLoopModel });

      // 17. 计算压缩后 token 数
      const postCompactTokens = PZ([summaryResponse]); // countTokens
      const compactionUsage = wp(summaryResponse);      // getUsage

      // 18. 记录分析日志
      console.log("tengu_compact", {
          preCompactTokenCount: preCompactTokens,
          postCompactTokenCount: postCompactTokens,
          compactionInputTokens: compactionUsage?.input_tokens,
          compactionOutputTokens: compactionUsage?.output_tokens,
          compactionCacheReadTokens: compactionUsage?.cache_read_input_tokens ?? 0,
          compactionCacheCreationTokens: compactionUsage?.cache_creation_input_tokens ?? 0,
          ...breakdownStats
      });

      // 19. 创建边界标记消息
      const boundaryMarker = kU1(
          isAutoCompact ? "auto" : "manual",
          preCompactTokens ?? 0,
          messages[messages.length - 1]?.uuid
      );

      // 20. 获取 transcript 路径
      const transcriptPath = AO(getSessionId()); // getTranscriptPath

      // 21. 构建摘要消息
      const summaryMessages = [
          createUserMessage({
              content: ox1(summaryText, true, transcriptPath), // buildCompactedContextMessage
              isCompactSummary: true,
              isVisibleInTranscriptOnly: true
          })
      ];

      // 22. 清除 prompt cache 状态
      QOA(toolUseContext.options.querySource ?? "compact", toolUseContext.agentId);

      // 23. 返回压缩结果
      return {
          boundaryMarker,
          summaryMessages,
          attachments,
          hookResults: [],
          userDisplayMessage,
          preCompactTokenCount: preCompactTokens,
          postCompactTokenCount: postCompactTokens,
          compactionUsage
      };

  } catch (error) {
      Hs4(error, toolUseContext); // handleCompactError
      throw error;

  } finally {
      toolUseContext.setStreamMode?.("requesting");
      toolUseContext.setResponseLength?.(() => 0);
      toolUseContext.onCompactProgress?.({ type: "compact_end" });
      toolUseContext.setSDKStatus?.(null);
  }
```

}

六、Session Memory 轻量压缩（mZ6）

async function mZ6(messages, agentId, threshold) { // trySessionMemoryCompact - 轻量级 Session Memory 压缩

```
  if (!BZ6()) { // isSessionMemoryEnabled
      return null;
  }

  await fFY(); // initializeSessionMemory
  await Zs4(); // syncSessionState

  const lastSummarizedId = Ps4(); // getLastSummarizedMessageId
  const sessionMemory = CZ6();     // getCurrentSessionMemory

  if (!sessionMemory) {
      trackEvent("session_memory_compact_no_memory");
      return null;
  }

  // 检查是否为空模板
  if (await Ss4(sessionMemory)) { // isEmptyTemplate
      trackEvent("session_memory_compact_empty_template");
      return null;
  }

  try {
      let startIndex;

      if (lastSummarizedId) {
          startIndex = messages.findIndex(m => m.uuid === lastSummarizedId);
          if (startIndex === -1) {
              trackEvent("session_memory_compact_summarized_id_not_found");
              return null;
          }
      } else {
          startIndex = messages.length - 1;
          trackEvent("session_memory_compact_resumed_session");
      }

      // 计算起始偏移
      const startOffset = TFY(messages, startIndex);

      // 获取需要压缩的消息
      const recentMessages = messages
          .slice(startOffset)
          .filter(m => !lR(m)); // 过滤 meta 消息

      // 获取压缩模型
      const compactModel = await getCompactModel("compact", { model: getDefaultModel() });
      const model = parseModelId(getModelId());

      // 构建 Session Memory 压缩结果
      const compactResult = vFY(
          messages,
          sessionMemory,
          recentMessages,
          compactModel,
          model,
          agentId
      );

      // 转换为消息列表
      const summaryMessages = $t(compactResult); // flattenCompactResult

      // 计算压缩后 token 数
      const postCompactTokens = SU1(summaryMessages);

      // 检查是否超过阈值
      if (threshold !== undefined && postCompactTokens >= threshold) {
          trackEvent("session_memory_compact_threshold_exceeded", {
              postCompactTokenCount: postCompactTokens,
              autoCompactThreshold: threshold
          });
          return null;
      }

      return {
          ...compactResult,
          postCompactTokenCount: postCompactTokens
      };

  } catch (error) {
      trackEvent("session_memory_compact_error");
      return null;
  }
```

}

七、压缩结果扁平化

function $t(compactionResult) { // flattenCompactResult - 将压缩结果转换为消息数组

```
  return [
      compactionResult.boundaryMarker,           // 压缩边界标记
      ...compactionResult.summaryMessages,       // 摘要消息
      ...(compactionResult.messagesToKeep ?? []), // 保留的最近消息
      ...compactionResult.attachments,           // 附件
      ...compactionResult.hookResults            // Hook 结果
  ];
```

}

八、压缩后上下文消息构建

function ox1(summaryText, includeTranscriptPath, transcriptPath, hasRetainedContext) { // buildCompactedContextMessage - 构建压缩后的上下文消息

```
  let message = `This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.
```

${B99(summaryText)}`; // processSummaryTags - 处理

标签

```
  if (transcriptPath) {
      message += `
```

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: ${transcriptPath}`; }

```
  if (hasRetainedContext) {
      message += `
```

Recent messages are preserved verbatim.`; }

```
  return message;
```

}

function B99(text) { // processSummaryTags - 处理摘要文本中的标签

```
  let result = text;

  // 移除 <analysis> 块，转换为普通文本
  const analysisMatch = result.match(/<analysis>([sS]*?)</analysis>/);
  if (analysisMatch) {
      const analysisContent = analysisMatch[1] || "";
      result = result.replace(
          /<analysis>[sS]*?</analysis>/,
          `Analysis:n${analysisContent.trim()}`
      );
  }

  // 移除 <summary> 块，转换为普通文本
  const summaryMatch = result.match(/<summary>([sS]*?)</summary>/);
  if (summaryMatch) {
      const summaryContent = summaryMatch[1] || "";
      result = result.replace(
          /<summary>[sS]*?</summary>/,
          `Summary:n${summaryContent.trim()}`
      );
  }

  // 清理多余空行
  result = result.replace(/nn+/g, "nn");

  return result.trim();
```

}

九、Token 使用分析

function ea4(messages) { // analyzeTokenUsage - 分析 token 使用情况

```
  const breakdown = {
      toolRequests: new Map(),
      toolResults: new Map(),
      humanMessages: 0,
      assistantMessages: 0,
      localCommandOutputs: 0,
      other: 0,
      attachments: new Map(),
      duplicateFileReads: new Map(),
      total: 0
  };

  const toolIdToName = new Map();
  const fileReadIds = new Map();
  const fileReadCounts = new Map();

  // 统计附件
  for (const msg of messages) {
      if (msg.type === "attachment") {
          const type = msg.attachment.type || "unknown";
          breakdown.attachments.set(type, (breakdown.attachments.get(type) || 0) + 1);
      }
  }

  // 统计消息 token
  for (const msg of getVisibleMessages(messages)) {
      const content = msg.message.content;

      if (typeof content === "string") {
          const tokens = w2(content);
          breakdown.total += tokens;

          if (msg.type === "user" && content.includes("local-command-stdout")) {
              breakdown.localCommandOutputs += tokens;
          } else if (msg.type === "user") {
              breakdown.humanMessages += tokens;
          } else {
              breakdown.assistantMessages += tokens;
          }
      } else if (Array.isArray(content)) {
          for (const block of content) {
              processContentBlock(block, msg, breakdown, toolIdToName, fileReadIds, fileReadCounts);
          }
      }
  }

  // 计算重复读取开销
  for (const [filePath, info] of fileReadCounts) {
      if (info.count > 1) {
          const wastedTokens = Math.floor(info.totalTokens / info.count) * (info.count - 1);
          breakdown.duplicateFileReads.set(filePath, {
              count: info.count,
              tokens: wastedTokens
          });
      }
  }

  return breakdown;
```

}

function As4(breakdown) { // formatBreakdownStats - 转换为可记录的统计对象

```
  const stats = {
      total_tokens: breakdown.total,
      human_message_tokens: breakdown.humanMessages,
      assistant_message_tokens: breakdown.assistantMessages,
      local_command_output_tokens: breakdown.localCommandOutputs,
      other_tokens: breakdown.other
  };

  // 添加附件统计
  breakdown.attachments.forEach((count, type) => {
      stats[`attachment_${type}_count`] = count;
  });

  // 添加工具请求统计
  breakdown.toolRequests.forEach((tokens, name) => {
      stats[`tool_request_${name}_tokens`] = tokens;
  });

  // 添加工具结果统计
  breakdown.toolResults.forEach((tokens, name) => {
      stats[`tool_result_${name}_tokens`] = tokens;
  });

  // 计算重复读取统计
  const duplicateTokens = [...breakdown.duplicateFileReads.values()]
      .reduce((sum, info) => sum + info.tokens, 0);

  stats.duplicate_read_tokens = duplicateTokens;
  stats.duplicate_read_file_count = breakdown.duplicateFileReads.size;

  // 计算百分比
  if (breakdown.total > 0) {
      stats.human_message_percent = Math.round(breakdown.humanMessages / breakdown.total * 100);
      stats.assistant_message_percent = Math.round(breakdown.assistantMessages / breakdown.total * 100);
      stats.local_command_output_percent = Math.round(breakdown.localCommandOutputs / breakdown.total * 100);
      stats.duplicate_read_percent = Math.round(duplicateTokens / breakdown.total * 100);
  }

  return stats;
```

}

十、PreCompact Hooks 系统

async function sW6(config, signal, timeoutMs = 30000) { // executePreCompactHooks

```
  const hookInput = {
      ...createBaseHookInput(),
      hook_event_name: "PreCompact",
      trigger: config.trigger,           // "manual" | "auto"
      custom_instructions: config.customInstructions
  };

  const results = await executeHooksOutsideREPL({
      hookInput,
      matchQuery: config.trigger,
      signal,
      timeoutMs
  });

  if (results.length === 0) {
      return {};
  }

  // 收集成功的 hook 输出
  const newInstructions = results
      .filter(r => r.succeeded && r.output.trim().length > 0)
      .map(r => r.output.trim());

  // 记录日志
  const logs = [];
  for (const result of results) {
      if (result.succeeded) {
          if (result.output.trim()) {
              logs.push(`PreCompact [${result.command}] completed: ${result.output.trim()}`);
          } else {
              logs.push(`PreCompact [${result.command}] completed`);
          }
      } else {
          logs.push(`PreCompact [${result.command}] failed: ${result.output.trim() || "No output"}`);
      }
  }

  return {
      newCustomInstructions: newInstructions.length > 0
          ? newInstructions.join("n")
          : undefined,
      logs
  };
```

}

十一、Transcript 路径管理

function AO(sessionId) { // getTranscriptPath - 获取 transcript 文件路径

```
  const projectDir = getProjectDir();     // ~/.claude/projects/<project-hash>
  const transcriptsDir = path.join(projectDir, "transcripts");

  // 确保目录存在
  ensureDir(transcriptsDir);

  // 使用 session ID 构建文件名
  return path.join(transcriptsDir, `${sessionId}.json`);
```

}

function p6() { // getSessionId - 获取当前 session ID return currentSessionId; }

function H8() { // getProjectDir - 获取项目目录 const cwd = process.cwd(); const hash = createHash('sha256').update(cwd).digest('hex').slice(0, 16); return path.join(os.homedir(), '.claude', 'projects', hash); }

十二、模型上下文窗口配置

// 模型上下文窗口大小映射 const MODEL_CONTEXT_WINDOWS = { // Claude 系列 "claude-3-opus": 200000, "claude-3-sonnet": 200000, "claude-3-haiku": 200000, "claude-3.5-sonnet": 200000, "claude-3.5-haiku": 200000, "claude-opus-4": 200000, "claude-sonnet-4": 200000,

```
  // 默认
  "default": 200000
```

};

function yG(model, provider) { // getContextWindowSize const modelLower = model.toLowerCase();

```
  for (const [key, size] of Object.entries(MODEL_CONTEXT_WINDOWS)) {
      if (modelLower.includes(key)) {
          return size;
      }
  }

  return MODEL_CONTEXT_WINDOWS.default;
```

}

function UP() { // getCurrentProvider return currentProvider; }

十三、环境变量支持

// 压缩相关环境变量

DISABLE_COMPACT // 完全禁用压缩 DISABLE_AUTO_COMPACT // 禁用自动压缩（保留手动） CLAUDE_AUTOCOMPACT_PCT_OVERRIDE // 覆盖自动压缩阈值百分比 (0-100) CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE // 覆盖阻塞限制

---

这就是 Claude Code 压缩工具的完整实现。核心流程是：

1.  检测触发：qc() 检查当前 token 使用率，RFY() 判断是否触发
2.  优先 Session Memory：mZ6() 尝试轻量级 Session Memory 压缩
3.  回退 LLM 摘要：MW1() 使用完整 LLM 生成结构化摘要
4.  保存 Transcript：压缩前保存完整对话到文件
5.  重建上下文：ox1() 构建压缩后的上下文消息






⏺ ┌─────────────────────────────────────────────────────────────────────────────────────────┐                                                                                                                        
  │                        Claude Code 上下文压缩系统完整架构                                  │
  └─────────────────────────────────────────────────────────────────────────────────────────┘                                                                                                                        
                                                                                 
  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗                                                                                                                      
  ║                                    主循环 (Agent Loop)                                     ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   ┌─────────────────┐    每次 LLM 调用前检查    ┌─────────────────────────────────────┐   ║
  ║   │   用户消息输入   │ ─────────────────────────▶│        上下文状态检查模块            │   ║
  ║   └─────────────────┘                           │                                     │   ║
  ║                                                 │  qc(usedTokens, model)              │   ║
  ║                                                 │  ├── percentLeft                    │   ║
  ║                                                 │  ├── isAboveWarningThreshold        │   ║
  ║                                                 │  ├── isAboveErrorThreshold          │   ║
  ║                                                 │  ├── isAboveAutoCompactThreshold ◀──┼───╋──┐
  ║                                                 │  └── isAtBlockingLimit              │   ║  │
  ║                                                 └─────────────────────────────────────┘   ║  │
  ║                                                              │                            ║  │
  ║                                                              ▼                            ║  │
  ║                                                 ┌─────────────────────────────────────┐   ║  │
  ║                                                 │    RFY(messages, model, source)     │   ║  │
  ║                                                 │    shouldTriggerAutoCompact()       │   ║  │
  ║                                                 │                                     │   ║  │
  ║                                                 │  检查条件:                           │   ║  │
  ║                                                 │  1. source != "compact"             │   ║  │
  ║                                                 │  2. um() == true (自动压缩开启)      │   ║  │
  ║                                                 │  3. usedTokens >= threshold         │   ║  │
  ║                                                 └─────────────────────────────────────┘   ║  │
  ║                                                              │                            ║  │
  ║                                         ┌────────────────────┴────────────────────┐       ║  │
  ║                                         ▼                                         ▼       ║  │
  ║                                      [false]                                   [true]     ║  │
  ║                                         │                                         │       ║  │
  ║                                         ▼                                         ▼       ║  │
  ║                               ┌──────────────────┐               ┌────────────────────┐   ║  │
  ║                               │   继续正常对话    │               │ Qs4() 执行自动压缩  │   ║  │
  ║                               └──────────────────┘               │ performAutoCompact │   ║  │
  ║                                                                  └────────────────────┘   ║  │
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝  │
                                                                             │                    │
  ┌──────────────────────────────────────────────────────────────────────────┼────────────────────┘
  │                                                                          │
  │  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  │  ║                              阈值计算模块 (Threshold)                                  ║
  │  ╠═══════════════════════════════════════════════════════════════════════════════════════╣
  │  ║                                                                                       ║
  │  ║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
  │  ║   │  常量定义                                                                    │     ║
  │  ║   │  ─────────────────────────────────────────────────────────────────────────  │     ║
  │  ║   │  OUTPUT_BUFFER_RESERVED  = 20000   (EFY)  预留输出缓冲                       │     ║
  │  ║   │  AUTOCOMPACT_BUFFER      = 13000   (XSA)  自动压缩缓冲                       │     ║
  │  ║   │  WARNING_THRESHOLD       = 20000   (kFY)  警告阈值缓冲                       │     ║
  │  ║   │  ERROR_THRESHOLD         = 20000   (LFY)  错误阈值缓冲                       │     ║
  │  ║   │  BLOCKING_LIMIT          = 3000    (DSA)  阻塞限制缓冲                       │     ║
  │  ║   └─────────────────────────────────────────────────────────────────────────────┘     ║
  │  ║                                                                                       ║
  │  ║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
  │  ║   │  a51(model) - getEffectiveContextWindow                                     │     ║
  │  ║   │  ─────────────────────────────────────────────────────────────────────────  │     ║
  │  ║   │  effectiveWindow = contextWindowSize - min(maxOutputTokens, 20000)          │     ║
  │  ║   │                                                                             │     ║
  │  ║   │  例: Claude Sonnet 4                                                        │     ║
  │  ║   │      200,000 - 20,000 = 180,000 tokens                                      │     ║
  │  ║   └─────────────────────────────────────────────────────────────────────────────┘     ║
  │  ║                                           │                                           ║
  │  ║                                           ▼                                           ║
  │  ║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
  │  ║   │  lg1(model) - getAutoCompactThreshold                                       │     ║
  │  ║   │  ─────────────────────────────────────────────────────────────────────────  │     ║
  │  ║   │  threshold = effectiveWindow - AUTOCOMPACT_BUFFER                           │     ║
  │  ║   │                                                                             │     ║
  │  ║   │  例: 180,000 - 13,000 = 167,000 tokens                                      │     ║
  │  ║   │                                                                             │     ║
  │  ║   │  支持环境变量覆盖: CLAUDE_AUTOCOMPACT_PCT_OVERRIDE                           │     ║
  │  ║   │  如设置 80, 则 threshold = effectiveWindow * 0.8                            │     ║
  │  ║   └─────────────────────────────────────────────────────────────────────────────┘     ║
  │  ║                                                                                       ║
  │  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
  │
  │
  │  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  │  ║                              Token 计数模块 (Token Counting)                           ║
  │  ╠═══════════════════════════════════════════════════════════════════════════════════════╣
  │  ║                                                                                       ║
  │  ║   ┌───────────────────────────────────┐     ┌───────────────────────────────────┐     ║
  │  ║   │  w2(text) - estimateTokens        │     │  F1(obj) - stringify              │     ║
  │  ║   │  ───────────────────────────────  │     │  ───────────────────────────────  │     ║
  │  ║   │  return Math.ceil(text.length / 4)│     │  return JSON.stringify(obj)       │     ║
  │  ║   │                                   │     │                                   │     ║
  │  ║   │  简单估算: 4 字符 ≈ 1 token       │     │  序列化后再估算 token             │     ║
  │  ║   └───────────────────────────────────┘     └───────────────────────────────────┘     ║
  │  ║                         │                                   │                         ║
  │  ║                         └─────────────┬─────────────────────┘                         ║
  │  ║                                       ▼                                               ║
  │  ║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
  │  ║   │  vv(messages) - countMessageTokens                                          │     ║
  │  ║   │  ─────────────────────────────────────────────────────────────────────────  │     ║
  │  ║   │  遍历消息列表，累加每条消息的 token:                                         │     ║
  │  ║   │  - string content: w2(content)                                              │     ║
  │  ║   │  - text block: w2(block.text)                                               │     ║
  │  ║   │  - tool_use: w2(name) + w2(F1(input))                                       │     ║
  │  ║   │  - tool_result: w2(content) 或 w2(F1(content))                              │     ║
  │  ║   │  - image: 固定 ~1000 tokens                                                 │     ║
  │  ║   └─────────────────────────────────────────────────────────────────────────────┘     ║
  │  ║                                                                                       ║
  │  ║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
  │  ║   │  lx1(messages, tools) - API Token Count (精确)                              │     ║
  │  ║   │  ─────────────────────────────────────────────────────────────────────────  │     ║
  │  ║   │  POST /v1/messages/count_tokens?beta=true                                   │     ║
  │  ║   │  Header: "anthropic-beta": "token-counting-2024-11-01"                      │     ║
  │  ║   │                                                                             │     ║
  │  ║   │  pU1() - 带回退的 token 计数                                                │     ║
  │  ║   │  先尝试 API，失败则回退到 Haiku 模型                                        │     ║
  │  ║   └─────────────────────────────────────────────────────────────────────────────┘     ║
  │  ║                                                                                       ║
  │  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
  │
  ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              自动压缩执行模块 (Auto Compact)                               ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   Qs4(messages, toolUseContext, systemContext, source)                                    ║
  ║   performAutoCompact()                                                                    ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │                                                                                 │     ║
  ║   │   ┌────────────────────────────────┐                                            │     ║
  ║   │   │  1. 检查 DISABLE_COMPACT 环境变量                                           │     ║
  ║   │   └────────────────────────────────┘                                            │     ║
  ║   │                    │                                                            │     ║
  ║   │                    ▼                                                            │     ║
  ║   │   ┌────────────────────────────────┐                                            │     ║
  ║   │   │  2. RFY() 检查是否需要压缩      │                                            │     ║
  ║   │   └────────────────────────────────┘                                            │     ║
  ║   │                    │                                                            │     ║
  ║   │          ┌────────┴────────┐                                                    │     ║
  ║   │          ▼                 ▼                                                    │     ║
  ║   │       [false]           [true]                                                  │     ║
  ║   │          │                 │                                                    │     ║
  ║   │          ▼                 ▼                                                    │     ║
  ║   │   ┌──────────────┐  ┌──────────────────────────────────────────────────────┐    │     ║
  ║   │   │返回 wasCompacted│  │  3. 优先尝试 Session Memory 压缩                    │    │     ║
  ║   │   │    = false    │  │     mZ6(messages, agentId, threshold)               │    │     ║
  ║   │   └──────────────┘  │     trySessionMemoryCompact()                        │    │     ║
  ║   │                     └──────────────────────────────────────────────────────┘    │     ║
  ║   │                                        │                                        │     ║
  ║   │                              ┌─────────┴─────────┐                              │     ║
  ║   │                              ▼                   ▼                              │     ║
  ║   │                          [成功]              [失败/null]                        │     ║
  ║   │                              │                   │                              │     ║
  ║   │                              ▼                   ▼                              │     ║
  ║   │                     ┌──────────────┐  ┌──────────────────────────────────┐      │     ║
  ║   │                     │ 返回压缩结果  │  │  4. 回退到完整 LLM 摘要压缩       │      │     ║
  ║   │                     │              │  │     MW1() performFullCompact()   │      │     ║
  ║   │                     └──────────────┘  └──────────────────────────────────┘      │     ║
  ║   │                                                  │                              │     ║
  ║   │                                                  ▼                              │     ║
  ║   │                                       ┌──────────────────┐                      │     ║
  ║   │                                       │   返回压缩结果    │                      │     ║
  ║   │                                       └──────────────────┘                      │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
  ╔════════════════════════════════════════════╗  ╔════════════════════════════════════════════╗
  ║   Session Memory 压缩 (轻量级)              ║  ║   完整 LLM 摘要压缩 (重量级)                ║
  ╠════════════════════════════════════════════╣  ╠════════════════════════════════════════════╣
  ║                                            ║  ║                                            ║
  ║  mZ6(messages, agentId, threshold)         ║  ║  MW1(messages, ctx, sys, isAuto, inst)     ║
  ║  trySessionMemoryCompact()                 ║  ║  performFullCompact()                      ║
  ║                                            ║  ║                                            ║
  ║  ┌──────────────────────────────────────┐  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  │ 1. 检查 Session Memory 是否启用       │  ║  ║  │ 1. 计算压缩前 token 数               │  ║
  ║  │    BZ6() isSessionMemoryEnabled      │  ║  ║  │    preCompactTokens = vv(messages)  │  ║
  ║  └──────────────────────────────────────┘  ║  ║  └──────────────────────────────────────┘  ║
  ║                   │                        ║  ║                   │                        ║
  ║                   ▼                        ║  ║                   ▼                        ║
  ║  ┌──────────────────────────────────────┐  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  │ 2. 初始化并同步 Session 状态          │  ║  ║  │ 2. 分析 token 使用情况               │  ║
  ║  │    fFY() initializeSessionMemory     │  ║  ║  │    ea4(messages) analyzeTokenUsage  │  ║
  ║  │    Zs4() syncSessionState            │  ║  ║  └──────────────────────────────────────┘  ║
  ║  └──────────────────────────────────────┘  ║  ║                   │                        ║
  ║                   │                        ║  ║                   ▼                        ║
  ║                   ▼                        ║  ║  ┌──────────────────────────────────────┐  ║
  ║  ┌──────────────────────────────────────┐  ║  ║  │ 3. 执行 PreCompact Hooks             │  ║
  ║  │ 3. 获取当前 Session Memory            │  ║  ║  │    sW6() executePreCompactHooks     │  ║
  ║  │    CZ6() getCurrentSessionMemory     │  ║  ║  │    - trigger: "auto" | "manual"     │  ║
  ║  └──────────────────────────────────────┘  ║  ║  │    - customInstructions              │  ║
  ║                   │                        ║  ║  └──────────────────────────────────────┘  ║
  ║                   ▼                        ║  ║                   │                        ║
  ║  ┌──────────────────────────────────────┐  ║  ║                   ▼                        ║
  ║  │ 4. 检查是否为空模板                   │  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  │    Ss4() isEmptyTemplate             │  ║  ║  │ 4. 构建摘要 Prompt                   │  ║
  ║  └──────────────────────────────────────┘  ║  ║  │    UOA() buildFullSummaryPrompt      │  ║
  ║                   │                        ║  ║  │    或 YR7() buildIncrementalPrompt   │  ║
  ║                   ▼                        ║  ║  └──────────────────────────────────────┘  ║
  ║  ┌──────────────────────────────────────┐  ║  ║                   │                        ║
  ║  │ 5. 找到上次摘要的位置                 │  ║  ║                   ▼                        ║
  ║  │    Ps4() getLastSummarizedMessageId  │  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  └──────────────────────────────────────┘  ║  ║  │ 5. 调用 LLM 生成摘要                 │  ║
  ║                   │                        ║  ║  │    $s4() callSummarizationLLM        │  ║
  ║                   ▼                        ║  ║  └──────────────────────────────────────┘  ║
  ║  ┌──────────────────────────────────────┐  ║  ║                   │                        ║
  ║  │ 6. 切片需要压缩的消息                 │  ║  ║                   ▼                        ║
  ║  │    recentMessages = messages.slice() │  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  │    过滤掉 meta 消息                   │  ║  ║  │ 6. 提取摘要文本                      │  ║
  ║  └──────────────────────────────────────┘  ║  ║  │    o51() extractSummaryText          │  ║
  ║                   │                        ║  ║  │    B99() processSummaryTags          │  ║
  ║                   ▼                        ║  ║  └──────────────────────────────────────┘  ║
  ║  ┌──────────────────────────────────────┐  ║  ║                   │                        ║
  ║  │ 7. 构建压缩结果                       │  ║  ║                   ▼                        ║
  ║  │    vFY() buildSessionMemoryCompact   │  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  └──────────────────────────────────────┘  ║  ║  │ 7. 收集附件                          │  ║
  ║                   │                        ║  ║  │    Os4() collectReadStateAttachments │  ║
  ║                   ▼                        ║  ║  │    Xs4() collectOtherAttachments     │  ║
  ║  ┌──────────────────────────────────────┐  ║  ║  │    _s4() getSessionMemoryAttachment  │  ║
  ║  │ 8. 检查压缩后是否超阈值               │  ║  ║  │    RZ6() getMemoryAttachment         │  ║
  ║  │    postCompactTokens >= threshold?   │  ║  ║  └──────────────────────────────────────┘  ║
  ║  │    如果超过，返回 null，回退到 MW1   │  ║  ║                   │                        ║
  ║  └──────────────────────────────────────┘  ║  ║                   ▼                        ║
  ║                   │                        ║  ║  ┌──────────────────────────────────────┐  ║
  ║                   ▼                        ║  ║  │ 8. 创建边界标记                      │  ║
  ║  ┌──────────────────────────────────────┐  ║  ║  │    kU1() createBoundaryMarker        │  ║
  ║  │ 返回压缩结果                          │  ║  ║  └──────────────────────────────────────┘  ║
  ║  │ {                                    │  ║  ║                   │                        ║
  ║  │   boundaryMarker,                    │  ║  ║                   ▼                        ║
  ║  │   summaryMessages,                   │  ║  ║  ┌──────────────────────────────────────┐  ║
  ║  │   messagesToKeep,                    │  ║  ║  │ 9. 构建上下文恢复消息                │  ║
  ║  │   attachments,                       │  ║  ║  │    ox1() buildCompactedContextMessage│  ║
  ║  │   postCompactTokenCount              │  ║  ║  └──────────────────────────────────────┘  ║
  ║  │ }                                    │  ║  ║                   │                        ║
  ║  └──────────────────────────────────────┘  ║  ║                   ▼                        ║
  ║                                            ║  ║  ┌──────────────────────────────────────┐  ║
  ║  特点: 快速、低成本、适合增量压缩          ║  ║  │ 10. 返回压缩结果                     │  ║
  ║                                            ║  ║  │ {                                    │  ║
  ╚════════════════════════════════════════════╝  ║  │   boundaryMarker,                    │  ║
                                                  ║  │   summaryMessages,                   │  ║
                                                  ║  │   attachments,                       │  ║
                                                  ║  │   hookResults,                       │  ║
                                                  ║  │   preCompactTokenCount,              │  ║
                                                  ║  │   postCompactTokenCount,             │  ║
                                                  ║  │   compactionUsage                    │  ║
                                                  ║  │ }                                    │  ║
                                                  ║  └──────────────────────────────────────┘  ║
                                                  ║                                            ║
                                                  ║  特点: 完整摘要、保留关键信息、适合首次压缩 ║
                                                  ║                                            ║
                                                  ╚════════════════════════════════════════════╝
                                                                      │
                                                                      ▼
  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              压缩结果处理模块 (Result Processing)                          ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   $t(compactionResult) - flattenCompactResult                                             ║
  ║   将压缩结果转换为消息数组                                                                 ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │                                                                                 │     ║
  ║   │   new_messages = [                                                              │     ║
  ║   │       boundaryMarker,           // 压缩边界标记 (带 UUID，用于追踪)             │     ║
  ║   │       ...summaryMessages,       // 摘要消息 (包含结构化摘要)                    │     ║
  ║   │       ...messagesToKeep,        // 保留的最近消息 (最近几轮对话)                │     ║
  ║   │       ...attachments,           // 附件 (Session Memory, 文件状态等)           │     ║
  ║   │       ...hookResults            // Hook 执行结果                               │     ║
  ║   │   ]                                                                             │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   ox1(summaryText, includeTranscript, transcriptPath, hasRetainedContext)                 ║
  ║   buildCompactedContextMessage - 构建压缩后的上下文消息                                    ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │                                                                                 │     ║
  ║   │   输出格式:                                                                      │     ║
  ║   │   ────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │   This session is being continued from a previous conversation that ran out    │     ║
  ║   │   of context. The summary below covers the earlier portion of the conversation.│     ║
  ║   │                                                                                 │     ║
  ║   │   [处理后的摘要内容]                                                             │     ║
  ║   │                                                                                 │     ║
  ║   │   If you need specific details from before compaction (like exact code         │     ║
  ║   │   snippets, error messages, or content you generated), read the full           │     ║
  ║   │   transcript at: /path/to/transcript.json                                       │     ║
  ║   │                                                                                 │     ║
  ║   │   Recent messages are preserved verbatim.                                       │     ║
  ║   │   ────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              摘要 Prompt 模板 (Summary Prompts)                            ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   UOA(customInstructions) - buildFullSummaryPrompt (首次压缩)                              ║
  ║   YR7(customInstructions) - buildIncrementalSummaryPrompt (增量压缩)                       ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  必须输出的 9 个 Section:                                                        │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  1. Primary Request and Intent     - 用户的主要请求和意图                        │     ║
  ║   │  2. Key Technical Concepts         - 关键技术概念、框架、技术栈                  │     ║
  ║   │  3. Files and Code Sections        - 涉及的文件和代码片段                        │     ║
  ║   │  4. Errors and Fixes               - 遇到的错误和修复方法                        │     ║
  ║   │  5. Problem Solving                - 问题解决过程和正在进行的调试                │     ║
  ║   │  6. All User Messages              - 所有用户消息 (非工具结果)                   │     ║
  ║   │  7. Pending Tasks                  - 待完成的任务                               │     ║
  ║   │  8. Current Work                   - 当前正在进行的工作                         │     ║
  ║   │  9. Optional Next Step             - 下一步建议 (含直接引用)                     │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  输出格式要求:                                                                   │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  <analysis>                                                                     │     ║
  ║   │  [分析思考过程]                                                                  │     ║
  ║   │  </analysis>                                                                    │     ║
  ║   │                                                                                 │     ║
  ║   │  <summary>                                                                      │     ║
  ║   │  1. Primary Request and Intent:                                                 │     ║
  ║   │     [详细描述]                                                                   │     ║
  ║   │                                                                                 │     ║
  ║   │  2. Key Technical Concepts:                                                     │     ║
  ║   │     - [概念 1]                                                                  │     ║
  ║   │     - [概念 2]                                                                  │     ║
  ║   │     ...                                                                         │     ║
  ║   │  </summary>                                                                     │     ║
  ║   │                                                                                 │     ║
  ║   │  重要: 不使用任何工具，只输出 <summary> 块                                       │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              Token 使用分析模块 (Token Analysis)                           ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   ea4(messages) - analyzeTokenUsage                                                       ║
  ║   分析消息列表中各类型的 token 消耗                                                        ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  返回结构:                                                                       │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  {                                                                              │     ║
  ║   │      total: number,                    // 总 token 数                          │     ║
  ║   │      humanMessages: number,            // 用户消息 token                        │     ║
  ║   │      assistantMessages: number,        // 助手消息 token                        │     ║
  ║   │      localCommandOutputs: number,      // 本地命令输出 token                    │     ║
  ║   │      other: number,                    // 其他类型 token                        │     ║
  ║   │                                                                                 │     ║
  ║   │      toolRequests: Map<name, tokens>,  // 按工具名分类的请求 token               │     ║
  ║   │      toolResults: Map<name, tokens>,   // 按工具名分类的结果 token               │     ║
  ║   │      attachments: Map<type, count>,    // 按类型分类的附件数量                   │     ║
  ║   │                                                                                 │     ║
  ║   │      duplicateFileReads: Map<path, {   // 重复文件读取检测                       │     ║
  ║   │          count: number,                // 读取次数                              │     ║
  ║   │          tokens: number                // 浪费的 token 数                       │     ║
  ║   │      }>                                                                         │     ║
  ║   │  }                                                                              │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   As4(breakdown) - formatBreakdownStats                                                   ║
  ║   将分析结果转换为可记录的统计对象，用于日志和遥测                                          ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              Hook 系统 (Hooks)                                             ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   sW6(config, signal, timeoutMs) - executePreCompactHooks                                 ║
  ║   在压缩前执行用户定义的 hooks                                                             ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  Hook 输入 (hookInput):                                                         │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  {                                                                              │     ║
  ║   │      hook_event_name: "PreCompact",                                             │     ║
  ║   │      trigger: "manual" | "auto",       // 触发方式                              │     ║
  ║   │      custom_instructions: string|null, // 自定义指令                            │     ║
  ║   │      session_id: string,               // 会话 ID                               │     ║
  ║   │      transcript_path: string,          // Transcript 文件路径                   │     ║
  ║   │      cwd: string,                      // 当前工作目录                          │     ║
  ║   │      permission_mode: string           // 权限模式                              │     ║
  ║   │  }                                                                              │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  Hook 类型:                                                                      │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  - command: 执行 shell 命令                                                     │     ║
  ║   │  - prompt: 执行 prompt hook                                                     │     ║
  ║   │  - agent: 执行 agent hook                                                       │     ║
  ║   │  - callback: 执行 callback 函数                                                 │     ║
  ║   │  - function: 执行函数 hook                                                      │     ║
  ║   │                                                                                 │     ║
  ║   │  返回值可以包含 newCustomInstructions，会合并到摘要 prompt 中                    │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              Transcript 管理 (Transcript)                                  ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   AO(sessionId) - getTranscriptPath                                                       ║
  ║   获取 transcript 文件路径                                                                 ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  路径结构:                                                                       │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  ~/.claude/                                                                     │     ║
  ║   │  └── projects/                                                                  │     ║
  ║   │      └── <project-hash>/              # H8() - 基于 cwd 的 SHA256 前 16 位      │     ║
  ║   │          ├── transcripts/                                                       │     ║
  ║   │          │   └── <session-id>.json    # 完整对话记录                            │     ║
  ║   │          ├── session-memory/                                                    │     ║
  ║   │          │   └── notes.md             # Session Memory 笔记                     │     ║
  ║   │          └── CLAUDE.md                # 项目级配置                              │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   p6() - getSessionId                                                                     ║
  ║   获取当前会话 ID                                                                          ║
  ║                                                                                           ║
  ║   H8() - getProjectDir                                                                    ║
  ║   获取项目目录 (基于 cwd 的 hash)                                                          ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
  ║                              环境变量配置 (Environment Variables)                          ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  压缩控制:                                                                       │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  DISABLE_COMPACT                     # 完全禁用压缩功能                         │     ║
  ║   │  DISABLE_AUTO_COMPACT               # 禁用自动压缩 (保留手动 /compact)          │     ║
  ║   │  CLAUDE_AUTOCOMPACT_PCT_OVERRIDE    # 覆盖自动压缩阈值 (0-100 百分比)           │     ║
  ║   │  CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE # 覆盖阻塞限制                             │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ║   ┌─────────────────────────────────────────────────────────────────────────────────┐     ║
  ║   │  使用示例:                                                                       │     ║
  ║   │  ─────────────────────────────────────────────────────────────────────────────  │     ║
  ║   │                                                                                 │     ║
  ║   │  # 在 80% 上下文使用时触发自动压缩                                               │     ║
  ║   │  export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80                                      │     ║
  ║   │                                                                                 │     ║
  ║   │  # 禁用自动压缩，只保留手动 /compact 命令                                        │     ║
  ║   │  export DISABLE_AUTO_COMPACT=1                                                  │     ║
  ║   │                                                                                 │     ║
  ║   └─────────────────────────────────────────────────────────────────────────────────┘     ║
  ║                                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝


  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    核心类/函数速查表                                         │
  ├─────────────────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                             │
  │  【Token 计数】                                                                              │
  │  ├── w2(text)                    简单估算: chars / 4                                        │
  │  ├── F1(obj)                     JSON.stringify                                             │
  │  ├── vv(messages)                消息列表总 token                                           │
  │  ├── SU1(messages)               压缩后 token 数                                            │
  │  ├── lx1(messages, tools)        API 精确计数                                               │
  │  └── pU1(messages, tools)        带回退的计数                                               │
  │                                                                                             │
  │  【阈值计算】                                                                                │
  │  ├── a51(model)                  有效上下文窗口                                             │
  │  ├── lg1(model)                  自动压缩阈值                                               │
  │  ├── qc(tokens, model)           上下文状态检查                                             │
  │  └── um()                        是否启用自动压缩                                           │
  │                                                                                             │
  │  【触发判断】                                                                                │
  │  ├── RFY(messages, model, src)   是否应该触发                                               │
  │  └── Qs4(...)                    执行自动压缩                                               │
  │                                                                                             │
  │  【压缩执行】                                                                                │
  │  ├── MW1(...)                    完整 LLM 摘要压缩                                          │
  │  ├── mZ6(...)                    Session Memory 轻量压缩                                    │
  │  ├── $t(result)                  结果扁平化                                                 │
  │  └── ox1(...)                    构建上下文消息                                             │
  │                                                                                             │
  │  【Prompt 构建】                                                                             │
  │  ├── UOA(instructions)           完整摘要 Prompt                                            │
  │  ├── YR7(instructions)           增量摘要 Prompt                                            │
  │  └── B99(text)                   处理 <summary> 标签                                        │
  │                                                                                             │
  │  【Token 分析】                                                                              │
  │  ├── ea4(messages)               分析 token 使用                                            │
  │  └── As4(breakdown)              格式化统计                                                 │
  │                                                                                             │
  │  【Hook 系统】                                                                               │
  │  └── sW6(config, signal)         PreCompact Hooks                                           │
  │                                                                                             │
  │  【路径管理】                                                                                │
  │  ├── AO(sessionId)               Transcript 路径                                            │
  │  ├── p6()                        当前 Session ID                                            │
  │  └── H8()                        项目目录                                                   │
  │                                                                                             │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘

  这个流程图包含了：

  1. 主循环：Agent Loop 中的上下文检查入口
  2. 阈值计算：所有常量和计算公式
  3. Token 计数：简单估算和 API 精确计数
  4. 自动压缩触发：Qs4 主函数流程
  5. 两种压缩路径：Session Memory (轻量) 和 LLM 摘要 (完整)
  6. 结果处理：消息重建和上下文恢复
  7. Prompt 模板：9 个必须输出的 section
  8. Token 分析：使用情况统计
  9. Hook 系统：PreCompact 钩子
  10. 环境变量：所有可配置项

