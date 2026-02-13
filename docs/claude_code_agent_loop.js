/**
 * UR 函数 - Claude Code Agent Loop 核心
 * 原始位置: cli.js:3407
 *
 * 这是 Claude Code CLI 的核心 agent loop 实现
 * 使用 async generator 模式，通过 yield 流式输出各种事件
 */

async function* UR({
  messages: A,                    // 消息历史数组
  systemPrompt: q,                // 系统提示词
  userContext: K,                 // 用户上下文
  systemContext: Y,               // 系统上下文
  canUseTool: z,                  // 工具权限检查函数
  toolUseContext: w,              // 工具使用上下文（包含 options, abortController 等）
  fallbackModel: H,               // 降级模型（主模型失败时使用）
  querySource: $,                 // 查询来源标识
  maxOutputTokensOverride: O,     // 最大输出 token 覆盖值
  maxTurns: _,                    // 最大轮数限制（防止无限循环）
}) {
  // ========== 状态变量初始化 ==========
  let J,                          // autoCompactTracking - 自动压缩追踪
    X,                            // stopHookActive - 停止钩子激活标记
    j = 0,                        // maxOutputTokensRecoveryCount - 输出截断恢复计数
    D = 1,                        // turnCount - 当前轮数（从1开始）
    M;                            // pendingToolUseSummary - 待处理的工具使用摘要

  // ========== 主循环 ==========
  while (!0) {  // 无限循环，通过 return 退出

    // --- 轮次开始信号 ---
    if (
      (yield { type: "stream_request_start" },  // 发出开始信号
       k3("query_fn_entry"),                     // 性能追踪
       !w.agentId)                               // 如果不是子 agent
    )
      y91("query_started");  // 记录查询开始

    // --- 查询追踪设置 ---
    let P = w.queryTracking
        ? { chainId: w.queryTracking.chainId, depth: w.queryTracking.depth + 1 }
        : { chainId: _8q(), depth: 0 },  // _8q() 生成唯一 ID
      W = P.chainId;
    w = { ...w, queryTracking: P };

    // --- 消息处理 ---
    let G = [...xN(A)],  // xN: 复制/标准化消息数组
      V = J;             // 保存上一轮的压缩追踪状态

    // ========== 微压缩（Micro Compaction）==========
    k3("query_microcompact_start");
    let Z = await qF(G, void 0, w);  // qF: 执行微压缩
    if (((G = Z.messages), Z.compactionInfo?.boundaryMessage))
      yield Z.compactionInfo.boundaryMessage;  // 输出压缩边界消息
    k3("query_microcompact_end");

    // --- 合并系统提示 ---
    let N = C6q(q, Y);  // C6q: 合并 systemPrompt 和 systemContext

    // ========== 自动压缩（Auto Compaction）==========
    k3("query_autocompact_start");
    let { compactionResult: T } = await vt4(
      G,
      w,
      {
        systemPrompt: q,
        userContext: K,
        systemContext: Y,
        toolUseContext: w,
        forkContextMessages: G,
      },
      $,
    );

    // 处理压缩结果
    if ((k3("query_autocompact_end"), T)) {
      let {
        preCompactTokenCount: V1,   // 压缩前 token 数
        postCompactTokenCount: E1,  // 压缩后 token 数
        compactionUsage: a,         // 压缩消耗
      } = T;

      // 记录压缩成功事件
      if (
        (c("tengu_auto_compact_succeeded", {
          originalMessageCount: A.length,
          compactedMessageCount:
            T.summaryMessages.length +
            T.attachments.length +
            T.hookResults.length,
          preCompactTokenCount: V1,
          postCompactTokenCount: E1,
        }),
        !V?.compacted)  // 如果之前没有压缩过
      )
        V = { compacted: !0, turnId: _8q(), turnCounter: 0 };

      let q1 = xt(T);  // xt: 转换压缩结果为消息
      for (let K1 of q1) yield K1;  // 输出压缩消息
      ((G = q1), Zt4());  // 更新消息，重置某些状态
    }

    w = { ...w, messages: G };  // 更新上下文中的消息

    // ========== API 调用准备 ==========
    let k = [],  // 本轮的 assistant 消息
      y = [];    // 本轮的工具结果消息

    k3("query_setup_start");

    // --- 流式工具执行器 ---
    let S = t2("tengu_streaming_tool_execution2")  // t2: 特性开关检查
        ? new Gp1(w.options.tools, z, w)  // Gp1: 流式并行工具执行器
        : null,
      m = await w.getAppState(),          // 获取应用状态
      u = m.toolPermissionContext.mode,   // 权限模式
      U = iA1({                           // iA1: 选择使用的模型
        permissionMode: u,
        mainLoopModel: w.options.mainLoopModel,
        exceeds200kTokens: u === "plan" && ZH6(G),  // ZH6: 检查是否超过200k
      });
    k3("query_setup_end");

    let Q = void 0;  // fetchOverride

    // --- 检查是否达到 token 限制 ---
    if (!T) {  // 如果没有发生压缩
      let { isAtBlockingLimit: V1 } = Wc(Fv(G), w.options.mainLoopModel);
      if (V1) {
        yield CY({ content: Hp, error: "invalid_request" });  // Hp: 错误消息
        return;  // 达到硬限制，退出
      }
    }

    // ========== API 流式调用 ==========
    let x = !0;  // 重试标记
    k3("query_api_loop_start");

    try {
      while (x) {  // 重试循环（用于模型降级）
        x = !1;    // 默认不重试

        try {
          let V1 = !1;  // 流式降级标记
          k3("query_api_streaming_start");

          // --- 调用 Claude API ---
          for await (let E1 of cG1({  // cG1: API 调用函数
            messages: WZ1(G, K),       // WZ1: 合并消息和用户上下文
            systemPrompt: N,
            maxThinkingTokens: w.options.maxThinkingTokens,
            tools: w.options.tools,
            signal: w.abortController.signal,  // 中断信号
            options: {
              async getToolPermissionContext() {
                return (await w.getAppState()).toolPermissionContext;
              },
              model: U,
              toolChoice: void 0,
              isNonInteractiveSession: w.options.isNonInteractiveSession,
              fallbackModel: H,
              onStreamingFallback: () => {
                V1 = !0;  // 标记发生了流式降级
              },
              querySource: $,
              agents: w.options.agentDefinitions.activeAgents,
              allowedAgentTypes: w.options.agentDefinitions.allowedAgentTypes,
              hasAppendSystemPrompt: !!w.options.appendSystemPrompt,
              maxOutputTokensOverride: O,
              fetchOverride: Q,
              mcpTools: m.mcp.tools,
              queryTracking: P,
              effortValue: m.effortValue,
              agentId: w.agentId,
            },
          })) {
            // --- 处理流式降级 ---
            if (V1) {
              for (let a of k) yield { type: "tombstone", message: a };  // 标记旧消息为废弃
              ((k.length = 0), (y.length = 0));  // 清空累积的消息
              if (S) (S.discard(), (S = new Gp1(w.options.tools, z, w)));  // 重置执行器
            }

            // --- 处理 assistant 消息 ---
            if ((yield E1, E1.type === "assistant")) {
              if ((k.push(E1), S && !w.abortController.signal.aborted)) {
                // 提取工具调用
                let a = E1.message.content.filter(
                  (q1) => q1.type === "tool_use",
                );
                // 添加到流式执行器（并行执行）
                for (let q1 of a) S.addTool(q1, E1);
              }
            }

            // --- 处理已完成的工具结果（并行执行时）---
            if (S && !w.abortController.signal.aborted) {
              for (let a of S.getCompletedResults())
                if (a.message)
                  (yield a.message,
                    y.push(
                      ...VJ([a.message], w.options.tools).filter(
                        (q1) => q1.type === "user",
                      ),
                    ));
            }
          }
          k3("query_api_streaming_end");

        } catch (V1) {
          // ========== 模型降级处理 ==========
          if (V1 instanceof Pw6 && H) {  // Pw6: 特定错误类型
            ((U = H),           // 切换到降级模型
              (x = !0),         // 标记需要重试
              yield* TI8(k, "Model fallback triggered"),  // TI8: 生成错误消息
              (k.length = 0),   // 清空消息
              (y.length = 0));
            if (S) (S.discard(), (S = new Gp1(w.options.tools, z, w)));
            w.options.mainLoopModel = H;  // 更新主模型
            continue;  // 重试
          }
          throw V1;  // 其他错误继续抛出
        }
      }
    } catch (V1) {
      // ========== 顶层错误处理 ==========
      Y1(V1 instanceof Error ? V1 : Error(String(V1)));  // Y1: 错误记录
      let E1 = V1 instanceof Error ? V1.message : String(V1);

      // 特定错误类型直接返回
      if (V1 instanceof tj1 || V1 instanceof vq1) {
        yield CY({ content: V1.message });
        return;
      }

      (yield* TI8(k, E1), yield TZ1({ toolUse: !1 }));  // TZ1: 结束消息
      return;
    }

    // ========== 中断处理 ==========
    if (w.abortController.signal.aborted) {
      if (S) {
        // 收集已完成的工具结果
        for await (let V1 of S.getRemainingResults())
          if (V1.message) yield V1.message;
      } else yield* TI8(k, "Interrupted by user");

      if (w.abortController.signal.reason !== "interrupt")
        yield TZ1({ toolUse: !1 });
      return;
    }

    // ========== 提取工具调用 ==========
    let l = k.flatMap((V1) =>
      V1.message.content.filter((E1) => E1.type === "tool_use"),
    );

    // 处理待处理的工具摘要
    if (M) {
      let V1 = await M;
      if (V1) yield V1;
    }

    // ========== 无工具调用时的处理 ==========
    if (!k.length || !l.length) {

      // --- max_output_tokens 恢复 ---
      if (k[k.length - 1]?.apiError === "max_output_tokens" && j < 3) {
        // 输出被截断，提示继续（最多重试3次）
        let a = c6({
            content:
              "Your response was cut off. Continue from where you left off.",
            isMeta: !0,
          }),
          q1 = {
            messages: [...G, ...k, a],
            toolUseContext: w,
            autoCompactTracking: V,
            maxOutputTokensRecoveryCount: j + 1,
            turnCount: D,
          };
        // 更新状态，继续下一轮
        ((A = q1.messages),
          (w = q1.toolUseContext),
          (J = q1.autoCompactTracking),
          (j = q1.maxOutputTokensRecoveryCount),
          (D = q1.turnCount));
        continue;
      }

      // --- 停止钩子处理 ---
      let E1 = yield* w8q(G, k, q, K, Y, w, $, X);  // w8q: 处理停止条件
      if (E1.preventContinuation) return;

      if (E1.blockingErrors.length > 0) {
        let a = {
          messages: [...G, ...k, ...E1.blockingErrors],
          toolUseContext: w,
          autoCompactTracking: V,
          maxOutputTokensRecoveryCount: 0,
          stopHookActive: !0,
          turnCount: D,
        };
        ((A = a.messages),
          (w = a.toolUseContext),
          (J = a.autoCompactTracking),
          (j = a.maxOutputTokensRecoveryCount),
          (X = a.stopHookActive),
          (D = a.turnCount));
        continue;  // 有阻塞错误，继续处理
      }
      return;  // 正常结束
    }

    // ========== 工具执行 ==========
    let r = !1,   // hookStoppedContinuation 标记
      t = w;      // 更新后的上下文

    if ((k3("query_tool_execution_start"), S)) {
      // --- 并行工具执行（流式执行器模式）---
      for await (let V1 of S.getRemainingResults()) {
        let E1 = V1.message;
        if (!E1) continue;

        if (
          (yield E1,
          E1 &&
            E1.type === "attachment" &&
            E1.attachment.type === "hook_stopped_continuation")
        )
          r = !0;  // 钩子停止了继续执行

        y.push(...VJ([E1], w.options.tools).filter((a) => a.type === "user"));
      }
      t = { ...S.getUpdatedContext(), queryTracking: P };

    } else {
      // --- 串行工具执行（回退模式）---
      for await (let V1 of Ff6(l, k, z, w)) {  // Ff6: 串行执行工具
        if (V1.message) {
          if (
            (yield V1.message,
            V1.message.type === "attachment" &&
              V1.message.attachment.type === "hook_stopped_continuation")
          )
            r = !0;

          y.push(
            ...VJ([V1.message], w.options.tools).filter(
              (E1) => E1.type === "user",
            ),
          );
        }
        if (V1.newContext) t = { ...V1.newContext, queryTracking: P };
      }
    }
    k3("query_tool_execution_end");

    // ========== 工具执行后的中断检查 ==========
    if (w.abortController.signal.aborted) {
      if (w.abortController.signal.reason !== "interrupt")
        yield TZ1({ toolUse: !0 });

      // 中断时也检查 max_turns
      let V1 = D + 1;
      if (_ && V1 > _)
        yield Eq({ type: "max_turns_reached", maxTurns: _, turnCount: V1 });
      return;
    }

    // 钩子停止了继续执行
    if (r) return;

    // ========== 准备下一轮 ==========
    let s = { ...t, queryTracking: P },
      _1 = D + 1;  // 递增轮数

    // --- max_turns 检查 ---
    if (_ && _1 > _) {
      yield Eq({ type: "max_turns_reached", maxTurns: _, turnCount: _1 });
      return;  // 达到最大轮数，退出
    }

    k3("query_recursive_call");

    // --- 构建下一轮状态 ---
    let W1 = {
      messages: [...G, ...k, ...y],  // 合并所有消息
      toolUseContext: s,
      autoCompactTracking: V,
      turnCount: _1,
      maxOutputTokensRecoveryCount: 0,
      pendingToolUseSummary: void 0,
      stopHookActive: X,
    };

    // --- 更新状态变量 ---
    ((A = W1.messages),
      (w = W1.toolUseContext),
      (J = W1.autoCompactTracking),
      (D = W1.turnCount),
      (j = W1.maxOutputTokensRecoveryCount),
      (M = W1.pendingToolUseSummary),
      (X = W1.stopHookActive));

    // 继续下一轮循环
  }
}

/**
 * ========== 关键辅助函数说明 ==========
 *
 * cG1()  - Claude API 调用，返回流式消息
 * qF()   - 微压缩：处理单条消息的压缩
 * vt4()  - 自动压缩：当上下文过大时自动总结历史
 * Gp1    - 流式并行工具执行器类
 * Ff6()  - 串行工具执行函数
 * VJ()   - 将工具结果转换为消息格式
 * TI8()  - 生成错误/墓碑消息
 * TZ1()  - 生成结束消息
 * CY()   - 生成错误响应消息
 * Eq()   - 生成事件消息（如 max_turns_reached）
 * c6()   - 创建用户消息
 * w8q()  - 处理停止条件和钩子
 * xN()   - 复制/标准化消息数组
 * WZ1()  - 合并消息和用户上下文
 * C6q()  - 合并系统提示和系统上下文
 * iA1()  - 选择使用的模型
 * ZH6()  - 检查消息是否超过 200k tokens
 * Wc()   - 检查是否达到 token 限制
 * Fv()   - 计算消息 token 数
 * _8q()  - 生成唯一 ID
 * k3()   - 性能追踪标记
 * t2()   - 特性开关检查
 * Y1()   - 错误日志记录
 * c()    - 事件记录
 * y91()  - 查询状态记录
 * Zt4()  - 重置某些状态
 *
 * ========== 关键类/错误类型 ==========
 *
 * Pw6 - 触发模型降级的错误类型
 * tj1 - 特定终止错误类型
 * vq1 - 特定终止错误类型
 * Gp1 - 流式并行工具执行器
 *
 * ========== 核心流程 ==========
 *
 * 1. 初始化状态
 * 2. while(true) 主循环:
 *    a. 发出开始信号
 *    b. 微压缩 + 自动压缩（处理上下文过长）
 *    c. 调用 Claude API（流式）
 *    d. 处理模型降级（如果需要）
 *    e. 收集 assistant 消息和工具调用
 *    f. 执行工具（并行或串行）
 *    g. 检查中断、max_turns
 *    h. 构建下一轮状态
 *    i. 继续循环或退出
 *
 * ========== 关键特性 ==========
 *
 * - Async Generator: 通过 yield 流式输出各种事件
 * - max_turns: 防止无限循环
 * - 模型降级: 主模型失败时自动切换到备用模型
 * - 自动压缩: 上下文过长时自动总结历史
 * - 并行工具执行: 通过 Gp1 类实现
 * - 中断处理: 通过 AbortController 支持用户中断
 * - max_output_tokens 恢复: 输出截断时自动提示继续（最多3次）
 * - Hook 系统: 支持停止钩子干预执行
 */
