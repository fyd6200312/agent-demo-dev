# 金融 RAG 设计方案

## 1. GraphRAG vs LightRAG

### GraphRAG（微软）
- 用 LLM 从文档中抽取实体和关系，构建知识图谱
- 对图谱做社区检测（Leiden 算法），生成多层级的社区摘要
- 查询时结合社区摘要做全局性问答
- 擅长回答需要跨文档、全局视角的问题
- 缺点：索引成本高，延迟较大

### LightRAG（香港大学）
- 同样构建知识图谱，但简化了索引流程
- 采用双层检索：低层级（具体实体/关系）+ 高层级（主题/摘要）
- 去掉了昂贵的社区检测和多层摘要生成
- 支持增量更新，新文档可以直接融入已有图谱
- 索引和查询成本显著低于 GraphRAG

### 核心对比

| 维度 | GraphRAG | LightRAG |
|------|----------|----------|
| 索引成本 | 高（社区检测+摘要生成） | 低（简化流程） |
| 增量更新 | 不支持/困难 | 原生支持 |
| 全局问答 | 强（社区摘要） | 通过高层级检索近似实现 |
| 查询延迟 | 较高 | 较低 |
| 实现复杂度 | 高 | 中等 |

### 开源方案
- GraphRAG（微软官方）：`github.com/microsoft/graphrag`
- LightRAG：`github.com/HKUDS/LightRAG`
- nano-graphrag：`github.com/gusye1234/nano-graphrag`（轻量复现）
- fast-graphrag：GraphRAG 性能优化版本

### 结论
除非有非常明确的全局摘要需求，LightRAG 是更务实的选择——效果相当甚至更好，成本和复杂度低很多。

---

## 2. 文章切分方案

### 主流切分算法

1. **固定窗口 + 重叠**：chunk_size=512, overlap=128，简单但粗暴，金融场景效果差
2. **递归字符切分**（LangChain RecursiveCharacterTextSplitter）：按段落 > 句子 > 字符逐级切
3. **语义切分**（Semantic Chunking）：用 embedding 相似度检测语义断点，相邻句子相似度骤降时切分
4. **文档结构感知切分**：按 Markdown 标题、PDF 的 section/subsection 切分
5. **Agentic Chunking**：用 LLM 判断每个段落是否应该归入当前 chunk，成本高但效果好

### 金融场景推荐
- 年报/研报：结构感知切分，按章节切分，表格单独处理
- 公告/新闻：语义切分，篇幅短，语义切分够用
- 财务报表：不要切分，整表作为一个 chunk，或转成结构化数据直接查询

---

## 3. 保序极大团切分算法（当前方案）

### 算法流程
1. 将文章拆分成句子 list
2. 在局部窗口（N=5~10）内计算所有句子对的 embedding 余弦相似度
3. 相似度超过阈值 θ 的句子之间连边，构建相似度图
4. 在图上求极大团（Bron-Kerbosch 算法）
5. 保持原始文档顺序不变，极大团决定哪些连续句子应合并为一个 chunk
6. chunk 长度到达上限时，在当前句子结束处切断

### 重叠句子归属策略
- 当某个句子同时属于多个极大团时，根据 chunk 长度决定归属
- 如果前一个 chunk 长度不够，将重叠句子向前合并

### 与主流算法对比

| 维度 | 固定窗口+重叠 | 语义切分（相邻突变检测） | 保序极大团 | Agentic Chunking |
|------|-------------|---------------------|-----------|-----------------|
| 切分依据 | 字符/token 数 | 相邻句子相似度骤降 | 局部窗口内两两相似度 | LLM 判断 |
| 语义完整性 | 差，硬切 | 中等，只看相邻关系 | 较好，团内两两内聚 | 最好 |
| 计算成本 | 几乎为零 | 低（N次相似度计算） | 中等（窗口内两两计算+求团） | 高（大量LLM调用） |
| 可读性 | 好（保序） | 好（保序） | 好（保序） | 好 |
| 实现复杂度 | 极低 | 低 | 中等 | 低但贵 |

### 优势
1. 比语义切分更严格：要求团内所有句子两两相似，不只是相邻相似
2. 比纯极大团聚合务实：保留原始语序，避免乱序拼接
3. 性价比好：比固定窗口智能，比 Agentic Chunking 便宜

### 劣势与注意事项
1. 对 embedding 模型依赖重：金融专业术语如果模型没见过，相似度计算不准
2. 阈值 θ 和窗口 N 需要调参：不同类型文档最优参数可能不同
3. 无法感知文档结构：纯语义驱动，忽略标题、章节、表格等显式结构信号
4. 对短句不友好："单位：万元"等短句 embedding 信息量少，归属可能不稳定

### 建议优化
- **结构信号融合**：如果两个句子跨了章节标题，直接不连边，用文档结构做硬约束
- **切断点优化**：到达长度上限时，往回看几句，找团内相似度最低的边切分，而非硬切

---

## 4. 更前沿的切分/检索方案

### Late Chunking（Jina AI）
- 先不切分，整篇文档过长上下文 embedding 模型，每个 token 拿到全文上下文表征，然后再切分做 pooling
- 每个 chunk 的 embedding 天然包含全文上下文，不存在指代丢失问题
- 依赖长上下文 embedding 模型（jina-embeddings-v3 支持 8K）

### Contextual Retrieval（Anthropic）
- 给每个 chunk 用 LLM 生成一段上下文描述，拼在 chunk 前面再做 embedding
- 信息只增不减，不存在压缩丢失
- 实验显示检索失败率降低 49%（结合 BM25 和 Reranker 可达 67%）

### ColBERT / ColPali（多表征索引）
- 保留 token 级别的多向量表征，检索时做 late interaction
- 细粒度匹配，对"一段话包含多个指标"的场景特别有效

### Proposition-based Chunking（Dense X Retrieval，清华）
- 用 LLM 把文档拆成独立命题，每个命题是自包含的事实陈述
- 不存在指代问题，金融场景天然适合
- 缺点：LLM 调用成本高

### RAPTOR（斯坦福）
- 递归聚类生成多层摘要树，检索时可命中摘要层或原文层
- 在 QuALITY 长文档问答上比传统 RAG 提升 20% 准确率

---

## 5. 实体指代与关联关系破碎问题

### 实体指代问题
切分后"该公司"、"该指标"等代词丢失指代对象。

解决方案：
1. **上下文注入**：切分后给每个 chunk 加元数据前缀（文档名、章节、主题）
2. **指代消解预处理**：切分前用 LLM 将代词替换为具体实体名
3. **Parent-Child 索引**：小 chunk 用于检索，命中后返回父级大 chunk
4. **知识图谱辅助**：用 LightRAG/GraphRAG 抽取实体关系，查询时同时检索相关实体上下文

### 关联关系破碎问题
"A公司收购B公司"被切到两个 chunk 里。

解决方案：
1. **增大 overlap**：简单有效，但增加存储和噪声
2. **关系感知切分**：切分前先做关系抽取，确保关联实体在同一 chunk
3. **多粒度索引**：同时维护段落级和文档级索引，查询时融合
4. **Graph RAG**：把关系显式存储在图谱中，不依赖 chunk 完整性

---

## 6. RAG 内容选择策略

### 适合 RAG 的内容
- 管理层讨论与分析（MD&A）
- 研报核心观点、逻辑推导
- 风险因素描述、行业分析
- 公告中的事件描述
- 会议纪要、调研记录
- 监管政策、法规条文

### 不适合 RAG 的内容

| 内容 | 更好的方案 |
|------|-----------|
| 结构化财务数据（营收、利润、PE等） | Text-to-SQL |
| 实时行情、K线数据 | API 直接查询 |
| 需要精确计算的（同比增长率、估值） | Code Interpreter |
| 大量时间序列对比 | 专门的分析引擎 |
| 目录、页眉页脚、免责声明 | 直接过滤掉 |

### 按信息价值分

**高价值（必须入库）**
- 包含因果关系："因为…所以…"、"受…影响"
- 包含判断/预测："预计"、"展望"、"我们认为"
- 关键数据+解释：数字背后的原因
- 差异化信息：分析师独到见解

**低价值（可不入库或降权）**
- 纯模板化内容：每份年报都有的固定表述
- 重复信息：同一事件在摘要和正文中重复出现，只保留详细版本
- 纯数字罗列：没有解释的数据表

### 推荐架构：混合路由

```
查询进来
  ↓
路由判断（Agent/分类器）
  ↓
├── 定性问题 → RAG 检索文本
├── 定量问题 → SQL 查结构化数据
├── 计算问题 → SQL + Code Interpreter
└── 混合问题 → SQL 取数据 + RAG 取解释，合并给 LLM
```

---

## 7. Chunk 入库内容结构

### 方案对比

| 方案 | embedding 基于 | 送给 LLM | 优点 | 缺点 |
|------|--------------|---------|------|------|
| 基础方案 | meta + chunk 原文 | meta + chunk 原文 | 无信息损失 | 概括性查询召回一般 |
| 摘要检索 | meta + 摘要 | meta + chunk 原文 | 概括性查询召回好 | 细节查询召回下降 |
| 上下文增强 | meta + context + chunk 原文 | meta + chunk 原文 | 信息只增不减 | 需要 LLM 生成 context |
| 多路召回 | 同时索引摘要和原文 | meta + chunk 原文 | 两者互补，效果最好 | 工程复杂度翻倍 |

### 摘要做检索的实际验证

| 查询类型 | 摘要做检索 | 原文做检索 |
|---------|-----------|-----------|
| 概括性查询（"核心竞争力是什么"） | 更好 | 一般 |
| 细节查询（"2024年Q3营收多少"） | 更差 | 更好 |
| 趋势分析（"行业发展趋势"） | 更好 | 一般 |
| 精确条款（"违约金比例"） | 更差 | 更好 |

中文社区实践反馈：摘要检索在 top-K recall 上约提升 5-15%，但主要来自概括性查询，细节查询是下降的。

### 推荐方案：多路召回

```
查询进来
  ↓
同时检索两个索引
├── 索引1：meta + 摘要 embedding → 召回概括性相关的 chunk
├── 索引2：meta + 原文 embedding → 召回细节匹配的 chunk
  ↓
合并去重 + Reranker 精排
  ↓
top-K 原文送给 LLM
```

如果只能选一个，**保留原文做检索更安全**，金融场景细节查询占比高。

### 推荐存储结构

```json
{
  "meta": "腾讯2024年报 | 管理层讨论与分析",
  "context": "本段来自腾讯2024年报第三章，讨论游戏业务海外营收情况",
  "content": "原始chunk内容...",
  "embedding": "基于 meta + context + content 生成"
}
```

---

## 8. 数据分类处理方案

不同类型的数据采用不同的处理和检索策略，不能一刀切。

### 8.1 正文文本处理

#### 处理流程

```
PDF 原文
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  PDF 清洗层  (pdf_parsing_optimized.py)              │
│                                                      │
│  PDFParser / _OptimizedPDFSegmentsMixin              │
│  ├── 跳过目录页        is_directory_page()           │
│  ├── 首页特殊处理      _process_first_page()         │
│  ├── 去重复行          remove_duplicate_text()       │
│  ├── 去免责声明        remove_disclaimer()           │
│  ├── 去页脚            remove_page_footer()          │
│  ├── 去联系方式        remove_contact_info()         │
│  ├── 去表格            _remove_tables()              │
│  ├── 去纯符号行        _remove_symbol_lines()        │
│  └── 去图表标注        graph_check()                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  句子切分  (topic_segmentation.py)                    │
│                                                      │
│  cut_sentences() — spaCy 优先，正则兜底               │
│  ├── 标题预切  detect_heading_line() — 遇标题先切一刀 │
│  └── QA 模式预切  pattern_segment_array()            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  语义切分层  (topic_segmentation.py)                  │
│                                                      │
│  calculate_similarity_matrix()                       │
│  ├── 局部窗口内计算 embedding 余弦相似度              │
│  ├── 阈值过滤，构建相似度图 (NetworkX)                │
│  ├── Bron-Kerbosch 求极大团  find_cliques_recursive()│
│  ├── 重叠处理  handle_overlap()                      │
│  ├── 间隙填充 + 连续序列检测  post_process           │
│  └── 长度归一化  [min_length, max_length]            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  入库结构                                            │
│                                                      │
│  {                                                   │
│    "meta": "公司名+时间+章节" (20-40字),              │
│    "content": "chunk原文" (~500字),                   │
│    "embedding": 基于 meta+content 生成                │
│  }                                                   │
│                                                      │
│  检索方式：embedding 语义匹配                         │
└─────────────────────────────────────────────────────┘
```

#### 核心代码位置

| 功能 | 文件 | 核心方法 |
|------|------|---------|
| PDF 清洗 | `pdf_parser/pdf_parsing_optimized.py` | `PDFParser.process_text()` |
| 标题检测 | `pdf_parser/pdf_parsing_optimized.py:189` | `detect_heading_line()` |
| 表格剔除 | `pdf_parser/pdf_parsing_optimized.py:831` | `_remove_tables()` |
| 句子切分 | `topic_method/topic_segmentation.py` | `cut_sentences()` |
| 标题预切 | `topic_method/topic_segmentation.py:242` | `pattern_segment_array()` |
| 极大团切分 | `topic_method/topic_segmentation.py:77` | `find_cliques_recursive()` |
| 重叠处理 | `topic_method/helpers/handle_conflict.py:38` | `handle_overlap()` |
| 相似度计算 | `topic_method/helpers/calculate_similarity.py` | `calculate_similarity_matrix()` |

#### 关键设计决策

- 使用原始版 `topic_segmentation.py`（极大团），不用优化版（连通分量），保证内聚性
- 标题预切在极大团之前执行，天然实现了 section 约束，不会跨章节聚合
- meta 占比 4-8%（20-40字 / 500字），对 embedding 质量影响可忽略
- meta 拼接是必要的：金融文档大量使用"本公司""该指标"等代词，不拼 meta 会丢失实体信息

### 8.2 表格处理

#### 处理流程

```
PDF 页面
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  表格检测与提取  (pdf_parsing_optimized.py)           │
│                                                      │
│  _extract_tables_as_segments()                       │
│  ├── page.find_tables()  — PyMuPDF 边框检测          │
│  ├── tab.extract()       — 提取全部 cell 数据        │
│  ├── tab.to_pandas()     — 兜底提取                  │
│  └── _rows_to_markdown() — 转完整 markdown           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  文本/表格混排  (pdf_parsing_optimized.py)            │
│                                                      │
│  parse_segments_from_url()                           │
│  ├── 文本块 bbox 与表格 bbox 重叠检测                 │
│  │   _rect_overlap() — 排除表格区域内的文本           │
│  ├── 按 y 坐标排序（阅读顺序）                        │
│  └── 线性扫描：连续文本合并，遇表格切断               │
│                                                      │
│  输出：                                               │
│  ┌─────────────────────────────────────┐             │
│  │ {"type":"text",  "content":"表格前正文(含标题)"}   │
│  │ {"type":"table", "content":"完整markdown表格"}     │
│  │ {"type":"text",  "content":"表格后正文(含尾注)"}   │
│  └─────────────────────────────────────┘             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  RAG 骨架提取（入库阶段）                             │
│                                                      │
│  从已有数据中组装：                                    │
│  ├── meta        — 上层拼接（公司名+时间）            │
│  ├── 表格标题    — 前一个 text segment 末尾            │
│  ├── cell 标题   — rows[0] 单独提取（需新增）         │
│  └── 表格尾注    — 后一个 text segment 开头            │
│                                                      │
│  入库结构：                                           │
│  {                                                   │
│    "meta": "公司名+时间",                             │
│    "skeleton": "meta+标题+cell标题+尾注",             │
│    "full_table": "完整markdown表格",                  │
│    "bm25_index": skeleton,                           │
│    "embedding": 基于 skeleton 生成                    │
│  }                                                   │
│                                                      │
│  检索方式：BM25 + embedding 双路召回                  │
│  ├── BM25  → 精确匹配（数值、列名、实体名）          │
│  └── 向量  → 语义匹配（发现性查询）                  │
│  生成时：命中后取 full_table + meta 送给 LLM          │
└─────────────────────────────────────────────────────┘
```

#### 核心代码位置

| 功能 | 文件 | 核心方法 |
|------|------|---------|
| 表格检测 | `pdf_parsing_optimized.py:368` | `_extract_tables_as_segments()` |
| markdown 转换 | `pdf_parsing_optimized.py:316` | `_rows_to_markdown()` |
| 重叠检测 | `pdf_parsing_optimized.py:330` | `_rect_overlap()` |
| 混排输出 | `pdf_parsing_optimized.py:434` | `parse_segments_from_url()` |
| 页码过滤 | `pdf_parsing_optimized.py:359` | `_should_filter_text()` |

#### 关键设计决策

- 表格骨架（标题+列头+尾注）做 RAG 索引，不含具体数值——数值对 embedding 无语义贡献
- 表格用 BM25 + embedding 双路召回——BM25 擅长精确匹配（数值、列名），embedding 擅长语义发现（用户措辞与表格内容不一致时）
- TARGET 基准测试显示：dense embedding 在表格级发现任务上显著优于 BM25，但 BM25 在精确数值匹配上更强，双路互补
- 命中后返回完整原始表格给 LLM——骨架只负责召回，生成靠完整数据
- cell 标题从 `rows[0]` 提取，是当前唯一需要新增的逻辑
- 券商研报表格 80-90% 能被 PyMuPDF 正确提取，无边框/跨页/合并单元格的边界 case 比例低
- Markdown 序列化是合理的默认选择，格式选择可带来 5-15% 的准确率差异

#### 表格检索效果数据（来自公开评测）

| 方案 | Recall@5 / 准确率 | 来源 |
|------|-------------------|------|
| 朴素 RAG（单路向量） | 74% Recall@5 | 生产系统报告 |
| 摘要检索 + 结构化存储（双层） | 87% Recall@5（+13pp） | 生产系统报告 |
| 加 Reranker | 再提升 15-30% | 跨多个评测 |
| TableRAG on WTQ | 57.03% 准确率 | NeurIPS 2024 |
| TableRAG on 1000x1000 大表 | 68.4% 准确率 | 其他方法超出上下文限制 |

#### 值得关注的前沿方案

- **TableRAG（NeurIPS 2024）**：将表格检索拆为 Schema 检索（找相关列）+ Cell 检索（定位具体值），在大表场景优势明显
- **T2-RAGBench**：32,908 条金融文本+表格混合问答，验证了 hybrid BM25（dense + sparse）对混合数据的有效性
- **多向量检索（ColBERT 风格）**：不把整表压缩成单向量，而是保留 token 级表征，避免"向量稀释"问题

### 8.3 系统级检索架构

```
用户查询
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  查询路由（Agent / 分类器）                           │
│  ├── 定性问题 → RAG                                  │
│  ├── 定量问题 → SQL                                  │
│  ├── 计算问题 → Code Interpreter                     │
│  └── 混合问题 → 多路                                 │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 正文 chunk   │ │ 表格骨架     │ │ 结构化数据   │
│              │ │              │ │              │
│ ES + 向量    │ │ ES + 向量    │ │ Text-to-SQL  │
│ 双路召回     │ │ 双路召回     │ │ 精确查询     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────┐
│  合并去重 + Reranker 精排                             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  送给 LLM 生成回答                                    │
│  ├── 正文 chunk → 直接使用 meta + content             │
│  ├── 表格 → 取完整原始表格 + meta                     │
│  └── SQL 结果 → 结构化数据                            │
└─────────────────────────────────────────────────────┘
```

### 8.4 不同数据类型处理对比

| 维度 | 正文文本 | 表格 | 结构化数据 |
|------|---------|------|-----------|
| 清洗方式 | PDF 清洗流水线 | PyMuPDF find_tables | ETL 入库 |
| 切分方式 | 保序极大团 | 整表不切分 | 不切分 |
| 索引内容 | meta + chunk 原文 | meta + 骨架（标题+列头+尾注） | 表结构 + 字段 |
| 检索方式 | ES + 向量双路召回 | ES + 向量双路召回 | Text-to-SQL |
| 送给 LLM | meta + chunk 原文 | meta + 完整原始表格 | SQL 查询结果 |
| 适合的查询 | 定性分析、观点、趋势 | "XX公司营收表""分业务数据" | 精确数值、计算 |

---

## 9. 评测体系

### 检索质量评测
- Recall@K：top-K 结果中包含正确答案的比例
- MRR（Mean Reciprocal Rank）：正确答案的排名
- 用人工标注的 query-document 对做 ground truth

### 生成质量评测
- **RAGAS**（`github.com/explodinggradients/ragas`）：Faithfulness、Answer Relevancy、Context Precision/Recall
- **DeepEval**：支持幻觉检测
- **LLM-as-Judge**：用 GPT-4/Claude 打分

### 金融专项指标
- 数字准确率（财务数据是否正确）
- 时间一致性（是否混淆了不同时期的数据）
- 实体归属准确率（指标是否对应到正确的公司）

### 在线评测
- 用户满意度、点击率、采纳率
- Bad case 分析，持续迭代

---

## 9. 参考资料

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RAPTOR 论文](https://arxiv.org/abs/2401.18059)
- [Dense-X Retrieval 论文](https://arxiv.org/abs/2312.06648)
- [HyDE 论文](https://arxiv.org/abs/2212.10496)
- [LangChain MultiVector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [LlamaIndex Document Summary Index](https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/)
- [GraphRAG](https://github.com/microsoft/graphrag)
- [LightRAG](https://github.com/HKUDS/LightRAG)
- [TARGET: Table Retrieval for Generative Tasks](https://arxiv.org/abs/2406.04473)
- [TableRAG (NeurIPS 2024)](https://arxiv.org/abs/2410.04739)
- [T2-RAGBench: 金融文本+表格混合问答评测](https://arxiv.org/abs/2501.13032)
