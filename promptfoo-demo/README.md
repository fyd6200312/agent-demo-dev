# Promptfoo 评测示例

## 快速开始

```bash
cd promptfoo-demo

# 设置 API Key（使用你的代理）
export ANTHROPIC_AUTH_TOKEN=你的key

# 运行评测
promptfoo eval

# 查看结果（Web UI）
promptfoo view
```

## 文件说明

- `promptfooconfig.yaml` - 主配置文件，包含 prompt、测试用例、断言规则

## 断言类型

| 类型 | 说明 | 需要 LLM |
|------|------|---------|
| contains | 包含字符串 | ❌ |
| not-contains | 不包含 | ❌ |
| regex | 正则匹配 | ❌ |
| javascript | JS 表达式 | ❌ |
| llm-rubric | 语义评分 | ✅ |

## 扩展

添加更多测试用例，编辑 `promptfooconfig.yaml` 的 `tests` 部分。
