# BM25 baseline 配置

These settings describe the BM25 baseline. The [reproduction guide](reproduction.md) covers the optional `RETRIEVAL_MODE` settings and their local assets.

`config.py` 定义默认值和校验，`.env.example` 提供相同的发布配置。运行时环境变量优先于仓库根目录 `.env`，最后才使用默认值。密钥只在本机设置。

## 模型与规划

| 变量 | 发布值 | 含义 |
| --- | --- | --- |
| `LLM_API_KEY` | `${DEEPSEEK_API_KEY}` | 引用本机环境变量中的密钥 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | 模型请求地址 |
| `LLM_MODEL` | `deepseek-v4-flash` | 课程基线模型 ID |
| `LLM_TEMPERATURE` | `0` | 请求温度 |
| `LLM_MAX_TOKENS` | `0` | 不向 API 指定该输出上限 |
| `LLM_ENABLE_THINKING` | `true` | 主流程启用思考 |
| `LLM_REASONING_EFFORT` | `high` | 思考强度 |
| `LLM_THINKING_ROUNDS` | `10` | 主流程全部十轮保持该配置 |
| `LLM_ENABLE_QUERY_PLANNER` | `true` | 开始搜索前规划互补查询 |
| `LLM_PLANNER_MAX_TOKENS` | `0` | 规划器不显式限制输出 Token |
| `LLM_TIMEOUT` | `1800` | 单次请求超时，单位秒 |
| `LLM_MAX_RETRIES` | `2` | Agent 显式处理暂时性模型错误的重试次数 |

规划器及强制工具选择路径使用其所需的请求设置；`high` 覆盖的是常规 Agent 思考轮次。温度为 0 仍不能保证模型输出、用量和成绩完全一致。

## 检索与文档工具

| 变量 | 发布值 | 含义 |
| --- | --- | --- |
| `CORPUS_PATH` | `./data/corpus` | Hugging Face `save_to_disk` 格式语料目录 |
| `BM25_DB_PATH` | `./indexes/bm25_full.sqlite` | SQLite FTS5 与文档定位表 |
| `RETRIEVAL_METHOD` | `bm25` | 本发布版唯一允许的检索值 |
| `RETRIEVAL_TOP_K` | `8` | 一次搜索默认返回数 |
| `RETRIEVAL_CANDIDATE_K` | `50` | 每个查询变体候选池深度 |
| `RETRIEVAL_RRF_K` | `60` | RRF 排名融合常数 |
| `MAX_READ_CHARS` | `4000` | 文档分页读取长度上限 |
| `MAX_MATCHES` | `10` | 单次文档内定位匹配数上限 |
| `MAX_CONTEXT_CHARS` | `500` | 单个关键词匹配的周边上下文上限 |

相对路径以仓库根目录为基准。已有索引可使用绝对路径，避免复制数 GB 文件；构建器与运行时使用同一组路径配置。修改语料后需要构建与之对应的新索引。

## 搜索与预算

| 变量 | 发布值 | 含义 |
| --- | --- | --- |
| `MAX_AGENT_STEPS` | `10` | Agent 模型轮次上限 |
| `MAX_TOOL_CALLS` | `15` | 工具请求预算 |
| `MAX_TOTAL_TOKENS_PER_RUN` | `95000` | 单题请求前检查的累计 Token 软阈值 |
| `MIN_REMAINING_TOKENS_FOR_LLM_CALL` | `10000` | 继续发送模型请求所需的剩余 Token 余量 |
| `MAX_STALLED_SEARCHES` | `3` | 连续无有效进展的轮数阈值 |
| `MAX_REPEATED_CANDIDATES` | `2` | 重复无效候选的停止阈值 |
| `SKILLS_PATH` | `./skills` | 按需加载的技能目录 |
| `LOG_LEVEL` | `INFO` | 日志配置 |

95,000 Token 与历史基线的单题配置一致。它是软阈值，已经发出的请求仍可能使实际用量超出。设置为 `0` 可关闭单题 Token 检查，但会改变历史评测条件。

批次的 `--max-total-tokens` 与单题配置不同：它只在调度下一题前检查已完成题目的总用量。在途题目可能继续增加 Token。完整 100 题出包命令显式使用 `--max-total-tokens 0`，防止批次提前停止；单题 95,000 的检查仍生效。

检索只保留普通 BM25 命中摘要，不添加标题开头、不启用额外的线索恢复检查或两轮停滞收尾，也不放宽答案证据要求。常规技能恢复、上下文筛选与证据充分时的停止属于 baseline 本身。
