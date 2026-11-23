# BM25 baseline 评测

## 成绩来源与范围

本发布版采用本机已完成对照中准确率优先的优化 BM25 配置。以下是 2026-11-18 课程验证集运行的历史记录，来源为本地 `bm25_100_endpoints_20261118/baseline` 的运行配置、摘要和逐题轨迹。本次清理完成了离线回归验证，没有重新调用付费模型测量准确率。

原计划运行 100 道有答案题，受批次调度额度限制，只完成并计分 **94 道**；答对 **40 道**，标准化 EM 为 **42.55%**。未完成题号为 `944`、`95`、`960`、`979`、`980`、`983`。不能把这个成绩表述为完整 100 题 EM，也不应用不同题集或单次小样本成绩替代它。

| 指标 | 历史基线实测 |
| --- | ---: |
| 已计分题目 | 94/100 |
| 标准化 EM | 40/94（42.55%） |
| 答案包含匹配率 | 47.87% |
| 具体答案 / `Unknown` | 53 / 41 |
| 发出的工具调用 | 1,224 |
| 实际执行的工具调用 | 1,195 |
| 题均发出的工具调用 | 13.02 |
| API 总 Token | 3,828,663 |
| 题均 API Token | 40,730.46 |
| 输入 / 输出 Token | 2,577,776 / 1,250,887 |
| 单题耗时中位数 / P95 | 63.81 / 407.67 秒 |
| 判为运行故障而未计分 | 0 |

历史运行使用 DeepSeek、温度 0、high 思考十轮、查询规划、Top‑8、候选池 50、RRF 常数 60、10 个 Agent 步骤、15 次工具调用和单题 95,000 Token 软阈值。按验证集原顺序、2 题并发运行。发布默认值与这些模型及单题配置一致。

原始多版本结果、逐题轨迹、预算账本与比较分析只保存在本机研究工作区，当前仓库不随附这些生成文件。

## 指标定义

- **标准化 EM**：NFKC Unicode 规范化、忽略大小写、将标点替换为空格、去除英文冠词并合并空白后，预测与参考完全相同。无答案测试集不计算正确率。
- **Contains**：规范化后的非空预测与参考中，一方包含另一方。它是辅助指标，不能代替 EM。
- **工具调用**：按轨迹中发出的 `tool_call` 计数，包括后来被拦截的请求；实际执行数另行记录。
- **Token**：使用 API 回传用量，包含查询规划器与主 Agent 请求，不使用本地字符数估算替代。
- **停止状态**：保留正常作答、证据不足、预算停止与网络故障等原因。`Unknown` 是可计分结果，不应被过滤掉来抬高 EM。

## 离线回归

```powershell
python -m pytest -q
```

测试覆盖 BM25 查询、排名融合和去重、文档分页和定位、技能路由、证据及答案校验、搜索停止、上下文构建、模型错误重试、API 用量对账、出包校验和 Web 返回字段。测试使用临时数据库与模型替身，不需要真实密钥、全量语料或模型 API 请求。

## 运行验证集

将课程发布的有答案验证题放在 `data/validation.jsonl`。它与 `prepare_corpus.py` 下载的检索语料是不同文件。数据加载器支持 `id/question/answer`，也接受 `query_id/query/answer`；无答案测试文件可以省略 `answer`。

只检查题目选择，不运行 Agent：

```powershell
python scripts/evaluate_agent.py --dataset data/validation.jsonl --limit 100 --no-shuffle --plan-only
```

运行完整的 100 题验证集：

```powershell
python scripts/evaluate_agent.py --dataset data/validation.jsonl --limit 100 --no-shuffle --workers 2 --max-total-tokens 0 --confirm-paid-run --output output/bm25-validation100
```

默认 CLI 只选择 1 题，批次调度阈值为 25,000 Token；完整运行必须显式指定以上参数。`--confirm-paid-run` 表示该命令允许发出实际模型请求。新输出目录用于新一轮运行；中断后使用原目录和相同参数，加上 `--resume`，以题号跳过已完成结果并重试运行故障。

每次运行在本地生成 `run_config.json`、`results.jsonl`、`summary.json`、`test_prediction.jsonl` 和 `official_run/`。`.gitignore` 排除整个 `output/`，不会将问题、参考答案、轨迹或运行用量上传。

## 正式测试集出包

将恰好 100 道不含答案、题号唯一的正式题保存到 `data/test.jsonl`：

```powershell
python scripts/evaluate_agent.py --dataset data/test.jsonl --course-test --limit 100 --no-shuffle --workers 2 --max-total-tokens 0 --confirm-paid-run --output output/bm25-course-test
```

`--course-test` 会在写出最终文件前检查完整题号与问题对应、100 条记录、非空答案、轨迹时间顺序、工具调用数，以及规划器和模型 API Token 对账。预算停止后的 `Unknown` 仍保留真实轨迹。批次没有跑满或有运行故障时不能作为有效全量提交。

提交文件每行恰好六个顶层字段：`id`、`question`、`answer`、`trajectory`、`tool_calls`、`total_tokens`。正式文件为 `output/bm25-course-test/test_prediction.jsonl`，必须由程序生成，不手工改写答案或轨迹。
