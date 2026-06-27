# FNLP Lab 4 实验流水线

本项目用于完成 FNLP Lab 4：使用 Qwen3-8B 解决大学物理和化学填空题，并生成可复现的 Kaggle 提交文件。

## 文件说明

- `scripts/run_pipeline.py`：运行 baseline 或教材 RAG 推理流程。
- `scripts/validate_submission.py`：检查提交文件格式是否符合要求。
- `src/lab4/extract.py`：从模型输出中抽取最终答案字符串。
- `src/lab4/retrieval.py`：基于允许使用的 `.tex` 教材建立 BM25 风格检索索引。
- `outputs/submission.csv`：生成的 Kaggle 提交文件。
- `outputs/traces.jsonl`：保存原始推理轨迹、检索片段和抽取出的答案。
- `REPORT_TEMPLATE.md`：实验报告模板。

## API 设置

作业要求 Kaggle 提交结果必须由 Qwen3-8B 生成。运行真实实验前，请先设置 API key：

```powershell
$env:DASHSCOPE_API_KEY="sk-ws-H.RPYRERE.m79C.MEQCIH6iUjF69ExfBeCmhmrQO0SlEw3p4UBbgHUlWvwrRSlOAiBpQzEl6WAN-zomno3ZQrHvx0ePfaa9EUMO5eYUg-fqWA"
$env:LLM_MODEL="qwen3-8b"
```

默认接口地址为：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

如果你的百炼控制台提供了不同的 OpenAI-compatible 接口地址，请额外设置：

```powershell
$env:LLM_BASE_URL="https://your-endpoint/v1"
```

本项目默认使用非流式调用，并设置 `enable_thinking=false`。这是 Qwen3 非流式接口的要求。

如需对照 Qwen3 thinking 模式，可以开启流式调用：

```powershell
python scripts/run_pipeline.py --method baseline --enable-thinking --temperature 0 --max-tokens 4096 --limit 5 --submission-out outputs/thinking_test_submission.csv --trace-out outputs/thinking_test_traces.jsonl
```

如果你使用环境变量开启过 thinking 模式，可以这样关闭：

```powershell
Remove-Item Env:\LLM_ENABLE_THINKING
```

## 运行 No-Thinking Baseline

合理 baseline：不开 thinking mode，直接 chain-of-thought，但完整使用题目文本、学科、子领域、原理提示和目标单位要求。运行：

```powershell
python scripts/run_pipeline.py --method baseline --disable-thinking --prompt-style baseline --samples 1 --temperature 0 --max-tokens 4096 --submission-out outputs/baseline_no_thinking_submission.csv --trace-out outputs/baseline_no_thinking_traces.jsonl
```

这里的 `--disable-thinking` 会强制关闭 Qwen3 thinking，即使环境变量里残留了 `LLM_ENABLE_THINKING=1` 也不会被误开。

注意：正常 baseline 不启用提交后单位自动修正。`--normalize-units` 只作为调试/消融工具，不建议用于报告中的 baseline。

如果想加速，可以并发运行。建议先从 `--workers 4` 开始，太高可能触发 API 限流或远端断连：

```powershell
python scripts/run_pipeline.py --method baseline --disable-thinking --prompt-style baseline --workers 4 --samples 1 --temperature 0 --max-tokens 4096 --submission-out outputs/baseline_no_thinking_submission.csv --trace-out outputs/baseline_no_thinking_traces.jsonl
```

如果中途断开，可以继续同一个 trace：

```powershell
python scripts/run_pipeline.py --method baseline --disable-thinking --prompt-style baseline --workers 4 --samples 1 --temperature 0 --max-tokens 4096 --resume-from-trace --submission-out outputs/baseline_no_thinking_submission.csv --trace-out outputs/baseline_no_thinking_traces.jsonl
```

## 运行教材 RAG

这是后续改进方法。RAG 版本会从作业允许使用的 `.tex` 教材中检索相关片段，再交给 Qwen3-8B 解题：

```powershell
python scripts/run_pipeline.py --method rag --top-k 4 --samples 1 --temperature 0 --max-tokens 4096 --submission-out outputs/rag_submission.csv --trace-out outputs/rag_traces.jsonl
```

第一次运行 RAG 时会建立教材索引，索引文件保存在 `cache/textbook_index.json`。

## 运行两阶段自检

`verify` 是一个改进方法，不属于必需 baseline。它先让 Qwen3-8B 正常解题，再让同一个模型审查公式、单位、数量级和符号，并输出修正后的最终答案。该方法不开 thinking mode，但每题通常需要两次 API 调用：

```powershell
python scripts/run_pipeline.py --method verify --disable-thinking --workers 3 --temperature 0 --max-tokens 4096 --submission-out outputs/verify_submission.csv --trace-out outputs/verify_traces.jsonl
python scripts/validate_submission.py --submission outputs/verify_submission.csv
```

如果想把教材检索和自检合在一起，可以运行：

```powershell
python scripts/run_pipeline.py --method rag-verify --disable-thinking --workers 3 --top-k 4 --temperature 0 --max-tokens 4096 --submission-out outputs/rag_verify_submission.csv --trace-out outputs/rag_verify_traces.jsonl
```

trace 中会同时保存 `draft_response`/`draft_answer` 和最终审查后的 `response`/`answer`，便于在报告中分析自检修改了哪些题。

## 运行 Self-Consistency

使用多次采样和多数投票得到最终答案：

```powershell
python scripts/run_pipeline.py --method rag --top-k 4 --samples 5 --temperature 0.7 --submission-out outputs/rag_sc_submission.csv --trace-out outputs/rag_sc_traces.jsonl
```

该方法调用 API 次数更多，成本也更高。建议先用 `--limit` 小规模试跑。

## 无 API 本地冒烟测试

如果还没有 API key，可以使用 mock 模式检查代码流程、输出文件和提交格式：

```powershell
$env:LAB4_MOCK_LLM="1"
python scripts/run_pipeline.py --method rag --limit 3 --submission-out outputs/mock_submission.csv --trace-out outputs/mock_traces.jsonl
python -m unittest discover -s tests
python scripts/validate_submission.py --submission outputs/mock_submission.csv
```

真实实验前请关闭 mock 模式：

```powershell
Remove-Item Env:\LAB4_MOCK_LLM
```

## 验证提交文件

生成 Kaggle 提交文件后，先运行格式检查：

```powershell
python scripts/validate_submission.py --submission outputs/rag_submission.csv
```

检查内容包括：

- 是否包含 230 行预测；
- 列名是否为 `id,answer`；
- `id` 是否与原数据顺序一致；
- 是否存在空答案；
- 答案中是否疑似残留单位；
- 答案中是否出现 `无法确定`、解释性文本或变量赋值。

如果只修改了答案抽取逻辑，可以不重新调用 API，直接从已有 trace 重建提交文件：

```powershell
python scripts/rebuild_submission_from_traces.py --traces outputs/baseline_traces.jsonl --submission-out outputs/baseline_submission_reextract.csv
python scripts/validate_submission.py --submission outputs/baseline_submission_reextract.csv
```

如果只有少数题答案格式明显异常，可以只重跑这些题，并用已有提交文件填充其他题：

```powershell
python scripts/run_pipeline.py --method baseline --ids 13,161,192 --fill-from-submission outputs/baseline_submission_reextract.csv --submission-out outputs/baseline_submission_fixed.csv --trace-out outputs/baseline_fixed_traces.jsonl
python scripts/validate_submission.py --submission outputs/baseline_submission_fixed.csv
```

## 建议实验顺序

1. 先运行 baseline，确认准确率接近作业给出的参考范围。
2. 再运行 RAG，对比是否有提升。
3. 在 RAG 基础上尝试 self-consistency。
4. 每次提交 Kaggle 后，把方法、提交文件、trace 文件和排行榜分数记录到报告中。
