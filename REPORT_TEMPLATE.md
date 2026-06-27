# 实验报告模板

## 1. 方法概述

说明完整实验流程，包括使用的模型、prompt 格式、答案抽取方式，以及是否使用了教材检索、self-consistency、验证器或计算工具。

需要写清楚：

- 模型固定为 Qwen3-8B；
- 输入题目的字段如何组织；
- 模型输出如何转换成最终提交答案；
- 是否使用外部知识，若使用，只能来自作业允许的教材。

## 2. Baseline

报告直接 chain-of-thought prompting 的 Qwen3-8B baseline 结果，并与作业说明中的约 50% 参考准确率进行比较。

可以包含：

- baseline prompt；
- Kaggle 分数；
- 与参考准确率的差异；
- 如果差异较大，说明调试过哪些内容，例如 prompt、答案抽取或提交格式。

## 3. 改进系统

说明在 baseline 之上加入的组件，例如：

- 从允许的 `.tex` 教材中进行检索；
- 多次采样并用 self-consistency 投票；
- 数值计算或符号计算；
- 答案验证与格式归一化；
- 针对物理和化学题目的不同处理策略。

请重点解释这些组件为什么可能提升 Qwen3-8B 的表现，而不是简单罗列模块。

## 4. 排行榜结果

记录 Kaggle 提交结果、提交时间、方法和分数。

| 方法 | 采样次数 | 是否检索 | Kaggle 分数 | 备注 |
|---|---:|---|---:|---|
| Baseline | 1 | 否 |  |  |
| RAG | 1 | 是 |  |  |
| RAG + self-consistency | 5 | 是 |  |  |

## 5. 实验分析

比较 baseline 和改进方法的表现，并进行有证据支持的分析。

建议分析：

- 相比 baseline 的提升幅度；
- 物理题和化学题的表现差异；
- 主要错误类型，例如知识回忆错误、单位换算错误、计算错误、答案抽取错误；
- 检索成功或失败的代表性案例；
- 工具或验证模块成功或失败的代表性案例。

## 6. 可复现性

说明最终提交文件如何生成。请写出对应命令，以及提交文件和 trace 文件路径。

示例：

```powershell
python scripts/run_pipeline.py --method rag --top-k 4 --samples 5 --temperature 0.7 --submission-out outputs/rag_sc_submission.csv --trace-out outputs/rag_sc_traces.jsonl
python scripts/validate_submission.py --submission outputs/rag_sc_submission.csv
```

最终提交材料应包括：

- 报告；
- 可复现代码；
- 与选定 Kaggle 提交对应的 `.csv` 预测文件；
- 运行说明或 README；
- 原始 reasoning traces。

