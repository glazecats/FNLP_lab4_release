from __future__ import annotations

from .data import Question
from .retrieval import RetrievedChunk


BASELINE_SYSTEM_PROMPT = """你是大学物理和大学化学解题助手。请简洁推理，最后给出可提交的最终答案。

要求：
1. 先判断题型和应使用的公式，确认每个代入量的含义、单位和题目是否匹配。
2. 使用题目给出的所有条件；必要时使用常见常数和教材标准约定。未给温度时，物理室温题可取 298 K。
3. 特别检查题目真正问的是完整物理量、缩放系数、比例、概率、角度、每天/每秒次数，还是非负大小。
4. 最终答案必须换算到题目要求单位，但 FINAL_ANSWER 中不要写单位；科学计数法必须保留 e 指数。
5. 只有目标单位本身明确含 10^n 时，才输出该缩放单位下的系数。
6. 不要回答“无法确定”“缺少数据”等文字；若需常见教材数据，请采用合理标准值继续。
7. 最后一行必须严格写成：FINAL_ANSWER: <answer>。<answer> 只能是一个数值或数学表达式，不要含解释、变量名、等号、单位或多个候选值。"""


def get_system_prompt(prompt_style: str) -> str:
    if prompt_style != "baseline":
        raise ValueError(f"Only the baseline prompt is supported here; got: {prompt_style}")
    return BASELINE_SYSTEM_PROMPT


VERIFIER_SYSTEM_PROMPT = """你是严格的答案审查器。请独立检查第一次解法，而不是简单复述。

重点检查：
1. 公式、量纲、单位换算、10 的幂和 SI 前缀是否正确。
2. 是否使用了题目所有数值条件，目标量是否读对。
3. 数值计算和数量级是否合理。
4. 答案是否应为非负大小，或是否需要保留符号。
5. FINAL_ANSWER 是否只有一个数值或数学表达式，且不含单位、解释、变量名或等号。

若第一次答案正确就保留；若发现明确错误，请重新计算并修正。不要输出“无法确定”“缺少数据”等文字；不能更可靠修正时保留第一次抽取出的答案。最后一行必须严格写成：FINAL_ANSWER: <answer>"""


def build_user_prompt(
    question: Question,
    context: list[RetrievedChunk] | None = None,
    *,
    prompt_style: str = "baseline",
) -> str:
    if prompt_style != "baseline":
        raise ValueError(f"Only the baseline prompt is supported here; got: {prompt_style}")

    parts = [
        f"题目编号：{question.id}",
        f"学科：{question.field}",
        f"子领域：{question.subfield or '未给出'}",
        f"相关定律/原理提示：{question.theorem or '未给出'}",
        f"题目：{question.question}",
    ]

    if question.unit:
        parts.extend(
            [
                f"目标答案单位：{question.unit}",
                "请把最终结果换算到这个目标单位，只输出该单位下的数值，不要写单位。只有当目标答案单位明确含有 10 的幂时，才输出对应缩放系数；普通单位必须保留完整数值和科学计数法指数。",
            ]
        )
    else:
        parts.append(
            "如果题目文字本身指定了单位，请按题目指定单位给出最终数值；若需要科学计数法，必须保留 e 指数。"
        )

    if context:
        snippets = []
        for i, chunk in enumerate(context, start=1):
            snippets.append(f"[{i}] source={chunk.source}\n{chunk.text}")
        parts.append("允许参考的教材片段：\n" + "\n\n".join(snippets))

    parts.append(
        "请逐步推理并计算。最后一行只输出 FINAL_ANSWER: <answer>，"
        "<answer> 中不要写单位、解释、变量名、等号、Markdown 加粗或多个候选值。"
    )
    return "\n".join(parts)


def build_verifier_prompt(
    question: Question,
    first_response: str,
    first_answer: str,
    context: list[RetrievedChunk] | None = None,
) -> str:
    parts = [
        f"题目编号：{question.id}",
        f"学科：{question.field}",
        f"子领域：{question.subfield or '未给出'}",
        f"相关定律/原理提示：{question.theorem or '未给出'}",
        f"题目：{question.question}",
        f"目标答案单位：{question.unit or '题目文字指定或无单位'}",
    ]
    if context:
        snippets = []
        for i, chunk in enumerate(context, start=1):
            snippets.append(f"[{i}] source={chunk.source}\n{chunk.text}")
        parts.append("允许参考的教材片段：\n" + "\n\n".join(snippets))
    parts.extend(
        [
            f"第一次抽取出的答案：{first_answer}",
            "第一次完整解题过程：",
            first_response,
            "请审查上面的解法。特别注意普通科学计数法不要丢指数；只有目标单位本身含 10^n 时才输出缩放系数。"
            "如果题目问大小/数量/比值，答案应为非负大小；如果题目问势能/自由能/像高/方向量，按物理符号给出。"
            "如果参考片段与题目不直接相关，请忽略参考片段。不要因为参考片段缺少信息就否定题目本身。"
            f"如果你无法给出更可靠的修正数值，请保留第一次抽取出的答案：{first_answer}。"
            "最后一行只输出 FINAL_ANSWER: <answer>，不要写单位、解释、变量名、等号、Markdown 加粗或多个候选值。",
        ]
    )
    return "\n".join(parts)
