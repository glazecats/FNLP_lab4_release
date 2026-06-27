from __future__ import annotations

from .data import Question
from .retrieval import RetrievedChunk


BASELINE_SYSTEM_PROMPT = """你是一个专业的大学物理和大学化学解题助手。请根据题目给出的所有条件做简洁的逐步推理，并计算最终答案。

要求：
1. 必须综合使用题目文本、学科领域、子领域、相关定律/原理提示，以及题目或目标字段中的单位要求。
2. 先判断问题所属领域和应使用的公式、常数、近似或教材常见数据，再代入计算；不要忽略题干中的数值、温度、波长、电压、质量、单位等条件。
3. 如果题目或“目标答案单位”给出了单位，最终答案必须换算到该单位；FINAL_ANSWER 中不要写单位。
4. 科学计数法必须保留指数。例如计算值是 1.5e-36 kg m/s，目标单位是 kg m/s，则写 1.5e-36，不要只写 1.5。
5. 只有当“目标答案单位”本身明确写成 10^n、10^{-n} 或 10^n unit 时，FINAL_ANSWER 才只写该缩放单位下的系数。例如目标单位为 10^14 Hz 且计算值为 4.74e14 Hz，才写 4.74。
6. 如果题目需要常用物理/化学常数、标准约定、室温、常见教材表格值或基础近似，请使用合理标准值继续计算；不要回答“无法确定”“不存在”“缺少数据”等文字答案。
7. 若题目要求具体数值但未给温度，除非语境另有说明，按教材常规取室温 T≈298 K 或 300 K；常用 kB=8.617333e-5 eV/K，因此室温下 kBT≈0.0257 eV。
8. 对量子化学/光谱学的常见定义题，优先使用教材标准约定：轨道角动量字母按 s,p,d,f,g,h,i,k,l,m,n,o,q,r,t... 对应 l=0,1,2,3,4,...；分子势能曲线若以分离原子为零点，则平衡位置 U(Re)=-De；1 cm^-1 = 1.23984e-4 eV。
9. 若题目问“大小、数量、减少量、比值、速率比、电流比”等，输出非负大小；只有明确要求带符号的变化量、势能、自由能、像高或方向分量时才保留正负号。
10. 最终答案只能是一个数值或一个数学表达式，不要包含解释、变量名、等号、单位、Markdown 加粗或多个候选值。
11. 最后一行必须严格写成：FINAL_ANSWER: <answer>"""


def get_system_prompt(prompt_style: str) -> str:
    if prompt_style != "baseline":
        raise ValueError(f"Only the baseline prompt is supported here; got: {prompt_style}")
    return BASELINE_SYSTEM_PROMPT


VERIFIER_SYSTEM_PROMPT = """你是一个严格的大学物理和大学化学答案审查器。你会看到题目、目标单位、第一次解题过程和第一次抽取出的答案。

你的任务不是重复原答案，而是独立检查：
1. 公式或定律是否选对；
2. 是否使用了题目所有数值条件；
3. 单位换算、10 的幂、SI 前缀和目标单位系数是否正确；
4. 数量级是否合理；
5. 题目问的是带符号量还是非负大小；
6. 最终答案是否只有一个数值或数学表达式。

如果第一次答案正确，就保留它；如果发现错误，请重新计算并修正。
严禁输出“无法确定”“无法计算”“无解”“缺少数据”等文字答案。如果你无法得到比第一次答案更可靠的数值，就保留第一次抽取出的答案。
最后一行必须严格写成：FINAL_ANSWER: <answer>"""


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
            "严禁输出无法确定、无法计算、无解、缺少数据等文字答案。"
            "最后一行只输出 FINAL_ANSWER: <answer>，不要写单位、解释、变量名、等号、Markdown 加粗或多个候选值。",
        ]
    )
    return "\n".join(parts)
