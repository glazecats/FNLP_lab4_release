from __future__ import annotations

import re

from .data import Question
from .retrieval import RetrievedChunk
from .units import infer_target_unit


def normalize_question_text_for_prompt(text: str) -> str:
    """Repair a small class of TeX/OCR exponent losses before sending to the LLM."""

    return re.sub(r"\b10([1-9])-(?=[A-Za-zμµ\\])", r"10^\1-", text)


BASELINE_SYSTEM_PROMPT = """你是一个专业的大学物理和大学化学解题助手。请根据题目给出的所有条件做简洁的逐步推理，并计算最终答案。

要求：
1. 必须综合使用题目文本、学科领域、子领域、相关定律/原理提示，以及题目或目标字段中的单位要求。
2. 先判断问题所属领域和应使用的公式、常数、近似或教材常见数据，再代入计算；不要忽略题干中的数值、温度、波长、电压、质量、单位等条件。
3. 套用公式时，请先仔细回忆公式中的每个量的含义，确保与题目中的物理量和意义匹配，不要只是讲题目中的数值随意的代入公式，你的过程应先指出你用的原始公式，然后指出你将要带入的每个数值以及他们对应的含义，并自己检查是否跟公式对应，你需要在过程中写明你带入的合理性；套用公式时请回忆公式对应的单位是什么，并检查题目中的单位和你要的单位是否一致，并且你需要在写过程的时候把单位也代入并检查量纲。
4. 如果题目或“目标答案单位”给出了单位，最终答案必须换算到该单位；FINAL_ANSWER 中不要写单位。
5. 科学计数法必须保留指数。例如计算值是 1.5e-36 kg m/s，目标单位是 kg m/s，则写 1.5e-36，不要只写 1.5。
6. 只有当“目标答案单位”本身明确写成 10^n、10^{-n} 或 10^n unit 时，FINAL_ANSWER 才只写该缩放单位下的系数。例如目标单位为 10^14 Hz 且计算值为 4.74e14 Hz，才写 4.74。
7. 如果题目需要常用物理/化学常数、标准约定、室温、常见教材表格值或基础近似，请使用合理标准值继续计算；不要回答“无法确定”“不存在”“缺少数据”等文字答案。
8. 若题目要求具体数值但未给温度，除非语境另有说明，物理室温题取 T≈293 K，化学标准状态/热力学题取 T≈298 K；常用 kB=8.617333e-5 eV/K。
9. 对量子化学/光谱学的常见定义题，优先使用教材标准约定：轨道角动量字母按 s,p,d,f,g,h,i,k,l,m,n,o,q,r,t... 对应 l=0,1,2,3,4,...；分子势能曲线若以分离原子为零点，则平衡位置 U(Re)=-De；1 cm^-1 = 1.23984e-4 eV。
10. 重读题目真正要求的量，如果问角度、比例、概率、系数或每天/每秒发生次数，必须输出该目标量，而不是中间物理量。
11. 若题目问“大小、数量、减少量、比值、速率比、电流比、概率、高度、距离”等，输出非负大小；只有明确要求带符号的变化量、势能、自由能或方向分量时才保留正负号。
11a. 若题目要求给出多个单位的结果，但“目标答案单位/形式”或 unit 字段只指定其中一个单位，FINAL_ANSWER 只输出该目标单位下的一个数值；若没有目标单位字段，优先输出题目最先要求的单位。
12. 如果给了教材片段，先判断它是否真的匹配题目；教材片段可能只是相近主题或例题。只借用公式、概念和通用常数，不要复制与题干不一致的例题数值、角度定义或条件。
13. 最终前自查：目标单位/形式、题目真正问的量、每个粒子/每摩尔/总量、角度参照、SI 前缀和 10 的幂、符号还是大小、数量级是否合理。
14. 若题目区分单个分子/粒子与一摩尔，总是先决定使用 k_B 还是 R；单个分子平均平动动能用 (3/2)k_B T，每摩尔理想气体平均平动动能用 (3/2)RT。
15. 核反应 Q 值按释放能量为正的约定检查，即 Q = 最终总动能 - 初始总动能（在只给动能变化时）；不要无理由取相反号。
16. 最终答案只能是一个数值，优先用普通小数或 e 科学计数法；不要包含解释、变量名、等号、单位、Markdown 加粗或多个候选值。
17. 最后一行必须严格写成：FINAL_ANSWER: <answer>"""


CONCISE_SYSTEM_PROMPT = """你是大学物理和大学化学数值题解题助手。请独立重读题目，先判断题目真正要求的目标量，再选择标准公式并计算。
核心要求：
1. 题目、学科、子领域和定律提示都只是线索；若召回片段或提示与题干不完全匹配，以题干条件和标准公式为准。
2. 写出所用公式，检查变量含义、单位、角度参照、每粒子/每摩尔/总量、比例/百分数、符号还是大小。
3. 如果题目或目标单位给出单位，最终只提交该目标单位下的一个数值；若单位含 10^n，只提交该缩放单位下的系数。
4. 普通科学计数法必须保留 e 指数；不要把 1.5e-36 写成 1.5。
5. 常用常数、室温或教材标准值可直接采用；不要回答无法确定。
6. 单个分子/粒子与一摩尔要分清：单个分子平均平动动能用 (3/2)k_B T，每摩尔用 (3/2)RT。
7. 核反应 Q 值按释放能量为正检查，若只给初末动能，Q = 最终总动能 - 初始总动能。
8. 最后一行必须严格写成：FINAL_ANSWER: <answer>。<answer> 只能是一个数值，不写单位、解释、等号或多个候选。"""


def get_system_prompt(prompt_style: str) -> str:
    if prompt_style == "concise":
        return CONCISE_SYSTEM_PROMPT
    if prompt_style != "baseline":
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    return BASELINE_SYSTEM_PROMPT


VERIFIER_SYSTEM_PROMPT = """你是一个严格的大学物理和大学化学答案审查器。你会看到题目、目标单位、第一次解题过程和第一次抽取出的答案。

你的任务不是重复原答案，而是独立检查：
1. 公式或定律是否选对，着重检查是否套用了正确的公式以及将数值带入公式的意义是否正确，检查带入数值的单位和公式中的单位的一致性，检查计算过程中的量纲是否算错；
2. 列竖式检查计算过程，检查数值计算是否正确，着重检查乘法运算（你可以通过除法来验证）；
3. 是否正确使用了题目的数值条件，着重检查数值代入的合理性以及单位一致性；
4. 单位换算、10 的幂、SI 前缀和目标单位系数是否正确；
5. 数量级是否合理，数值大小是否符合常识性结果；
6. 题目问的是完整物理量还是缩放系数/比值/概率/角度；
7. 题目问的是带符号量还是非负大小；
8. 最终答案是否只有一个数值或数学表达式，并检查数值计算是否正确。

如果第一次答案正确，就保留它；如果发现错误，请重新计算并修正。
修正时必须先独立重读题意，不要只围绕第一次答案做局部修改。
如果需要修正：请遵循以下规则：
1. 必须综合使用题目文本、学科领域、子领域、相关定律/原理提示，以及题目或目标字段中的单位要求。
2. 先判断问题所属领域和应使用的公式、常数、近似或教材常见数据，再代入计算；不要忽略题干中的数值、温度、波长、电压、质量、单位等条件。
3. 套用公式时，请先仔细回忆公式中的每个量的含义，确保与题目中的物理量和意义匹配，不要只是讲题目中的数值随意的代入公式，你的过程应先指出你用的原始公式，然后指出你将要带入的每个数值以及他们对应的含义，并自己检查是否跟公式对应，你需要在过程中写明你带入的合理性。
4. 如果题目或“目标答案单位”给出了单位，最终答案必须换算到该单位；FINAL_ANSWER 中不要写单位。
5. 科学计数法必须保留指数。例如计算值是 1.5e-36 kg m/s，目标单位是 kg m/s，则写 1.5e-36，不要只写 1.5。
6. 只有当“目标答案单位”本身明确写成 10^n、10^{-n} 或 10^n unit 时，FINAL_ANSWER 才只写该缩放单位下的系数。例如目标单位为 10^14 Hz 且计算值为 4.74e14 Hz，才写 4.74。
7. 如果题目需要常用物理/化学常数、标准约定、室温、常见教材表格值或基础近似，请使用合理标准值继续计算；不要回答“无法确定”“不存在”“缺少数据”等文字答案。
8. 若题目要求具体数值但未给温度，除非语境另有说明，物理室温题取 T≈293 K，化学标准状态/热力学题取 T≈298 K；常用 kB=8.617333e-5 eV/K。
9. 对量子化学/光谱学的常见定义题，优先使用教材标准约定：轨道角动量字母按 s,p,d,f,g,h,i,k,l,m,n,o,q,r,t... 对应 l=0,1,2,3,4,...；分子势能曲线若以分离原子为零点，则平衡位置 U(Re)=-De；1 cm^-1 = 1.23984e-4 eV。
10. 重读题目真正要求的量，如果问角度、比例、概率、系数或每天/每秒发生次数，必须输出该目标量，而不是中间物理量。
11. 若题目问“大小、数量、减少量、比值、速率比、电流比、概率、高度、距离”等，输出非负大小；只有明确要求带符号的变化量、势能、自由能或方向分量时才保留正负号。
11a. 若题目要求给出多个单位的结果，但“目标答案单位/形式”或 unit 字段只指定其中一个单位，FINAL_ANSWER 只输出该目标单位下的一个数值；若没有目标单位字段，优先输出题目最先要求的单位。
12. 如果给了教材片段，先判断它是否真的匹配题目；教材片段可能只是相近主题或例题。只借用公式、概念和通用常数，不要复制与题干不一致的例题数值、角度定义或条件。
13. 最终前自查：目标单位/形式、题目真正问的量、每个粒子/每摩尔/总量、角度参照、SI 前缀和 10 的幂、符号还是大小、数量级是否合理。
14. 若题目区分单个分子/粒子与一摩尔，总是先决定使用 k_B 还是 R；单个分子平均平动动能用 (3/2)k_B T，每摩尔理想气体平均平动动能用 (3/2)RT。
15. 核反应 Q 值按释放能量为正的约定检查，即 Q = 最终总动能 - 初始总动能（在只给动能变化时）；不要无理由取相反号。
16. 最终答案只能是一个数值，优先用普通小数或 e 科学计数法；不要包含解释、变量名、等号、单位、Markdown 加粗或多个候选值。
17. 最后一行必须严格写成：FINAL_ANSWER: <answer>

严禁输出“无法确定”“无法计算”“无解”“缺少数据”等文字答案。如果你无法得到比第一次答案更可靠的数值，就保留第一次抽取出的答案。
最后一行必须严格写成：FINAL_ANSWER: <answer>"""


ARBITER_SYSTEM_PROMPT = """你是最终审题专家。你会看到同一道题的两个候选解法：候选A使用了教材召回片段，候选B不使用教材召回片段。

请先独立重读题目并从零开始解题，候选解法只能作为交叉检查材料，不能先入为主。
工作方式：
1. 先确认题目要求的目标量和目标单位/形式。
2. 独立选择公式，检查变量含义、单位、数量级、角度参照、SI 前缀、每个粒子/每摩尔/总量、符号还是大小。
3. 再比较候选A/B：如果候选使用了不匹配的教材片段、复制例题数值、读错目标量、漏单位换算或数量级异常，应拒绝它。
4. 如果两个候选都不可靠，按你的独立推导给出答案。
5. 严禁输出无法确定、无法计算、无解、缺少数据等文字答案。
6. 最后一行必须严格写成：FINAL_ANSWER: <answer>。<answer> 只能是一个数值，优先小数或 e 科学计数法，不写单位、解释、等号、Markdown 或多个候选。"""


def build_user_prompt(
    question: Question,
    context: list[RetrievedChunk] | None = None,
    *,
    prompt_style: str = "baseline",
) -> str:
    if prompt_style not in {"baseline", "concise"}:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    parts = [
        f"题目编号：{question.id}",
        f"学科：{question.field}",
        f"子领域：{question.subfield or '未给出'}",
        f"相关定律/原理提示：{question.theorem or '未给出'}",
        f"题目：{normalize_question_text_for_prompt(question.question)}",
    ]

    target_unit = infer_target_unit(question.question, question.unit)
    if target_unit:
        parts.extend(
            [
                f"目标答案单位/形式：{target_unit}",
            "请把最终结果换算到这个目标单位，只输出该单位下的一个数值，不要写其他单位的并列答案，不要写单位。只有当目标答案单位明确含有 10 的幂时，才输出对应缩放系数；普通单位必须保留完整数值和科学计数法指数。",
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
        "如果教材片段与题目不直接匹配，请以题目条件和标准公式为准。"
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
        f"题目：{normalize_question_text_for_prompt(question.question)}",
        f"目标答案单位/形式：{infer_target_unit(question.question, question.unit) or '题目文字指定或无单位'}",
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
            "如果题目问大小/数量/比值/高度/距离，答案应为非负大小；只有题目明确问有符号变化量、势能、自由能或方向量时，才按物理符号给出。"
            "如果参考片段与题目不直接相关，请忽略参考片段。不要因为参考片段缺少信息就否定题目本身。"
            f"如果你无法给出更可靠的修正数值，请保留第一次抽取出的答案：{first_answer}。"
            "严禁输出无法确定、无法计算、无解、缺少数据等文字答案。"
            "如果要修改答案，必须说明你使用了哪个公式、哪个单位换算或哪个题意修正；不要凭感觉改变数量级。"
            "最后一行只输出 FINAL_ANSWER: <answer>，不要写单位、解释、变量名、等号、Markdown 加粗或多个候选值。",
        ]
    )
    return "\n".join(parts)


def build_arbiter_prompt(
    question: Question,
    *,
    rag_response: str,
    rag_answer: str,
    no_rag_response: str,
    no_rag_answer: str,
    context: list[RetrievedChunk] | None = None,
) -> str:
    parts = [
        f"题目编号：{question.id}",
        f"学科：{question.field}",
        f"子领域：{question.subfield or '未给出'}",
        f"相关定律/原理提示：{question.theorem or '未给出'}",
        f"题目：{normalize_question_text_for_prompt(question.question)}",
        f"目标答案单位/形式：{infer_target_unit(question.question, question.unit) or '题目文字指定或无单位'}",
    ]
    if context:
        snippets = []
        for i, chunk in enumerate(context, start=1):
            snippets.append(f"[{i}] source={chunk.source}\n{chunk.text}")
        parts.append("候选A使用过的教材片段，仅供判断是否匹配题目：\n" + "\n\n".join(snippets))
    parts.extend(
        [
            f"候选A（使用教材召回）最终答案：{rag_answer}",
            "候选A过程摘录：",
            rag_response[-2400:],
            f"候选B（不使用教材召回）最终答案：{no_rag_answer}",
            "候选B过程摘录：",
            no_rag_response[-2400:],
            "请先独立重解，再用候选A/B交叉检查，最后给出唯一可提交答案。",
        ]
    )
    return "\n".join(parts)
