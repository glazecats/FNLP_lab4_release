from __future__ import annotations

from .data import Question
from .retrieval import RetrievedChunk


FINAL_PROTOCOL = """输出协议：
- 如果需要计算三位有效数字乘除、开方、指数、对数、三角函数、积分、复杂分数或多步数值运算，不要心算。只输出一行：TOOL_CALC: <python_expression>
- <python_expression> 只能使用数字、+ - * / **、括号，以及 sqrt/log/ln/log10/exp/sin/cos/tan/asin/acos/atan/sind/cosd/tand/asind/acosd/atand/sinh/cosh/tanh/atanh/radians/degrees/integral/pi/e/c/h/k_B/kB_eV/N_A/R/F/G/g 等常量或函数；也可使用 math. 或 np. 前缀；不要写单位。
- 定积分写成 integral(expr, x, lower, upper)，例如 integral(x**2, x, 0, 1)；被积变量必须是单个字母名。
- Python 的 sin/cos/tan/asin/acos/atan 使用弧度。题目角度若以 degree/度/° 给出，优先使用 sind/cosd/tand/asind/acosd/atand，或显式乘 pi/180。
- 对整个表达式开方时必须写 sqrt(<whole_expression>)，不要写 a**3**0.5 或 a**3 ** 0.5 这类连续幂。
- 你绝对不要自己输出 TOOL_RESULT；TOOL_RESULT 只能由程序返回。若你输出 TOOL_CALC，本轮不要同时输出 FINAL_ANSWER。
- 收到 TOOL_RESULT 后继续推理，可以再次请求 TOOL_CALC。
- 当你已经得到最终结果时，最后一行必须严格写成：FINAL_ANSWER: <answer>
- <answer> 只能是一个数值或 LaTeX 数学表达式，不要写单位、变量名、等号、解释、多个候选值或 Markdown。"""


GENERAL_SOLVER_RULES = """通用解题规则：
1. 先判断题目真正要求的量，再选择公式；不要把中间量当最终答案。
2. 使用题目文本、field、subfield、theorem、unit 中所有有用信息。
3. 如果题目或 unit 字段给出目标单位/缩放形式，最终答案必须换算到该形式，但 FINAL_ANSWER 里不要写单位。
4. 科学计数法不要丢指数；只有目标单位本身明确含 10^n 缩放时，才输出该缩放下的系数。
5. 代入公式前检查每个量的物理/化学含义和单位；必要时做量纲检查。
6. 常数、标准状态、室温和教材常见约定可以使用合理标准值，但要保持一致。
7. 如果题目问大小、数量、概率、比例、速率比或减少量，通常输出非负大小；只有题目明确要带符号变化量、势能、自由能或方向分量时保留符号。
8. 给出的检索知识可能不匹配题目。先判断相关性，只借用通用公式、定义和常数，不要复制不一致的例题数值。
9. 对几何量、角度、符号、比例、概率、速率和时间区间，先从题目定义出发确认参考对象和目标量，不要把相近但定义不同的教材例子直接套用。
10. 对含背景值、初始值、容器值、净量、总量、每粒子、每摩尔、每秒、每天等条件的题，先区分原始量和题目真正要求的量。
11. 密度、质量、力、摩尔量、体积量、能量、功率、频率和时间不能凭名称混用；先用量纲判断公式两边是否一致，必要时完整转换到同一单位体系。
12. 对需要积分、微分、极限、平衡或渐近值的题，先建立目标量与变量的关系，再决定是否需要积分或取极限，不要把中间变量当最终答案。
13. 对概率、比例、衰减、吸收、散射、增长或减少类问题，先明确“单次/总量/单位时间/单位长度/单位面积”的定义，再由定义推导表达式。
14. 最终前自查：目标单位/形式、每粒子还是每摩尔、每秒还是每天、角度参照、SI 前缀、10 的幂、符号、数量级。"""


DIRECT_SYSTEM = f"""你是 DIRECT_SOLVER，一个谨慎的大学物理和物理化学解题助手。
你的目标是独立解题，不使用外部检索片段。

{GENERAL_SOLVER_RULES}

{FINAL_PROTOCOL}"""


RAG_CURATOR_SYSTEM = """你是 RAG_CURATOR。你的任务不是直接给最终答案，而是从允许的教材片段中提取对当前题目可能有用的信息。

要求：
1. 先判断每个片段是否真的与题目匹配。
2. 只提取公式、定义、常数、单位换算、适用条件和容易误用的限制。
3. 如果片段像相近主题但条件不一致，明确写出“不应照搬的地方”。
4. 不要编造片段里没有的教材内容；如果片段没用，直接说相关性低。
5. 输出简洁的 RAG_NOTES 项目符号，不要给最终答案。"""


RAG_SOLVER_SYSTEM = f"""你是 RAG_SOLVER，一个使用教材笔记辅助解题的大学物理和物理化学解题助手。
你会看到题目和 RAG_CURATOR 提炼出的笔记。笔记可能不完整或不完全匹配，必须自己判断。

{GENERAL_SOLVER_RULES}

{FINAL_PROTOCOL}"""


ARBITER_SYSTEM = f"""你是 ARBITER。你会看到同一道题的 DIRECT_SOLVER 和 RAG_SOLVER 解法。
你的任务是比较两者，选择更可靠的答案；如果两者都有问题，则重新解题。

裁决规则：
1. 优先检查题目真正问什么、目标单位、量纲、数量级和符号。
2. 如果 RAG 笔记与题目不匹配，不要因为有检索就偏向 RAG。
3. 如果两个答案不同，找出分歧来自公式、单位换算、计算、题意理解还是答案抽取。
4. 如果两个答案相同，也要检查它们是否共享了同一个未经验证的公式、常数、角度定义或单位假设；一致不等于可靠。
5. 需要复杂计算时使用 TOOL_CALC，不要心算。
6. 如果 verifier feedback 明确指出上一轮公式或单位错误，必须按 feedback 重新推导，不要继续选择被指出错误的候选。

{FINAL_PROTOCOL}"""


VERIFIER_SYSTEM = f"""你是 VERIFIER，一个严格的答案审查器。
你会看到题目、候选答案、前面各角色的推理和工具计算结果。

检查项目：
1. 公式/定律是否对应题目要求的量。
2. 数值是否全部来自题目或合理标准常数。
3. 单位换算、量纲、SI 前缀、10 的幂是否正确。
4. 科学计数法指数是否被错误丢掉。
5. 答案是每粒子、每摩尔还是总量。
6. 时间单位是否符合题目要求，例如中间量是每秒、每年还是总次数。
7. 如果题目涉及积分、微分、极限、比例或时间区间，推导出的中间量是否确实对应最终要求的目标量。
8. 工具表达式是否存在连续幂、漏括号、漏开方、把一整个表达式只开了部分方等优先级问题。
9. 符号是否符合题目要求。
10. FINAL_ANSWER 是否只含一个数值或数学表达式，且不含单位/解释/等号。

输出协议：
- 如果候选答案可靠，输出：
PASS
FINAL_ANSWER: <answer>
- 如果只需小修正且你能可靠修正，输出：
FIX
FINAL_ANSWER: <answer>
- 如果发现核心公式、题意或单位路径不可靠，需要前面解题者重做，输出：
LOOP
REASON: <short reason>

如果你能从前面任一解法、工具结果或自己的检查中确定正确答案，就输出 FIX，而不是 LOOP。只有无法可靠修正时才输出 LOOP。
需要复杂计算时也可以先输出 TOOL_CALC: <python_expression>。"""


def question_block(question: Question) -> str:
    return "\n".join(
        [
            f"id: {question.id}",
            f"field: {question.field}",
            f"subfield: {question.subfield or 'N/A'}",
            f"theorem: {question.theorem or 'N/A'}",
            f"target_unit: {question.unit or 'N/A'}",
            f"question: {question.question}",
        ]
    )


def direct_user_prompt(question: Question, feedback: str | None = None) -> str:
    parts = ["Solve this problem independently.", question_block(question)]
    if feedback:
        parts.append("Previous verifier feedback:\n" + feedback)
    return "\n\n".join(parts)


def curator_user_prompt(question: Question, chunks: list[RetrievedChunk]) -> str:
    snippets = []
    for i, chunk in enumerate(chunks, 1):
        snippets.append(f"[{i}] source={chunk.source} score={chunk.score:.3f}\n{chunk.text}")
    return "\n\n".join(
        [
            "RAG_CURATOR task: extract only useful textbook information for this problem.",
            question_block(question),
            "Retrieved textbook snippets:\n" + "\n\n".join(snippets),
        ]
    )


def rag_solver_user_prompt(question: Question, notes: str, feedback: str | None = None) -> str:
    parts = [
        "Solve this problem using the curated textbook notes only when they are relevant.",
        question_block(question),
        "Curated RAG notes:\n" + notes,
    ]
    if feedback:
        parts.append("Previous verifier feedback:\n" + feedback)
    return "\n\n".join(parts)


def arbiter_user_prompt(question: Question, direct: dict, rag: dict, feedback: str | None = None) -> str:
    parts = [
        "ARBITER task: compare both solutions and produce the best final answer.",
        question_block(question),
        f"DIRECT_SOLVER answer: {direct.get('answer')}\nDIRECT_SOLVER transcript:\n{direct.get('transcript')}",
        f"RAG_SOLVER answer: {rag.get('answer')}\nRAG_SOLVER transcript:\n{rag.get('transcript')}",
    ]
    if feedback:
        parts.append("Previous verifier feedback:\n" + feedback)
    return "\n\n".join(parts)


def verifier_user_prompt(question: Question, candidate_answer: str, history: str) -> str:
    return "\n\n".join(
        [
            "VERIFIER task: check the candidate answer and decide PASS, FIX, or LOOP.",
            question_block(question),
            f"Candidate FINAL_ANSWER: {candidate_answer}",
            "Full previous work:\n" + history,
        ]
    )
