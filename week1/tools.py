from typing import Dict
import re


PAPER_KNOWLEDGE_BASE: Dict[str, str] = {
    "react": (
        "ReAct 的核心思想是把“推理（Reasoning）”与“行动（Acting）”交替结合。"
        "模型先思考下一步该做什么，再执行动作获取外部信息，然后根据观察结果继续推理。"
        "这种方式适合问答、工具调用、检索增强等任务。"
    ),
    "toolformer": (
        "Toolformer 的核心思想是：让语言模型学会在合适的时候调用外部工具。"
        "外部工具可以是搜索、计算器、翻译器或数据库。"
        "它强调模型不仅要会回答，还要会决定何时借助工具。"
    ),
    "reflexion": (
        "Reflexion 的核心思想是：让智能体在失败后生成文字形式的反思，"
        "再把这些反思作为下一轮决策的依据。"
        "它强调通过语言反馈而不是重新训练参数来提升表现。"
    ),
}


CODE_KNOWLEDGE_BASE: Dict[str, str] = {
    "think_clean": (
        "可以使用 Python 的 re.sub 清理 <think>...</think> 内容。"
        "常见写法是：re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)。"
        "如果还担心残留标签，可以再额外删除 </?think>。"
    ),
    "python_function": (
        "Python 函数通常用 def 定义。"
        "建议把可复用逻辑封装成独立函数，例如 build_llm()、clean_model_output()。"
        "这样主流程会更清晰。"
    ),
    "debug": (
        "调试时建议同时打印输入、工具返回、模型原始输出和清洗后的最终输出。"
        "这样更容易定位是工具问题、prompt 问题，还是输出清洗问题。"
    ),
}


def search_paper_topic(topic: str) -> str:
    key = topic.strip().lower()
    return PAPER_KNOWLEDGE_BASE.get(
        key,
        f"没有检索到主题 {topic} 的结果。你可以尝试：ReAct / Toolformer / Reflexion"
    )


def get_code_help(code_topic: str) -> str:
    key = code_topic.strip().lower()
    return CODE_KNOWLEDGE_BASE.get(
        key,
        "当前代码工具信息有限。你可以尝试：think_clean / python_function / debug"
    )


def normalize_math_expression(text: str) -> str:
    """
    从用户输入中粗略提取四则运算表达式，并规范化为 Python 可计算形式。
    仅用于教学 demo。
    """
    if not text:
        return ""

    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("＝", "=")

    chars = []
    allowed = set("0123456789+-*/(). =")
    for ch in text:
        if ch in allowed:
            chars.append(ch)

    expr = "".join(chars)
    expr = re.sub(r"\s+", "", expr)

    # 如果有等号，只取等号左边
    if "=" in expr:
        expr = expr.split("=")[0]

    return expr


def calculate_math_expression(expr: str) -> str:
    """
    计算简单四则运算表达式。
    仅适用于教学 demo。
    """
    if not expr:
        return ""

    if not re.fullmatch(r"[0-9+\-*/().]+", expr):
        return ""

    try:
        value = eval(expr, {"__builtins__": {}}, {})
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)
    except Exception:
        return ""
