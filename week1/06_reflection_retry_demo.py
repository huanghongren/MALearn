from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import (
    RESEARCH_ASSISTANT_PROMPT,
    CODER_ASSISTANT_PROMPT,
    FIRST_FINALIZER_PROMPT,
    RETRY_FINALIZER_PROMPT,
)
from tools import (
    search_paper_topic,
    get_code_help,
    normalize_math_expression,
    calculate_math_expression,
)
from utils import (
    get_llm,
    clean_model_output,
    print_section_line,
    extract_number_from_answer,
)


class AgentState(TypedDict):
    user_input: str
    task_type: str

    need_research: bool
    need_coder: bool
    need_math: bool

    research_topic: str
    code_topic: str
    math_expression: str

    research_tool_result: str
    code_tool_result: str
    math_result: str

    research_result: str
    code_result: str

    first_answer: str
    reflection_passed: bool
    reflection_note: str
    final_answer: str


def planner_node(state: AgentState):
    user_input = state["user_input"].lower()

    research_keywords = [
        "react", "toolformer", "reflexion",
        "论文", "方法", "思想", "区别", "学习建议", "介绍"
    ]
    coder_keywords = [
        "代码", "python", "函数", "脚本", "实现", "示例",
        "demo", "langgraph", "清洗", "正则", "utils.py"
    ]
    math_keywords = ["计算", "求值", "等于多少", "+", "-", "*", "/", "×", "÷", "（", "）", "("]

    need_research = any(keyword in user_input for keyword in research_keywords)
    need_coder = any(keyword in user_input for keyword in coder_keywords)
    need_math = any(keyword in state["user_input"] for keyword in math_keywords)

    if need_math:
        task_type = "math"
    elif need_research and need_coder:
        task_type = "hybrid"
    elif need_coder:
        task_type = "code"
    else:
        task_type = "research"

    if not need_research and not need_coder and not need_math:
        need_research = True
        task_type = "research"

    if "react" in user_input:
        research_topic = "ReAct"
    elif "toolformer" in user_input:
        research_topic = "Toolformer"
    elif "reflexion" in user_input:
        research_topic = "Reflexion"
    else:
        research_topic = "ReAct"

    if "think" in user_input or "清洗" in user_input or "标签" in user_input:
        code_topic = "think_clean"
    elif "debug" in user_input or "报错" in user_input:
        code_topic = "debug"
    else:
        code_topic = "python_function"

    math_expression = normalize_math_expression(state["user_input"]) if need_math else ""

    return {
        "task_type": task_type,
        "need_research": need_research,
        "need_coder": need_coder,
        "need_math": need_math,
        "research_topic": research_topic,
        "code_topic": code_topic,
        "math_expression": math_expression,
    }


def executor_node(state: AgentState):
    llm = get_llm()

    research_tool_result = ""
    code_tool_result = ""
    math_result = ""

    research_result = ""
    code_result = ""

    if state["need_research"]:
        research_tool_result = search_paper_topic(state["research_topic"])

        research_prompt = f"""
{RESEARCH_ASSISTANT_PROMPT}

用户问题：
{state['user_input']}

研究主题：
{state['research_topic']}

工具信息：
{research_tool_result}

请输出 research_agent 的中间结果。
""".strip()

        research_response = llm.invoke(research_prompt)
        research_result = clean_model_output(research_response.content)

    if state["need_coder"]:
        code_tool_result = get_code_help(state["code_topic"])

        code_prompt = f"""
{CODER_ASSISTANT_PROMPT}

用户问题：
{state['user_input']}

代码主题：
{state['code_topic']}

工具信息：
{code_tool_result}

请输出 coder_agent 的中间结果。
""".strip()

        code_response = llm.invoke(code_prompt)
        code_result = clean_model_output(code_response.content)

    if state["need_math"]:
        math_result = calculate_math_expression(state["math_expression"])

    return {
        "research_tool_result": research_tool_result,
        "code_tool_result": code_tool_result,
        "math_result": math_result,
        "research_result": research_result,
        "code_result": code_result,
    }


def first_finalizer_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
{FIRST_FINALIZER_PROMPT}

用户问题：
{state['user_input']}

任务类型：
{state['task_type']}

research_result：
{state['research_result']}

code_result：
{state['code_result']}

math_result：
{state['math_result']}

如果是数学题，请直接给出结果，并简要说明计算步骤。
请输出第一版答案。
""".strip()

    response = llm.invoke(prompt)
    first_answer = clean_model_output(response.content)

    return {"first_answer": first_answer}


def checker_node(state: AgentState):
    """
    用规则做最小质量检查，并加入数学题专项检查。
    """
    answer = state["first_answer"]

    bad_patterns = [
        "<think>",
        "</think>",
        "首先我需要",
        "接下来我需要",
        "最后检查一下",
        "让我来一步一步思考",
    ]

    for pattern in bad_patterns:
        if pattern in answer:
            return {
                "reflection_passed": False,
                "reflection_note": f"第一版答案包含草稿式分析或思考痕迹：{pattern}",
            }

    # 数学题专项检查：调用 tools.py 里的数学计算器得到标准答案，再和模型答案对比
    if state["need_math"]:
        expected = calculate_math_expression(state["math_expression"])
        predicted = extract_number_from_answer(answer)

        if not expected:
            return {
                "reflection_passed": False,
                "reflection_note": "数学工具未能成功计算表达式，请重新生成明确答案。",
            }

        if predicted != expected:
            return {
                "reflection_passed": False,
                "reflection_note": (
                    f"数学题答案错误。当前答案中的结果是 {predicted or '未提取到'}，"
                    f"正确结果应为 {expected}。请重新计算，并给出简要步骤。"
                ),
            }

    if state["need_coder"]:
        code_signals = ["代码", "函数", "示例", "re.sub", "python", "正则"]
        if not any(signal.lower() in answer.lower() for signal in code_signals):
            return {
                "reflection_passed": False,
                "reflection_note": "用户请求涉及代码实现，但第一版答案缺少明确的代码实现信号。",
            }

    if state["need_research"] and len(answer.strip()) < 40:
        return {
            "reflection_passed": False,
            "reflection_note": "第一版答案过短，不足以支撑科研学习场景。",
        }

    return {
        "reflection_passed": True,
        "reflection_note": "第一版答案通过基本检查，可直接作为最终答案。",
    }


def check_edge(state: AgentState):
    if state["reflection_passed"]:
        return "go_end"
    return "go_retry"


def retry_finalizer_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
{RETRY_FINALIZER_PROMPT}

用户问题：
{state['user_input']}

第一版答案：
{state['first_answer']}

反思意见：
{state['reflection_note']}

research_result：
{state['research_result']}

code_result：
{state['code_result']}

math_result（正确参考值）：
{state['math_result']}

请重写最终答案。
如果是数学题，请明确写出正确结果，并简要说明步骤。
""".strip()

    response = llm.invoke(prompt)
    final_answer = clean_model_output(response.content)

    return {"final_answer": final_answer}


def pass_through_node(state: AgentState):
    """
    如果检查通过，直接把 first_answer 作为 final_answer。
    """
    return {"final_answer": state["first_answer"]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("first_finalizer", first_finalizer_node)
    builder.add_node("checker", checker_node)
    builder.add_node("retry_finalizer", retry_finalizer_node)
    builder.add_node("pass_through", pass_through_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "first_finalizer")
    builder.add_edge("first_finalizer", "checker")

    builder.add_conditional_edges(
        "checker",
        check_edge,
        {
            "go_end": "pass_through",
            "go_retry": "retry_finalizer",
        },
    )

    builder.add_edge("pass_through", END)
    builder.add_edge("retry_finalizer", END)

    return builder.compile()


def run_case(user_input: str):
    graph = build_graph()

    result = graph.invoke(
        {
            "user_input": user_input,
            "task_type": "",
            "need_research": False,
            "need_coder": False,
            "need_math": False,
            "research_topic": "",
            "code_topic": "",
            "math_expression": "",
            "research_tool_result": "",
            "code_tool_result": "",
            "math_result": "",
            "research_result": "",
            "code_result": "",
            "first_answer": "",
            "reflection_passed": False,
            "reflection_note": "",
            "final_answer": "",
        }
    )

    print_section_line(90)
    print(f"[用户问题]\n{user_input}\n")
    print(f"[task_type]\n{result['task_type']}\n")
    print(f"[need_research]\n{result['need_research']}\n")
    print(f"[need_coder]\n{result['need_coder']}\n")
    print(f"[need_math]\n{result['need_math']}\n")
    print(f"[math_expression]\n{result['math_expression']}\n")
    print(f"[math_result]\n{result['math_result']}\n")
    print(f"[research_result]\n{result['research_result']}\n")
    print(f"[code_result]\n{result['code_result']}\n")
    print(f"[第一版答案]\n{result['first_answer']}\n")
    print(f"[是否通过检查]\n{result['reflection_passed']}\n")
    print(f"[反思意见]\n{result['reflection_note']}\n")
    print("[最终答案]")
    print(result["final_answer"])
    print_section_line(90)


def main():
    demo_cases = [
        "请介绍一下 ReAct 的核心思想，并告诉我应该怎么开始学习。",
        "请给我一个 Python 示例，演示如何清洗 <think> 标签。",
        "请介绍 ReAct 的核心思想，并给我一个最小 LangGraph 学习建议。",
        "请比较 Toolformer 和 ReAct 的差异，并告诉我如果我要写 demo，应该先做什么。",
        "请计算：37 × 48 + 125 = ?",
        "请计算：(15 + 27) × 4 - 18 = ?",
        "请计算 12*(8+5)-(7*-8)+(-9*-9)，并简要说明计算步骤。"
    ]

    for user_input in demo_cases[-1:]:
        run_case(user_input)


if __name__ == "__main__":
    main()
