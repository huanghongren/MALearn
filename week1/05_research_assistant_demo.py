from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import (
    RESEARCH_ASSISTANT_PROMPT,
    CODER_ASSISTANT_PROMPT,
    FINALIZER_PROMPT,
)
from tools import search_paper_topic, get_code_help
from utils import get_llm, clean_model_output, print_section_line


class AgentState(TypedDict):
    user_input: str
    need_research: bool
    need_coder: bool
    research_topic: str
    code_topic: str
    research_tool_result: str
    code_tool_result: str
    research_result: str
    code_result: str
    final_answer: str


def planner_node(state: AgentState):
    """
    对用户任务进行最小拆解：
    判断是否需要 research / coder。
    """
    user_input = state["user_input"].lower()

    research_keywords = [
        "react", "toolformer", "reflexion",
        "论文", "方法", "思想", "区别", "学习建议", "介绍"
    ]
    coder_keywords = [
        "代码", "python", "函数", "脚本", "实现", "示例",
        "demo", "langgraph", "清洗", "正则", "utils.py"
    ]

    need_research = any(keyword in user_input for keyword in research_keywords)
    need_coder = any(keyword in user_input for keyword in coder_keywords)

    # 如果都没识别到，默认走 research
    if not need_research and not need_coder:
        need_research = True

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

    return {
        "need_research": need_research,
        "need_coder": need_coder,
        "research_topic": research_topic,
        "code_topic": code_topic,
    }


def executor_node(state: AgentState):
    """
    按 planner 的判断，依次调用 research_agent / coder_agent。
    这里把多智能体逻辑收敛在一个执行节点里，便于学习任务拆解。
    """
    llm = get_llm()

    research_tool_result = ""
    code_tool_result = ""
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

    return {
        "research_tool_result": research_tool_result,
        "code_tool_result": code_tool_result,
        "research_result": research_result,
        "code_result": code_result,
    }


def finalizer_node(state: AgentState):
    """
    整合 research_result 和 code_result，生成最终回答。
    """
    llm = get_llm()

    prompt = f"""
{FINALIZER_PROMPT}

用户问题：
{state['user_input']}

是否需要 research：
{state['need_research']}

是否需要 coder：
{state['need_coder']}

research_result：
{state['research_result']}

code_result：
{state['code_result']}

请输出最终回答。
""".strip()

    response = llm.invoke(prompt)
    final_answer = clean_model_output(response.content)

    return {"final_answer": final_answer}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("finalizer", finalizer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "finalizer")
    builder.add_edge("finalizer", END)

    return builder.compile()


def run_case(user_input: str):
    graph = build_graph()

    result = graph.invoke(
        {
            "user_input": user_input,
            "need_research": False,
            "need_coder": False,
            "research_topic": "",
            "code_topic": "",
            "research_tool_result": "",
            "code_tool_result": "",
            "research_result": "",
            "code_result": "",
            "final_answer": "",
        }
    )

    print_section_line(80)
    print(f"[用户问题]\n{user_input}\n")
    print(f"[need_research]\n{result['need_research']}\n")
    print(f"[need_coder]\n{result['need_coder']}\n")
    print(f"[research_topic]\n{result['research_topic']}\n")
    print(f"[code_topic]\n{result['code_topic']}\n")
    print(f"[research_result]\n{result['research_result']}\n")
    print(f"[code_result]\n{result['code_result']}\n")
    print("[最终回答]")
    print(result["final_answer"])
    print_section_line(80)


def main():
    demo_cases = [
        "请介绍一下 ReAct 的核心思想，并告诉我应该怎么开始学习。",
        "请给我一个 Python 示例，演示如何清洗 <think> 标签。",
        "请介绍 ReAct 的核心思想，并给我一个最小 LangGraph 学习建议。",
        "请比较 Toolformer 和 ReAct 的差异，并告诉我如果我要写 demo，应该先做什么。",
    ]

    for user_input in demo_cases:
        run_case(user_input)


if __name__ == "__main__":
    main()
