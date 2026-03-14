from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import (
    FINAL_ANSWER_PROMPT,
)
from tools import search_paper_topic, get_code_help
from utils import (
    build_llm,
    clean_model_output,
    print_section_line,
)


class AgentState(TypedDict):
    user_input: str
    route: str
    topic: str
    tool_result: str
    answer: str


def router_node(state: AgentState):
    """
    路由节点：
    用规则先做最小可控路由，而不是一开始就交给 LLM。
    """
    user_input = state["user_input"].lower()

    code_keywords = ["代码", "python", "函数", "脚本", "报错", "debug", "清洗", "正则"]

    if any(keyword in user_input for keyword in code_keywords):
        route = "coder"
    else:
        route = "research"

    return {"route": route}


def research_node(state: AgentState):
    """
    科研解释节点：
    根据用户问题粗略提取主题，并调用论文知识工具。
    """
    user_input = state["user_input"].lower()

    if "react" in user_input:
        topic = "ReAct"
    elif "toolformer" in user_input:
        topic = "Toolformer"
    elif "reflexion" in user_input:
        topic = "Reflexion"
    else:
        topic = "ReAct"

    tool_result = search_paper_topic(topic)
    return {
        "topic": topic,
        "tool_result": tool_result,
    }


def coder_node(state: AgentState):
    """
    代码帮助节点：
    根据问题选择代码知识工具主题。
    """
    user_input = state["user_input"].lower()

    if "think" in user_input or "清洗" in user_input or "标签" in user_input:
        topic = "think_clean"
    elif "debug" in user_input or "报错" in user_input:
        topic = "debug"
    else:
        topic = "python_function"

    tool_result = get_code_help(topic)
    return {
        "topic": topic,
        "tool_result": tool_result,
    }


def answer_node(state: AgentState):
    """
    最终回答节点：
    把前面节点准备好的信息组织成用户可读答案。
    """
    llm = build_llm()

    prompt = f"""
{FINAL_ANSWER_PROMPT}

用户问题：
{state['user_input']}

当前路由：
{state['route']}

当前主题：
{state['topic']}

辅助信息：
{state['tool_result']}

请生成最终回答。
""".strip()

    response = llm.invoke(prompt)
    final_answer = clean_model_output(response.content)

    return {"answer": final_answer}


def route_edge(state: AgentState):
    """
    条件边函数：
    读取当前 state 中的 route，
    返回“分支标签”，而不是直接返回目标节点名。
    后续会由 add_conditional_edges 中的映射表，
    将分支标签映射到真正的目标节点。
    """
    if state["route"] == "coder":
        return "go_coder"
    return "go_researcher"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", coder_node)
    builder.add_node("answer", answer_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        route_edge,
        {
            "go_researcher": "researcher",
            "go_coder": "coder",
        },
    )

    builder.add_edge("researcher", "answer")
    builder.add_edge("coder", "answer")
    builder.add_edge("answer", END)

    graph = builder.compile()
    return graph


def run_case(user_input: str):
    graph = build_graph()

    result = graph.invoke(
        {
            "user_input": user_input,
            "route": "",
            "topic": "",
            "tool_result": "",
            "answer": "",
        }
    )

    print_section_line(70)
    print(f"[用户问题]\n{user_input}\n")
    print(f"[路由结果]\n{result['route']}\n")
    print(f"[识别主题]\n{result['topic']}\n")
    print(f"[工具返回]\n{result['tool_result']}\n")
    print("[最终回答]")
    print(result["answer"])
    print_section_line(70)


def main():
    demo_cases = [
        "请介绍一下 ReAct 的核心思想，并给我学习建议。",
        "请给我一个 Python 示例，演示如何清洗 <think> 标签。",
        "Toolformer 和普通问答模型有什么区别？",
        "请解释一下为什么要把通用函数写到 utils.py 里面。",
    ]

    for user_input in demo_cases:
        run_case(user_input)


if __name__ == "__main__":
    main()
