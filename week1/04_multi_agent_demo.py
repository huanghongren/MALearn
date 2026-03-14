from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import (
    RESEARCH_AGENT_PROMPT,
    CODER_AGENT_PROMPT,
    SUPERVISOR_FINALIZE_PROMPT,
)
from tools import search_paper_topic, get_code_help
from utils import get_llm, clean_model_output, print_section_line


class AgentState(TypedDict):
    user_input: str
    route: str
    topic: str
    tool_result: str
    specialist_result: str
    final_answer: str


def supervisor_route_node(state: AgentState):
    """
    supervisor 先决定把任务交给哪个 specialist。
    这里仍然采用规则路由，便于学习。
    """
    user_input = state["user_input"].lower()
    code_keywords = ["代码", "python", "函数", "脚本", "报错", "debug", "清洗", "正则"]

    if any(keyword in user_input for keyword in code_keywords):
        return {"route": "coder"}
    return {"route": "research"}


def route_edge(state: AgentState):
    """
    根据 supervisor 给出的 route，返回条件边分支标签。
    """
    if state["route"] == "coder":
        return "go_coder_agent"
    return "go_research_agent"


def research_agent_node(state: AgentState):
    """
    科研型 specialist。
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

    llm = get_llm()
    prompt = f"""
{RESEARCH_AGENT_PROMPT}

用户问题：
{state['user_input']}

当前主题：
{topic}

工具信息：
{tool_result}

请输出 research_agent 的专业中间结果。
""".strip()

    response = llm.invoke(prompt)
    specialist_result = clean_model_output(response.content)

    return {
        "topic": topic,
        "tool_result": tool_result,
        "specialist_result": specialist_result,
    }


def coder_agent_node(state: AgentState):
    """
    代码型 specialist。
    """
    user_input = state["user_input"].lower()

    if "think" in user_input or "清洗" in user_input or "标签" in user_input:
        topic = "think_clean"
    elif "debug" in user_input or "报错" in user_input:
        topic = "debug"
    else:
        topic = "python_function"

    tool_result = get_code_help(topic)

    llm = get_llm()
    prompt = f"""
{CODER_AGENT_PROMPT}

用户问题：
{state['user_input']}

当前主题：
{topic}

工具信息：
{tool_result}

请输出 coder_agent 的专业中间结果。
""".strip()

    response = llm.invoke(prompt)
    specialist_result = clean_model_output(response.content)

    return {
        "topic": topic,
        "tool_result": tool_result,
        "specialist_result": specialist_result,
    }


def supervisor_finalize_node(state: AgentState):
    """
    supervisor 汇总 specialist 的结果，生成最终回答。
    """
    llm = get_llm()

    prompt = f"""
{SUPERVISOR_FINALIZE_PROMPT}

用户问题：
{state['user_input']}

任务分配结果：
{state['route']}

当前主题：
{state['topic']}

工具信息：
{state['tool_result']}

specialist 中间结果：
{state['specialist_result']}

请输出最终回答。
""".strip()

    response = llm.invoke(prompt)
    final_answer = clean_model_output(response.content)

    return {"final_answer": final_answer}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor_route", supervisor_route_node)
    builder.add_node("research_agent", research_agent_node)
    builder.add_node("coder_agent", coder_agent_node)
    builder.add_node("supervisor_finalize", supervisor_finalize_node)

    builder.set_entry_point("supervisor_route")

    builder.add_conditional_edges(
        "supervisor_route",
        route_edge,
        {
            "go_research_agent": "research_agent",
            "go_coder_agent": "coder_agent",
        },
    )

    builder.add_edge("research_agent", "supervisor_finalize")
    builder.add_edge("coder_agent", "supervisor_finalize")
    builder.add_edge("supervisor_finalize", END)

    return builder.compile()


def run_case(user_input: str):
    graph = build_graph()

    result = graph.invoke(
        {
            "user_input": user_input,
            "route": "",
            "topic": "",
            "tool_result": "",
            "specialist_result": "",
            "final_answer": "",
        }
    )

    print_section_line(70)
    print(f"[用户问题]\n{user_input}\n")
    print(f"[supervisor 路由结果]\n{result['route']}\n")
    print(f"[识别主题]\n{result['topic']}\n")
    print(f"[工具返回]\n{result['tool_result']}\n")
    print(f"[specialist 中间结果]\n{result['specialist_result']}\n")
    print("[最终回答]")
    print(result["final_answer"])
    print_section_line(70)


def main():
    demo_cases = [
        "请介绍一下 ReAct 的核心思想，并给我学习建议。",
        "Toolformer 和普通问答模型有什么区别？",
        "请给我一个 Python 示例，演示如何清洗 <think> 标签。",
        "请解释一下为什么要把通用函数写到 utils.py 里面。",
    ]

    for user_input in demo_cases:
        run_case(user_input)


if __name__ == "__main__":
    main()
