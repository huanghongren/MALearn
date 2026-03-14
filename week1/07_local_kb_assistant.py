from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import (
    KB_FILE_PATH,
    KB_ANSWER_PROMPT,
    KB_RETRY_PROMPT,
)
from utils import (
    get_llm,
    clean_model_output,
    print_section_line,
    load_local_kb,
    split_text_to_chunks,
    retrieve_top_chunks,
)


class AgentState(TypedDict):
    user_input: str
    retrieved_chunks: str
    retrieved_chunk_ids: str
    first_answer: str
    reflection_passed: bool
    reflection_note: str
    final_answer: str


def retriever_node(state: AgentState):
    kb_text = load_local_kb(KB_FILE_PATH)
    chunks = split_text_to_chunks(kb_text)

    chunk_ids, selected_chunks = retrieve_top_chunks(
        query=state["user_input"],
        chunks=chunks,
        top_k=2,
    )

    retrieved_chunks = "\n\n".join(selected_chunks)
    retrieved_chunk_ids = ", ".join(chunk_ids)

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


def answer_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
{KB_ANSWER_PROMPT}

用户问题：
{state['user_input']}

检索到的知识片段：
{state['retrieved_chunks']}

请生成第一版回答。
""".strip()

    response = llm.invoke(prompt)
    first_answer = clean_model_output(response.content)

    return {"first_answer": first_answer}


def checker_node(state: AgentState):
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

    if len(answer.strip()) < 30:
        return {
            "reflection_passed": False,
            "reflection_note": "第一版答案过短，缺少有效信息。",
        }

    if "当前检索信息有限" not in answer and len(state["retrieved_chunks"].strip()) > 0:
        # 很轻量的 groundedness 检查：至少应该围绕知识片段说点东西
        query_lower = state["user_input"].lower()
        if "react" in query_lower and "react" not in answer.lower():
            return {
                "reflection_passed": False,
                "reflection_note": "用户问题涉及 ReAct，但第一版答案没有明确回应该主题。",
            }
        if "toolformer" in query_lower and "toolformer" not in answer.lower():
            return {
                "reflection_passed": False,
                "reflection_note": "用户问题涉及 Toolformer，但第一版答案没有明确回应该主题。",
            }
        if "langgraph" in query_lower and "langgraph" not in answer.lower():
            return {
                "reflection_passed": False,
                "reflection_note": "用户问题涉及 LangGraph，但第一版答案没有明确回应该主题。",
            }

    return {
        "reflection_passed": True,
        "reflection_note": "第一版答案通过基本检查，可直接作为最终答案。",
    }


def check_edge(state: AgentState):
    if state["reflection_passed"]:
        return "go_end"
    return "go_retry"


def retry_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
{KB_RETRY_PROMPT}

用户问题：
{state['user_input']}

检索到的知识片段：
{state['retrieved_chunks']}

第一版答案：
{state['first_answer']}

反思意见：
{state['reflection_note']}

请重写最终答案。
""".strip()

    response = llm.invoke(prompt)
    final_answer = clean_model_output(response.content)

    return {"final_answer": final_answer}


def pass_through_node(state: AgentState):
    return {"final_answer": state["first_answer"]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retriever", retriever_node)
    builder.add_node("answer", answer_node)
    builder.add_node("checker", checker_node)
    builder.add_node("retry", retry_node)
    builder.add_node("pass_through", pass_through_node)

    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "answer")
    builder.add_edge("answer", "checker")

    builder.add_conditional_edges(
        "checker",
        check_edge,
        {
            "go_end": "pass_through",
            "go_retry": "retry",
        },
    )

    builder.add_edge("pass_through", END)
    builder.add_edge("retry", END)

    return builder.compile()


def run_case(user_input: str):
    graph = build_graph()

    result = graph.invoke(
        {
            "user_input": user_input,
            "retrieved_chunks": "",
            "retrieved_chunk_ids": "",
            "first_answer": "",
            "reflection_passed": False,
            "reflection_note": "",
            "final_answer": "",
        }
    )

    print_section_line(90)
    print(f"[用户问题]\n{user_input}\n")
    print(f"[命中的 chunk ids]\n{result['retrieved_chunk_ids']}\n")
    print(f"[检索到的知识片段]\n{result['retrieved_chunks']}\n")
    print(f"[第一版答案]\n{result['first_answer']}\n")
    print(f"[是否通过检查]\n{result['reflection_passed']}\n")
    print(f"[反思意见]\n{result['reflection_note']}\n")
    print("[最终答案]")
    print(result["final_answer"])
    print_section_line(90)


def main():
    demo_cases = [
        "请介绍一下 ReAct 的核心思想。",
        "Toolformer 和普通问答模型有什么区别？",
        "请解释 LangGraph 中的 State、Node 和 Edge。",
        "多智能体中的 supervisor 和 specialist 是什么关系？",
        "什么是 Reflection + Retry 机制？",
    ]

    for user_input in demo_cases:
        run_case(user_input)


if __name__ == "__main__":
    main()
