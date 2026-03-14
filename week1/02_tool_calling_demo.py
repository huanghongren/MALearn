from config import TOOL_AGENT_SYSTEM_PROMPT
from tools import search_paper_topic, list_supported_topics
from utils import build_llm, clean_model_output, print_section_line


def build_prompt(user_query: str, topic: str, tool_result: str, supported_topics: str) -> str:
    """
    构造最终发给模型的提示词。
    """
    prompt = f"""
{TOOL_AGENT_SYSTEM_PROMPT}

用户问题：
{user_query}

当前检索主题：
{topic}

工具1：主题检索结果
{tool_result}

工具2：当前支持的主题
{supported_topics}

请严格根据以上工具信息作答。
如果工具信息不足，请明确说明“当前工具信息有限”，不要自行虚构细节。
""".strip()
    return prompt


def run_demo(user_query: str, topic: str):
    """
    运行单条演示样例。
    """
    llm = build_llm()

    tool_result = search_paper_topic(topic)
    supported_topics = list_supported_topics()

    prompt = build_prompt(
        user_query=user_query,
        topic=topic,
        tool_result=tool_result,
        supported_topics=supported_topics,
    )

    response = llm.invoke(prompt)
    raw_output = response.content
    final_output = clean_model_output(raw_output)

    print_section_line(70)
    print(f"[用户问题]\n{user_query}\n")
    print(f"[调用的主题工具]\n{topic}\n")
    print(f"[工具返回]\n{tool_result}\n")
    print("[模型原始输出]")
    print(raw_output)
    print("\n[清洗后的最终回答]")
    print(final_output)
    print_section_line(70)


def main():
    demo_cases = [
        ("请介绍一下 ReAct 的核心思想，并告诉我应该怎么开始学习。", "ReAct"),
        ("Toolformer 和普通问答模型有什么区别？", "Toolformer"),
        ("Reflexion 对 agent 有什么启发？", "Reflexion"),
    ]

    for user_query, topic in demo_cases:
        run_demo(user_query, topic)


if __name__ == "__main__":
    main()
