from config import SYSTEM_PROMPT_ZH
from utils import build_llm, print_section_line


def main():
    print_section_line(60)
    print("Day1 Demo: LangChain 连接本地 vLLM")
    print_section_line(60)

    llm = build_llm()

    user_query = "请你用 5 句话介绍什么是多智能体系统，以及它和单智能体的区别。"
    prompt = f"""
{SYSTEM_PROMPT_ZH}

用户问题：
{user_query}
""".strip()

    print(f"\n[用户问题]\n{user_query}\n")
    print("[模型回答]")

    response = llm.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()
