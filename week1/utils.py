import re
from langchain_openai import ChatOpenAI
from config import (
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    VLLM_API_KEY,
    DEFAULT_TEMPERATURE,
)

# 模块级单例
_LLM_INSTANCE = None


def build_llm() -> ChatOpenAI:
    """
    构造连接本地 vLLM 的 LangChain Chat 模型。
    注意：这个函数负责“创建实例”，通常不建议在每个节点里反复直接调用。
    """
    return ChatOpenAI(
        model=VLLM_MODEL_NAME,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=DEFAULT_TEMPERATURE,
    )


def get_llm() -> ChatOpenAI:
    """
    获取单例 LLM 对象。
    首次调用时创建，后续直接复用。
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = build_llm()
    return _LLM_INSTANCE


def clean_model_output(text: str) -> str:
    """
    清理模型输出中的思考痕迹和多余空白。
    """
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def print_section_line(width: int = 70):
    """
    打印分隔线。
    """
    print("=" * width)

import re

def extract_number_from_answer(text: str) -> str:
    """
    从回答中提取最后一个数字（整数或小数）作为粗略答案候选。
    用于数学题最小检查。
    """
    if not text:
        return ""

    import re
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return ""

    return numbers[-1]

#DAY7
def load_local_kb(file_path: str) -> str:
    """
    读取本地知识库文本。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_to_chunks(text: str):
    """
    按空行切分知识块。
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


def extract_keywords(query: str):
    """
    从问题中提取最小关键词集合。
    这里只做非常轻量的规则化处理。
    """
    query = query.lower()
    raw_keywords = [
        "react", "toolformer", "reflexion",
        "langgraph", "state", "node", "edge",
        "多智能体", "supervisor", "specialist",
        "reflection", "retry", "工具", "推理", "行动"
    ]

    keywords = [kw for kw in raw_keywords if kw in query]
    if not keywords:
        # 兜底：按空格切简单词
        keywords = [word.strip() for word in query.split() if word.strip()]

    return keywords


def retrieve_top_chunks(query: str, chunks, top_k: int = 2):
    """
    用关键词匹配对 chunk 打分，返回最相关的 top_k。
    """
    keywords = extract_keywords(query)
    scored = []

    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            if kw and kw in chunk_lower:
                score += 1

        scored.append((idx, chunk, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    selected = [item for item in scored if item[2] > 0][:top_k]

    # 如果一个都没匹配上，兜底返回前 1 个
    if not selected and chunks:
        selected = [(0, chunks[0], 0)]

    selected_ids = [str(item[0]) for item in selected]
    selected_chunks = [item[1] for item in selected]

    return selected_ids, selected_chunks
