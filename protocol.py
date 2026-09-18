"""定义 Qwen3.5 使用的搜索工具协议。


支持两个场景，由环境变量 SEARCH_R1_SCENE 切换：
- 默认：英文 Wikipedia 事实问答（Search-R1 原版协议）
- ecommerce：中文电商客服（检索商品手册 / 服务政策，最终答案必须附文档编号）
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


WIKI_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for evidence. Use a concise English query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A concise English search query."}
            },
            "required": ["query"],
        },
    },
}

WIKI_SYSTEM_PROMPT = """You answer factual questions with help from a search tool.
Search when you need evidence. You may call search several times with concise English queries.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
When ready, end with exactly one non-empty line in this format:
Answer: <your short answer>
Do not call a tool and give the final answer in the same turn.

Example of a question that needs two searches:
Question: What country was the author of The Little Prince born in?
1. Call search with query "The Little Prince author".
2. From the result, identify Antoine de Saint-Exupery.
3. Call search with query "Antoine de Saint-Exupery birthplace country".
4. From the result, identify France.
5. In a new assistant turn, give the final answer:
Answer: France"""

ECOMMERCE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "检索电商商品知识库（商品手册、服务政策）。请使用简短的中文关键词查询。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "简短的中文检索关键词，如：蓝曜T1 电池容量"}
            },
            "required": ["query"],
        },
    },
}

ECOMMERCE_SYSTEM_PROMPT = """你是电商平台的智能客服助手，必须依据知识库中的《商品手册》和《服务政策》回答用户问题，禁止编造文档中没有的信息。
你可以使用 search 工具检索知识库，查询词用简短中文（例如“蓝曜T1Pro 电池容量”或“七天无理由 运费”）。每轮 assistant 回合只调用一次 search，等待工具返回结果后，再决定继续检索还是作答。
信息充分后，在新的 assistant 回合中严格输出两行最终答案，不要在同一回合同时调用工具和作答：
Answer: <简短答案，须与文档表述一致，如 5000mAh、支持、蓝曜T1Pro>
Source: <答案依据的文档编号，如 P003；跨多篇文档时用逗号分隔，如 P003,S004>
需要多篇文档才能回答的问题（如原厂保修叠加延保、激活限制、两款商品参数对比），应分别检索、综合后再作答。"""

ECOMMERCE_SCENE = os.getenv("SEARCH_R1_SCENE") == "ecommerce"
SEARCH_TOOL = ECOMMERCE_SEARCH_TOOL if ECOMMERCE_SCENE else WIKI_SEARCH_TOOL
SYSTEM_PROMPT = ECOMMERCE_SYSTEM_PROMPT if ECOMMERCE_SCENE else WIKI_SYSTEM_PROMPT

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAssistant:
    """保存一次 assistant 输出的协议解析结果。"""

    kind: str  # 解析类型："tool"、"answer" 或 "invalid"
    content: str  # 普通文本：工具调用前的规划，或完整答案/非法输出
    query: str | None = None


def initial_messages(question: str) -> list[dict[str, Any]]:
    """为一道问题创建初始对话。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """渲染消息并把 tokenizer 的不同返回类型统一成一维 token 列表。"""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """用模型原生 chat template 构建带工具定义的生成 prompt。"""
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回 tokens 末尾与 suffix 开头的最长重叠长度。"""
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-length:] == suffix[:length]:
            return length
    return 0


def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """用真实采样 token 接上 assistant 结束符和新的 tool observation。"""
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    # Qwen3.5 会 trim 消息 content，不能靠重新编码采样文本定位结束符。
    # 用空 assistant 单独提取模板闭合 token，真实采样 token 始终原样保留。
    empty_assistant_end = _render_chat(
        tokenizer,
        [*messages_before_assistant, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
    )
    if empty_assistant_end[: len(canonical_prompt)] != canonical_prompt:
        raise ValueError("chat template 无法从空 assistant 提取结束边界")
    assistant_closing_tokens = empty_assistant_end[len(canonical_prompt) :]

    assistant_message = {"role": "assistant", "content": assistant_text}
    messages_with_assistant = [*messages_before_assistant, assistant_message]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )

    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[: len(canonical_assistant_end)] != canonical_assistant_end:
        raise ValueError("加入 tool observation 后 chat template 改写了历史消息")
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]

    # sampler 可能已经返回部分或全部 assistant 结束符，只补尚未包含的部分。
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


def parse_assistant(text: str) -> ParsedAssistant:
    """把 assistant 文本识别成搜索调用、最终回答或非法调用。"""
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = "invalid" if "<tool_call>" in text else "answer"
        return ParsedAssistant(kind=kind, content=text.strip())
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return ParsedAssistant(kind="invalid", content=text.strip())
    query = matches[0].group(1).strip()
    if not query or "<" in query or ">" in query:
        return ParsedAssistant(kind="invalid", content=text.strip())
    content = text[: matches[0].start()].strip()
    return ParsedAssistant(kind="tool", content=content, query=query)


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化搜索结果消息。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "search",
        "content": content,
    }


def stop_sequences(tokenizer: Any) -> list[str]:
    """返回模型结束一轮 assistant 输出时使用的停止字符串。"""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []
