"""计算 Search-R1 的奖励：答案精确匹配 + 格式 + 引用可溯源（电商场景）。

电商场景奖励 = 答案正确性 EM（1.0）
             + 引用存在（0.1）
             + 引用编号确实出现在检索结果中（0.2，防止编造引用）
             + 引用编号命中 gold_docs（0.2，保证答案有据可查）
             - 检索效率成本（每次搜索 -0.05，最多计 4 次）
未提供 gold_doc_ids 时（英文 Wikipedia 场景）退化为原版 EM / Format 奖励。
"""

import re
import unicodedata
from dataclasses import dataclass


ANSWER_PATTERN = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
SOURCE_PATTERN = re.compile(r"^\s*Source:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
DOC_ID_PATTERN = re.compile(r"[PS]\d{3}")
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardResult:
    """保存最终 reward 及其判定细节。"""

    reward: float
    valid_format: bool
    exact_match: bool
    answer: str | None
    has_source: bool = False
    source_retrieved: bool = False  # 引用编号在轨迹实际检索到的文档中
    source_gold: bool = False  # 引用编号命中 gold_docs


def normalize_answer(text: str) -> str:
    """统一答案格式：转小写、去除标点和英文冠词，并合并多余空格。"""
    lowered = text.lower()
    without_punctuation = "".join(
        char for char in lowered if not unicodedata.category(char).startswith("P")
    )
    without_articles = ARTICLE_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str | None:
    """只接受恰好一行非空的 Answer: 最终答案。"""
    matches = ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def extract_source_ids(text: str) -> tuple[bool, list[str]]:
    """要求恰好一行 Source 并提取其中的文档编号；多行 / 无编号视为不合规。"""
    lines = SOURCE_PATTERN.findall(text)
    if len(lines) != 1:
        return False, []
    ids = DOC_ID_PATTERN.findall(lines[0])
    return (bool(ids), ids)


def score_answer(
    text: str,
    references: list[str],
    retrieved_doc_ids: list[str] | None = None,
    gold_doc_ids: list[str] | None = None,
    search_calls: int = 0,
) -> RewardResult:
    """计算轨迹奖励；无 gold_doc_ids 时按原版 EM / Format 规则。"""
    answer = extract_answer(text)
    if answer is None:
        return RewardResult(-0.1, False, False, None)
    normalized = normalize_answer(answer)
    exact_match = any(normalized == normalize_answer(reference) for reference in references)

    if gold_doc_ids is None:
        return RewardResult(float(exact_match), True, exact_match, answer)

    has_source, cited_ids = extract_source_ids(text)
    retrieved = retrieved_doc_ids or []
    source_retrieved = has_source and any(doc_id in retrieved for doc_id in cited_ids)
    source_gold = has_source and any(doc_id in gold_doc_ids for doc_id in cited_ids)

    reward = (
        float(exact_match)
        + 0.1 * float(has_source)
        + 0.2 * float(source_retrieved)
        + 0.2 * float(source_gold)
        - 0.05 * min(search_calls, 4)
    )
    return RewardResult(
        reward=reward,
        valid_format=has_source,  # 电商场景：Answer + Source 两行齐全才算格式合法
        exact_match=exact_match,
        answer=answer,
        has_source=has_source,
        source_retrieved=source_retrieved,
        source_gold=source_gold,
    )
