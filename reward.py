"""计算 Search-R1 的奖励：答案精确匹配 + 格式 + 引用可溯源（电商场景）。

电商场景奖励 = 答案正确性 EM（w_em）
             + 引用存在（w_has_source）
             + 引用编号确实出现在检索结果中（w_source_retrieved，防止编造引用）
             + 引用编号命中 gold_docs（w_source_gold，保证答案有据可查）
             - 搜索成本（w_search_penalty × min(搜索次数, cap)）
未提供 gold_doc_ids 时（英文 Wikipedia 场景）退化为原版 EM / Format 奖励。

所有权重集中在 RewardWeights 中；默认值即线上训练配置，因此旧调用行为完全不变。
`--reward-config` 可按预设名切换权重，用于「奖励项消融」：把某一项权重置 0 后重训，
观察该分项是否真的提供了训练信号（见 reward_ablation.py 的离线归因）。
"""

import re
import unicodedata
from dataclasses import dataclass

ANSWER_PATTERN = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
SOURCE_PATTERN = re.compile(r"^\s*Source:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
DOC_ID_PATTERN = re.compile(r"[PS]\d{3}")
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardWeights:
    """复合奖励的权重配置；默认值即当前线上训练使用的配置。"""

    exact_match: float = 1.0  # 答案精确匹配
    has_source: float = 0.1  # 输出中包含合法的 Source 行
    source_retrieved: float = 0.2  # 引用编号确实出现在本轮检索结果中
    source_gold: float = 0.2  # 引用编号命中 gold_docs
    search_penalty: float = 0.05  # 每次搜索的惩罚
    search_penalty_cap: int = 4  # 搜索惩罚封顶次数

    def to_dict(self) -> dict[str, float]:
        """转成可写进 JSONL / SwanLab 的扁平字典。"""
        return {
            "exact_match": self.exact_match,
            "has_source": self.has_source,
            "source_retrieved": self.source_retrieved,
            "source_gold": self.source_gold,
            "search_penalty": self.search_penalty,
            "search_penalty_cap": float(self.search_penalty_cap),
        }

    @property
    def uses_citation_terms(self) -> bool:
        """是否启用了任何引用相关分项；全为 0 时退化为原版 EM/Format 奖励。"""
        return any(
            weight > 0.0
            for weight in (self.has_source, self.source_retrieved, self.source_gold)
        )


DEFAULT_WEIGHTS = RewardWeights()

# 消融实验预设：每一项只改一个分项的权重，保证单变量可比。
REWARD_PRESETS: dict[str, RewardWeights] = {
    "full": DEFAULT_WEIGHTS,
    "no_source_gold": RewardWeights(source_gold=0.0),
    "no_source_retrieved": RewardWeights(source_retrieved=0.0),
    "no_has_source": RewardWeights(has_source=0.0),
    "no_citation": RewardWeights(
        has_source=0.0, source_retrieved=0.0, source_gold=0.0
    ),
    "em_plus_format": RewardWeights(
        source_retrieved=0.0, source_gold=0.0, search_penalty=0.0
    ),
    "em_only": RewardWeights(
        has_source=0.0,
        source_retrieved=0.0,
        source_gold=0.0,
        search_penalty=0.0,
    ),
    "no_search_penalty": RewardWeights(search_penalty=0.0),
}


def resolve_weights(name: str) -> RewardWeights:
    """按预设名取权重；名称非法时给出可选值列表。"""
    if name not in REWARD_PRESETS:
        raise ValueError(
            f"未知的奖励预设 {name!r}；可选：{', '.join(sorted(REWARD_PRESETS))}"
        )
    return REWARD_PRESETS[name]


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
    weights: RewardWeights | None = None,
) -> RewardResult:
    """计算轨迹奖励；无 gold_doc_ids 时按原版 EM / Format 规则。"""
    active = weights or DEFAULT_WEIGHTS
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
        active.exact_match * float(exact_match)
        + active.has_source * float(has_source)
        + active.source_retrieved * float(source_retrieved)
        + active.source_gold * float(source_gold)
        - active.search_penalty * min(search_calls, active.search_penalty_cap)
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


def reward_contributions(
    result: RewardResult,
    search_calls: int,
    weights: RewardWeights | None = None,
) -> dict[str, float]:
    """把复合奖励拆成逐分项贡献，供奖励归因 / 消融分析使用。

    返回的键与 RewardWeights 字段同名，再加一个 `search_penalty` 负项；
    各项直接相加（格式非法时 Answer 缺失，此时只有格式惩罚项非零）。
    """
    active = weights or DEFAULT_WEIGHTS
    if result.answer is None:
        return {
            "exact_match": 0.0,
            "has_source": 0.0,
            "source_retrieved": 0.0,
            "source_gold": 0.0,
            "search_penalty": -0.1,
        }
    return {
        "exact_match": active.exact_match * float(result.exact_match),
        "has_source": active.has_source * float(result.has_source),
        "source_retrieved": active.source_retrieved * float(result.source_retrieved),
        "source_gold": active.source_gold * float(result.source_gold),
        "search_penalty": -active.search_penalty
        * min(search_calls, active.search_penalty_cap),
    }
