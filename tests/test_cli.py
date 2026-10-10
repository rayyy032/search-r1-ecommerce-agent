"""CLI 参数自检：保证每个入口脚本的 parse_args() 在无可选参数时都能跑通。

## 为什么需要这个文件

`pyflakes` / `mypy` 都抓不到一类错误：**代码里用了 `args.xxx`，但 parser 里从没
`add_argument("--xxx")`**。这类错误只在真正调用 `parse_args()` 时才暴露。

历史上就这么漏过两次：
- `train.py` / `eval.py` 引用了 `args.reward_config`，但两个 parser 都没定义
  `--reward-config` → 训练/评测跑完全程后，写 summary 那一刻才 `AttributeError` 崩；
- `rollout.py` 的 `score_trajectory` 签名与调用点不一致 → 跑到评分阶段才 `TypeError`。

本文件只做「参数解析 + 关键属性存在性」检查：**不需要 GPU、不联网、不加载模型**。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (模块名, 必须补的最小参数, 该 CLI 必须在解析结果里出现的属性)
CLI_ENTRIES = [
    ("train", ["--max-steps", "1"], ["reward_config", "run_name", "search_backend"]),
    ("eval", [], ["reward_config", "output", "top_k", "search_backend"]),
    ("sft", [], ["questions", "samples_per_question", "min_reward", "run_name"]),
]


@pytest.mark.parametrize("module_name,argv_extra,required_attrs", CLI_ENTRIES)
def test_parse_args_smoke(
    module_name: str, argv_extra: list[str], required_attrs: list[str], monkeypatch
) -> None:
    """入口脚本的 parse_args() 必须能解析，且关键属性存在。"""
    monkeypatch.setattr(sys, "argv", [f"{module_name}.py", *argv_extra])
    module = importlib.import_module(module_name)
    args = module.parse_args()

    missing = [name for name in required_attrs if not hasattr(args, name)]
    assert not missing, (
        f"{module_name}.py 的 parse_args() 缺少属性 {missing}；"
        "通常是引用了 args.xxx 但 parser 里没定义 --xxx"
    )


@pytest.mark.parametrize(
    "module_name,argv_extra", [("train", ["--max-steps", "1"]), ("eval", [])]
)
def test_reward_config_defaults_to_full(
    module_name: str, argv_extra: list[str], monkeypatch
) -> None:
    """奖励消融默认值必须是线上配置 full，避免静默改变实验口径。"""
    from reward import DEFAULT_WEIGHTS, resolve_weights

    monkeypatch.setattr(sys, "argv", [f"{module_name}.py", *argv_extra])
    args = importlib.import_module(module_name).parse_args()

    assert args.reward_config == "full"
    assert resolve_weights(args.reward_config) is DEFAULT_WEIGHTS
