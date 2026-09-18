"""从训练日志提取的逐步指标重绘训练曲线（与 SwanLab 云端数据同源）。

数据文件: docs/training_data/train-{ecom,baseline}-steps.txt
每行格式: step=1/20 step_time=29.9s loss_mean=0.0068 mean_reward=1.096
           correct_rate=0.688 mean_search_calls=1.28 input_tokens=43802
           loss_tokens=2932 padded_tokens=57968
（loss_mean=skipped 表示该步全组零方差，被 Dynamic Sampling 跳过）

用法: python plot_curves.py [--out-dir docs/images]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

KV_RE = re.compile(r"(\w+)=([0-9.e+-]+|skipped)")


def parse(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        kv = dict(KV_RE.findall(line))
        if "step" not in kv:
            continue
        kv["step"] = int(kv["step"].split("/")[0])
        for k in ("mean_reward", "correct_rate", "mean_search_calls"):
            kv[k] = float(kv[k])
        kv["loss_mean"] = float(kv["loss_mean"]) if kv["loss_mean"] != "skipped" else None
        rows.append(kv)
    return rows


def plot_run(rows: list[dict], title: str, out: Path) -> None:
    steps = [r["step"] for r in rows]
    reward = [r["mean_reward"] for r in rows]
    correct = [r["correct_rate"] for r in rows]
    searches = [r["mean_search_calls"] for r in rows]
    loss_steps = [r["step"] for r in rows if r["loss_mean"] is not None]
    loss = [r["loss_mean"] for r in rows if r["loss_mean"] is not None]
    n_skipped = len(rows) - len(loss)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), dpi=160)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def style(ax, ylabel):
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlabel("训练步数 (step)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, 21, 2))

    ax = axes[0][0]
    ax.plot(steps, reward, "o-", color="#2563eb", lw=1.8, ms=4)
    ax.annotate(f"{reward[-1]:.2f}", (steps[-1], reward[-1]),
                textcoords="offset points", xytext=(-4, 8), color="#2563eb", fontsize=9)
    style(ax, "组内平均奖励 mean_reward")

    ax = axes[0][1]
    ax.plot(steps, correct, "s-", color="#16a34a", lw=1.8, ms=4)
    ax.set_ylim(-0.03, 1.05)
    ax.annotate(f"{correct[-1]*100:.0f}%", (steps[-1], correct[-1]),
                textcoords="offset points", xytext=(-14, 8), color="#16a34a", fontsize=9)
    style(ax, "答案正确率 correct_rate")

    ax = axes[1][0]
    ax.plot(steps, searches, "^-", color="#d97706", lw=1.8, ms=4)
    style(ax, "平均搜索次数 mean_search_calls")

    ax = axes[1][1]
    ax.plot(loss_steps, loss, "o-", color="#64748b", lw=1.8, ms=4)
    if n_skipped:
        ax.set_title(f"{n_skipped} 步零方差被 Dynamic Sampling 跳过", fontsize=9, color="#94a3b8")
    style(ax, "策略损失 loss_mean")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/images")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("docs/training_data/train-baseline-steps.txt",
         "Wikipedia 基线复现 · GRPO 20 步（search-r1-baseline-2）", "curves-baseline.png"),
        ("docs/training_data/train-ecom-steps.txt",
         "中文电商客服场景 · GRPO 20 步（ecom-r1-20step-4）", "curves-ecom.png"),
    ]
    for data, title, name in runs:
        rows = parse(Path(data))
        plot_run(rows, title, out_dir / name)


if __name__ == "__main__":
    main()
