"""生成电商客服场景的合成知识库与问答集。

设计要点：
1. 品牌 / 型号全部虚构（蓝曜、青岚、沐光……），模型无法依赖参数化知识作答，
   必须调用检索工具，避免知识泄漏污染 RL 信号。
2. 文档分两类：商品手册（P 开头编号）与服务政策（S 开头编号）。
3. 问答分三类：单跳事实题（ecommerce_single）、政策题（ecommerce_policy）、
   多跳题（ecommerce_multihop：延保叠加、激活限制、两款商品对比）。
4. 每题带 gold_docs：答案依据的文档编号，供引用可溯源奖励校验。

输出：datasets/ecommerce/{docs,train,dev}.jsonl（dev 固定 70 题）。
"""

import itertools
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "ecommerce"
DEV_SIZE = 70
SEED = 42

# ----------------------------- 服务政策文档 -----------------------------

POLICIES: list[dict] = [
    {
        "doc_id": "S001",
        "title": "七天无理由退货政策",
        "content": (
            "签收后 7 天内、商品完好不影响二次销售的，可申请七天无理由退货。"
            "定制商品、激活后的手机不支持七天无理由退货。无理由退货的往返运费由买家承担；"
            "订单带有运费险的，首重（1kg 以内）运费由平台承担。"
        ),
    },
    {
        "doc_id": "S002",
        "title": "运费险说明",
        "content": (
            "带有运费险的订单发生退换货时，首重运费（1kg 以内）由平台承担，超出首重的部分由买家承担。"
            "理赔金额在退款完成后 72 小时内到账，可在订单页查看。"
        ),
    },
    {
        "doc_id": "S003",
        "title": "价格保护规则",
        "content": (
            "签收后 30 天内，同一商品出现降价的，可在订单详情页申请价保补差价。"
            "大促秒杀、使用优惠券造成的差价不支持价保。"
        ),
    },
    {
        "doc_id": "S004",
        "title": "延长保修服务条款",
        "content": (
            "支付 199 元可购买延长保修服务，整机保修时长在原厂保修基础上延长 1 年，"
            "与原厂保修连续计算，每个订单限购一次。人为损坏、进液、私自拆机不在延保范围内。"
        ),
    },
    {
        "doc_id": "S005",
        "title": "三包通用规则",
        "content": (
            "保修期自商品签收次日起计算。签收后 7 天内出现性能故障可退货，"
            "15 天内出现性能故障可换货，超过 15 天按保修政策处理。"
        ),
    },
    {
        "doc_id": "S006",
        "title": "发货时效说明",
        "content": (
            "现货商品付款后 48 小时内发出；新疆、西藏等偏远地区 72 小时内发出；"
            "预售商品的发货时间以商品页面标注为准。"
        ),
    },
    {
        "doc_id": "S007",
        "title": "上门安装服务说明",
        "content": (
            "空调、洗衣机提供免费上门安装，安装人工费全免；"
            "支架、加长铜管、专用水龙头等辅材需另行收费。"
        ),
    },
    {
        "doc_id": "S008",
        "title": "发票开具规则",
        "content": (
            "签收后可在订单页申请电子发票，支持个人或企业抬头。"
            "发票内容按实际购买商品开具，不支持变更发票类目。"
        ),
    },
]

# ----------------------------- 商品规格配置 -----------------------------
# 每个品类：保修年限、型号字母、属性生成器（field, 标签, 取值函数, 单位, 问题问法）

BRANDS = ["蓝曜", "青岚", "沐光", "赤焰", "墨羽", "云穗"]
SUFFIXES = ["Pro", "Max", "青春版", "旗舰版", "SE"]


def build_catalog(rng: random.Random) -> list[dict]:
    """按品类生成虚构商品，数值属性在品类内刻意拉开差距，保证对比题有唯一答案。"""
    products: list[dict] = []

    def numeric(lo: int, hi: int, step: int) -> int:
        return rng.randrange(lo, hi + 1, step)

    def add(category: str, code_letter: str, warranty: int, n: int, specs: list[dict]) -> None:
        rng.shuffle(BRANDS)
        for i in range(n):
            brand = BRANDS[i % len(BRANDS)]
            suffix = SUFFIXES[(i * 2 + len(products)) % len(SUFFIXES)]
            model = f"{brand}{code_letter}{i + 1} {suffix}".replace(" ", "")
            values = {spec["field"]: spec["gen"](rng) for spec in specs}
            products.append(
                {
                    "doc_id": f"P{len(products) + 1:03d}",
                    "category": category,
                    "brand": brand,
                    "model": model,
                    "warranty": warranty,
                    "specs": specs,
                    "values": values,
                }
            )

    add(
        "手机", "T", 1, 5,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(1999, 6999, 100), "ask": "售价是多少元"},
            {"field": "battery", "label": "电池容量", "unit": "mAh", "gen": lambda r: numeric(4200, 5600, 100), "ask": "电池容量是多少"},
            {"field": "charge", "label": "有线快充功率", "unit": "W", "gen": lambda r: r.choice([33, 66, 90, 120]), "ask": "有线快充功率是多少瓦"},
            {"field": "screen", "label": "屏幕尺寸", "unit": "英寸", "gen": lambda r: r.choice([6.1, 6.3, 6.5, 6.7]), "ask": "屏幕尺寸是多少英寸"},
            {"field": "weight", "label": "重量", "unit": "g", "gen": lambda r: numeric(165, 230, 1), "ask": "重量是多少克"},
            {"field": "wireless", "label": "无线充电", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持无线充电吗"},
            {"field": "waterproof", "label": "防水等级", "unit": None, "gen": lambda r: r.choice(["IP68", "IP65", "生活防水"]), "ask": "防水等级是什么"},
        ],
    )
    add(
        "洗衣机", "W", 3, 5,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(1599, 5999, 100), "ask": "售价是多少元"},
            {"field": "capacity", "label": "洗涤容量", "unit": "kg", "gen": lambda r: r.choice([8, 9, 10, 12]), "ask": "洗涤容量是多少公斤"},
            {"field": "noise", "label": "洗涤噪音", "unit": "dB", "gen": lambda r: numeric(42, 58, 1), "ask": "洗涤噪音是多少分贝"},
            {"field": "airwash", "label": "空气洗", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持空气洗吗"},
            {"field": "washer_dryer", "label": "洗烘一体", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持洗烘一体吗"},
        ],
    )
    add(
        "空调", "A", 6, 5,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(2199, 7999, 100), "ask": "售价是多少元"},
            {"field": "horsepower", "label": "匹数", "unit": "匹", "gen": lambda r: r.choice([1.0, 1.5, 2.0, 3.0]), "ask": "是几匹的"},
            {"field": "apf", "label": "APF 能效比", "unit": None, "gen": lambda r: r.choice([3.6, 4.0, 4.5, 5.0]), "ask": "APF 能效比是多少"},
            {"field": "self_clean", "label": "自清洁", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持自清洁吗"},
            {"field": "fresh_air", "label": "新风功能", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持新风功能吗"},
        ],
    )
    add(
        "笔记本电脑", "L", 2, 5,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(3999, 9999, 100), "ask": "售价是多少元"},
            {"field": "memory", "label": "内存", "unit": "GB", "gen": lambda r: r.choice([8, 16, 32]), "ask": "内存是多少GB"},
            {"field": "battery_life", "label": "续航时间", "unit": "小时", "gen": lambda r: numeric(6, 18, 1), "ask": "续航时间是多少小时"},
            {"field": "weight", "label": "重量", "unit": "kg", "gen": lambda r: r.choice([1.2, 1.4, 1.6, 1.8, 2.0]), "ask": "重量是多少公斤"},
            {"field": "refresh", "label": "屏幕刷新率", "unit": "Hz", "gen": lambda r: r.choice([60, 90, 120, 144]), "ask": "屏幕刷新率是多少Hz"},
        ],
    )
    add(
        "蓝牙耳机", "E", 1, 4,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(199, 1499, 50), "ask": "售价是多少元"},
            {"field": "anc", "label": "降噪深度", "unit": "dB", "gen": lambda r: r.choice([25, 32, 40, 48]), "ask": "降噪深度是多少分贝"},
            {"field": "total_battery", "label": "总续航", "unit": "小时", "gen": lambda r: numeric(18, 40, 2), "ask": "配合充电盒的总续航是多少小时"},
            {"field": "waterproof", "label": "防水等级", "unit": None, "gen": lambda r: r.choice(["IPX4", "IPX5", "IPX7"]), "ask": "防水等级是什么"},
        ],
    )
    add(
        "扫地机器人", "R", 2, 4,
        [
            {"field": "price", "label": "售价", "unit": "元", "gen": lambda r: numeric(1299, 5499, 100), "ask": "售价是多少元"},
            {"field": "suction", "label": "吸力", "unit": "Pa", "gen": lambda r: r.choice([4000, 6000, 8000, 12000]), "ask": "吸力是多少Pa"},
            {"field": "battery", "label": "电池容量", "unit": "mAh", "gen": lambda r: numeric(3200, 6400, 200), "ask": "电池容量是多少"},
            {"field": "navigation", "label": "避障方式", "unit": None, "gen": lambda r: r.choice(["激光避障", "视觉避障", "机械避障"]), "ask": "用的是哪种避障方式"},
            {"field": "auto_empty", "label": "自动集尘", "unit": None, "gen": lambda r: r.choice(["支持", "不支持"]), "ask": "支持自动集尘吗"},
        ],
    )
    return products


def render_product_doc(p: dict) -> str:
    """把商品属性渲染成手册式中文段落（知识库文档正文）。"""
    lines = [
        f"{p['model']}是{p['brand']}品牌推出的一款{p['category']}产品。",
        f"售后政策：整机保修 {p['warranty']} 年。核心参数如下：",
    ]
    for spec in p["specs"]:
        value = p["values"][spec["field"]]
        unit = spec["unit"] or ""
        lines.append(f"- {spec['label']}：{value}{unit}")
    return "\n".join(lines)


# ----------------------------- 问答生成 -----------------------------

def numeric_answers(value: object, unit: str | None) -> list[str]:
    """为数值答案生成多种等价表述，适配 EM 归一化。"""
    text = str(value)
    answers = {text}
    if unit:
        answers.add(f"{text}{unit}")
    if unit == "mAh":
        answers.add(f"{text}毫安时")
    return list(answers)


def build_questions(products: list[dict]) -> list[dict]:
    qa: list[dict] = []

    def add_qa(question: str, answers: list[str], source: str, gold: list[str]) -> None:
        qa.append(
            {
                "id": f"ecom_{len(qa):04d}",
                "question": question,
                "answers": answers,
                "data_source": source,
                "gold_docs": gold,
            }
        )

    # 1) 单跳商品事实题：每件商品取前 3 个数值属性 + 全部布尔 / 选择属性提问
    for p in products:
        numeric_specs = [s for s in p["specs"] if s["unit"]][:3]
        choice_specs = [s for s in p["specs"] if not s["unit"]]
        for spec in [*numeric_specs, *choice_specs]:
            value = p["values"][spec["field"]]
            if spec["unit"]:
                add_qa(f"{p['model']}的{spec['label']}是多少？",
                       numeric_answers(value, spec["unit"]),
                       "ecommerce_single", [p["doc_id"]])
            elif spec["ask"].endswith("吗"):
                add_qa(f"{p['model']}{spec['ask']}？",
                       [str(value)], "ecommerce_single", [p["doc_id"]])
            else:
                add_qa(f"{p['model']}的{spec['label']}是什么？",
                       [str(value)], "ecommerce_single", [p["doc_id"]])

    # 2) 政策题（围绕同一政策文档的不同条款提问）
    policy_qa = [
        ("七天无理由退货需要在签收后几天内申请？", ["7天", "7"], ["S001"]),
        ("无理由退货时没有运费险的订单运费谁承担？", ["买家承担", "买家"], ["S001"]),
        ("有运费险的订单退换货，首重运费谁承担？", ["平台承担", "平台"], ["S001", "S002"]),
        ("价保期是签收后多少天？", ["30天", "30"], ["S003"]),
        ("购买延长保修服务需要多少钱？", ["199元", "199"], ["S004"]),
        ("延保服务把整机保修延长多久？", ["1年", "1"], ["S004"]),
        ("商品保修期从哪一天开始计算？", ["签收次日"], ["S005"]),
        ("现货商品付款后多久发货？", ["48小时"], ["S006"]),
        ("空调免费上门安装时，辅材费用怎么算？", ["另行收费", "收费"], ["S007"]),
        ("签收后可以申请开具什么类型的发票？", ["电子发票"], ["S008"]),
    ]
    for question, answers, gold in policy_qa:
        add_qa(question, answers, "ecommerce_policy", gold)

    # 3a) 多跳：原厂保修 + 延保叠加
    for p in products:
        total = p["warranty"] + 1
        add_qa(f"购买延保服务后，{p['model']}的整机保修总时长是多久？",
               [f"{total}年", str(total)],
               "ecommerce_multihop", [p["doc_id"], "S004"])

    # 3b) 多跳：手机激活后不支持无理由（商品 + 政策）
    for p in products:
        if p["category"] == "手机":
            add_qa(f"{p['model']}激活之后还能七天无理由退货吗？",
                   ["不能", "不支持"],
                   "ecommerce_multihop", [p["doc_id"], "S001"])

    # 3c) 多跳：同品类两款商品数值属性对比
    compare_attrs = {
        "手机": [("battery", "电池容量", "更大"), ("price", "售价", "更便宜"), ("weight", "重量", "更轻")],
        "洗衣机": [("capacity", "洗涤容量", "更大"), ("price", "售价", "更便宜")],
        "空调": [("apf", "APF 能效比", "更高"), ("price", "售价", "更便宜")],
        "笔记本电脑": [("battery_life", "续航", "更长"), ("weight", "重量", "更轻")],
        "蓝牙耳机": [("anc", "降噪深度", "更高"), ("total_battery", "总续航", "更长")],
        "扫地机器人": [("suction", "吸力", "更大"), ("price", "售价", "更便宜")],
    }
    by_category: dict[str, list[dict]] = {}
    for p in products:
        by_category.setdefault(p["category"], []).append(p)
    for category, items in by_category.items():
        for field, label, comparator in compare_attrs[category]:
            ordered = sorted(items, key=lambda x: x["values"][field])
            for lo, hi in itertools.combinations(ordered, 2):
                if lo["values"][field] == hi["values"][field]:
                    continue  # 数值相同则对比题无唯一答案，跳过
                if comparator == "更便宜" or comparator == "更轻":
                    winner = lo  # 价格 / 重量越小越好
                else:
                    winner = hi
                add_qa(f"{lo['model']}和{hi['model']}哪个{label}{comparator}？",
                       [winner["model"]],
                       "ecommerce_multihop", [lo["doc_id"], hi["doc_id"]])

    return qa


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    products = build_catalog(rng)

    docs = [
        {
            "doc_id": p["doc_id"],
            "title": f"{p['model']} 商品手册",
            "category": p["category"],
            "content": render_product_doc(p),
        }
        for p in products
    ] + POLICIES

    questions = build_questions(products)
    rng.shuffle(questions)
    dev = questions[:DEV_SIZE]
    train = questions[DEV_SIZE:]

    for name, rows in [("docs", docs), ("train", train), ("dev", dev)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(rows)} 条")

    sources = {q["data_source"] for q in questions}
    print(f"文档 {len(docs)} 篇（商品 {len(products)} + 政策 {len(POLICIES)}）；"
          f"问题来源：{sorted(sources)}")


if __name__ == "__main__":
    main()
