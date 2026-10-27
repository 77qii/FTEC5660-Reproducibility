"""把多轮的 rescore JSON 聚合成"论文口径"的 mean ± std 表。

配套 rescore_c5.py --json 使用：
    ./venv/bin/python rescore_c5.py result_a.json --json
    ./venv/bin/python rescore_c5.py result_b.json --json
    ./venv/bin/python aggregate_legacy.py result_a_rescore.json result_b_rescore.json
"""
import json
import statistics
import sys

REPORTED = {"Multi-Hop": 27.02, "Temporal": 45.85, "Open-Domain": 12.14,
            "Single-Hop": 44.65, "Adversarial": 50.03}
ORDER = ["Multi-Hop", "Temporal", "Open-Domain", "Single-Hop", "Adversarial", "Overall"]

# rescore_c5.py 输出的键是小写连字符，这里做一次映射
KEYMAP = {"Multi-Hop": "multi-hop", "Temporal": "temporal", "Open-Domain": "open-domain",
          "Single-Hop": "single-hop", "Adversarial": "adversarial", "Overall": "Overall"}


def main(paths):
    trials = [json.load(open(p)) for p in paths]
    n = len(trials)
    print(f"\n## 论文口径（legacy_c5_reference=True）聚合，{n} 轮\n")
    print("| Category | Reported | Mine (paper's reference) | Δ |")
    print("|---|---|---|---|")
    for name in ORDER:
        k = KEYMAP[name]
        vals = [t["old"][k] for t in trials if k in t["old"]]
        if not vals:
            continue
        cell = (f"{vals[0]:.2f} (single)" if len(vals) == 1
                else f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}")
        rep = REPORTED.get(name)
        if rep is None:
            print(f"| {name} | — | {cell} | — |")
        else:
            print(f"| {name} | {rep:.2f} | {cell} | {statistics.mean(vals)-rep:+.2f} |")
    print("\n注：论文的 Adversarial 列用错误答案当参考答案，衡量的是与 trap 答案的 token 重叠；"
          "在样本上 trap overlap 低与成功拒答相关，但不能替代 refusal accuracy。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
