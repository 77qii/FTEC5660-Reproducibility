"""把多次 trial 的结果聚合成 mean ± std，直接输出可粘贴进报告的 Markdown 表格。

用法：
    ./venv/bin/python aggregate_trials.py result_t1.json result_t2.json result_t3.json
    ./venv/bin/python aggregate_trials.py --reported result_*.json
"""
import json
import os
import statistics
import sys

CATS = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}

# 论文 Table 1，GPT-4o-mini 骨干
REPORTED = {"Multi-Hop": 27.02, "Temporal": 45.85, "Open-Domain": 12.14,
            "Single-Hop": 44.65, "Adversarial": 50.03}


def load(path):
    d = json.load(open(path))
    out = {}
    for cat, name in CATS.items():
        key = f"category_{cat}"
        m = d["aggregate_metrics"].get(key, {}).get("f1", {})
        if m:
            out[name] = (m["mean"] * 100, int(m["count"]))
    overall = d["aggregate_metrics"].get("overall", {}).get("f1", {})
    if overall:
        out["Overall"] = (overall["mean"] * 100, int(overall["count"]))
    out["_meta"] = {
        "model": d.get("model"),
        "ratio": d.get("ratio"),
        "seed": d.get("seed"),
        "legacy_c5": d.get("legacy_c5_reference"),
        "failures": d.get("llm_failures") or {},
    }
    return out


def main(paths):
    trials = [load(p) for p in paths]
    n = len(trials)
    meta = trials[0]["_meta"]

    print(f"\n## Trials: {n}  |  model: {meta['model']}  |  ratio: {meta['ratio']}  "
          f"|  legacy_c5: {meta['legacy_c5']}")
    for i, t in enumerate(trials, 1):
        print(f"  trial {i}: seed={t['_meta']['seed']}  failures={t['_meta']['failures'] or 'none'}")

    print(f"\n| Category | n | Reported (GPT-4o-mini) | Mine (mean ± std) | Δ |")
    print("|---|---|---|---|---|")

    names = [CATS[c] for c in (1, 2, 3, 4, 5)] + ["Overall"]
    legacy = meta["legacy_c5"]
    for name in names:
        vals, counts = [], []
        for t in trials:
            if name in t:
                vals.append(t[name][0])
                counts.append(t[name][1])
        if not vals:
            continue
        cnt = counts[0]
        if len(vals) == 1:
            cell = f"{vals[0]:.2f} (single trial)"
        else:
            cell = f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"
        rep = REPORTED.get(name)
        # 论文的 Adversarial 列是按"错误答案"当参考答案算的（见报告 §5）。
        # 只有用 --legacy_c5_reference 跑出来的数才能和它比；新口径不能直接对照。
        if name == "Adversarial" and not legacy:
            print(f"| {name} | {cnt} | {rep:.2f} (paper's reference) | "
                  f"{cell} (corrected reference) | ⚠ not comparable |")
            continue
        if rep is None:
            print(f"| {name} | {cnt} | — (not reported) | {cell} | — |")
        else:
            d = statistics.mean(vals) - rep
            print(f"| {name} | {cnt} | {rep:.2f} | {cell} | {d:+.2f} |")

    print("\n注：Overall 论文未报告，不能直接对照。")
    if not legacy:
        print("注：Adversarial 行口径不同（论文用错误答案当参考答案），Δ 不可算。"
              "用 --legacy_c5_reference 重跑后再比。")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--reported"]
    if not args:
        print(__doc__)
        sys.exit(1)
    main(args)
