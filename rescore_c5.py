"""同一批预测，两套 cat-5 打分口径，看指标会差多少。

这是纯后处理：读已经存下来的 predictions，用 legacy_c5=False / True 各打一遍分，
不调用任何 LLM，所以零成本、可重复。

用法：
    ./venv/bin/python rescore_c5.py A-mem/dry_run_result.json
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from load_dataset import load_locomo_dataset  # noqa: E402
from utils import score_prediction  # noqa: E402


def main(result_path):
    res = json.load(open(result_path))
    samples = load_locomo_dataset("data/locomo10.json")

    # 结果里的 sample_id 是"被评测的样本序号"；ratio<1 时只评了前 N 段，
    # 所以只对结果里真正出现过的样本建索引。
    used = {row["sample_id"] for row in res["individual_results"]}
    flat = []
    for si, sample in enumerate(samples):
        if si not in used:
            continue
        for qa in sample.qa:
            if int(qa.category) in (1, 2, 3, 4, 5):
                flat.append((si, qa))

    # 结果文件里的顺序与评测循环一致
    preds = res["individual_results"]
    assert len(preds) == len(flat), f"结果 {len(preds)} 条 vs 数据集 {len(flat)} 条，对不上"

    buckets = {}
    for (si, qa), row in zip(flat, preds):
        assert row["question"] == qa.question, "问题顺序错位，无法可靠重打分"
        _, m_new = score_prediction(row["prediction"], qa, legacy_c5=False)
        _, m_old = score_prediction(row["prediction"], qa, legacy_c5=True)
        b = buckets.setdefault(qa.category, {"new": [], "old": []})
        b["new"].append(m_new["f1"])
        b["old"].append(m_old["f1"])

    names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    print(f"\n文件: {result_path}")
    print(f"题数: {len(preds)}\n")
    print(f"{'类别':14s} {'n':>4s} {'新口径 F1':>11s} {'论文口径 F1':>13s} {'差值':>9s}")
    print("-" * 56)
    tot_new = tot_old = 0.0
    for cat in sorted(buckets):
        n = len(buckets[cat]["new"])
        fn = sum(buckets[cat]["new"]) / n
        fo = sum(buckets[cat]["old"]) / n
        tot_new += sum(buckets[cat]["new"])
        tot_old += sum(buckets[cat]["old"])
        print(f"{names[cat]:14s} {n:4d} {fn*100:11.2f} {fo*100:13.2f} {(fn-fo)*100:+9.2f}")
    N = len(preds)
    print("-" * 56)
    print(f"{'overall':14s} {N:4d} {tot_new/N*100:11.2f} {tot_old/N*100:13.2f} "
          f"{(tot_new-tot_old)/N*100:+9.2f}")

    # 机器可读输出，供 aggregate_legacy.py 聚合多轮
    if "--json" in sys.argv:
        out = {"file": result_path, "n": N,
               "new": {names[c]: sum(buckets[c]["new"]) / len(buckets[c]["new"]) * 100
                       for c in buckets},
               "old": {names[c]: sum(buckets[c]["old"]) / len(buckets[c]["old"]) * 100
                       for c in buckets}}
        out["new"]["Overall"] = tot_new / N * 100
        out["old"]["Overall"] = tot_old / N * 100
        jpath = result_path.replace(".json", "") + "_rescore.json"
        with open(jpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n已写入: {jpath}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dry_run_result.json")
