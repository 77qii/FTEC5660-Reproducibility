"""量化 Temporal 类别的"形式错配"：预测用相对时间表达、金标准是绝对日期。

不调用 LLM，只读已存下的预测，所以随时可重跑。
用法：
    ./venv/bin/python analyze_temporal.py A-mem/result_conv26_new.json \
        A-mem/result_conv26_seed43.json A-mem/result_conv26_seed44.json
"""
import json
import os
import re
import sys

REL = re.compile(
    r"\b(yesterday|today|tomorrow|tonight|last night|last week|last month|last year|"
    r"next week|next month|next year|this week|this month|this year|the other day|"
    r"a few days ago|recently|just now)\b",
    re.I,
)
ABS = re.compile(
    r"\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.I,
)


def main(paths):
    print(f"{'trial file':30s} {'n':>4s} {'错配':>5s} {'占比':>7s} "
          f"{'错配F1':>8s} {'其余F1':>8s} {'整体F1':>8s}")
    print("-" * 78)
    for p in paths:
        d = json.load(open(p))
        rows = [r for r in d["individual_results"] if int(r["category"]) == 2]
        mis, ok = [], []
        for r in rows:
            f1 = float(r["metrics"]["f1"])
            if REL.search(str(r["prediction"] or "")) and ABS.search(str(r["reference"] or "")):
                mis.append(f1)
            else:
                ok.append(f1)
        n = len(rows)
        print(f"{os.path.basename(p):30s} {n:4d} {len(mis):5d} {len(mis)/n*100:6.1f}% "
              f"{sum(mis)/len(mis)*100:8.2f} {sum(ok)/len(ok)*100:8.2f} "
              f"{(sum(mis)+sum(ok))/n*100:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
