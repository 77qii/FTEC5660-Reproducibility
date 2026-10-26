"""抓 BERTScore 基准值：改 utils.py 之前先存一份，改完逐位比对，
证明"只是变快，数值一模一样"。
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from utils import calculate_bert_scores  # noqa: E402

PAIRS = [
    ("7 May 2023", "7 May 2023"),
    ("8 May 2023", "7 May 2023"),
    ("Not mentioned in the conversation", "Not mentioned in the conversation"),
    ("playing guitar", "playing the guitar with friends"),
    ("a support group", "a support group for LGBTQ people"),
    ("", "7 May 2023"),
    ("Paris", "London"),
    ("she went to the beach last summer", "last summer she visited the beach"),
]

out = []
for pred, ref in PAIRS:
    s = calculate_bert_scores(pred, ref)
    out.append({"pred": pred, "ref": ref, "scores": s})
    print(f"{pred[:40]!r:44} vs {ref[:30]!r:34} -> "
          f"P={s['bert_precision']:.10f} R={s['bert_recall']:.10f} F={s['bert_f1']:.10f}")

tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bert_baseline_{tag}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n已保存: {path}")
