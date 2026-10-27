#!/usr/bin/env python3
"""Surface the category-5 (adversarial) refusal metric that the evaluator computes but never reports.

`utils.score_prediction` writes `adversarial_correct` into every category-5 question's metric dict,
and `aggregate_metrics` carries it into `aggregate_metrics["category_5"]`. Nothing downstream reads
it: every table in the paper, the README and the report's main results table reports token-level F1
instead. On category 5 that F1 is scored against a reference string that the model is *supposed* to
contradict, so it measures string overlap with the trap, not whether the model handled the trap.

This script reads the stored per-question predictions (no API calls, no model downloads) and:

  1. re-derives `adversarial_correct` from scratch, independently of utils.py, and asserts it matches
     the stored value on every question -- so a drift in either implementation shows up as a failure
     rather than as a quietly different number;
  2. reports the refusal rate per trial with a Wilson 95% interval, plus mean +/- std across trials;
  3. decomposes the failures into mutually exclusive behavioural modes;
  4. cross-tabulates token F1 against the binary verdict under both reference conventions, which is
     where the two instruments turn out to disagree.

Usage
-----
    python analyze_c5.py result_conv26_new.json result_conv26_seed43.json result_conv26_seed44.json
    python analyze_c5.py --json c5.json --md c5.md result_conv26_*.json

Exit status is non-zero if any self-check fails.
"""

import argparse
import glob
import json
import math
import statistics
import sys

# load_dataset.NOT_MENTIONED -- the correct answer to a category-5 question
NOT_MENTIONED = "Not mentioned in the conversation"

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- metrics
# Both helpers below are re-implementations of utils.py, on purpose. If the two ever disagree the
# assertion in load_trial() fires; importing utils.py instead would make the check vacuous (and
# pull in torch).

def tokenize(text):
    """utils.simple_tokenize, verbatim."""
    return str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def token_f1(prediction, reference):
    """utils.calculate_metrics' `f1`: set-overlap F1 over lowercased word tokens."""
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def refusal_verdict(prediction, trap):
    """Split utils.is_not_mentioned into its two clauses so failures can be attributed.

    Returns (declines, gives_trap, correct). `correct` reproduces the stored
    `adversarial_correct` flag exactly: the model must say the answer is not in the conversation
    and must not also hand back the trap answer.
    """
    text = str(prediction).lower()
    declines = ("not mentioned" in text) or ("no information available" in text)
    trap_norm = str(trap or "").lower().strip(" .")
    gives_trap = bool(trap_norm) and not trap_norm.startswith("not mentioned") and trap_norm in text
    return declines, gives_trap, int(declines and not gives_trap)


def wilson(k, n, z=Z95):
    """95% Wilson score interval -- the right interval for n=47, where normal-approx is too narrow."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- loading

def load_trial(path):
    """Load one result file, keep category 5, and assert the stored numbers are reproducible."""
    with open(path) as fh:
        data = json.load(fh)

    if data.get("legacy_c5_reference"):
        sys.exit(
            f"{path}: legacy_c5_reference is true. That run scored category 5 against the trap "
            f"answer, so 'reference' is not the refusal string and this analysis does not apply."
        )

    rows = []
    for r in data.get("individual_results", []):
        if int(r.get("category", 0)) != 5:
            continue

        prediction = r.get("prediction")
        reference = r.get("reference")
        trap = r.get("adversarial_answer")
        metrics = r.get("metrics", {})

        if reference != NOT_MENTIONED:
            sys.exit(f"{path}: category-5 reference is {reference!r}, expected {NOT_MENTIONED!r}")

        if "adversarial_correct" not in metrics:
            sys.exit(f"{path}: no adversarial_correct on a category-5 question -- older result file?")

        # (1) the stored F1 must be reproducible from the stored prediction and reference
        recalc_f1 = token_f1(prediction, reference)
        if abs(recalc_f1 - float(metrics["f1"])) > 1e-9:
            sys.exit(
                f"{path}: f1 mismatch on {r.get('question', '?')[:60]!r} -- "
                f"stored {metrics['f1']:.6f}, recomputed {recalc_f1:.6f}"
            )

        # (2) the stored flag must be reproducible from the stored prediction and trap
        declines, gives_trap, correct = refusal_verdict(prediction, trap)
        if correct != int(metrics["adversarial_correct"]):
            sys.exit(
                f"{path}: adversarial_correct mismatch on {r.get('question', '?')[:60]!r} -- "
                f"stored {metrics['adversarial_correct']}, recomputed {correct}"
            )

        rows.append({
            "seed": data.get("seed"),
            "question": r.get("question"),
            "prediction": prediction,
            "trap": trap,
            "declines": declines,
            "gives_trap": gives_trap,
            "correct": correct,
            "f1_refusal": recalc_f1,                       # reference = NOT_MENTIONED
            "f1_trap": token_f1(prediction, trap),         # reference = adversarial answer
            "mode": classify(declines, gives_trap),
        })

    if not rows:
        sys.exit(f"{path}: no category-5 questions found")
    return data.get("seed"), rows


def classify(declines, gives_trap):
    """Mutually exclusive failure modes. `clean_refusal` is the only success mode."""
    if declines and gives_trap:
        return "declined_but_restated_trap"
    if declines:
        return "clean_refusal"
    if gives_trap:
        return "took_the_bait"
    return "other"


MODES = ["clean_refusal", "took_the_bait", "declined_but_restated_trap", "other"]

MODE_NOTE = {
    "clean_refusal": "declines, and does not echo the trap",
    "took_the_bait": "returns the trap answer",
    "declined_but_restated_trap": "declines but also echoes the trap (hedging)",
    "other": "neither declines nor echoes the trap",
}


# --------------------------------------------------------------------------- reporting

def summarize(trials):
    per_trial = []
    for seed, rows in trials:
        n = len(rows)
        k = sum(r["correct"] for r in rows)
        lo, hi = wilson(k, n)
        per_trial.append({
            "seed": seed,
            "n": n,
            "correct": k,
            "rate": 100.0 * k / n,
            "ci_low": 100.0 * lo,
            "ci_high": 100.0 * hi,
            "f1_refusal": 100.0 * statistics.mean(r["f1_refusal"] for r in rows),
            "f1_trap": 100.0 * statistics.mean(r["f1_trap"] for r in rows),
            "modes": {m: sum(1 for r in rows if r["mode"] == m) for m in MODES},
        })

    pooled_rows = [r for _, rows in trials for r in rows]
    K, N = sum(r["correct"] for r in pooled_rows), len(pooled_rows)
    lo, hi = wilson(K, N)
    rates = [t["rate"] for t in per_trial]

    return {
        "per_trial": per_trial,
        "pooled": {
            "n": N,
            "correct": K,
            "rate": 100.0 * K / N,
            "ci_low": 100.0 * lo,
            "ci_high": 100.0 * hi,
            "modes": {m: sum(1 for r in pooled_rows if r["mode"] == m) for m in MODES},
        },
        "across_trials": {
            "n_trials": len(per_trial),
            "mean": statistics.mean(rates),
            "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "min": min(rates),
            "max": max(rates),
        },
        "disagreement": disagreement(pooled_rows),
        "rows": pooled_rows,
    }


def disagreement(rows):
    """Where token F1 and the binary verdict point in opposite directions."""
    total = len(rows)
    high_right = [r for r in rows if r["f1_trap"] >= 0.5 and r["correct"]]
    high_wrong = [r for r in rows if r["f1_trap"] >= 0.5 and not r["correct"]]
    low_right = [r for r in rows if r["f1_trap"] < 0.5 and r["correct"]]
    low_wrong = [r for r in rows if r["f1_trap"] < 0.5 and not r["correct"]]
    return {
        "n": total,
        "f1_trap_high_and_right": len(high_right),
        "f1_trap_high_but_wrong": len(high_wrong),
        "f1_trap_low_but_right": len(low_right),
        "f1_trap_low_and_wrong": len(low_wrong),
        "n_disagree": len(high_wrong) + len(low_right),
        "pct_disagree": 100.0 * (len(high_wrong) + len(low_right)) / total,
        "f1_trap_correct": 100.0 * statistics.mean(r["f1_trap"] for r in rows if r["correct"]),
        "f1_trap_incorrect": 100.0 * statistics.mean(r["f1_trap"] for r in rows if not r["correct"]),
        "f1_refusal_correct": 100.0 * statistics.mean(r["f1_refusal"] for r in rows if r["correct"]),
        "f1_refusal_incorrect": 100.0 * statistics.mean(r["f1_refusal"] for r in rows if not r["correct"]),
    }


def render_markdown(s):
    out = []
    a = s["across_trials"]
    out.append("### Refusal accuracy on category 5 (adversarial)")
    out.append("")
    out.append("| Trial | n | Correct | Rate | 95% Wilson CI | F1 vs refusal | F1 vs trap |")
    out.append("|---|---|---|---|---|---|---|")
    for t in s["per_trial"]:
        out.append(
            f"| seed {t['seed']} | {t['n']} | {t['correct']} | {t['rate']:.2f} | "
            f"[{t['ci_low']:.1f}, {t['ci_high']:.1f}] | {t['f1_refusal']:.2f} | {t['f1_trap']:.2f} |"
        )
    p = s["pooled"]
    out.append(
        f"| **pooled** | {p['n']} | {p['correct']} | **{p['rate']:.2f}** | "
        f"[{p['ci_low']:.1f}, {p['ci_high']:.1f}] | — | — |"
    )
    out.append("")
    out.append(
        f"Across {a['n_trials']} trials: **{a['mean']:.2f} +/- {a['std']:.2f}** "
        f"(range {a['min']:.2f}-{a['max']:.2f})."
    )
    out.append("")

    out.append("### What the failures actually are")
    out.append("")
    out.append("| Mode | Questions | Share |")
    out.append("|---|---|---|")
    for m in MODES:
        c = p["modes"][m]
        out.append(f"| `{m}` — {MODE_NOTE[m]} | {c} | {100.0 * c / p['n']:.1f}% |")
    out.append("")

    d = s["disagreement"]
    out.append("### Where token F1 and the binary verdict disagree")
    out.append("")
    out.append("| | F1 vs trap >= 0.5 | F1 vs trap < 0.5 |")
    out.append("|---|---|---|")
    out.append(f"| **handled the trap** | {d['f1_trap_high_and_right']} | {d['f1_trap_low_but_right']} |")
    out.append(f"| **did not** | {d['f1_trap_high_but_wrong']} | {d['f1_trap_low_and_wrong']} |")
    out.append("")
    out.append(
        f"Mean F1 against the trap answer: {d['f1_trap_correct']:.2f} when the model refused, "
        f"{d['f1_trap_incorrect']:.2f} when it did not. "
        f"{d['n_disagree']} of {d['n']} questions ({d['pct_disagree']:.1f}%) fall on the wrong side "
        f"of the 0.5 line under one instrument or the other."
    )
    return "\n".join(out)


def render_console(s):
    a = s["across_trials"]
    p = s["pooled"]
    print("Category 5 (adversarial) -- refusal accuracy")
    print("-" * 78)
    print(f"{'trial':>10} {'n':>4} {'ok':>4} {'rate':>7} {'95% Wilson CI':>18} {'F1|refusal':>11} {'F1|trap':>9}")
    for t in s["per_trial"]:
        print(
            f"{'seed ' + str(t['seed']):>10} {t['n']:>4} {t['correct']:>4} {t['rate']:>6.2f}% "
            f"{'[' + format(t['ci_low'], '.1f') + ', ' + format(t['ci_high'], '.1f') + ']':>18} "
            f"{t['f1_refusal']:>10.2f} {t['f1_trap']:>9.2f}"
        )
    print(
        f"{'pooled':>10} {p['n']:>4} {p['correct']:>4} {p['rate']:>6.2f}% "
        f"{'[' + format(p['ci_low'], '.1f') + ', ' + format(p['ci_high'], '.1f') + ']':>18}"
    )
    print(f"\nacross {a['n_trials']} trials: {a['mean']:.2f} +/- {a['std']:.2f} "
          f"(range {a['min']:.2f}-{a['max']:.2f})")

    print("\nfailure modes")
    print("-" * 78)
    for m in MODES:
        c = p["modes"][m]
        print(f"  {m:<28} {c:>4}  {100.0 * c / p['n']:>5.1f}%   {MODE_NOTE[m]}")

    d = s["disagreement"]
    print("\nF1 vs behaviour")
    print("-" * 78)
    print(f"  mean F1 vs trap  | refused {d['f1_trap_correct']:>6.2f}   did not {d['f1_trap_incorrect']:>6.2f}")
    print(f"  mean F1 vs refusal | refused {d['f1_refusal_correct']:>6.2f}   did not {d['f1_refusal_incorrect']:>6.2f}")
    print(f"  F1-vs-trap >= 0.5 but wrong : {d['f1_trap_high_but_wrong']:>4}")
    print(f"  F1-vs-trap <  0.5 but right : {d['f1_trap_low_but_right']:>4}")
    print(f"  disagreement                : {d['n_disagree']}/{d['n']} = {d['pct_disagree']:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result_*.json files from test_advanced_robust.py")
    ap.add_argument("--json", metavar="PATH", help="write the full summary as JSON")
    ap.add_argument("--md", metavar="PATH", help="write markdown tables")
    args = ap.parse_args()

    paths = []
    for pattern in args.results:
        hits = sorted(glob.glob(pattern)) or [pattern]
        paths.extend(hits)

    trials = [load_trial(p) for p in paths]
    summary = summarize(trials)
    render_console(summary)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v for k, v in summary.items() if k != "rows"}, fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(render_markdown(summary) + "\n")
        print(f"wrote {args.md}")

    # a run where the two reference conventions happen to agree is not an error, but a run with
    # zero category-5 questions is
    return 0 if summary["pooled"]["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
