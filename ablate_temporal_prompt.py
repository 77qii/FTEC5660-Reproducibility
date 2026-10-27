#!/usr/bin/env python3
"""37-question fixed-context prompt ablation; no memory construction or query calls.

prepare: extract exact answer prompts from the completed seed-42 evaluation log.
run: replay original and absolute-date prompts, 74 answer calls, with identical
system/temperature/max_tokens. Gold is used only by utils.calculate_metrics.
Saved request records contain usage/model metadata, never credentials.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = ("Use DATE of CONVERSATION to answer with an approximate date.\n"
            "Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.")
ABSOLUTE = ("Use the DATE of the relevant evidence CONVERSATION session to answer with an absolute date. "
            "Convert relative expressions such as yesterday, last year, or next month using that session's date. "
            "Return a day as D Month YYYY, a month as Month YYYY, or a year as YYYY; keep only the precision supported by the evidence. "
            "Do not answer with relative date words or invent a missing date.\n"
            "Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.")
SYSTEM = "Follow the format specified in the prompt exactly. Do not add extra commentary."
LOG_PREFIX = r"^\d{4}-\d\d-\d\d .*? - INFO - "


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args):
    d = json.loads(Path(args.baseline).read_text())
    parts = re.split(LOG_PREFIX + r"Question (\d+): ", Path(args.log).read_text(), flags=re.M)
    assert (len(parts) - 1) // 2 == len(d["individual_results"]) == 199
    rows = []
    for i in range(1, len(parts), 2):
        idx, block = int(parts[i]), parts[i + 1]
        r = d["individual_results"][idx - 1]
        question = block.split("\n", 1)[0].strip()
        assert question == r["question"]
        if int(r["category"]) != 2:
            continue
        m = re.search(LOG_PREFIX + r"User Prompt: (.*?)(?=" + LOG_PREFIX + r"Category:)", block, flags=re.M | re.S)
        assert m, f"missing prompt for question {idx}"
        prompt = m.group(1).rstrip("\r\n")
        assert prompt.endswith(f"Question: {question} Short answer:")
        assert prompt.count(ORIGINAL) == 1
        context = prompt.split(", answer the following question. " + ORIGINAL)[0]
        assert context.startswith("Based on the context: ")
        rows.append({"question_index": idx, "sample_id": r["sample_id"], "question": question,
                     "reference": r["reference"], "category": 2,
                     "historical_prediction": r["prediction"], "historical_metrics": r["metrics"],
                     "original_prompt": prompt,
                     "absolute_prompt": prompt.replace(ORIGINAL, ABSOLUTE, 1),
                     "context_sha256": hashlib.sha256(context.encode()).hexdigest()})
    assert len(rows) == 37
    assert len({(r["sample_id"], r["question"]) for r in rows}) == 37
    save(args.out, {"source_prediction_sha256": sha(args.baseline), "source_log_sha256": sha(args.log),
                    "source_seed": 42, "n": 37, "retrieval": "exact answer-prompt context replay; k=10; no new query calls",
                    "original_instruction": ORIGINAL, "absolute_instruction": ABSOLUTE, "rows": rows})
    print("Prepared 37 exact logged prompts; only the date instruction differs.")


def read_env(path):
    if path:
        for line in Path(path).read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, val = s.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def run(args):
    read_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    model = args.model
    if not key or not endpoint:
        raise SystemExit("API key and base URL must be set in the environment or --env-file.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import nltk
    nltk.download = lambda *a, **kw: False  # already-installed corpora; no import-time downloads
    from openai import OpenAI
    from utils import calculate_metrics
    from analyze_c5 import token_f1
    from analyze_temporal import REL, ABS
    inputs = json.loads(Path(args.inputs).read_text())
    input_hash = sha(args.inputs)
    assert inputs["n"] == len(inputs["rows"]) == 37
    for r in inputs["rows"]:
        assert r["absolute_prompt"] == r["original_prompt"].replace(ORIGINAL, ABSOLUTE, 1)
        actual = calculate_metrics(r["historical_prediction"], r["reference"])["f1"]
        assert abs(actual - r["historical_metrics"]["f1"]) < 1e-12
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.resume:
        raise SystemExit("Checkpoint exists; use --resume explicitly to avoid duplicate paid calls.")
    records = [json.loads(s) for s in checkpoint.read_text().splitlines() if s.strip()] if checkpoint.exists() else []
    assert all(r["inputs_sha256"] == input_hash and r["model_requested"] == model for r in records)
    done = {(r["question_index"], r["arm"]): r for r in records}
    assert len(done) == len(records)
    client = OpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=120)
    start = datetime.now(timezone.utc).isoformat()
    with checkpoint.open("a", encoding="utf-8") as f:
        for i, row in enumerate(inputs["rows"]):
            arms = ("original", "absolute") if i % 2 == 0 else ("absolute", "original")
            for arm in arms:
                ident = (row["question_index"], arm)
                if ident in done:
                    continue
                total_cost = sum((r["usage"]["prompt_tokens"] * 0.28 + r["usage"]["completion_tokens"] * 0.42) / 1e6 for r in records)
                if total_cost >= args.max_meter_usd:
                    raise SystemExit("Meter budget reached; saved checkpoint, no additional calls made.")
                response = client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": row[arm + "_prompt"]}],
                    temperature=0.7, max_tokens=1000)
                prediction = response.choices[0].message.content
                assert prediction is not None
                rec = {"inputs_sha256": input_hash, "question_index": row["question_index"],
                       "arm": arm, "model_requested": model, "model_returned": response.model,
                       "response_id": response.id, "response_created": response.created,
                       "finish_reason": response.choices[0].finish_reason,
                       "prediction": prediction, "usage": response.usage.model_dump(),
                       "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records.append(rec)
                done[ident] = rec
                print(f"Completed {len(records)}/74 answer calls", flush=True)
    out_rows = []
    for row in inputs["rows"]:
        out = {k: row[k] for k in ("sample_id", "question", "reference", "category", "question_index", "context_sha256")}
        out["historical"] = {"prediction": row["historical_prediction"], "metrics": row["historical_metrics"]}
        for arm in ("original", "absolute"):
            rec = done[(row["question_index"], arm)]
            metrics = calculate_metrics(rec["prediction"], row["reference"])
            assert abs(metrics["f1"] - token_f1(rec["prediction"], row["reference"])) < 1e-12
            out[arm] = {"prediction": rec["prediction"], "metrics": metrics, "model_returned": rec["model_returned"]}
        out_rows.append(out)
    summary = {}
    for arm in ("historical", "original", "absolute"):
        candidates = sum(bool(REL.search(r[arm]["prediction"]) and ABS.search(str(r["reference"]))) for r in out_rows)
        summary[arm] = {"n": 37, "f1": sum(r[arm]["metrics"]["f1"] for r in out_rows) / 37 * 100,
                        "format_candidates": candidates, "format_candidate_pct": candidates / 37 * 100}
    prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in records)
    completion_tokens = sum(r["usage"]["completion_tokens"] for r in records)
    save(args.out, {"experiment": "paired fixed-context Temporal prompt replay", "started_at_utc": start,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": input_hash,
                    "model_requested": model, "models_returned": sorted({r["model_returned"] for r in records}),
                    "system_message": SYSTEM, "temperature": 0.7, "max_tokens": 1000,
                    "order": "interleaved; original first on even indices, absolute first on odd indices",
                    "calls": len(records), "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "meter_usd": (prompt_tokens * 0.28 + completion_tokens * 0.42) / 1e6,
                    "price_note": "same historical meter rates; cache discounts ignored; not an invoice",
                    "scope": "37 questions, seed-42 logged contexts; one answer per arm; no API sampling seed; no causal/significance claim",
                    "format_note": "unchanged analyze_temporal.REL/ABS regex; candidate rate, not semantic error rate",
                    "summary": summary, "rows": out_rows})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--baseline", required=True)
    prep.add_argument("--log", required=True)
    prep.add_argument("--out", required=True)
    prep.set_defaults(func=prepare)
    rr = sub.add_parser("run")
    rr.add_argument("--inputs", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--checkpoint", required=True)
    rr.add_argument("--env-file")
    rr.add_argument("--model", default="deepseek-chat")
    rr.add_argument("--resume", action="store_true")
    rr.add_argument("--max-meter-usd", type=float, default=1.0)
    rr.set_defaults(func=run)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
