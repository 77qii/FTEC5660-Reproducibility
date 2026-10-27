#!/usr/bin/env python3
"""Recompute the report's main tables and the cost block from shipped evidence.

Standard library only. No API calls, no network, no model downloads. The script reads the
whitelisted prediction files and sanitised logs under ``evidence/``, re-derives every number it
reports, and refuses to print a table when a stored value does not reproduce.

What it checks (any failure aborts the report and exits non-zero):

  1. set-token F1 recomputed from ``prediction``/``reference`` equals the stored ``metrics.f1``
     on every question of all three trials;
  2. the same ``(sample_id, question)`` carries the same gold reference and category in all
     three trials, and the trial question sets are identical;
  3. ``aggregate_metrics`` mean/std/median/min/max/count equal a recomputation from the rows;
  4. category 5 goes through ``analyze_c5.load_trial`` (which re-asserts F1 and the refusal
     flag) and ``analyze_c5.summarize`` (question-cluster bootstrap, 20,000 resamples, seed
     20261026);
  5. the shipped evidence files match ``evidence/manifest.json``;
  6. the sanitised logs contain no absolute machine path, no ANSI escape and no credential-like
     string, and the meter block is arithmetically self-consistent.

What it reports, but cannot prove:

  * the model identity. The logs are console stdout and never carry the API's ``model`` field,
    so "the API returned deepseek-flash" is not evidenced anywhere in this evidence set. This is
    reported as unresolved instead of being asserted.
  * wall-clock runtime beyond the first/last timestamped log line.

Usage
-----
    # 1. third party, from the shipped evidence only
    python3 reproduce_tables.py --out-dir repro-out

    # 2. maintainer, rebuild evidence/ from the experiment directory (whitelist copy, log
    #    sanitisation, provenance, manifest). Never writes into the source directory.
    python3 reproduce_tables.py --prepare --source-dir /path/to/repro-amem

    # 3. zero-dependency self-test: real data must pass, four tampered inputs must be rejected
    python3 reproduce_tables.py --selftest

``--out-dir`` may not be the evidence directory or a subdirectory of it: the script never
overwrites its own inputs.
"""

import argparse
import difflib
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_c5  # noqa: E402  stdlib-only; the F1 and refusal implementations live there
import analyze_temporal  # noqa: E402  stdlib-only; REL/ABS are reused verbatim

REL = analyze_temporal.REL
ABS = analyze_temporal.ABS
token_f1 = analyze_c5.token_f1

CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}

# The paper's reported column (GPT-4o-mini, full benchmark) as transcribed in REPORT.md §4.
REPORTED_F1 = {"Multi-Hop": 27.02, "Temporal": 45.85, "Open-Domain": 12.14,
               "Single-Hop": 44.65, "Adversarial": 50.03}

TRIALS = [
    {"label": "seed 42", "seed": 42, "predictions": "result_conv26_new.json", "log": "run_real_t1.log"},
    {"label": "seed 43", "seed": 43, "predictions": "result_conv26_seed43.json", "log": "run_real_seed43.log"},
    {"label": "seed 44", "seed": 44, "predictions": "result_conv26_seed44.json", "log": "run_real_seed44.log"},
]

BASELINE_LOGS = ["baseline_before.log", "baseline_after.log"]
BERT_FILES = ["bert_baseline_before.json", "bert_baseline_after.json"]
BERT_EXPECTED_VALUES = 24  # 8 pairs x (precision, recall, f1)

# Scripts whose hash belongs in the manifest: everything needed to re-derive a number above.
MANIFEST_SCRIPTS = ["reproduce_tables.py", "analyze_c5.py", "analyze_temporal.py", "aggregate_legacy.py", "utils.py",
                    "meter.py", "run_metered.py", "rescore_c5.py", "capture_bert_baseline.py",
                    "ablate_temporal_prompt.py", "run_trials.sh"]

F1_TOL = 1e-9
AGG_TOL = 1e-9


class VerificationError(Exception):
    """A stored value did not reproduce. Raised instead of printing a wrong table."""


# --------------------------------------------------------------------------- small helpers

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fail(message):
    raise VerificationError(message)


def check_float(stored, recomputed, where, tol=F1_TOL):
    stored = float(stored)
    if abs(stored - recomputed) > tol * max(1.0, abs(stored), abs(recomputed)):
        fail(f"{where}: stored {stored!r}, recomputed {recomputed!r}")


def mean_or_none(values):
    values = list(values)
    return statistics.mean(values) if values else None


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- prediction checks

def _stats(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def verify_predictions(pred_dir, trials_spec=TRIALS):
    """Load the three trials, re-derive F1, cross-trial gold/category, and aggregate metrics."""
    trials = []
    for spec in trials_spec:
        path = Path(pred_dir) / spec["predictions"]
        if not path.exists():
            fail(f"missing evidence file: {path}")
        data = read_json(path)
        rows = data.get("individual_results") or []
        if not rows:
            fail(f"{path.name}: no individual_results")
        if data.get("legacy_c5_reference"):
            fail(f"{path.name}: legacy_c5_reference is true; this evidence set is the corrected run")

        n_expected = int(data.get("total_questions") or len(rows))
        if len(rows) != n_expected:
            fail(f"{path.name}: {len(rows)} rows but total_questions={n_expected}")

        for i, r in enumerate(rows):
            where = f"{path.name}[{i}] {str(r.get('question'))[:60]!r}"
            if not r.get("question"):
                fail(f"{where}: missing question text")
            if r.get("sample_id") is None:
                fail(f"{where}: missing sample_id")
            category = int(r.get("category", 0))
            if category not in CATEGORY_NAMES:
                fail(f"{where}: unexpected category {category!r}")
            metrics = r.get("metrics") or {}
            if "f1" not in metrics:
                fail(f"{where}: no stored f1")
            recomputed = token_f1(r.get("prediction"), r.get("reference"))
            check_float(metrics["f1"], recomputed, f"{where} f1")

        dist = {}
        for r in rows:
            dist[str(int(r["category"]))] = dist.get(str(int(r["category"])), 0) + 1
        if dist != data.get("category_distribution"):
            fail(f"{path.name}: category_distribution {data.get('category_distribution')} "
                 f"!= recomputed {dist}")

        trials.append({
            "label": spec["label"],
            "seed": data.get("seed"),
            "file": spec["predictions"],
            "model_field": data.get("model"),
            "n": len(rows),
            "rows": rows,
            "aggregate_metrics": data.get("aggregate_metrics") or {},
        })

    # cross-trial identity: same (sample_id, question) -> same gold and category
    def identity_map(trial):
        out = {}
        for r in trial["rows"]:
            key = (r["sample_id"], r["question"])
            if key in out:
                fail(f"{trial['file']}: duplicate question within a trial: {key!r}")
            out[key] = (r["reference"], int(r["category"]))
        return out

    maps = [identity_map(t) for t in trials]
    keys = set(maps[0])
    for t, m in zip(trials[1:], maps[1:]):
        if set(m) != keys:
            missing = sorted(set(keys) - set(m))[:3]
            extra = sorted(set(m) - set(keys))[:3]
            fail(f"{t['file']}: question set differs from {trials[0]['file']} "
                 f"(missing {missing}, extra {extra})")
        for key in keys:
            if m[key] != maps[0][key]:
                fail(f"{t['file']}: gold/category changed for {key!r}: "
                     f"{m[key]!r} != {maps[0][key]!r}")

    # aggregate metrics must equal a recomputation from the rows
    for trial in trials:
        stored = trial["aggregate_metrics"]
        if not stored:
            fail(f"{trial['file']}: no aggregate_metrics")
        overall, by_cat = {}, {}
        for r in trial["rows"]:
            cat = int(r["category"])
            for name, value in r["metrics"].items():
                overall.setdefault(name, []).append(value)
                by_cat.setdefault(cat, {}).setdefault(name, []).append(value)
        groups = {"overall": overall}
        for cat, metrics in by_cat.items():
            groups[f"category_{cat}"] = metrics
        for group_name, metrics in groups.items():
            if group_name not in stored:
                fail(f"{trial['file']}: aggregate_metrics lacks {group_name}")
            for name, values in metrics.items():
                if name not in stored[group_name]:
                    fail(f"{trial['file']}: aggregate_metrics[{group_name}] lacks {name}")
                computed = _stats(values)
                for field in ("mean", "std", "median", "min", "max", "count"):
                    where = f"{trial['file']} aggregate_metrics[{group_name}][{name}].{field}"
                    if field == "count":
                        if int(stored[group_name][name][field]) != computed[field]:
                            fail(f"{where}: stored {stored[group_name][name][field]}, "
                                 f"recomputed {computed[field]}")
                    else:
                        check_float(stored[group_name][name][field], computed[field], where, AGG_TOL)
        if set(stored) - set(groups):
            fail(f"{trial['file']}: aggregate_metrics has groups with no rows: "
                 f"{sorted(set(stored) - set(groups))}")

    return trials


# --------------------------------------------------------------------------- §4 tables

def _per_trial_values(trial, key):
    """key='corrected' uses the stored (corrected-reference) F1; 'legacy' re-scores category 5
    against the adversarial answer, which is what the paper's convention does."""
    out = {}
    for cat in sorted(CATEGORY_NAMES):
        values = []
        for r in trial["rows"]:
            if int(r["category"]) != cat:
                continue
            if key == "corrected":
                values.append(float(r["metrics"]["f1"]) * 100.0)
            elif cat == 5:
                values.append(token_f1(r.get("prediction"), r.get("adversarial_answer")) * 100.0)
            else:
                values.append(float(r["metrics"]["f1"]) * 100.0)
        out[cat] = values
    return out


def build_category_table(trials, key):
    """Mean of the per-trial category means, with the sample std across trials (n=3)."""
    per_trial = [_per_trial_values(t, key) for t in trials]
    rows = []
    for cat in sorted(CATEGORY_NAMES):
        trial_means = [statistics.mean(p[cat]) for p in per_trial]
        rows.append({
            "category": CATEGORY_NAMES[cat],
            "n": len(per_trial[0][cat]),
            "per_trial_mean": trial_means,
            "mean": statistics.mean(trial_means),
            "std": sample_std(trial_means),
            "min": min(trial_means),
            "max": max(trial_means),
            "reported": REPORTED_F1.get(CATEGORY_NAMES[cat]),
        })
    trial_overall = [statistics.mean([v for cat in p for v in p[cat]]) for p in per_trial]
    overall = {
        "n": sum(len(per_trial[0][cat]) for cat in per_trial[0]),
        "per_trial_mean": trial_overall,
        "mean": statistics.mean(trial_overall),
        "std": sample_std(trial_overall),
        "min": min(trial_overall),
        "max": max(trial_overall),
        "reported": None,
    }
    for row in rows:
        if row["reported"] is not None:
            row["delta"] = row["mean"] - row["reported"]
    overall["delta"] = None
    return {"rows": rows, "overall": overall}


# --------------------------------------------------------------------------- §4.1 temporal

def build_temporal(trials, reported_temporal):
    per_trial = []
    for trial in trials:
        rows = [r for r in trial["rows"] if int(r["category"]) == 2]
        candidates, others = [], []
        for r in rows:
            prediction = str(r.get("prediction") or "")
            reference = str(r.get("reference") or "")
            f1 = float(r["metrics"]["f1"]) * 100.0
            if REL.search(prediction) and ABS.search(reference):
                candidates.append(f1)
            else:
                others.append(f1)
        per_trial.append({
            "label": trial["label"],
            "n": len(rows),
            "candidates": len(candidates),
            "candidate_share": 100.0 * len(candidates) / len(rows),
            "candidate_f1": mean_or_none(candidates),
            "other_f1": mean_or_none(others),
            "overall_f1": statistics.mean([v for v in candidates + others]),
            "zero_f1_questions": sum(1 for r in rows if float(r["metrics"]["f1"]) == 0.0),
        })

    actual = statistics.mean([t["overall_f1"] for t in per_trial])
    # hypothetical: screened candidates score like the rest of the temporal questions
    counterfactual = statistics.mean([t["other_f1"] for t in per_trial])
    gap = reported_temporal - actual
    return {
        "regex": {"rel": REL.pattern, "abs": ABS.pattern,
                  "note": "unchanged analyze_temporal.REL/ABS; screening only, no date resolution"},
        "per_trial": per_trial,
        "candidates_total": sum(t["candidates"] for t in per_trial),
        "observations_total": sum(t["n"] for t in per_trial),
        "actual_mean": actual,
        "counterfactual_mean": counterfactual,
        "gain": counterfactual - actual,
        "reported_gap": gap,
        "share_of_gap": 100.0 * (counterfactual - actual) / gap if gap else None,
        "note": "hypothetical replacement of subgroup scores, not a date-normalisation re-score",
    }


# --------------------------------------------------------------------------- §5 category 5

def build_cat5(pred_dir, resamples, bootstrap_seed):
    trials = []
    for spec in TRIALS:
        path = Path(pred_dir) / spec["predictions"]
        try:
            seed, rows = analyze_c5.load_trial(str(path))
        except SystemExit as exc:  # load_trial reports through sys.exit
            fail(f"{spec['predictions']}: analyze_c5.load_trial rejected the file: {exc.code}")
        trials.append((seed, rows))
    try:
        summary = analyze_c5.summarize(trials, resamples, bootstrap_seed)
    except ValueError as exc:
        fail(f"analyze_c5.summarize rejected the trials: {exc}")
    summary.pop("rows", None)  # per-question predictions stay in evidence/, not in the table
    return summary


# --------------------------------------------------------------------------- cost

METER_PATTERNS = {
    "calls": r"成功调用\s*:\s*([\d,]+)",
    "failed_calls": r"失败调用\s*:\s*([\d,]+)",
    "prompt_tokens": r"输入 tokens\s*:\s*([\d,]+)",
    "cached_tokens": r"其中命中缓存\s*:\s*([\d,]+)",
    "completion_tokens": r"输出 tokens\s*:\s*([\d,]+)",
    "cost_usd": r"合计成本\s*:\s*\$([\d.]+)",
}
RATES_RE = re.compile(r"单价\(美元/M\)\s*:\s*in=([\d.]+)\s+out=([\d.]+)")
AVG_RE = re.compile(r"平均每次调用\s*:\s*([\d,]+) in / ([\d,]+) out")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - ", re.M)


def parse_meter(text, where):
    out = {}
    for name, pattern in METER_PATTERNS.items():
        m = re.search(pattern, text)
        if not m:
            return None
        out[name] = float(m.group(1).replace(",", "")) if name == "cost_usd" else int(m.group(1).replace(",", ""))
    m = RATES_RE.search(text)
    if not m:
        return None
    out["price_in"], out["price_out"] = float(m.group(1)), float(m.group(2))
    m = AVG_RE.search(text)
    if m:
        out["avg_in"] = int(m.group(1).replace(",", ""))
        out["avg_out"] = int(m.group(2).replace(",", ""))
    return out


def build_cost(evidence_dir, trials, checks):
    rows = []
    for spec, trial in zip(TRIALS, trials):
        path = Path(evidence_dir) / "logs" / spec["log"]
        if not path.exists():
            fail(f"missing evidence log: {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        meter = parse_meter(text, path.name)
        if meter is None:
            fail(f"{path.name}: no token/cost block found")
        expected = (meter["prompt_tokens"] * meter["price_in"]
                    + meter["completion_tokens"] * meter["price_out"]) / 1e6
        check_float(meter["cost_usd"], expected, f"{path.name} cost", 1e-4)
        if meter["calls"]:
            if meter.get("avg_in") is not None:
                if abs(meter["avg_in"] - meter["prompt_tokens"] / meter["calls"]) > 1.0:
                    fail(f"{path.name}: average input tokens per call does not match the totals")
                if abs(meter["avg_out"] - meter["completion_tokens"] / meter["calls"]) > 1.0:
                    fail(f"{path.name}: average output tokens per call does not match the totals")
        stamps = TIMESTAMP_RE.findall(text)
        qa_calls = 2 * trial["n"]  # two calls per question in the answer phase
        rows.append({
            "label": spec["label"],
            "log": spec["log"],
            **meter,
            "cached_share": 100.0 * meter["cached_tokens"] / meter["prompt_tokens"] if meter["prompt_tokens"] else None,
            "qa_calls_implied": qa_calls,
            "memory_calls_derived": meter["calls"] - qa_calls,
            "first_log_timestamp": stamps[0] if stamps else None,
            "last_log_timestamp": stamps[-1] if stamps else None,
            "timestamped_lines": len(stamps),
        })
    checks.append({"id": "cost.meter_arithmetic", "ok": True,
                   "detail": "recomputed cost from tokens x stated rates matches the logged total"})
    total = sum(r["cost_usd"] for r in rows)
    return {"per_trial": rows, "total_usd": total,
            "note": "tokens are measured from the API usage field; dollars price every input token "
                    "at list, so the total is an upper bound, not a bill"}


# --------------------------------------------------------------------------- BERTScore

def build_bertscore(evidence_dir):
    paths = [Path(evidence_dir) / "baseline" / name for name in BERT_FILES]
    for path in paths:
        if not path.exists():
            fail(f"missing BERTScore baseline: {path}")
    before, after = (read_json(p) for p in paths)
    if len(before) != len(after) or not before:
        fail("BERTScore baselines have different lengths")
    pairs, values, max_diff = 0, 0, 0.0
    for i, (a, b) in enumerate(zip(before, after)):
        if a.get("pred") != b.get("pred") or a.get("ref") != b.get("ref"):
            fail(f"BERTScore baseline pair {i} changed input text")
        for key in ("bert_precision", "bert_recall", "bert_f1"):
            values += 1
            delta = abs(float(a["scores"][key]) - float(b["scores"][key]))
            max_diff = max(max_diff, delta)
            if delta != 0.0:
                fail(f"BERTScore value {i}.{key} changed: {a['scores'][key]} -> {b['scores'][key]}")
        pairs += 1
    if values != BERT_EXPECTED_VALUES:
        fail(f"expected {BERT_EXPECTED_VALUES} BERTScore values, found {values}")
    return {"pairs": pairs, "values": values, "max_abs_diff": max_diff, "identical": max_diff == 0.0,
            "files": BERT_FILES,
            "note": "8 fixed (prediction, reference) pairs x 3 scores, captured before and after "
                    "the BERTScorer reuse change in utils.py"}


# --------------------------------------------------------------------------- model identity

MODEL_PROBE_RE = re.compile(r"(?i)deepseek[-a-z0-9._]*")


def build_model_identity(evidence_dir, trials, ablation=None):
    hits, flash = {}, 0
    for spec in TRIALS:
        path = Path(evidence_dir) / "logs" / spec["log"]
        text = path.read_bytes().decode("utf-8", errors="replace")
        tokens = MODEL_PROBE_RE.findall(text)
        hits[spec["log"]] = sorted({t.lower() for t in tokens})
        flash += len(re.findall(r"(?i)flash", text))
    fields = sorted({str(t["model_field"]) for t in trials})
    api_field_present = any("model_returned" in r for t in trials for r in t["rows"])

    returned = (ablation or {}).get("models_returned") or []
    ablation_ok = (ablation or {}).get("status") == "verified"
    status = "unresolved"
    if flash == 0 and not api_field_present:
        detail = (
            "The three run logs are console stdout from the evaluation harness and never contain "
            "the API response's `model` field, so they cannot show which model the server actually "
            "returned. No `deepseek-flash` string appears anywhere in those logs; the only model "
            "token present is `deepseek-chat`, and there it is a cache directory name "
            "(cached_memories_robust_openai_deepseek-chat). The `model` field in the result JSON is "
            "echoed by the harness from its command line, i.e. the model that was requested, not "
            "the model the server reported."
        )
        if ablation_ok and returned:
            detail += (
                f" The prompt-ablation records do carry an API-returned model field and report "
                f"{returned} for {ablation.get('calls')} calls requested as "
                f"`{ablation.get('model_requested')}`. That evidences the server's routing for that "
                f"74-call run only; it is a separate run from the three scored trials and cannot be "
                f"transferred to them. Model identity for the three scored trials therefore stays "
                f"unresolved."
            )
        else:
            detail += (" No record in this evidence set carries an API-returned model field, so a "
                       "claim that the runs used deepseek-flash is not supported; it stays "
                       "unresolved.")
    else:
        status = "evidenced"
        detail = "An API-returned model field is present in the evidence."
    return {
        "status": status,
        "deepseek_flash_mentions_in_logs": flash,
        "model_tokens_seen_in_logs": hits,
        "result_file_model_field": fields,
        "api_returned_model_field_present": api_field_present,
        "ablation_models_returned": returned,
        "ablation_calls": (ablation or {}).get("calls"),
        "detail": detail,
    }


# --------------------------------------------------------------------------- ablation (optional)

def build_ablation(evidence_dir, checks):
    ablation_path = Path(evidence_dir) / "temporal_ablation.json"
    if not ablation_path.exists():
        checks.append({"id": "ablation.present", "ok": True,
                       "detail": "temporal_ablation.json not shipped; section skipped"})
        return {"status": "absent",
                "detail": "evidence/temporal_ablation.json does not exist; the ablation table is "
                          "omitted rather than estimated"}

    problems = []
    ablation = read_json(ablation_path)
    contexts_path = Path(evidence_dir) / "temporal_contexts.json"
    if not contexts_path.exists():
        problems.append("temporal_contexts.json is missing, so inputs_sha256 cannot be verified")
        contexts = None
    else:
        contexts = read_json(contexts_path)
        expected = sha256_file(contexts_path)
        if ablation.get("inputs_sha256") != expected:
            problems.append(f"inputs_sha256 {ablation.get('inputs_sha256')} != {expected}")

    rows = ablation.get("rows") or []
    if len(rows) != 37:
        problems.append(f"expected 37 rows, found {len(rows)}")
    if ablation.get("calls") != 74:
        problems.append(f"expected 74 calls, found {ablation.get('calls')}")

    from ablate_temporal_prompt import ORIGINAL, ABSOLUTE
    context_rows = (contexts or {}).get("rows", [])
    by_index = {r["question_index"]: r for r in context_rows}
    if len(by_index) != 37 or len({r["question_index"] for r in rows}) != 37:
        problems.append("expected 37 unique question indices in contexts and outcomes")
    for ctx in context_rows:
        prompt = ctx["original_prompt"]
        if prompt.count(ORIGINAL) != 1 or ctx["absolute_prompt"] != prompt.replace(ORIGINAL, ABSOLUTE, 1):
            problems.append("ablation prompt differs beyond the specified date instruction")
        context = prompt.split(", answer the following question. " + ORIGINAL)[0]
        if sha256_bytes(context.encode()) != ctx["context_sha256"]:
            problems.append("recorded context hash does not reproduce from actual prompt bytes")
    historical = read_json(Path(evidence_dir) / "predictions/result_conv26_new.json")
    historical_map = {(r["sample_id"], r["question"]): r for r in historical["individual_results"]}
    if contexts and contexts["source_prediction_sha256"] != sha256_file(Path(evidence_dir) / "predictions/result_conv26_new.json"):
        problems.append("context source predictions differ from shipped seed 42")
    for ctx in context_rows:
        base = historical_map.get((ctx["sample_id"], ctx["question"]))
        if not base or base["reference"] != ctx["reference"] or base["prediction"] != ctx["historical_prediction"]:
            problems.append("historical context prediction/reference mismatch")
    request_path = Path(evidence_dir) / "temporal_requests.jsonl"
    requests = [json.loads(s) for s in request_path.read_text().splitlines() if s.strip()] if request_path.exists() else []
    request_map = {(r["question_index"], r["arm"]): r for r in requests}
    if len(requests) != 74 or len(request_map) != 74:
        problems.append("request log must have exactly 74 unique paired records")
    else:
        for request in requests:
            if request["inputs_sha256"] != ablation["inputs_sha256"]:
                problems.append("request inputs hash mismatch")
        pt = sum(r["usage"]["prompt_tokens"] for r in requests)
        ct = sum(r["usage"]["completion_tokens"] for r in requests)
        if pt != ablation["prompt_tokens"] or ct != ablation["completion_tokens"]:
            problems.append("ablation token totals do not match response usage logs")
        if abs((pt * 0.28 + ct * 0.42) / 1e6 - ablation["meter_usd"]) > 1e-12:
            problems.append("ablation meter arithmetic mismatch")
        if sorted({r["model_returned"] for r in requests}) != ablation["models_returned"]:
            problems.append("API-returned model set mismatch")
    for row in rows:
        where = f"ablation row {row.get('question_index')}"
        ctx = by_index.get(row.get("question_index"))
        if ctx is None:
            problems.append(f"{where}: question_index absent from temporal_contexts.json")
            continue
        if row.get("context_sha256") != ctx.get("context_sha256"):
            problems.append(f"{where}: context_sha256 mismatch")
        if row.get("question") != ctx.get("question") or row.get("reference") != ctx.get("reference"):
            problems.append(f"{where}: question/reference differ from the context file")
        for arm in ("original", "absolute"):
            arm_row = row.get(arm) or {}
            req = request_map.get((row["question_index"], arm))
            if req is None or req["prediction"] != arm_row.get("prediction") or req["model_returned"] != arm_row.get("model_returned"):
                problems.append(f"{where}: {arm} differs from logged API response")
            if "prediction" not in arm_row or "metrics" not in arm_row:
                problems.append(f"{where}: arm {arm} missing prediction/metrics")
                continue
            recomputed = token_f1(arm_row["prediction"], row["reference"])
            if abs(recomputed - float(arm_row["metrics"]["f1"])) > F1_TOL:
                problems.append(f"{where} arm {arm}: f1 stored {arm_row['metrics']['f1']}, "
                                f"recomputed {recomputed}")

    summary = ablation.get("summary") or {}
    recomputed_summary = {}
    for arm in ("historical", "original", "absolute"):
        if arm == "historical":
            predictions = [(r["historical"]["prediction"], r["historical"]["metrics"]["f1"]) for r in rows]
        else:
            predictions = [(r[arm]["prediction"], r[arm]["metrics"]["f1"]) for r in rows]
        candidates = sum(1 for (p, _), r in zip(predictions, rows)
                         if REL.search(str(p or "")) and ABS.search(str(r.get("reference") or "")))
        recomputed_summary[arm] = {
            "n": len(rows),
            "f1": statistics.mean([float(f1) for _, f1 in predictions]) * 100.0,
            "format_candidates": candidates,
            "format_candidate_pct": 100.0 * candidates / len(rows),
        }
        stored = summary.get(arm) or {}
        for key, value in recomputed_summary[arm].items():
            if key not in stored:
                problems.append(f"summary.{arm}.{key} missing")
            elif key == "n" or key.endswith("candidates"):
                if int(stored[key]) != int(value):
                    problems.append(f"summary.{arm}.{key}: stored {stored[key]}, recomputed {value}")
            else:
                if abs(float(stored[key]) - float(value)) > 1e-6:
                    problems.append(f"summary.{arm}.{key}: stored {stored[key]}, recomputed {value}")

    checks.append({"id": "ablation.integrity", "ok": not problems,
                   "detail": "rows/summary/candidate counts and both hashes verified" if not problems
                             else "; ".join(problems[:4])})
    return {
        "status": "verified" if not problems else "failed",
        "model_requested": ablation.get("model_requested"),
        "models_returned": ablation.get("models_returned"),
        "calls": ablation.get("calls"),
        "prompt_tokens": ablation.get("prompt_tokens"),
        "completion_tokens": ablation.get("completion_tokens"),
        "meter_usd": ablation.get("meter_usd"),
        "scope": ablation.get("scope"),
        "summary": recomputed_summary,
        "problems": problems,
    }


# --------------------------------------------------------------------------- log sanitisation

ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
SECRET_PATTERNS = [
    ("openai_key", re.compile(rb"sk-[A-Za-z0-9_\-]{12,}")),
    ("bearer", re.compile(rb"(?i)bearer\s+[A-Za-z0-9._\-]{12,}")),
    ("assignment", re.compile(rb"(?i)\b(?:api[_-]?key|secret|password|passwd|authorization|"
                              rb"access[_-]?token|auth[_-]?token)\b\s*[:=]\s*[^\s\"']{8,}")),
    ("url_credential", re.compile(rb"(?i)https?://[^\s\"']*[?&](?:key|token|sig|secret)=")),
]
LEFTOVER_PATH_RE = re.compile(rb"(?:/Users/|/home/|/private/var/|/var/folders/)")


def sanitise_log(raw, source_dir):
    """Strip machine absolute paths, ANSI escapes and credential-like strings.

    Timestamps, token counts and the cost block are untouched, so the meter numbers can still be
    re-derived from the sanitised file.
    """
    counts = {"ansi": 0, "abs_prefix": 0, "abs_parent": 0, "user_path": 0, "redacted": 0}
    data = raw

    ansi = ANSI_RE.findall(data)
    if ansi:
        counts["ansi"] = len(ansi)
        data = ANSI_RE.sub(b"", data)

    root = str(Path(source_dir).resolve()).encode()
    parent = str(Path(source_dir).resolve().parent).encode()
    if root and root in data:
        counts["abs_prefix"] = data.count(root)
        data = data.replace(root, b"<REPRO_ROOT>")
    if parent and parent in data:
        counts["abs_parent"] = data.count(parent)
        data = data.replace(parent, b"<WORKSPACE>")

    user_paths = re.findall(rb"/Users/[A-Za-z0-9._\-]+", data)
    if user_paths:
        counts["user_path"] = len(user_paths)
        data = re.sub(rb"/Users/[A-Za-z0-9._\-]+", b"/Users/<USER>", data)

    for name, pattern in SECRET_PATTERNS:
        hits = pattern.findall(data)
        if hits:
            counts["redacted"] += len(hits)
            data = pattern.sub(f"<REDACTED:{name}>".encode(), data)

    leftovers = LEFTOVER_PATH_RE.findall(data)
    if leftovers:
        fail(f"sanitisation left {len(leftovers)} absolute path fragment(s); refusing to ship the log")
    remaining = sum(len(p.findall(data)) for _, p in SECRET_PATTERNS)
    if remaining:
        fail(f"sanitisation left {remaining} credential-like string(s); refusing to ship the log")
    return data, counts


def write_if_unchanged(path, data):
    """Create the file, or accept an existing byte-identical copy. Never overwrite different bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        fail(f"{path} already exists with different content; refusing to overwrite evidence")
    path.write_bytes(data)
    return "created"


# --------------------------------------------------------------------------- prepare

def git(args, cwd):
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)
        return out.stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(source_dir, evidence_dir):
    source = Path(source_dir).resolve()
    evidence = Path(evidence_dir).resolve()
    if not source.is_dir():
        fail(f"source directory not found: {source}")

    manifest = {"upstream_commit": "b31a1dd", "entries": []}
    sanitisation = {"policy": "只删除机器绝对路径、ANSI 控制符与潜在凭据；时间戳、token 计数、"
                             "成本块与预测文本原样保留",
                    "source_dir_note": "absolute source paths are recorded here only; the shipped "
                                       "logs and predictions are scrubbed or verbatim as noted",
                    "files": []}

    # 1. whitelist copy of the three prediction files, byte for byte
    for spec in TRIALS:
        src = source / "A-mem" / spec["predictions"]
        if not src.exists():
            fail(f"missing source prediction file: {src}")
        raw = src.read_bytes()
        dest = evidence / "predictions" / spec["predictions"]
        state = write_if_unchanged(dest, raw)
        manifest["entries"].append({
            "kind": "prediction", "source": f"A-mem/{spec['predictions']}",
            "evidence": f"predictions/{spec['predictions']}",
            "sha256": sha256_bytes(raw), "bytes": len(raw),
            "copied_verbatim": True, "state": state,
        })

    # 2. logs: original hashes recorded, sanitised copies shipped
    for name in [spec["log"] for spec in TRIALS] + BASELINE_LOGS:
        src = source / name
        if not src.exists():
            fail(f"missing source log: {src}")
        raw = src.read_bytes()
        clean, counts = sanitise_log(raw, source)
        dest = evidence / "logs" / name
        state = write_if_unchanged(dest, clean)
        manifest["entries"].append({
            "kind": "log", "source": name, "evidence": f"logs/{name}",
            "original_sha256": sha256_bytes(raw), "original_bytes": len(raw),
            "sha256": sha256_bytes(clean), "bytes": len(clean), "state": state,
        })
        sanitisation["files"].append({
            "source": name, "evidence": f"logs/{name}",
            "original_sha256": sha256_bytes(raw), "sanitized_sha256": sha256_bytes(clean),
            "original_bytes": len(raw), "sanitized_bytes": len(clean),
            "replacements": counts,
            "steps": ["strip ANSI CSI escape sequences",
                      "replace the experiment root with <REPRO_ROOT>",
                      "replace the workspace root with <WORKSPACE>",
                      "replace /Users/<name> with /Users/<USER>",
                      "redact credential-like strings if any (count recorded)"],
        })

    # 3. BERTScore baselines
    for name in BERT_FILES:
        src = source / name
        if not src.exists():
            fail(f"missing BERTScore baseline: {src}")
        raw = src.read_bytes()
        dest = evidence / "baseline" / name
        state = write_if_unchanged(dest, raw)
        manifest["entries"].append({
            "kind": "bertscore_baseline", "source": name, "evidence": f"baseline/{name}",
            "sha256": sha256_bytes(raw), "bytes": len(raw), "state": state,
        })

    # 4. dataset and scripts: hashes only, not shipped
    for rel, kind in [("A-mem/data/locomo10.json", "dataset")] + [(s, "script") for s in MANIFEST_SCRIPTS]:
        src = source / rel if rel.startswith("A-mem/") else HERE / rel
        if not src.exists():
            src = source / rel
        if not src.exists():
            continue
        raw = src.read_bytes()
        manifest["entries"].append({"kind": kind, "source": rel, "sha256": sha256_bytes(raw),
                                    "bytes": len(raw), "shipped": False})

    # 5. optional ablation artifacts: hash them if the main thread has produced them
    for name, kind in [("temporal_contexts.json", "ablation_inputs"),
                       ("temporal_ablation.json", "ablation_result"),
                       ("temporal_requests.jsonl", "ablation_requests")]:
        full = evidence / name
        if full.exists():
            manifest["entries"].append({
                "kind": kind, "source": name, "evidence": name,
                "sha256": sha256_file(full), "bytes": full.stat().st_size, "shipped": True,
                "note": "produced outside this script; hashed so a later edit invalidates the report",
            })

    # 6. provenance: real git state, never a hardcoded "clean"
    upstream_head = git(["rev-parse", "HEAD"], source / "A-mem") if (source / "A-mem" / ".git").exists() else None
    upstream_full = git(["rev-parse", "b31a1dd^{commit}"], source / "A-mem") if (source / "A-mem" / ".git").exists() else None
    upstream_subject = git(["log", "-1", "--format=%s", "b31a1dd"], source / "A-mem") if upstream_full else None
    local_head = git(["rev-parse", "HEAD"], HERE)
    local_subject = git(["log", "-1", "--format=%s", "HEAD"], HERE) if local_head else None
    local_status = git(["status", "--porcelain"], HERE)
    upstream_status = git(["status", "--porcelain"], source / "A-mem") if upstream_full else None

    src_utils = source / "A-mem" / "utils.py"
    repo_utils = HERE / "utils.py"
    local_vs_source = {"source": "A-mem/utils.py", "local": "utils.py"}
    if src_utils.exists() and repo_utils.exists():
        src_raw, repo_raw = src_utils.read_bytes(), repo_utils.read_bytes()
        identical = src_raw == repo_raw
        changed = []
        if not identical:
            diff = difflib.unified_diff(src_raw.decode("utf-8", "replace").splitlines(),
                                        repo_raw.decode("utf-8", "replace").splitlines(),
                                        "A-mem/utils.py", "utils.py", lineterm="", n=0)
            changed = [line for line in diff
                       if line[:1] in "+-" and not line.startswith(("+++", "---"))]
        comment_only = bool(changed) and all(l[1:].lstrip().startswith("#") for l in changed)
        local_vs_source.update({
            "utils_py_identical": identical,
            "utils_py_sha256_source": sha256_bytes(src_raw),
            "utils_py_sha256_local": sha256_bytes(repo_raw),
            "changed_lines": len(changed),
            "differences_are_comments_only": comment_only,
        })
        if identical:
            local_vs_source["note"] = (
                "byte-identical: the local BERTScorer-reuse patch is present in both copies")
        elif comment_only:
            local_vs_source["note"] = (
                f"differ in {len(changed)} comment line(s) and nothing else; no executable change, "
                f"so both score identically")
            local_vs_source["changed_examples"] = [l[:160] for l in changed[:4]]
        else:
            local_vs_source["note"] = (
                f"differ in {len(changed)} line(s) that are not comment-only; which side is "
                f"authoritative has to be re-checked before trusting either")
            local_vs_source["changed_examples"] = [l[:160] for l in changed[:8]]
    else:
        local_vs_source.update({"utils_py_identical": None,
                                "note": "utils.py missing on one side; comparison not possible"})

    upstream_utils = git(["show", "b31a1dd:utils.py"], source / "A-mem") if upstream_full else None
    if upstream_utils is not None:
        local_vs_source["upstream_utils_py_sha256"] = sha256_bytes(upstream_utils.encode())
        local_vs_source["upstream_has_bert_scorer_reuse"] = "_BERT_SCORER" in upstream_utils

    provenance = {
        "upstream": {
            "repository": "A-mem",
            "commit": upstream_full or "b31a1dd (could not resolve to a full sha in the local clone)",
            "commit_short": "b31a1dd",
            "commit_subject": upstream_subject,
            "resolved_locally": bool(upstream_full),
            "note": "commit b31a1dd is the upstream fix for the category-5 reference; this "
                    "repository builds on it rather than re-deriving it",
        },
        "local": {
            "head": local_head,
            "head_subject": local_subject,
            "working_tree_uncommitted": bool(local_status),
            "dirty_paths": local_status.splitlines() if local_status else [],
            "note": "computed at prepare time from git; not asserted clean",
        },
        "source_environment": {
            "path": str(source),
            "upstream_head": upstream_head,
            "upstream_dirty_paths": upstream_status.splitlines() if upstream_status else [],
        },
        "local_vs_source": local_vs_source,
    }

    manifest["prepared_from"] = str(source)
    manifest["sanitisation"] = "sanitization.json"

    # the optional ablation chain points at the harness's own log, not the console capture
    contexts_path = evidence / "temporal_contexts.json"
    if contexts_path.exists():
        contexts = read_json(contexts_path)
        source_log_sha = contexts.get("source_log_sha256")
        harness_logs = {}
        harness_dir = source / "A-mem" / "logs"
        if harness_dir.is_dir() and source_log_sha:
            for candidate in sorted(harness_dir.glob("*.log")):
                if sha256_file(candidate) == source_log_sha:
                    harness_logs[str(candidate.relative_to(source))] = source_log_sha
        manifest["temporal_contexts"] = {
            "evidence": "temporal_contexts.json",
            "sha256": sha256_file(contexts_path),
            "source_log_sha256": source_log_sha,
            "source_log_matches": harness_logs,
            "note": "the harness also writes its own log under A-mem/logs/; temporal_contexts.json "
                    "was extracted from that file, which is a subset of the console capture in "
                    "logs/run_real_t1.log, so their hashes differ by construction",
        }

    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "sanitization.json").write_text(json.dumps(sanitisation, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    (evidence / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    (evidence / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"prepared evidence in {evidence}")
    for entry in manifest["entries"]:
        print(f"  {entry['kind']:<20} {entry.get('evidence') or entry['source']:<40} {entry['sha256'][:12]}")
    return 0


# --------------------------------------------------------------------------- evidence hashes

def verify_manifest(evidence_dir, checks):
    path = Path(evidence_dir) / "manifest.json"
    if not path.exists():
        fail("evidence/manifest.json is missing; run --prepare on the experiment directory first")
    manifest = read_json(path)
    verified = 0
    for entry in manifest.get("entries", []):
        target = entry.get("evidence")
        if target:
            full = Path(evidence_dir) / target
        elif entry.get("kind") == "script":
            target = entry["source"]
            full = HERE / target
        elif entry.get("kind") == "dataset":
            target = "data/locomo10.json"
            full = HERE / target
        else:
            continue
        if not full.exists():
            fail(f"manifest lists {target} but the file is missing")
        actual = sha256_file(full)
        if actual != entry["sha256"]:
            fail(f"{target}: sha256 {actual} != manifest {entry['sha256']}")
        verified += 1
    checks.append({"id": "manifest.sha256", "ok": True,
                   "detail": f"{verified} shipped evidence file(s) match manifest.json"})
    return manifest


def verify_sanitisation(evidence_dir, checks):
    path = Path(evidence_dir) / "sanitization.json"
    if not path.exists():
        fail("evidence/sanitization.json is missing")
    doc = read_json(path)
    for entry in doc.get("files", []):
        full = Path(evidence_dir) / entry["evidence"]
        if not full.exists():
            fail(f"sanitised log missing: {full}")
        raw = full.read_bytes()
        if sha256_bytes(raw) != entry["sanitized_sha256"]:
            fail(f"{entry['evidence']}: sha256 does not match sanitized_sha256")
        if ANSI_RE.search(raw):
            fail(f"{entry['evidence']}: still contains ANSI escapes")
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(raw):
                fail(f"{entry['evidence']}: still contains a credential-like string ({name})")
        if LEFTOVER_PATH_RE.search(raw):
            fail(f"{entry['evidence']}: still contains an absolute machine path")
    checks.append({"id": "logs.sanitised", "ok": True,
                   "detail": f"{len(doc.get('files', []))} log(s) hash-verified and free of absolute "
                             f"paths, ANSI escapes and credential-like strings"})
    return doc


# --------------------------------------------------------------------------- report

def build_report(evidence_dir, resamples, bootstrap_seed):
    evidence = Path(evidence_dir)
    checks = []
    manifest = verify_manifest(evidence, checks)
    verify_sanitisation(evidence, checks)

    trials = verify_predictions(evidence / "predictions")
    checks.append({"id": "predictions.f1", "ok": True,
                   "detail": f"{sum(t['n'] for t in trials)} stored F1 values reproduce from "
                             f"prediction/reference (set-token F1)"})
    checks.append({"id": "predictions.gold_category", "ok": True,
                   "detail": f"{len(trials[0]['rows'])} questions share gold and category across "
                             f"all three trials"})
    checks.append({"id": "predictions.aggregate_metrics", "ok": True,
                   "detail": "mean/std/median/min/max/count of every metric and category reproduce"})

    legacy = build_category_table(trials, "legacy")
    corrected = build_category_table(trials, "corrected")
    temporal = build_temporal(trials, REPORTED_F1["Temporal"])
    cat5 = build_cat5(evidence / "predictions", resamples, bootstrap_seed)
    checks.append({"id": "cat5.load_trial_summarize", "ok": True,
                   "detail": f"analyze_c5.load_trial accepted all 3 trials; bootstrap "
                             f"{cat5['cluster_bootstrap']['resamples']} resamples, seed "
                             f"{cat5['cluster_bootstrap']['seed']}"})
    cost = build_cost(evidence, trials, checks)
    bert = build_bertscore(evidence)
    checks.append({"id": "bertscore.invariance", "ok": True,
                   "detail": f"{bert['values']} values bit-identical before/after the scorer change"})
    ablation = build_ablation(evidence, checks)
    identity = build_model_identity(evidence, trials, ablation)
    checks.append({"id": "model.identity", "ok": identity["status"] == "evidenced",
                   "informational": True,
                   "detail": "unresolved for the three scored trials: the logs are stdout without "
                             "the API model field, and no deepseek-flash string appears"
                             if identity["status"] != "evidenced"
                             else "API-returned model field present"})

    unresolved = []
    if identity["status"] != "evidenced":
        unresolved.append({
            "item": "model identity (deepseek-flash)",
            "reason": identity["detail"],
        })
    for row in cost["per_trial"]:
        if row["first_log_timestamp"] is None:
            unresolved.append({"item": f"{row['log']} timestamps",
                               "reason": "no timestamped INFO lines; runtime not derivable"})
    unresolved.append({
        "item": "wall-clock runtime",
        "reason": "only the first/last timestamped log line is recorded; process start-up and "
                  "model loading happen before the first line and are not measurable here",
    })
    if ablation["status"] == "failed":
        unresolved.append({"item": "temporal prompt ablation",
                           "reason": "integrity check failed: " + "; ".join(ablation["problems"][:3])})

    return {
        "inputs": {
            "evidence_dir": str(evidence),
            "manifest_entries": len(manifest.get("entries", [])),
            "prediction_sha256": {t["file"]: sha256_file(evidence / "predictions" / t["file"]) for t in trials},
        },
        "checks": checks,
        "s4_paper_reference": legacy,
        "s4_corrected_reference": corrected,
        "s4_temporal": temporal,
        "s5_cat5": cat5,
        "cost": cost,
        "bertscore": bert,
        "model_identity": identity,
        "ablation": ablation,
        "unresolved": unresolved,
        "notes": [
            "No API calls are made; every number is derived from shipped evidence.",
            "Per-category uncertainty is the sample std of the three per-trial category means, "
            "not a per-question std.",
            "The paper's category-5 convention scores against the trap answer, so lower is not "
            "better: it measures overlap with the trap, not refusal accuracy.",
            "The corrected-vs-paper adversarial delta is calculated before rounding: "
            "37.46499 displays as +37.46; subtracting rounded display columns would give +37.47.",
        ],
    }


def render_markdown(report):
    out = []
    add = out.append

    add("# Reproduced tables")
    add("")
    add("Generated by `reproduce_tables.py` from `evidence/` only. No API calls, no network. "
        "Category-5 interval: question-cluster percentile bootstrap, "
        f"{report['s5_cat5']['cluster_bootstrap']['resamples']:,} resamples, RNG seed "
        f"{report['s5_cat5']['cluster_bootstrap']['seed']}.")
    add("")

    add("## 0. Verification checks")
    add("")
    add("| Check | Result | Detail |")
    add("|---|---|---|")
    for c in report["checks"]:
        add(f"| `{c['id']}` | {'pass' if c['ok'] else 'UNRESOLVED'} | {c['detail']} |")
    add("")

    add("## 1. §4 — the paper's category-5 reference, three trials")
    add("")
    add("| Category | n | Reported (GPT-4o-mini) | Mine (paper's reference) | Δ |")
    add("|---|---|---|---|---|")
    for row in report["s4_paper_reference"]["rows"]:
        add(f"| {row['category']} | {row['n']} | {row['reported']:.2f} | "
            f"**{row['mean']:.2f} ± {row['std']:.2f}** | {row['delta']:+.2f} |")
    overall = report["s4_paper_reference"]["overall"]
    add(f"| Overall | {overall['n']} | not reported | {overall['mean']:.2f} ± {overall['std']:.2f} | — |")
    add("")
    add("Per-trial category means (the ± above is their sample std, n=3):")
    add("")
    add("| Category | seed 42 | seed 43 | seed 44 |")
    add("|---|---|---|---|")
    for row in report["s4_paper_reference"]["rows"]:
        add("| " + row["category"] + " | " + " | ".join(f"{v:.2f}" for v in row["per_trial_mean"]) + " |")
    add("| Overall | " + " | ".join(f"{v:.2f}" for v in overall["per_trial_mean"]) + " |")
    add("")

    add("## 2. §4 — corrected category-5 reference")
    add("")
    add("| Category | n | Corrected | Paper's reference | Δ |")
    add("|---|---|---|---|---|")
    paper = {row["category"]: row for row in report["s4_paper_reference"]["rows"]}
    for row in report["s4_corrected_reference"]["rows"]:
        delta = row["mean"] - paper[row["category"]]["mean"]
        add(f"| {row['category']} | {row['n']} | {row['mean']:.2f} ± {row['std']:.2f} | "
            f"{paper[row['category']]['mean']:.2f} | {delta:+.2f} |")
    oc = report["s4_corrected_reference"]["overall"]
    add(f"| Overall | {oc['n']} | {oc['mean']:.2f} ± {oc['std']:.2f} | "
        f"{overall['mean']:.2f} | {oc['mean'] - overall['mean']:+.2f} |")
    add("")

    t = report["s4_temporal"]
    add("## 3. §4.1 — Temporal regex screening (screening only, no date resolution)")
    add("")
    add("| | " + " | ".join(row["label"] for row in t["per_trial"]) + " |")
    add("|---|" + "---|" * len(t["per_trial"]))
    add("| Temporal questions | " + " | ".join(str(row["n"]) for row in t["per_trial"]) + " |")
    add("| Relative prediction and absolute gold | " +
        " | ".join(f"**{row['candidates']}** ({row['candidate_share']:.0f}%)" for row in t["per_trial"]) + " |")
    add("| F1 of that subset | " + " | ".join(fmt(row["candidate_f1"]) for row in t["per_trial"]) + " |")
    add("| F1 of the remaining questions | " + " | ".join(fmt(row["other_f1"]) for row in t["per_trial"]) + " |")
    add("| Temporal F1 overall | " + " | ".join(fmt(row["overall_f1"]) for row in t["per_trial"]) + " |")
    add("")
    add(f"Sensitivity estimate: if the {t['candidates_total']} screened candidates scored like the "
        f"remaining questions, Temporal would be **{t['counterfactual_mean']:.2f}** instead of "
        f"{t['actual_mean']:.2f} (gain {t['gain']:+.2f}, "
        f"{fmt(t['share_of_gap'], 0)}% of the {t['reported_gap']:.2f}-point gap to the reported "
        f"{REPORTED_F1['Temporal']:.2f}). {t['note']}; the candidates may be wrong and may differ in "
        f"difficulty, so format is a plausible partial contributor, not an established cause.")
    add("")

    c5 = report["s5_cat5"]
    add("## 4. §5 — category-5 refusal, failure modes, cross-tab")
    add("")
    add("| Trial | n | Correct | Rate | 95% Wilson CI | F1 vs refusal | F1 vs trap |")
    add("|---|---|---|---|---|---|---|")
    for row in c5["per_trial"]:
        add(f"| seed {row['seed']} | {row['n']} | {row['correct']} | {row['rate']:.2f} | "
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}] | {row['f1_refusal']:.2f} | {row['f1_trap']:.2f} |")
    p = c5["pooled"]
    add(f"| **pooled (naive)** | {p['n']} | {p['correct']} | **{p['rate']:.2f}** | "
        f"[{p['ci_low']:.1f}, {p['ci_high']:.1f}] | — | — |")
    add("")
    a = c5["across_trials"]
    b = c5["cluster_bootstrap"]
    add(f"Across {a['n_trials']} trials: **{a['mean']:.2f} ± {a['std']:.2f}** "
        f"(range {a['min']:.2f}–{a['max']:.2f}). Question-cluster 95% percentile interval: "
        f"**[{b['ci_low']:.1f}, {b['ci_high']:.1f}]** over {b['n_questions']} clusters x "
        f"{b['n_trials']} trials. {c5['pooled_interval_note']}.")
    add("")
    add("| Mode | Questions | Share |")
    add("|---|---|---|")
    for mode, count in p["modes"].items():
        add(f"| `{mode}` — {analyze_c5.MODE_NOTE[mode]} | {count} | {100.0 * count / p['n']:.1f}% |")
    add("")
    d = c5["disagreement"]
    add("| | F1 vs trap ≥ 0.5 | F1 vs trap < 0.5 |")
    add("|---|---|---|")
    add(f"| handled the trap | {d['f1_trap_high_and_right']} | {d['f1_trap_low_but_right']} |")
    add(f"| did not | {d['f1_trap_high_but_wrong']} | {d['f1_trap_low_and_wrong']} |")
    add("")
    add(f"Mean F1 against the trap: {fmt(d['f1_trap_correct'])} when the model refused, "
        f"{fmt(d['f1_trap_incorrect'])} when it did not. The two instruments disagree on "
        f"{d['n_disagree']}/{d['n']} observations ({d['pct_disagree']:.1f}%).")
    add("")

    cost = report["cost"]
    add("## 5. Cost (tokens measured, dollars modelled)")
    add("")
    add("| Item | " + " | ".join(row["label"] for row in cost["per_trial"]) + " |")
    add("|---|" + "---|" * len(cost["per_trial"]))
    fields = [("LLM calls", lambda r: f"{r['calls']:,}"),
              ("Failed calls", lambda r: f"{r['failed_calls']:,}"),
              ("Input tokens", lambda r: f"{r['prompt_tokens']:,}"),
              ("… cache hits", lambda r: f"{r['cached_tokens']:,} ({r['cached_share']:.0f}%)"),
              ("Output tokens", lambda r: f"{r['completion_tokens']:,}"),
              ("Rate (USD/M in/out)", lambda r: f"{r['price_in']}/{r['price_out']}"),
              ("Cost (upper bound)", lambda r: f"${r['cost_usd']:.2f}")]
    for name, fn in fields:
        add(f"| {name} | " + " | ".join(fn(r) for r in cost["per_trial"]) + " |")
    add("")
    add(f"Total **${cost['total_usd']:.2f}**. {cost['note']}.")
    add("")
    add("Log span (first and last timestamped line; start-up is outside it):")
    add("")
    for row in cost["per_trial"]:
        add(f"- `{row['log']}`: {row['first_log_timestamp']} → {row['last_log_timestamp']} "
            f"({row['timestamped_lines']:,} timestamped lines)")
    add("")

    bert = report["bertscore"]
    add("## 6. BERTScore invariance")
    add("")
    add(f"{bert['pairs']} fixed pairs × 3 scores = **{bert['values']} values**, "
        f"max absolute difference before/after the scorer-reuse change: **{bert['max_abs_diff']}** "
        f"({'bit-identical' if bert['identical'] else 'DIFFERS'}). {bert['note']}.")
    add("")

    ab = report["ablation"]
    add("## 7. Temporal prompt ablation (optional)")
    add("")
    if ab["status"] == "absent":
        add(ab["detail"] + ".")
    else:
        add(f"Status: **{ab['status']}**. {ab['calls']} answer calls, "
            f"{ab['prompt_tokens']:,} input / {ab['completion_tokens']:,} output tokens, "
            f"metered ${ab['meter_usd']:.4f}. Requested model `{ab['model_requested']}`, "
            f"returned {ab['models_returned']}.")
        add("")
        add("| Arm | n | F1 | Format candidates |")
        add("|---|---|---|---|")
        for arm in ("historical", "original", "absolute"):
            row = ab["summary"][arm]
            add(f"| {arm} | {row['n']} | {row['f1']:.2f} | "
                f"{row['format_candidates']} ({row['format_candidate_pct']:.1f}%) |")
        if ab["problems"]:
            add("")
            add("Integrity problems: " + "; ".join(ab["problems"][:4]))
        add("")
        add(f"Scope: {ab['scope']}.")
    add("")

    mi = report["model_identity"]
    add("## 8. Model identity evidence")
    add("")
    add(f"Status: **{mi['status']}**. `deepseek-flash` occurrences in the three run logs: "
        f"{mi['deepseek_flash_mentions_in_logs']}. Model tokens seen in logs: "
        f"{json.dumps(mi['model_tokens_seen_in_logs'], ensure_ascii=False)}. "
        f"`model` field in the result files: {json.dumps(mi['result_file_model_field'])}. "
        f"API-returned model field present: {mi['api_returned_model_field_present']}.")
    add("")
    add(mi["detail"])
    add("")

    add("## 9. Unresolved")
    add("")
    for item in report["unresolved"]:
        add(f"- **{item['item']}** — {item['reason']}")
    add("")

    add("## 10. Notes")
    add("")
    for note in report["notes"]:
        add(f"- {note}")
    add("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- self-test

def selftest(evidence_dir):
    """Real data must pass; four tampered variants must be rejected."""
    evidence = Path(evidence_dir)
    if not evidence.is_dir():
        fail(f"self-test needs the shipped evidence at {evidence}")
    results = []

    def expect_pass(name, fn):
        try:
            fn()
        except VerificationError as exc:
            print(f"FAIL {name}: unexpectedly rejected: {exc}")
            return False
        print(f"ok   {name}")
        return True

    def expect_fail(name, fn, needle):
        try:
            fn()
        except VerificationError as exc:
            if needle not in str(exc):
                print(f"FAIL {name}: rejected for the wrong reason: {exc}")
                return False
            print(f"ok   {name} (rejected: {str(exc)[:90]})")
            return True
        print(f"FAIL {name}: accepted a tampered input")
        return False

    # 1. real data
    results.append(expect_pass("real evidence reproduces",
                               lambda: verify_predictions(evidence / "predictions")))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def copy_predictions(target):
            """Mirror evidence/predictions into <target>/predictions and return that path."""
            pred_dir = target / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            for spec in TRIALS:
                shutil.copyfile(evidence / "predictions" / spec["predictions"],
                                pred_dir / spec["predictions"])
            return pred_dir

        # 2. tampered F1
        tampered = copy_predictions(tmp / "f1")
        path = tampered / TRIALS[1]["predictions"]
        data = read_json(path)
        data["individual_results"][7]["metrics"]["f1"] = min(
            1.0, float(data["individual_results"][7]["metrics"]["f1"]) + 0.01)
        path.write_text(json.dumps(data), encoding="utf-8")
        results.append(expect_fail("tampered stored F1 rejected",
                                   lambda: verify_predictions(tampered), "f1"))

        # 3. a question removed from one trial, with the trial's own bookkeeping kept consistent,
        #    so only the cross-trial question-set comparison can catch it
        missing = copy_predictions(tmp / "missing")
        path = missing / TRIALS[2]["predictions"]
        data = read_json(path)
        dropped = data["individual_results"].pop(5)
        data["total_questions"] = len(data["individual_results"])
        key = str(int(dropped["category"]))
        data["category_distribution"][key] -= 1
        path.write_text(json.dumps(data), encoding="utf-8")
        results.append(expect_fail("missing question rejected",
                                   lambda: verify_predictions(missing), "question set differs"))

        # 4. hash mismatch against the manifest
        bad_manifest = tmp / "hash"
        copy_predictions(bad_manifest)
        manifest = read_json(evidence / "manifest.json")
        manifest["entries"] = [e for e in manifest["entries"]
                               if e.get("evidence", "").startswith("predictions/")]
        for entry in manifest["entries"]:
            entry["sha256"] = "0" * 64
        (bad_manifest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        checks = []
        results.append(expect_fail("hash mismatch rejected",
                                   lambda: verify_manifest(bad_manifest, checks), "sha256"))

    ok = all(results)
    print(f"\nself-test: {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="directory for tables.json and tables.md")
    ap.add_argument("--evidence-dir", default=str(HERE / "evidence"),
                    help="shipped evidence directory (default: <script>/evidence)")
    ap.add_argument("--prepare", action="store_true",
                    help="rebuild evidence/ from --source-dir (whitelist copy + log sanitisation)")
    ap.add_argument("--source-dir", help="experiment directory (contains A-mem/ and the logs)")
    ap.add_argument("--selftest", action="store_true", help="run the zero-dependency self-test")
    ap.add_argument("--bootstrap-resamples", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261026)
    args = ap.parse_args(argv)

    evidence = Path(args.evidence_dir).resolve()

    if args.selftest:
        try:
            return selftest(evidence)
        except VerificationError as exc:
            print(f"self-test setup failed: {exc}", file=sys.stderr)
            return 1

    if args.prepare:
        if not args.source_dir:
            ap.error("--prepare requires --source-dir")
        try:
            return prepare(args.source_dir, evidence)
        except VerificationError as exc:
            print(f"prepare failed: {exc}", file=sys.stderr)
            return 1

    if not args.out_dir:
        ap.error("--out-dir is required (or use --prepare / --selftest)")
    if args.bootstrap_resamples < 1000:
        ap.error("--bootstrap-resamples must be >= 1000")
    out_dir = Path(args.out_dir).resolve()
    if out_dir == evidence or evidence in out_dir.parents:
        ap.error("--out-dir must not be the evidence directory or inside it")

    try:
        report = build_report(evidence, args.bootstrap_resamples, args.bootstrap_seed)
    except VerificationError as exc:
        print(f"verification failed, no tables written: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    (out_dir / "tables.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {out_dir / 'tables.json'}")
    print(f"wrote {out_dir / 'tables.md'}")

    # informational checks document what the evidence cannot show; they do not fail the run
    blocking = [c for c in report["checks"] if not c["ok"] and not c.get("informational")]
    return 1 if blocking or report["ablation"]["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
