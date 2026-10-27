# Reproducibility Report: A-Mem (Agentic Memory for LLM Agents)

**Course:** FTEC5660 Agentic AI for Business and FinTech
**Name:** Qi Zheng  **Student ID:** 1155268235
**Repository reproduced:** `WujiangXu/A-mem` @ `b31a1dd` (2026-10-21)
**Scope and status.** Measured results come from **three QA trials sharing one memory build**
(`conv-26`, 199 questions, `deepseek-chat`, seeds 42/43/44). Paper values are reference points,
not matched-condition baselines. Standard deviations describe fixed-memory QA variation, not
end-to-end stability; cost projections and Temporal sensitivity estimates are labelled as estimates.

---

## 1. Project summary and reproduction target

**The system.** A-Mem is a memory layer for LLM agents built on the Zettelkasten principle.
Instead of storing turns verbatim and retrieving by cosine similarity, each turn becomes a
*note* with LLM-generated keywords, a context sentence, and tags. When a new note arrives, the
system retrieves its nearest neighbours and asks the LLM whether the new note should
**strengthen** itself, **update** its neighbours, or trigger **no evolution**. The result is a
memory graph whose nodes are rewritten over time.

**What I tried to reproduce.** Table 1 of the paper (NeurIPS 2025), the LoCoMo benchmark F1
scores, broken down by question category. LoCoMo is a long-conversation QA benchmark; each
conversation is ~400 turns across several sessions, and questions are labelled into five
categories.

**Scope decision.** I reproduce **one conversation** (`conv-26`, 419 turns, 199 questions) rather
than the full 10-conversation / 1986-question benchmark, and I compare against the paper's
**GPT-4o-mini** column. This is the "one dataset split" subset the assignment explicitly allows.

**Two modifications, ordered by role.** The **primary modification** is a controlled Temporal prompt ablation with a measured F1 change (§5.1). The **additional improvement** is a reusable category-5 evaluation workflow (§5.2). An upstream scoring fix by the authors is audited separately and is **not** claimed as mine (§5.3).

**Why this repo.** It is unusual in that the README states the repository exists *specifically*
to reproduce the paper's results, and it ships a `test_advanced_robust.py` harness that works
with any OpenAI-compatible backend. That makes the model substitution (see §2.2) a
configuration change rather than a code change, which keeps the comparison clean.

---

<figure style="margin:10px 0;break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 108" role="img" aria-label="Experiment flow (measured subset)" style="display:block;width:100%;height:auto">
<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#185FA5"/></marker></defs>
<rect x="1" y="12" width="120" height="57" rx="6" fill="#E6F1FB" stroke="#B5D4F4"/><rect x="139" y="12" width="120" height="57" rx="6" fill="#E6F1FB" stroke="#B5D4F4"/><rect x="277" y="12" width="120" height="57" rx="6" fill="#E6F1FB" stroke="#B5D4F4"/><rect x="415" y="12" width="120" height="57" rx="6" fill="#F4F2EC" stroke="#D8D6CC"/><rect x="553" y="12" width="126" height="57" rx="6" fill="#EAF3E2" stroke="#C5D8B4"/>
<g stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrow)"><path d="M123 40H135"/><path d="M261 40H273"/><path d="M399 40H411"/><path d="M537 40H549"/></g>
<g fill="#1F1F1D" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="12"><text x="61" y="35">LoCoMo</text><text x="199" y="35">A-Mem memory</text><text x="337" y="35">DeepSeek QA</text><text x="475" y="35">Evaluation</text><text x="616" y="35">analyze_c5.py</text></g>
<g fill="#57564F" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="10"><text x="61" y="54">conv-26 · 199 Q</text><text x="199" y="54">419 notes · one build</text><text x="337" y="54">three QA repeats</text><text x="475" y="54">stored F1 / flags</text><text x="616" y="54">checks · CI · cross-tabs</text><text x="340" y="94">Temporal supporting test: 37 exact logged contexts → paired original / absolute-date prompts</text></g>
</svg>
<figcaption style="font-size:8.5pt;color:#57564F">Figure 1. Experiment flow (measured subset). Post-processing changes evaluation quality, not stored predictions.</figcaption>
</figure>

## 2. Setup

### 2.1 Environment

| Item | Value |
|---|---|
| OS / hardware | macOS (Apple Silicon), CPU only, no GPU |
| Python | 3.13.12 (repo README says 3.9; works on 3.13 without modification) |
| Venv | isolated `venv/`, ~1.4 GB (torch + transformers + sentence-transformers) |
| Dataset | `data/locomo10.json` shipped in-repo, 2.7 MB |
| Data scale | 10 conversations, 5,882 turns, 1,986 QA pairs |

Category distribution of the full dataset: multi-hop 282, temporal 321, open-domain 96,
single-hop 841, adversarial 446. Adversarial questions are **22.5%** of the benchmark — this
matters in §5.

### 2.2 Model substitution (Edge case A)

The paper's main table uses GPT-4o-mini. I substitute **DeepSeek** (`deepseek-chat`), treating
the model as a controlled variable.

**This required no code change.** A-Mem's `RobustOpenAIController` constructs
`OpenAI(api_key=...)` without a `base_url`, but the `openai` Python SDK falls back to the
`OPENAI_BASE_URL` environment variable. Verified empirically:

```
$ OPENAI_BASE_URL=https://api.deepseek.com/v1 OPENAI_API_KEY=sk-... python -c \
    "from openai import OpenAI; print(OpenAI().base_url)"
https://api.deepseek.com/v1/
```

Recorded variables:

| Variable | Value |
|---|---|
| Model (requested) | `deepseek-chat` |
| Model (as returned by API) | `deepseek-flash` in the 74-call ablation only; original trials did not retain this field |
| Provider | DeepSeek |
| Base URL | `https://api.deepseek.com/v1` |
| temperature | repo defaults: 0.7 for categories 1–4, 0.5 for category 5 |
| max_tokens | 1000 (hard-coded in the repo) |
| top_p | not set (provider default) |
| Prompt templates | unmodified in original trials; only the date instruction changes in §4.1 / §5.4 |

**Model-field evidence.** The original result JSON records the *requested* model, `deepseek-chat`;
its logs did not retain the API-returned `model` field. The new ablation preserves that field in
all 74 response records: each returns `deepseek-flash`. This is an API response value, not an
independently verified checkpoint identity, and cannot retroactively identify the original trials.
Alias resolution may change server-side; comparison with the new paired runs is conditional on
this limitation.

**Explicit caveat.** Results are **not directly comparable to the paper**, because the
underlying LLM differs. I compare against the GPT-4o-mini column as the closest available
reference; any gap may reflect model, subset selection, memory construction, prompts or environment.
Without matched baselines, this setup cannot isolate A-Mem's causal contribution.

### 2.3 Compute and cost (tokens measured, dollars modelled)

Instrumented with `meter.py`, which patches the OpenAI SDK's `Completions.create` and reads
`response.usage`; the repository is untouched.

| Item | Trial 1 (seed 42) | Trial 2 (seed 43) | Trial 3 (seed 44) |
|---|---|---|---|
| LLM calls | **1,723** (memory + QA) | **398** (QA only, memories reused) | **398** (QA only, memories reused) |
| Failed calls | 0 | 0 | 0 |
| Input tokens | 3,843,050 | 2,596,009 | 2,599,264 |
| … of which cache hits | 165,504 (4%) | 1,018,236 (39%) | 1,345,400 (52%) |
| Output tokens | 135,040 | 4,107 | 4,182 |
| Average per call | 2,230 in / 78 out | 6,523 in / 10 out | 6,531 in / 11 out |
| Cost (upper bound) | **$1.13** | **$0.73** | **$0.73** |
| Wall-clock | 44 min | 14 min | 13 min 46 s |

**Total: $2.59** across the three original completed trials (the ablation adds ~$0.275), plus a few cents on an attempt that aborted
at question 8/199 before its meter report was written.

**Read those dollars as an upper bound, not a bill.** The token counts are exact — they come from
the API's own `usage` field — but `meter.py` prices *every* input token at list, and 39–52% of
the input in trials 2 and 3 came from the provider's cache, which is billed at a fraction of
list. The real charge is lower; I did not model the discount because I could not verify the rate
for this account tier.

The cache is keyed by `backend + model`, not seed. Trials 2 and 3 reuse trial 1's memory and
make **398 QA calls** each (two per question); the first run therefore adds **1,325 memory calls**
(1,723 − 398). The first-run average, 2,230 input tokens, mixes memory building and QA — it is
**not a memory-only average**. QA-only trials average ~6,530 input tokens and ~10 output tokens.

**My initial ≈$0.22 estimate was ~5× too low** against the $1.13 meter estimate. Full-benchmark
projection must separate phases: **5,882 / 419 ≈ 14.04×** as many turns, but **1,986 / 199 ≈
9.98×** as many questions. Assuming the measured memory calls per turn and QA calls per question
carry over gives ~18,600 memory calls + **3,972 QA calls**, or **~22,600 first-run calls**.
At the same rounded, no-cache unit-cost assumptions, QA reuse projects to **~$7.3** per full
benchmark trial; estimating memory cost as ($1.13 − $0.73) × 14.04 gives ~$5.6, hence **~$12.9
for a first run**. This subtraction uses QA-only trials as a proxy for trial 1's QA cost, not
phase-resolved metering. These are rough scenarios, not measured totals or guaranteed upper bounds
for unseen conversations. The $2.59 upper estimate applies only to the completed trials.

Time remains material: ~44 minutes for the first conversation and ~14 minutes per reused-memory
QA trial; full-benchmark runtime has not been measured.

---

## 3. Reproduction protocol

### 3.1 Target and metric

**Target claim:** A-Mem's per-category F1 on LoCoMo (Table 1).
**Metric:** token-level F1 between the predicted answer and the gold answer, averaged per
category. The repo also computes BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, SBERT similarity and
BERTScore; F1 is the headline metric in the paper and is what I report.

Reported values (GPT-4o-mini backbone, paper Table 1):

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial |
|---|---|---|---|---|---|
| LoCoMo | 25.02 | 18.41 | 12.04 | 40.36 | 69.23 |
| MemGPT | 26.65 | 25.52 | 9.15 | 41.04 | 43.29 |
| **A-Mem** | **27.02** | **45.85** | **12.14** | **44.65** | **50.03** |

### 3.2 Determinism controls

| Control | Setting |
|---|---|
| Seed | `--seed 42` (governs the order of the two options in category-5 prompts) |
| Retrieval k | `--retrieve_k 10` (repo default) |
| Memory scope | `--memory_scope conversation` (LoCoMo protocol, as in the paper) |
| Category-5 reference | `--legacy_c5_reference` off (see §5) |
| Dataset | pinned to the in-repo file; no download at run time |

### 3.3 Trials

**Three QA trials are complete**, seeds 42/43/44, zero failures in the completed runs. An earlier
seed-44 attempt aborted at question 8/199 with `HTTP 402 Insufficient Balance`; after a top-up it
completed in 13 min 46 s and only the completed result enters the statistics.

**These are not three independent end-to-end reproductions.** Memory was built once in trial 1
and reused in trials 2 and 3 (their result metadata records `cached_samples: [0]`). The local
`--seed` controls only category-5 option ordering, not provider-side LLM randomness. Mean ±
sample std therefore describes repeated stochastic QA **conditional on one fixed memory state**;
it does not estimate variation from stochastic memory construction or across conversations.
Wilson intervals per trial are descriptive binomial summaries. For combined category-5 results,
§5.3 retains each question's three outcomes together in a cluster bootstrap, rather than treating
47 × 3 observations as 141 independent questions.

### 3.4 Commands

```bash
# one conversation (conv-26)
./run_eval.sh --ratio 0.1 --backend openai --retrieve_k 10 --seed 42 \
              --output result_conv26_new.json

# same predictions, re-scored with the paper's category-5 reference
./venv/bin/python rescore_c5.py result_conv26_new.json
```

`run_metered.py` is a drop-in replacement for the above that additionally reports token counts
and cost (it patches the OpenAI SDK's `Completions.create`; the repository code is untouched).

---

## 4. Results: my numbers vs reported

One conversation (`conv-26`, 199 questions), `deepseek-chat`, `retrieve_k=10`, **three trials**
(seeds 42, 43, 44). F1 in percent, mean ± std.

The adversarial row uses the paper's reference convention, not matched experimental conditions;
the corrected score is underneath. Lower trap-overlap F1 is associated with successful refusal
in these stored outputs (§5), but is not an accuracy metric. All Δ values are **descriptive**:
a different model and one selected conversation are compared with the paper's full benchmark.

| Category | n | Reported (GPT-4o-mini) | Mine (paper's reference) | Δ |
|---|---|---|---|---|
| Multi-Hop | 32 | 27.02 | **27.04 ± 4.40** | **+0.02** |
| Temporal | 37 | 45.85 | **21.56 ± 2.44** | **−24.29** |
| Open-Domain | 13 | 12.14 | **13.88 ± 1.75** | +1.74 |
| Single-Hop | 70 | 44.65 | **43.08 ± 1.36** | −1.57 |
| Adversarial | 47 | 50.03 | **31.93 ± 2.05** | −18.10 |
| Overall | 199 | not reported | 31.96 ± 1.11 | — |

With the corrected category-5 reference: **Adversarial 69.40 ± 1.70**, **Overall 40.81 ± 1.43**.
Zero LLM failures, zero stale caches in any of the three trials.

**Multi-Hop's +0.02 mean difference is not an equivalence test.** Trials scored **31.37, 22.58
and 27.17**, an 8.79-point spread on the same 32 questions and memory. Numerical closeness to
27.02 does not establish a reproducible match or relative improvement over another method.

**Single-Hop is descriptively close** (−1.57, std 1.36, n=70). This is a smaller observed QA
spread, not a tight bound on the full-benchmark difference. **Open-Domain** is also numerically
close (+1.74, std 1.75), but n=13 is small. No matched-condition baseline was run for either.

**Temporal is substantially lower** in all three fixed-memory QA trials (−24.29 on the mean,
std 2.44). This persistence does not rule out question-selection effects, model differences or
memory-construction effects. §4.1 investigates one possible contributing mechanism.

**Adversarial** is 31.93 ± 2.05 under the paper's reference versus 50.03 in the paper. The −18.10
is a difference in trap-overlap F1, not an 18-point difference in bait rate. Backbone and dataset
scope still differ; sharing a scoring convention alone does not make the comparison controlled.

### 4.1 Temporal: a possible format contribution, not a complete explanation

Across the trials, **17 to 19 of 37 temporal predictions scored exactly 0**, and only one of
111 QA observations reached 1.0. Some answers use relative dates while the reference is absolute.
This can lower token F1 without proving a reasoning error; it also does **not** prove the relative
answer is correct. Examples below are format-mismatch candidates, not a validated correctness set:

| Question | Gold | DeepSeek's answer |
|---|---|---|
| When did Caroline go to the LGBTQ support group? | `7 May 2023` | `Yesterday` |
| When did Melanie paint a sunrise? | `2022` | `Last year` |
| When is Melanie planning on going camping? | `June 2023` | `Next month` |
| When is Caroline going to the transgender conference? | `July 2023` | `This month` |

I verified the first case against the dataset. `session_1` is dated **"1:56 pm on 8 May, 2023"**,
and the turn reads *"I went to a LGBTQ support group yesterday"*. **If anchored to that evidence
session**, `Yesterday` resolves to 7 May 2023, matching the gold, yet shares no tokens with it.
This verifies that one instance is recoverable with the right date anchor, not that every relative
answer is correct. `Last year`, `Next month` and `This month` each require the relevant evidence
session's date, not an assumed common anchor.

`analyze_temporal.py` performs **regex screening only** (relative prediction + absolute gold).
It neither resolves dates nor labels the remaining answers as genuine reasoning errors:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Temporal questions | 37 | 37 | 37 |
| Prediction uses a relative time expression **and** gold is absolute | **13** (35%) | **14** (38%) | **14** (38%) |
| F1 of that subset | **4.12** | **5.61** | **7.40** |
| F1 of the remaining questions | **29.97** | **28.01** | **34.53** |
| Temporal F1 overall | 20.89 | 19.54 | 24.27 |

**Sensitivity estimate, not a causal decomposition.** If screened candidates scored at the
same mean as the remaining questions, Temporal would be **≈30.84** rather than 21.56: a gain of
**≈9.28**, about **38% of the 24.29-point gap** (roughly 9 of 24). This is a hypothetical
replacement of subgroup scores, **not an actual date-normalisation re-score**. The candidates
may include incorrect answers and differ in difficulty. Thus format is a plausible partial
contributor; neither that ~9 nor the remaining ~15 can be assigned conclusively to format or
model capability without further investigation.

**From estimate to experiment.** §5.1 replaces this hypothetical subgroup calculation with an actual paired prompt intervention on the same 37 questions and the same logged retrieval contexts. The ~9.28 figure above stays a hypothetical sensitivity estimate; it is not the ablation result and not a causal decomposition.

### 4.2 Degenerate stub check

The zero-cost stub answers `Not mentioned in the conversation` to every question. It obtains
**100.00 category-5 F1** and **26.38 Overall**, despite failing the answerable task. Under the
fixed reference, 444/446 category-5 questions have no gold answer. This demonstrates a degenerate
response policy, not a matched-condition memory baseline; refusal must be read alongside
categories 1–4 (§5.5). Full stub logs are retained locally.

---

## 5. Two modifications, ordered by role

**Contribution boundary.** §5.1 is the **primary modification**: a single, isolated intervention
on the answer prompt, measured on fixed memory and fixed retrieval contexts. §5.2 is an
**additional evaluation improvement**: a reusable reporting and uncertainty workflow that changes
how results are read, not how the model answers. §5.3 audits an upstream scoring fix that belongs
to the **authors**. None of the three changes A-Mem's memory algorithm, retrieval or note
construction.

### 5.1 Primary modification — Temporal prompt ablation (23.66 → 57.49)

**Design.** The Temporal category is the largest gap to the paper (−24.29) and the diagnosis in
§4.1 pointed at answer *form* rather than retrieval. To test that directly I held everything
except one instruction fixed:

| Held fixed | Changed |
|---|---|
| the 37 Temporal questions of `conv-26` | one sentence of the category-2 answer instruction |
| the exact answer prompt and retrieved context recorded in the seed-42 log | — |
| stored memory (419 notes), `retrieve_k = 10`, no query regeneration | — |
| system message, requested model, temperature 0.7, `max_tokens` 1000 | — |
| `utils.calculate_metrics` (token F1) and `analyze_temporal.REL/ABS` | — |

Reusing the logged prompts means the comparison cannot be confounded by a different retrieval
draw: both arms see byte-identical context. The two instructions differ only in how a date must
be expressed:

```
control:   Use DATE of CONVERSATION to answer with an approximate date.
treatment: Use the DATE of the relevant evidence CONVERSATION session to answer with an
           absolute date. Convert relative expressions such as yesterday, last year, or next
           month using that session's date. Return a day as D Month YYYY, a month as
           Month YYYY, or a year as YYYY; keep only the precision supported by the evidence.
           Do not answer with relative date words or invent a missing date.
```

No gold date is inserted into either prompt. Arms are interleaved and the first arm alternates, so
drift in provider behaviour cannot systematically favour one arm: **37 answers per arm, 74 calls,
zero failed calls**, no API sampling seed.

**Result.** The intervention moves Temporal F1 by **+33.84 points** and removes most
format-screened candidates:

| Condition (same 37 questions, same contexts) | Temporal F1 | Format candidates | Share |
|---|---|---|---|
| Historical seed 42 — context source only | 20.89 | 13 / 37 | 35.1% |
| Control: original instruction, fresh run | **23.66** | **13 / 37** | **35.1%** |
| Treatment: absolute-date instruction | **57.49** | **2 / 37** | **5.4%** |
| Treatment − fresh control | **+33.84** | **−11** | **−29.7 pt** |

The correct reference point is the **fresh control**, not the three-trial mean: re-asking the same
questions already moves F1 from 20.89 to 23.66 with no prompt change, which is provider
stochasticity on identical inputs. Against that baseline the prompt accounts for +33.84.

**What this measures, and what it does not.** The candidate share is the unchanged regex screen, a
format proxy rather than a verified date-correctness rate; the two remaining candidates were not
manually adjudicated. Each arm is **one stochastic answer per question**, so the figure is a
measured paired difference, not an estimate of repeat-run variability, a confidence interval, or a
significance test. All 74 responses report API `model: deepseek-flash` (§2.2). **57.49 is not a
comparison with the paper's 45.85**, which used another model on the full benchmark, and the
historical 24.29-point gap is not thereby assigned to format. Usage: 980,412 input and 307 output
tokens, **$0.2746** at the original meter's no-cache rates — an upper estimate, not an invoice.

### 5.2 Additional evaluation improvement — category-5 diagnostic workflow

`analyze_c5.py` is the second, independent contribution: a **reusable post-processing workflow**,
not a new predicate and not a performance change. Upstream already stores `adversarial_correct`
(1 when the answer declines and does not echo the trap), but no table reads it. The workflow:

1. Independently recomputes token F1 and refusal flags and **asserts agreement with the stored
   values on all 141 QA observations** — a consistency check, not semantic validation.
2. Reports the binary refusal rate with per-trial Wilson summaries and a **question-cluster
   bootstrap** that keeps each question's three outcomes together.
3. Classifies exclusive failure modes and cross-tabulates refusal against F1 under both references,
   emitting JSON and tables that later runs can reuse.

| Evaluation aspect | Before: upstream reporting | After: this workflow |
|---|---|---|
| Consistency assertions | stored F1 / flags, no independent audit | F1 and refusal flags re-derived on 141 observations |
| Uncertainty | refusal interval not surfaced | 47 clusters, 20,000 draws, seed 20261026 → [56.7, 79.4] |
| Failure classification | no exclusive breakdown | 43 trap answers, 2 other, 0 refusal-plus-trap |
| Cross-tabulation | F1 and flag reported apart | dual-reference F1 × refusal; 139/141 disagreement at F1 ≥ 0.5 |

It reads stored predictions, makes **zero API calls** and changes no model output. Its benefit is
**evaluation quality, not model accuracy**.

| Trial | Correct / 47 | Rate | 95% Wilson CI |
|---|---|---|---|
| seed 42 | 31 | 65.96 | [51.7, 77.8] |
| seed 43 | 32 | 68.09 | [53.8, 79.6] |
| seed 44 | 33 | 70.21 | [56.0, 81.3] |
| **pooled (naive)** | **96 / 141 observations** | **68.09** | **[60.0, 75.2]** |

The pooled Wilson interval is **naive: it ignores repeated-question dependence**. Resampling the
**47 `(sample_id, question)` clusters**, each retaining all three outcomes, gives a 95% percentile
interval of **[56.7, 79.4]** (20,000 resamples, local RNG seed 20261026). It is exploratory and
conditional on the stored memory and trials; it does not model dependence between different
questions of one conversation or uncertainty from rebuilding memory, and it is not a full-benchmark
interval. Across fixed-memory QA trials: **68.09 ± 2.13**.

The cross-tab is the diagnostic that matters: of **96 successful refusals**, none has F1-vs-trap
≥ 0.5; of the 45 failures, **43** do. Mean trap F1 is **2.11** for refusals and **95.56** for
failures, so the instruments disagree on **139 / 141 observations (98.6%)** — a near-inverse
relationship on this sample, not proof of how the column ranks other models.

### 5.3 Upstream scoring fix: attribution and measured effect

This is the **authors'** change, included for completeness. Commit `b31a1dd` (2026-10-21) states
that "picking the trap scored high" and that `--legacy_c5_reference` restores "what the paper's
numbers used" [3]. I verified the parent commit's code rather than the message:

```python
# pre-fix runner
metrics = calculate_metrics(prediction, qa.final_answer)
return self.adversarial_answer      # category 5: trap option used as the reference
# post-fix runner
metrics = score_prediction(prediction, qa, legacy_c5_reference)   # -> NOT_MENTIONED
```

Re-scoring the **same 199 predictions** under both conventions isolates a pure metric effect:

| Category | n | Corrected reference | Paper's reference | Δ |
|---|---|---|---|---|
| Multi-Hop | 32 | 27.04 | 27.04 | 0.00 |
| Temporal | 37 | 21.56 | 21.56 | 0.00 |
| Open-Domain | 13 | 13.88 | 13.88 | 0.00 |
| Single-Hop | 70 | 43.08 | 43.08 | 0.00 |
| **Adversarial** | 47 | **69.40** | **31.93** | **+37.46** |
| **Overall** | 199 | **40.81** | **31.96** | **+8.85** |

Per trial the adversarial shift is +34.66, +36.26 and +41.47; four of five categories are
bit-identical, which is the direct check that the change is confined to category 5. No character of
model output changed, so a table mixing the conventions is not comparable. Whether the historical
*baseline* rows used the same convention is **not verifiable from inspected public artifacts**; I
did not locate their evaluation code. The repair is theirs, and the measured effect is reported
here as an audit.

### 5.4 Recommendation

Report category 5 separately as a **binary refusal rate with a dependence-aware interval**, and
keep F1 for categories 1–4 rather than mixing unlike instruments into a headline total. The
predicate is phrase-based and can miss paraphrases; a constant-refusal stub also scores **100** on
the binary metric, so read it beside the answerable categories, never alone. For Temporal, require
an explicit date form and re-test it on more than one conversation before treating the gain as a
property of the system rather than of this prompt on these contexts.

---

## 6. Debug diary: main blockers and verified fixes

| Blocker | Resolution / reproducibility lesson |
|---|---|
| `rouge-score` sdist extraction failed (`EEXIST`) | install from locally extracted PyPI sdist; keep exact version |
| NLTK download blocked / corpora rejected as writable | use official `nltk_data` packages and remove other-user write permission |
| Flat modules / script-relative output paths | use repo-root commands; insert repo on import path for external drivers |
| Interrupted background job / HTTP 402 at question 8 | managed job, check completion and usage; exclude aborted run from statistics |

### 6.1 BERTScore: diagnose, optimise, verify

A no-network stub run still took **18 min 32 s**. Profiling found **7.83 s/question** in
`calculate_metrics`, including **6.07 s (77%)** in BERTScore. The library reloaded ~1.4 GB
`roberta-large` weights for each question. In `utils.py`, I reuse one lazy `BERTScorer`
(+24/−1 executable patch). This is an **engineering fix**, separate from the formal evaluation
modification. Unoptimised metric time extrapolates to ~4.3 hours for 1,986 questions, ~12.9 hours
for three trials; that excludes memory-building and network work.

Verification: **8 pairs × 3 scores = 24 values bit-identical**; two complete 199-question dry runs
had **0 prediction and 0 metric differences**, max |Δ| = 0.00e+00. Metric time changed
**7.83 → 0.12 s/question (~65×)**; stub wall-clock changed **18:32 → 3:12**. Published before/after
logs and value files are included in `evidence/`. Original real-trial wall-clock durations are
approximate: saved timestamps cover the evaluation, not all process startup/model loading.

---

## 7. Conclusions

**What was reproduced.** One memory build plus three QA trials completed with zero failures
across 2,519 calls (1,723 + 398 + 398). Multi-Hop 27.04 vs 27.02, Single-Hop 43.08 vs 44.65 and
Open-Domain 13.88 vs 12.14 are numerically close, while Temporal and Adversarial differ markedly.
The paper does not report a directly comparable Overall F1; local category-level closeness must
not be described as overall agreement.

The reproduction demonstrates that A-Mem can achieve broadly comparable performance on selected
LoCoMo categories under a different backbone model. However, without rerunning the baseline
methods under identical conditions, this experiment cannot independently verify A-Mem's relative
improvement over naive retrieval. **Neither superiority nor failure of the memory method has
been established here.** The paper's LoCoMo and MemGPT rows were quoted, not reproduced.

**Uncertainty and scope.** Multi-Hop's 8.79-point spread makes the +0.02 mean difference fragile.
Single-Hop has smaller observed QA variation, but no category supports a matched-condition
performance claim. With only three trials sharing one memory state, std is weakly estimated and
says nothing about end-to-end rebuild stability. More QA repeats do not replace independent
memory builds or more conversations.

**What the diagnostics establish.**

1. **Temporal (primary modification):** the paired fixed-context prompt test measured
   **23.66 → 57.49 F1** (+33.84) and format candidates **13/37 → 2/37**. One stochastic pass per
   arm supports prompt sensitivity on these contexts, not a stable accuracy effect, a significance
   claim or paper outperformance. The earlier ~9.28 subgroup estimate remains hypothetical.
2. **Adversarial (additional improvement):** applying the authors' upstream reference fix to
   identical outputs changes F1 by **+37.46** (Overall **+8.85**) without changing behaviour. My
   added workflow reports **68.09 ± 2.13%**, exploratory 95% cluster interval **[56.7, 79.4]** and
   **98.6%** disagreement with trap F1, plus consistency assertions, failure attribution and
   cross-tabs. No performance gain or new refusal predicate is claimed.

**Resources.** Completed-trial no-cache meter estimates total **$2.59**, not the account bill.
A first full-benchmark run and a reused-memory QA trial have different costs: the rough scenarios
are **~22,600 calls / $12.9** and **3,972 calls / $7.3**, respectively (§2.3), not measured totals.
The BERTScore reuse fix removes a measured local bottleneck (~65× per-question speedup); overall
runtime still includes memory/LLM work and should not be attributed entirely to metric computation.

**Recommendations.** Pin the commit and retain API-returned model fields; report category 5
separately with its behavioural and dependence limitations; validate date anchors before diagnosing
Temporal errors; and cache BERTScore. A bounded prompt ablation was performed; **matched-condition
baseline runs were not**. No additional research scope is claimed.

---

## Appendix A: Reproduction artifacts and repository status

**Repository:** https://github.com/77qii/FTEC5660-Reproducibility
**Final commit:** `662402da83edaf57f8b9a56abd1cc4af2cf85d73`
All code, evidence and result files are in that commit. The immediately following commit changes only this line so the SHA can be recorded; `evidence/provenance.json` holds the upstream commit and the tree state at evidence-preparation time.

**One-command offline table reproduction** (Python 3, standard library only, no key/network):

```bash
python3 reproduce_tables.py --out-dir reproduced_tables
python3 reproduce_tables.py --selftest
```

| Artifact | Purpose |
|---|---|
| `evidence/predictions/*.json` | three original 199-question result files, copied byte-for-byte |
| `evidence/logs/`, `evidence/baseline/` | three metered runs and BERTScore before/after evidence; logs sanitised |
| `evidence/temporal_contexts.json` | exact 37 historical answer prompts and context hashes |
| `evidence/temporal_ablation.json`, `temporal_requests.jsonl` | paired predictions, all response usage / returned model fields |
| `evidence/manifest.json`, `provenance.json`, `sanitization.json` | SHA-256 fingerprints, upstream / local Git state, redaction record |
| `analyze_c5.py`, `analyze_temporal.py`, `reproduce_tables.py` | refusal diagnostics, unchanged format regex, verified main tables |
| `ablate_temporal_prompt.py` | prepare logged contexts; paid paired replay, 74 answer calls |
| `run_eval.sh`, `run_metered.py`, `meter.py`, `run_trials.sh` | original pipeline, environment setup, usage and QA-repeat wrappers |
| `rescore_c5.py`, `aggregate_trials.py`, `aggregate_legacy.py` | original and legacy-reference aggregation |

The offline entry checks all **597 stored F1 values**, question/gold/category identity, aggregate
statistics, category-5 flags, Bootstrap settings, BERTScore values, costs and ablation summaries.
It rejects altered F1, missing questions or mismatched hashes. Fingerprints establish internal
consistency, not independent attestation of experiment authorship. Numerical reporting rounds
*after* aggregation (e.g. the exact adversarial delta 37.46499 displays as **37.46**, not 37.47
obtained by subtracting rounded displayed means). Original prediction metadata retains a local
dataset path so that byte identity is preserved; sanitised logs remove machine paths. API keys,
`.env`, private memory caches, virtual environments and model weights are excluded.

Only `utils.py` is changed in the upstream source (BERTScorer reuse); added scripts supply the
evaluation workflow and the isolated prompt replay. The original harness/prompts are retained.
The main report tables must match the newly generated `tables.json` / `tables.md`.

## Appendix B: References and data sources

[1] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025.
*A-Mem: Agentic Memory for LLM Agents*. NeurIPS 2025. Version used for citation:
[arXiv:2502.12110v11](https://arxiv.org/abs/2502.12110v11).

[2] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. *Evaluating Very Long-Term Conversational Memory of LLM Agents*. ACL 2024, pp. 13851–13870.
[doi:10.18653/v1/2024.acl-long.747](https://aclanthology.org/2024.acl-long.747/).

[3] Xu et al. *A-Mem evaluation repository*, commit `b31a1dd` (2026-10-21).
[Repository](https://github.com/WujiangXu/A-mem) ·
[scoring-fix commit](https://github.com/WujiangXu/A-mem/commit/b31a1dd).
Full commit SHA is saved in `evidence/provenance.json`.

[4] Snap Research. *LoCoMo benchmark data*.
[Official repository](https://github.com/snap-research/locomo) ·
[project page](https://snap-research.github.io/locomo/).
The evaluated artifact is A-Mem's shipped `data/locomo10.json`; its SHA-256 and the conv-26
selection are recorded in the manifest. Dataset scale in this report is counted from that file,
not inferred from a different paper version.
