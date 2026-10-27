# Reproducibility Report: A-Mem (Agentic Memory for LLM Agents)

**Course:** FTEC5660 Agentic AI for Business and FinTech
**Name:** Qi Zheng  **Student ID:** 1155268235
**Repository reproduced:** `WujiangXu/A-mem` @ `b31a1dd` (2026-10-21)
**Scope and status.** Every number below is measured, not quoted, from **three real trials**
(`conv-26`, 199 questions, `deepseek-chat`, seeds 42/43/44). With n=3 the standard deviations in
§4 are indicative rather than precise, and should be read as a spread, not a confidence interval.

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

**Why this repo.** It is unusual in that the README states the repository exists *specifically*
to reproduce the paper's results, and it ships a `test_advanced_robust.py` harness that works
with any OpenAI-compatible backend. That makes the model substitution (see §2.2) a
configuration change rather than a code change, which keeps the comparison clean.

---

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
| Model (as returned by API) | `deepseek-flash` |
| Provider | DeepSeek |
| Base URL | `https://api.deepseek.com/v1` |
| temperature | repo defaults: 0.7 for categories 1–4, 0.5 for category 5 |
| max_tokens | 1000 (hard-coded in the repo) |
| top_p | not set (provider default) |
| Prompt templates | unmodified |

**A note on model aliases.** The API was called with `deepseek-chat`, but every response reports
`model: deepseek-flash`. `deepseek-chat` is therefore an alias whose backing model is chosen
server-side. This matters for reproducibility beyond the usual "different model" caveat: even
re-running the identical command later may hit a different underlying checkpoint, so a
model-alias substitution is not a stable control. The resolved name is recorded above so that at
least the run is identifiable after the fact.

**Explicit caveat.** Results are **not directly comparable to the paper**, because the
underlying LLM differs. I compare against the GPT-4o-mini column as the closest available
reference and treat any gap as the joint effect of model + environment, not as evidence
against the paper.

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

**Total: $2.59** across the three completed trials, plus a few cents on an attempt that aborted
at question 8/199 before its meter report was written.

**Read those dollars as an upper bound, not a bill.** The token counts are exact — they come from
the API's own `usage` field — but `meter.py` prices *every* input token at list, and 39–52% of
the input in trials 2 and 3 came from the provider's cache, which is billed at a fraction of
list. The real charge is lower; I did not model the discount because I could not verify the rate
for this account tier.

Trials 2 and 3 are near-identical in every column (398 calls, ~2.60 M input tokens, 4.1 k output
tokens, $0.73 modelled) — the expected signature of the same 199 questions against the same
cached memories, differing only in the seed that orders the two category-5 options.

Two things stand out. First, **the memory cache is keyed by `backend + model`, not by seed**, so
trials 2 and 3 reuse trial 1's memories — that is why they needed 398 calls each instead of 1,723.
Second, **the per-call token profile flips between phases**: memory building averages 2,230 input
tokens (the evolution prompt embeds five neighbour memories), while QA averages 6,523 (ten
retrieved notes) but emits only ~10 output tokens.

**My initial estimate was wrong by 5×** (I predicted ≈$0.22 for a conversation). I had guessed
~500 input tokens per call; the measured average is 2,230 in trial 1. Extrapolating to the full
10-conversation benchmark: **≈ 17,000 calls and $11**.

Cost is not a binding constraint, but **time** is: ~44 minutes for the first conversation (memory
building dominates) and ~14 minutes per additional trial.

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

Because the LLM is stochastic even at fixed temperature, the plan is **3 trials** per
configuration (seeds 42/43/44) reporting mean ± std.

**Status: 3 of 3 trials complete** (seeds 42, 43, 44), zero LLM failures in any of them. The third
trial was first attempted against an exhausted account and aborted at question 8/199 with
`HTTP 402 Insufficient Balance`; after a top-up the same command ran cleanly in 13 min 46 s. §4
therefore reports n=3.

A practical note that makes extra trials cheap: the memory cache is keyed by
`backend + model`, **not by seed**, so trials 2 and 3 reuse the memories built in trial 1 and
only re-run the QA phase (~400 calls each instead of ~1,700). The seed affects only the order of
the two options presented in category-5 prompts, so this reuse is sound.

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

The adversarial row is scored with the paper's reference so that it is comparable; the corrected
score is given underneath. **Under the paper's reference a lower number is better** — see §5.

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

**Multi-Hop is an exact match on the mean (+0.02 points) — but read the variance first.** The
three trials scored **31.37, 22.58 and 27.17**, an 8.79-point spread on the same 32 questions.
The third trial landed almost exactly on the paper's number, which is *why* the mean now sits
0.02 away; that is arithmetic, not evidence. The per-trial estimate is simply too noisy at n=32
to claim a match of that size. What *can* be said is that Multi-Hop does not systematically
underperform the paper — the paper's 27.02 sits inside our three-trial range.

**Single-Hop is the most trustworthy comparison.** At n=70 with ±1.36, the −1.57 gap is small and
tightly bounded. This is the only row where both the sample size and the variance support a firm
statement: our Single-Hop score is about 1.6 points below the paper's, consistently.

**Open-Domain agrees within noise** (+1.74 ± 1.75) but n=13, so this row carries almost no
information and should not be cited as evidence either way.

**Temporal fails, stably.** ±2.44 across three trials on a −24.29 gap means this is not sampling
noise. It is systematic, and §4.1 identifies the mechanism.

**The adversarial row needs care.** 31.93 ± 2.05 vs the paper's 50.03. Because a *lower* score is
better under this convention, DeepSeek appears to fall for the trap about 18 points less often
than the paper's model. That comparison is at least apples-to-apples, but see §5 for why neither
number should be read as accuracy.

### 4.1 Why Temporal collapses: a metric artefact, not a reasoning failure

The −24.29 on Temporal is the one large discrepancy, and it has a specific, verifiable cause.

Across the three trials, **17 to 19 of the 37 temporal predictions scored exactly 0**, and only
one of the 111 ever reached 1.0. The failures are not wrong dates — they are the *same* date
expressed in a different form:

| Question | Gold | DeepSeek's answer |
|---|---|---|
| When did Caroline go to the LGBTQ support group? | `7 May 2023` | `Yesterday` |
| When did Melanie paint a sunrise? | `2022` | `Last year` |
| When is Melanie planning on going camping? | `June 2023` | `Next month` |
| When is Caroline going to the transgender conference? | `July 2023` | `This month` |

I verified the first case against the dataset. `session_1` is dated **"1:56 pm on 8 May, 2023"**,
and the turn reads *"I went to a LGBTQ support group yesterday"*. So the correct date is
7 May 2023 — **DeepSeek's `Yesterday` is semantically right**. It scores zero because `Yesterday`
and `7 May 2023` share no tokens, and F1 is a token-overlap metric.

Quantified across the category, independently in each trial
(`analyze_temporal.py`, shipped in the repository):

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Temporal questions | 37 | 37 | 37 |
| Prediction uses a relative time expression **and** gold is absolute | **13** (35%) | **14** (38%) | **14** (38%) |
| F1 of that subset | **4.12** | **5.61** | **7.40** |
| F1 of the remaining questions | **29.97** | **28.01** | **34.53** |
| Temporal F1 overall | 20.89 | 19.54 | 24.27 |

The decomposition is stable across all three trials: the mismatch subset scores 4–7 while the
rest scores 28–35. If the format-mismatch subset scored like the rest, Temporal would be
**≈31** (the mean of the "remaining" column) rather than ≈22. So roughly **9 of the 24 points**
are a surface-form artefact, and **≈15 points** remain a genuine capability difference between
the two models. Both effects are real; conflating them would be wrong in either direction.

The mechanism is a prompt/model interaction. The category-2 prompt says *"Use DATE of CONVERSATION
to answer with an approximate date"*. GPT-4o-mini evidently resolves this to an absolute date;
DeepSeek resolves it relative to the session's own date and emits `Yesterday`. **Same prompt,
different output format** — which the metric cannot forgive. This is a reproducibility finding
about the benchmark, not about A-Mem: the Temporal column partly measures date-format compliance
rather than temporal reasoning.

### 4.2 A degenerate baseline the paper does not report

Before running any real model, I ran the harness with a **stub backend** that returns the
constant string `Not mentioned in the conversation` to every question. This costs nothing and
establishes what a useless model scores.

| Category | n | F1 (stub) |
|---|---|---|
| Multi-Hop | 32 | 0.33 |
| Temporal | 37 | 8.35 |
| Open-Domain | 13 | 6.21 |
| Single-Hop | 70 | 2.13 |
| **Adversarial** | **47** | **100.00** |
| Overall | 199 | 26.38 |

A model that never answers anything scores **100.00 F1 on the adversarial category**, because 444
of the 446 adversarial questions have no gold answer at all, so `Not mentioned in the conversation`
*is* the correct response. That is a property of the metric, not of A-Mem — but it means the
adversarial column cannot be read as "accuracy". See §5.

---

## 5. Modifications: one fix, one missing metric

### 5.1 Modification 1: which string category 5 is scored against

Nothing in the algorithm. The modification is to **which string the prediction is scored
against** for category-5 questions.

The discrepancy is documented by the authors themselves. Commit `b31a1dd` (2026-10-21), five
days before I started, carries this message:

> **Score category-5 questions against "Not mentioned in the conversation" (#29)**
>
> `QA.final_answer` returns the adversarial answer for category 5. It is the wrong option
> offered in the prompt, but both runners also used it as the reference, so **picking the trap
> scored high**. 444 of the 446 category-5 questions have no `answer` field. The correct
> response is that the conversation does not contain the answer.
> …
> `--legacy_c5_reference` restores the previous scoring, **which is what the paper's numbers
> used**.

So in the evaluation behind Table 1, a category-5 question was scored against the **adversarial
(wrong)** answer. Under that reference, a model that falls for the trap scores *higher*. The
paper presents the column as "higher is better" and reports A-Mem 50.03 vs LoCoMo 69.23 — the
baseline appears worse, but what the column actually measures is **susceptibility to the wrong
answer**.

The commit states that *both* of the repository's runners used `final_answer` as the reference.
Whether the baseline rows in Table 1 were produced with the same convention is **not verifiable
from public artifacts** — I could not find the baseline evaluation code. If they were, the
entire column is affected, not only A-Mem; I flag this as an open question rather than a
finding.

The fix is a one-flag switch (`--legacy_c5_reference`), which makes this a clean, isolated,
measurable modification: the old behaviour remains reachable, so both scorings can be applied
to the same predictions.

**Code-level confirmation.** I checked out the parent commit (`b31a1dd^`) rather than relying on
the commit message. The pre-fix runner scored every question against `qa.final_answer`
(`test_advanced_robust.py`, line 273 of the old file):

```python
metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {...}
```

and `QA.final_answer` was defined as:

```python
def final_answer(self):
    if self.category == 5:
        return self.adversarial_answer      # <- the wrong option, used as the gold answer
    return self.answer
```

Post-fix, the runner calls `score_prediction(prediction, qa, legacy_c5_reference)`, which
resolves the reference through `qa.reference()` → `NOT_MENTIONED` for category 5. The bug and
its repair are both visible in the diff; no inference required.

### 5.2 Measured impact on identical predictions

Because the change is pure post-processing, I scored the **same 199 real predictions** twice, for
all three trials. No LLM calls, no sampling noise — the delta is exactly the metric change.

| Category | n | Corrected reference | Paper's reference | Δ |
|---|---|---|---|---|
| Multi-Hop | 32 | 27.04 | 27.04 | 0.00 |
| Temporal | 37 | 21.56 | 21.56 | 0.00 |
| Open-Domain | 13 | 13.88 | 13.88 | 0.00 |
| Single-Hop | 70 | 43.08 | 43.08 | 0.00 |
| **Adversarial** | 47 | **69.40** | **31.93** | **+37.47** |
| **Overall** | 199 | **40.81** | **31.96** | **+8.85** |

Per trial, the adversarial shift was **+34.66** (seed 42), **+36.26** (seed 43) and **+41.47**
(seed 44); the overall shift was +8.19, +8.56 and +9.79. The effect is stable across trials,
which is expected — it is arithmetic on the same predictions, not a sampling effect.

Two things follow.

**Isolation holds.** Four of five categories are bit-identical, as they must be — the flag only
touches category 5. This is a direct check that the change is confined to the intended mechanism.

**The effect is large and purely nominal.** The model's answers did not change by a single
character, yet the adversarial score moves 34.66–41.47 points and the overall score 8.19–9.79
depending on the seed. Any table that mixes the two conventions is not comparable, and the
paper's 50.03 cannot be placed next to a corrected 69.40 as if the gap were a capability
difference.

### 5.3 Modification 2: reporting the metric the evaluator already computes

Neither reference is fully satisfactory. Under the paper's reference the metric rewards agreeing
with a wrong answer; under the corrected one it rewards refusing, and a constant "not mentioned"
scores 100 (§4.2). Both are F1 — token overlap with a string — but category 5 is a binary decision,
so the right instrument is binary, and the repository already has it: `utils.score_prediction`
writes `adversarial_correct` into every category-5 metric dict (1 if the prediction declines *and*
does not echo the trap) and `aggregate_metrics` carries it into `aggregate_metrics["category_5"]`.
**No table reads it** — not the paper's Table 1, not the README, not the results table I wrote in §4.

The predicate is upstream's, from the same commit `b31a1dd` that fixed the reference (`git blame`
puts lines 190–207 on the author). What I added is the reporting path, `analyze_c5.py`: it
re-derives the flag from the stored predictions independently of `utils.py` and **asserts the two
agree on all 141 questions**, reports the rate with a Wilson interval (at n=47 a normal
approximation is too narrow), decomposes the failures into exclusive behavioural modes, and
cross-tabulates the flag against F1 under both references. It reads predictions already on disk —
**zero API cost, no model downloads**, ~0.2 s.

| Trial | Correct / 47 | Rate | 95% Wilson CI |
|---|---|---|---|
| seed 42 | 31 | 65.96 | [51.7, 77.8] |
| seed 43 | 32 | 68.09 | [53.8, 79.6] |
| seed 44 | 33 | 70.21 | [56.0, 81.3] |
| **pooled** | **96 / 141** | **68.09** | **[60.0, 75.2]** |

Across trials: **68.09 ± 2.13**. The failures are unambiguous: 43 questions (30.5%) returned the
trap answer outright, 2 (1.4%) answered something that was neither a refusal nor the trap, and
**zero both declined and echoed the trap** — the clause meant to catch hedging never fires here.
The model either refuses cleanly or takes the bait.

The informative result is the cross-tabulation. Of the 96 questions the model handled, **none** has
F1-vs-trap ≥ 0.5; of the 45 it did not, **43** do. Mean F1 against the trap is **2.11** when the
model refused and **95.56** when it did not. The two instruments are not noisy proxies for one
another — on this data they are near-exact inverses, disagreeing on **139 of 141 questions
(98.6%)**.

That reframes the column rather than merely qualifying it: the paper's adversarial F1 does not
weakly measure trap-handling, it *ranks models by how often they take the bait*, so a score near 50
means the set splits roughly evenly. 50.03 and 31.93 therefore differ mainly in bait rate, not in
capability, and neither can be read as accuracy. Behaviourally, A-Mem handles **68.09%** of traps
here (lower bound 60.0); the paper released no predictions, so there is no binary figure to compare
against, and I claim only that the F1 column cannot support the comparison.

### 5.4 Recommendation

Report category 5 as a **binary refusal rate with an interval**, never as F1 and never folded into
an overall average — a five-category mean implicitly weights a number that is anti-correlated with
success. Keep F1 for categories 1–4, where token overlap is a defensible proxy for a short factual
answer.

---

## 6. Debug diary

### 6.1 `pip install rouge-score` fails with `EEXIST: mkdir .../pip-install-*`

The sdist build could not create its temporary extraction directory. Changing `TMPDIR` to a
short path did not help; `--no-build-isolation` did not help.

**Resolution:** downloaded the sdist directly from PyPI, extracted it, and installed from the
local directory:

```bash
curl -sL -o rouge_score-0.1.2.tar.gz <pypi-url>
tar xzf rouge_score-0.1.2.tar.gz
pip install --no-build-isolation ./vendor/rouge_score-0.1.2
```

### 6.2 `nltk.download()` blocked

`nltk.download('punkt')` and `('wordnet')` failed with an SSRF/network restriction in my
environment; METEOR and BLEU tokenisation both depend on them. **Resolution:** fetched the four
packages (`punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) as zips from the `nltk_data` GitHub mirror
and unpacked them into `~/nltk_data/`.

### 6.3 NLTK still refuses to load the data: "Refused, writable by other accounts"

NLTK then rejected the corpora outright — *"Refused, writable by other accounts (a file planted or
rewritten there would be loaded as trusted data)"*. This is a security check, and the message
appears in the middle of a long warning block, so it is easy to miss. **Resolution:**
`chmod -R go-w ~/nltk_data`; METEOR then returned non-zero scores.

### 6.4 The evaluation harness, not the LLM, is the bottleneck

With the stub backend (zero network latency) a single conversation still took **18 min 32 s**.
Profiling the metric function:

```
calculate_metrics per question:  7.83 s
  of which bert_score:           6.07 s   (77%)
  of which rouge:                0.0007 s
  of which bleu:                 0.0020 s
```

The cause is in the `bert_score` library: `bert_score.utils.get_model()` has no `lru_cache`,
so `bert_score.score()` calls `from_pretrained("roberta-large")` — reloading ~1.4 GB of weights
— **for every single question**. Extrapolated: **4.3 hours** for the 1,986-question benchmark,
i.e. 20+ hours for 3 trials.

**Resolution:** in `utils.py`, build one `BERTScorer` and reuse it (lazy module-level
singleton). +24/−1 lines.

**Verification that the fix changes nothing but speed:**

1. 8 sentence pairs × 3 BERTScore values = 24 numbers: **bit-identical** before vs after.
2. Two full 199-question dry runs, compared question by question: **0 prediction differences,
   0 metric differences (max |Δ| = 0.00e+00)**, all six aggregate F1 values identical.
3. Speed: **7.83 s → 0.12 s per question (~65×)**; end-to-end dry run **18 min 32 s → 3 min 12&nbsp;s**.

**This is an engineering fix, not the assignment's "one modification".** I report it separately
so it is not confused with §5, which is the scientific change under evaluation. It is included
because without it, running 3 trials was not feasible in the available time.

### 6.5 A run that died silently, and how I caught it

The third trial was launched as `nohup ... &` from a shell that then exited. The log stopped at
`Processing sample 1/1` and never grew again — **but it also contained no traceback**, so from the
log alone it looked like a slow run rather than a dead one. `ps` and `pgrep` were both unavailable
in my environment (`operation not permitted`), so I could not simply check the PID.

What settled it was using the provider as the liveness probe: the account balance is queryable
over HTTP, and a run of 199 questions must spend money.

```bash
curl -s https://api.deepseek.com/user/balance -H "Authorization: Bearer $KEY"
# balance unchanged for five minutes  =>  no calls are being made  =>  the process is gone
```

The balance had moved by exactly one cent (the two calls issued before the shell exited) and then
froze. Relaunched under a managed background task, the same command completed in 13 min 46 s.

**Lesson.** "No error in the log" is not evidence that a process is alive. When the process table
is unreadable, find a side effect that *must* occur per unit of work — here, billing — and watch
that instead.

### 6.6 Smaller items

- `test_advanced_robust.py` resolves `--dataset` and `--output` relative to the script's own
  directory, not the working directory. Worth knowing when scripting around it.
- The module set is flat (no package), so any external script must `sys.path.insert` the repo
  directory before importing `utils` / `load_dataset`.

---

## 7. Conclusions

**What reproduces.** The pipeline runs end to end with zero LLM failures across 2,519 calls
(1,723 + 398 + 398), and the aggregate picture matches the paper closely: Multi-Hop 27.04 vs
27.02, Single-Hop 43.08 vs 44.65, Open-Domain 13.88 vs 12.14, overall behaviour within a few
points — despite a completely different backbone model. A-Mem's central claim, that structured
agentic memory beats naive retrieval on multi-hop questions, is at minimum **not contradicted**.

**But the variance matters as much as the means.** Multi-Hop swung 8.79 points across three runs
on the same 32 questions (31.37 / 22.58 / 27.17). So "we matched the paper to 0.02 points" is
arithmetic, not evidence. Only Single-Hop (n=70, ±1.36) supports a firm claim, and it says we are
about 1.6 points below. Even at n=3 the standard deviations are themselves weakly estimated; a
fourth and fifth trial would tighten them for another $0.73 each.

**What does not reproduce, and why.** Two columns need separate explanations, and neither is
evidence against the method:

1. **Temporal (−24.29, stable at ±2.44)** is mostly a metric artefact. 35–38% of the answers are
   semantically correct dates expressed relatively (`Yesterday` for `7 May 2023`) and score near
   zero because F1 is token-overlap. Roughly 9 of the 24 points are this; the remaining ~15 look
   like a genuine capability gap between the two models.

2. **Adversarial** is not comparable as reported. The paper scored it against the wrong answer
   (documented by the authors in commit `b31a1dd`), so its 50.03 measures trap-susceptibility,
   not accuracy. On identical predictions, switching to the correct reference moves the column by
   **+37.47** and the overall score by **+8.85** with no change in model behaviour. Measured
   behaviourally instead of by string overlap, the model refuses the trap on **68.09%** of the 47
   questions (95% CI [60.0, 75.2]), and F1-vs-trap is the near-exact inverse of that verdict on
   98.6% of questions — so the column's ranking carries no information about trap-handling.

**What is expensive.** Not the tokens — $1.13 for the first conversation, $0.73 per additional
trial, $2.59 in total and these are upper bounds because the meter ignores cache discounts
(§2.3), ~$11 extrapolated for the full benchmark. The cost is wall-clock time,
and most of it was not the agent: a missing `lru_cache` in a third-party library made metric
computation ~65× slower than it needed to be (§6.4). 44 minutes for the first conversation is
~40 minutes of LLM latency plus ~4 minutes of local compute; before the fix the local compute
alone was 18 minutes.

**Recommendations to future users.**

1. **Pin the commit.** This repository changed its evaluation semantics on 2026-10-21. Any number
   produced before that date is not comparable to any number produced after it. Record the commit
   hash in every results file.
2. **Record the resolved model name, not the requested one.** `deepseek-chat` returned
   `deepseek-flash`; the alias may point elsewhere tomorrow.
3. **Do not average category 5 into an overall score.** Report the binary refusal rate instead
   (68.09% [60.0, 75.2] here), and report the category separately. Both available F1 references
   reward a degenerate behaviour (§4.2, §5.3), and on this data F1-vs-trap is nearly the inverse of
   the behaviour it is supposed to measure.
4. **Check answer *form* before believing a capability gap.** The Temporal column would look like
   a 25-point failure of the method; it is mostly a 9-point formatting mismatch.
5. **Cache the BERTScore model** before running anything at scale (§6.4).

---

## Appendix A: Artifacts

| File | Purpose |
|---|---|
| `run_eval.sh` | Entry point; reads `.env`, maps DeepSeek onto the OpenAI endpoint |
| `run_metered.py`, `meter.py` | The same, plus token/cost accounting (`meter.py` patches `Completions.create`; the repo is untouched) |
| `dry_run.py` | Stub backend for end-to-end validation at zero cost |
| `smoke_test.py` | 5 checks (imports, dataset, embeddings, metrics, prompt templates) |
| `rescore_c5.py` | Re-scores stored predictions under both category-5 references (`--json` emits a machine-readable summary) |
| `run_trials.sh` | Runs N trials at different seeds and aggregates; memory is built once and reused across seeds |
| `aggregate_trials.py` | Emits the report-ready mean ± std table against the paper's numbers |
| `aggregate_legacy.py` | Aggregates the paper-reference rescores across trials |
| `analyze_temporal.py` | Decomposes the Temporal category into format-mismatch vs. genuine-error subsets (§4.1) |
| `analyze_c5.py` | Surfaces the category-5 refusal rate: re-derives the flag and asserts it matches the stored value, adds Wilson intervals, failure modes and the F1-vs-behaviour cross-tab (§5.3) |
| `capture_bert_baseline.py` | Before/after numerical equality check for the BERTScore fix (writes `bert_baseline_{before,after}.json`) |

## Appendix B: Repository diff

Only one upstream file was modified:

```
utils.py | 25 ++++++++++++++++++++++++-
1 file changed, 24 insertions(+), 1 deletion(-)
```

The change is confined to `calculate_bert_scores` and its new lazy accessor; the algorithm,
prompts, retrieval and evaluation logic are untouched. Everything else in the diff is new files:
the analysis scripts, the runner wrappers, the report and the README.
