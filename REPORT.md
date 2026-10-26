# Reproducibility Report: A-Mem (Agentic Memory for LLM Agents)

**Course:** FTEC5660 Agentic AI for Business and FinTech
**Name:** Qi Zheng  **Student ID:** 1155268235
**Repository reproduced:** `WujiangXu/A-mem` @ `b31a1dd` (2026-10-21)
**Status of this draft:** All seven sections are written and filled with measured numbers from
**two real trials** (`conv-26`, 199 questions, `deepseek-chat`, seeds 42/43). A planned third
trial did not run — the API account ran out of credit mid-run (§2.3). With n=2 the standard
deviations in §4 are very weakly estimated and should be read as a spread, not a confidence
interval.

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

### 2.3 Compute and cost (measured)

Instrumented with `meter.py`, which patches the OpenAI SDK's `Completions.create` and reads
`response.usage`; the repository is untouched.

| Item | Trial 1 (seed 42) | Trial 2 (seed 43) |
|---|---|---|
| LLM calls | **1,723** (memory + QA) | **398** (QA only, memories reused) |
| Failed calls | 0 | 0 |
| Input tokens | 3,843,050 | 2,596,009 |
| Output tokens | 135,040 | 4,107 |
| Average per call | 2,230 in / 78 out | 6,523 in / 10 out |
| Cost | **$1.13** | **$0.73** |
| Wall-clock | 44 min | 14 min |

**Total spent: $1.89** across two trials plus an aborted third.

Two things stand out. First, **the memory cache is keyed by `backend + model`, not by seed**, so
trials 2 and 3 reuse trial 1's memories — that is why trial 2 needed 398 calls instead of 1,723.
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

`[[STATUS: 2 of 3 trials complete (seeds 42, 43). Trial 3 aborted at question 8/199 with
HTTP 402 Insufficient Balance; the account ran out of credit. §4 reports n=2.]]`

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

One conversation (`conv-26`, 199 questions), `deepseek-chat`, `retrieve_k=10`, **two trials**
(seeds 42 and 43). F1 in percent, mean ± std.

The adversarial row is scored with the paper's reference so that it is comparable; the corrected
score is given underneath. **Under the paper's reference a lower number is better** — see §5.

| Category | n | Reported (GPT-4o-mini) | Mine (paper's reference) | Δ |
|---|---|---|---|---|
| Multi-Hop | 32 | 27.02 | **26.98 ± 6.21** | **−0.04** |
| Temporal | 37 | 45.85 | **20.21 ± 0.96** | **−25.64** |
| Open-Domain | 13 | 12.14 | **13.52 ± 2.31** | +1.38 |
| Single-Hop | 70 | 44.65 | **42.58 ± 1.48** | −2.07 |
| Adversarial | 47 | 50.03 | **33.11 ± 0.19** | −16.92 |
| Overall | 199 | not reported | 31.78 ± 1.50 | — |

With the corrected category-5 reference: **Adversarial 68.58 ± 1.32**, **Overall 40.15 ± 1.24**.
Zero LLM failures, zero stale caches in either trial.

**Multi-Hop is an exact match on the mean (−0.04 points) — but read the variance first.** The two
trials scored 31.37 and 22.58, an 8.79-point swing on the same 32 questions. The near-perfect
agreement with the paper is therefore partly a coincidence of averaging: the per-trial estimate is
simply too noisy at n=32 to claim a 0.04-point match. What *can* be said is that Multi-Hop does not
systematically underperform the paper — the paper's 27.02 sits inside our two-trial range.

**Single-Hop is the most trustworthy comparison.** At n=70 with ±1.48, the −2.07 gap is small and
tightly bounded. This is the only row where both the sample size and the variance support a firm
statement: our Single-Hop score is about 2 points below the paper's, consistently.

**Open-Domain agrees within noise** (+1.38 ± 2.31) but n=13, so this row carries almost no
information and should not be cited as evidence either way.

**Temporal fails, stably.** ±0.96 across two trials means the −25.64 gap is not sampling noise.
It is systematic, and §4.1 identifies the mechanism.

**The adversarial row needs care.** 33.11 ± 0.19 vs the paper's 50.03. Because a *lower* score is
better under this convention, DeepSeek appears to fall for the trap about 17 points less often
than the paper's model. That comparison is at least apples-to-apples, but see §5 for why neither
number should be read as accuracy.

### 4.1 Why Temporal collapses: a metric artefact, not a reasoning failure

The −24.96 on Temporal is the one large discrepancy, and it has a specific, verifiable cause.

I inspected all 37 temporal predictions. **Not one scored F1 = 1.0**, and 18 scored exactly 0.
The failures are not wrong dates — they are the *same* date expressed in a different form:

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

Quantified across the category, in both trials independently:

| | seed 42 | seed 43 |
|---|---|---|
| Temporal questions | 37 | 37 |
| Prediction uses a relative time expression **and** gold is absolute | **14** (38%) | **15** (41%) |
| F1 of that subset | **3.83** | **5.24** |
| F1 of the remaining questions | **31.27** | **29.29** |
| Temporal F1 overall | 20.89 | 19.54 |

The decomposition is stable across trials. If the format-mismatch subset scored like the rest,
Temporal would be **≈31** rather than ≈20. So roughly **10 of the 25 points** are a surface-form
artefact, and **≈15 points** remain a genuine capability difference between the two models. Both
effects are real; conflating them would be wrong in either direction.

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

A model that never answers anything scores **100.00 F1 on the adversarial category**. The
reason is that 444 of the 446 adversarial questions have no gold answer at all, so
`Not mentioned in the conversation` *is* the correct response. This is a property of the
metric, not of A-Mem — but it means the adversarial column cannot be read as "accuracy" and
must not be averaged into an overall score without comment. See §5.

---

## 5. Modification: fixing the category-5 reference

### 5.1 What changed

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
both trials. No LLM calls, no sampling noise — the delta is exactly the metric change.

| Category | n | Corrected reference | Paper's reference | Δ |
|---|---|---|---|---|
| Multi-Hop | 32 | 26.98 | 26.98 | 0.00 |
| Temporal | 37 | 20.21 | 20.21 | 0.00 |
| Open-Domain | 13 | 13.52 | 13.52 | 0.00 |
| Single-Hop | 70 | 42.58 | 42.58 | 0.00 |
| **Adversarial** | 47 | **68.58** | **33.11** | **+35.47** |
| **Overall** | 199 | **40.15** | **31.78** | **+8.37** |

Per trial, the adversarial shift was **+34.66** (seed 42) and **+36.26** (seed 43); the overall
shift was +8.19 and +8.56. The effect is stable across trials, which is expected — it is
arithmetic on the same predictions, not a sampling effect.

Two things follow.

**Isolation holds.** Four of five categories are bit-identical, as they must be — the flag only
touches category 5. This is a direct check that the change is confined to the intended mechanism.

**The effect is large and purely nominal.** The model's answers did not change by a single
character, yet the adversarial score moves 34.66 points and the overall score 8.19. Any table
that mixes the two conventions is not comparable, and the paper's 50.03 cannot be placed next to
a corrected 67.64 as if the gap were a capability difference.

For calibration: with the stub backend (§4.2), which answers `Not mentioned in the conversation`
to *every* question, the same switch moves the adversarial score by **+97.25** and the overall
by +22.97. The real model shows a smaller swing (+34.66 / +8.19 for seed 42, +36.26 / +8.56 for
seed 43) precisely because it does not always decline — the two references partly agree when a
prediction is a real attempt at the answer. The stub case is the ceiling of the artefact, the real
case is its realistic size.

### 5.3 Which metric should be used

Neither reference is fully satisfactory. Under the paper's reference, the metric rewards
agreeing with a wrong answer. Under the corrected reference, it rewards refusing to answer — a
constant "not mentioned" scores 100 (§4.1). The repository already computes a third quantity,
`adversarial_correct`, which checks whether the prediction declined the trap *and* did not
echo the adversarial string. I recommend reporting that instead of F1 for category 5, and
reporting category-5 results separately rather than folding them into an overall average.

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
environment. METEOR and BLEU tokenisation both depend on these.

**Resolution:** fetched the four packages (`punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) as zips
from the `nltk_data` GitHub mirror and unpacked them into `~/nltk_data/`.

### 6.3 NLTK still refuses to load the data: "Refused, writable by other accounts"

After unpacking, NLTK rejected the corpora:

> Refused, writable by other accounts (a file planted or rewritten there would be loaded as
> trusted data)

This is a security check, and the message appears in the middle of a long warning block, so it
is easy to miss. **Resolution:** `chmod -R go-w ~/nltk_data`. After this, METEOR returned
non-zero scores.

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
3. Speed: **7.83 s → 0.12 s per question (~65×)**; end-to-end dry run **18 min 32 s → 3 min 12 s**.

**This is an engineering fix, not the assignment's "one modification".** I report it separately
so it is not confused with §5, which is the scientific change under evaluation. It is included
because without it, running 3 trials was not feasible in the available time.

### 6.5 Smaller items

- `test_advanced_robust.py` resolves `--dataset` and `--output` relative to the script's own
  directory, not the working directory. Worth knowing when scripting around it.
- The module set is flat (no package), so any external script must `sys.path.insert` the repo
  directory before importing `utils` / `load_dataset`.

---

## 7. Conclusions

**What reproduces.** The pipeline runs end to end with zero LLM failures across 2,121 calls
(1,723 + 398), and the aggregate picture matches the paper closely: Multi-Hop 26.98 vs 27.02,
Single-Hop 42.58 vs 44.65, Open-Domain 13.52 vs 12.14, overall behaviour within a few points —
despite a completely different backbone model. A-Mem's central claim, that structured agentic
memory beats naive retrieval on multi-hop questions, is at minimum **not contradicted**: our
multi-hop score sits on top of the paper's rather than below it.

**But the variance matters as much as the means.** Multi-Hop swung 8.79 points between two runs
on the same 32 questions. So "we matched the paper to 0.04 points" is arithmetic, not evidence.
Only Single-Hop (n=70, ±1.48) supports a firm claim, and it says we are about 2 points below.
With n=2 the standard deviations are themselves barely estimated; a third trial would be worth
its $0.73.

**What does not reproduce, and why.** Two columns need separate explanations, and neither is
evidence against the method:

1. **Temporal (−25.64, stable at ±0.96)** is mostly a metric artefact. 38–41% of the answers are
   semantically correct dates expressed relatively (`Yesterday` for `7 May 2023`) and score near
   zero because F1 is token-overlap. Roughly 10 of the 25 points are this; the remaining ~15 look
   like a genuine capability gap between the two models.

2. **Adversarial** is not comparable as reported. The paper scored it against the wrong answer
   (documented by the authors in commit `b31a1dd`), so its 50.03 measures trap-susceptibility,
   not accuracy. On identical predictions, switching to the correct reference moves the column by
   **+35.47** and the overall score by **+8.37** with no change in model behaviour.

**What is expensive.** Not the tokens — $1.13 for the first conversation, $0.73 per additional
trial, $1.89 total spent, ~$11 extrapolated for the full benchmark. The cost is wall-clock time,
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
3. **Do not average category 5 into an overall score.** Report `adversarial_correct`, and report
   the category separately. Both available references reward a degenerate behaviour (§4.2, §5.3).
4. **Check answer *form* before believing a capability gap.** The Temporal column would look like
   a 25-point failure of the method; it is mostly a 10-point formatting mismatch.
5. **Cache the BERTScore model** before running anything at scale (§6.4).

---

## Appendix A: Artifacts

| File | Purpose |
|---|---|
| `run_eval.sh` | Entry point; reads `.env`, maps DeepSeek onto the OpenAI endpoint |
| `run_metered.py` | Same, plus token/cost accounting |
| `meter.py` | Patches `Completions.create` to record usage; repo untouched |
| `dry_run.py` | Stub backend for end-to-end validation at zero cost |
| `smoke_test.py` | 5 checks (imports, dataset, embeddings, metrics, prompt templates) |
| `rescore_c5.py` | Re-scores stored predictions under both category-5 references (`--json` emits a machine-readable summary) |
| `run_trials.sh` | Runs N trials at different seeds and aggregates; memory is built once and reused across seeds |
| `aggregate_trials.py` | Emits the report-ready mean ± std table against the paper's numbers |
| `aggregate_legacy.py` | Aggregates the paper-reference rescores across trials |
| `capture_bert_baseline.py` | Before/after numerical equality check for the BERTScore fix |
| `bert_baseline_{before,after}.json` | The 24 compared values |

## Appendix B: Repository diff

Only one file in the upstream repository was modified:

```
utils.py | 25 ++++++++++++++++++++++++-
1 file changed, 24 insertions(+), 1 deletion(-)
```

The change is confined to `calculate_bert_scores` and its new lazy accessor; the algorithm,
prompts, retrieval and evaluation logic are untouched.
