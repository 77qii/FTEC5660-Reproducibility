# A-Mem — Reproducibility Study

This repository is a fork of [`WujiangXu/A-mem`](https://github.com/WujiangXu/A-mem) (commit
`b31a1dd`) prepared for the FTEC5660 *Reproducibility Work* assignment.

**Student:** Qi Zheng · 1155268235

> The upstream A-Mem README is preserved in full at the bottom of this file. Everything above
> the horizontal rule describes what I changed and how to run it.

---

## What this study does

It reproduces one conversation of the LoCoMo benchmark (`conv-26`, 419 turns, 199 questions)
from Table 1 of *A-Mem: Agentic Memory for LLM Agents* (NeurIPS 2025), using **DeepSeek** in
place of the paper's GPT-4o-mini, and measures one small modification to the evaluation.

## Install

```bash
git clone <this-repo>
cd <this-repo>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Two known install snags on a clean machine, and their fixes:

- **`pip install rouge-score` may fail** with `EEXIST: mkdir .../pip-install-*`. Build it from the
  sdist instead:
  ```bash
  curl -sLO https://files.pythonhosted.org/packages/e2/c5/9136736c37022a6ad27fea38f3111eb8f02fe75d067f9a985cc358653102/rouge_score-0.1.2.tar.gz
  tar xzf rouge_score-0.1.2.tar.gz && pip install --no-build-isolation ./rouge_score-0.1.2
  ```
- **NLTK data may not download** in a restricted network. Fetch it manually and make it private,
  or NLTK will refuse to load it:
  ```bash
  for p in tokenizers/punkt tokenizers/punkt_tab corpora/wordnet corpora/omw-1.4; do
    curl -sLO https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/$p.zip
    unzip -oq $(basename $p).zip -d ~/nltk_data/$(dirname $p)
  done
  chmod -R go-w ~/nltk_data
  ```

## Configure

```bash
cp .env.example .env
# edit .env: DEEPSEEK_API_KEY=sk-...
```

Switching the model requires **no code change**: the repo's `RobustOpenAIController` builds
`OpenAI(api_key=...)` without a `base_url`, and the `openai` SDK falls back to the
`OPENAI_BASE_URL` environment variable.

## Run

```bash
# one conversation (conv-26), 199 questions, ~$1.13, ~45 min
./run_eval.sh --ratio 0.1 --seed 42 --output result_seed42.json

# three trials at different seeds; memories are cached and reused, so trials 2-3
# only re-run QA (~$0.73 and ~14 min each)
./run_trials.sh 3 0.1

# same predictions, scored under the paper's category-5 reference
python rescore_c5.py result_seed42.json --json
python aggregate_legacy.py result_seed42_rescore.json
```

Sanity checks that cost nothing:

```bash
python smoke_test.py     # imports, dataset, embeddings, metrics, prompt templates
python dry_run.py --ratio 0.1   # full pipeline with a stub LLM, no API calls
```

## What changed

### 1. `utils.py` — reuse the BERTScore model (+24/−1 lines)

`bert_score.score()` calls `bert_score.utils.get_model()`, which has **no `lru_cache`**, so every
question re-loaded ~1.4 GB of `roberta-large` weights. Metric computation was 6.07 s per
question — 99.8% of the runtime — i.e. 4.3 hours for the 1,986-question benchmark.

The fix builds one `BERTScorer` and reuses it. Values are unchanged: 24 of 24 compared numbers are
bit-identical, and two full 199-question runs produced **0 prediction differences and 0 metric
differences** (max |Δ| = 0.00e+00). Speed: **7.83 s → 0.12 s per question (~65×)**.

This is an engineering fix, not the study's measured modification.

### 2. `tools` — evaluation instrumentation

| File | Purpose |
|---|---|
| `meter.py` | Patches `Completions.create` to record token usage; repo untouched |
| `run_metered.py` | Drop-in entry point that reports tokens and cost |
| `dry_run.py` | Stub backend for end-to-end validation at zero cost |
| `smoke_test.py` | Five environment checks |
| `rescore_c5.py` | Re-scores stored predictions under both category-5 references |
| `aggregate_trials.py` | mean ± std table against the paper's numbers |
| `aggregate_legacy.py` | Aggregates the paper-reference rescores across trials |
| `analyze_temporal.py` | Splits the Temporal category into date-format mismatches vs. genuine errors |
| `analyze_c5.py` | Category-5 refusal rate: Wilson intervals, failure modes, F1-vs-behaviour cross-tab |
| `capture_bert_baseline.py` | Before/after numerical equality check for the BERTScore fix |

## The measured modifications

### 1. Which string category 5 is scored against

Commit `b31a1dd` (2026-10-21) changed how category-5 (adversarial) questions are scored. Its
message states the old behaviour plainly:

> `QA.final_answer` returns the adversarial answer for category 5. It is the wrong option offered
> in the prompt, but both runners also used it as the reference, so **picking the trap scored
> high**. … `--legacy_c5_reference` restores the previous scoring, **which is what the paper's
> numbers used**.

So the paper's Adversarial column was scored against the *wrong* answer. On identical
predictions, switching to the corrected reference moves that column by **+37.47** and the overall
score by **+8.85**, with no change in model behaviour (mean over three trials; per-trial shifts
are +34.66, +36.26 and +41.47).

### 2. Reporting the metric the evaluator already computes

`utils.score_prediction` writes `adversarial_correct` into every category-5 metric dict, and
`aggregate_metrics` carries it into `aggregate_metrics["category_5"]` — but no table anywhere reads
it. `analyze_c5.py` does: it re-derives the flag from the stored predictions, asserts the two
implementations agree on every question, and reports the rate with a confidence interval instead of
a token-overlap F1.

```bash
python analyze_c5.py result_conv26_new.json result_conv26_seed43.json result_conv26_seed44.json
```

Reads predictions already on disk — no API calls, no model downloads.

```
     trial    n   ok    rate      95% Wilson CI  F1|refusal   F1|trap
   seed 42   47   31  65.96%       [51.7, 77.8]      67.64     32.98
   seed 43   47   32  68.09%       [53.8, 79.6]      69.51     33.25
   seed 44   47   33  70.21%       [56.0, 81.3]      71.05     29.58
    pooled  141   96  68.09%       [60.0, 75.2]
```

Across three trials the model refuses the trap on **68.09% ± 2.13** of the 47 questions. The
failures are 43 outright bait-takings, 2 unrelated answers, and **zero** cases of hedging (decline
plus echo). F1 against the trap answer is **2.11** when the model refused and **95.56** when it did
not: the two instruments disagree on **139 of 141 questions (98.6%)**, i.e. the F1 column ranks
models by bait rate rather than by trap-handling.

Full results, debug diary and conclusions are in [`REPORT.md`](REPORT.md).

## Secrets

`.env` is git-ignored. No API key is committed anywhere in this repository. Run artifacts
(`result_*.json`, caches, logs) are git-ignored as well.

---

# Upstream README

# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

> **Note:** This repository is specifically designed to reproduce the results presented in our paper. If you want to use the A-Mem system in building your agents, please refer to our official implementation at: [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

For more details, please refer to our paper: [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:

**Option A — Original evaluation (requires OpenAI JSON schema support):**
```bash
python test_advanced.py
```

**Option B — Robust evaluation (recommended, works with any LLM backend):**

The robust evaluation (`test_advanced_robust.py`) removes the JSON schema dependency and supports OpenAI API, vLLM, and Ollama backends.

```bash
# OpenAI models
python test_advanced_robust.py --backend openai --model gpt-4o-mini \
    --dataset data/locomo10.json --output results_robust_gpt-4o-mini.json

# vLLM-served open-source models (start vLLM server first)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json --output results_robust_qwen3b.json \
    --sglang_port 30000

# Ollama models
python test_advanced_robust.py --backend ollama --model qwen2.5:3b \
    --dataset data/locomo10.json --output results_robust_ollama_qwen3b.json
```

Key arguments:
- `--retrieve_k`: Number of memories to retrieve per query (default: 10). Tune this per model for best results.
- `--ratio`: Fraction of dataset to evaluate (e.g., `--ratio 0.1` for 10% quick test).
- `--backend`: One of `openai`, `vllm`, `ollama`.
- `--sglang_port`: Port for vLLM/SGLang server (default: 30000).

4. Run the full k-sweep to find optimal retrieval k per model:
```bash
bash run_k_sweep.sh
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter `k` value accordingly. Memories are cached after the first run, so subsequent k-sweep evaluations only re-run the QA answering step.

**Categories Information:** The LoCoMo dataset contains the following categories:
* Category 1: Multi-hop
* Category 2: Temporal
* Category 3: Open-domain
* Category 4: Single-hop
* Category 5: Adversarial

For more details about the categories, please refer to [this GitHub issue](https://github.com/snap-research/locomo/issues/6).

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
