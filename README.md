# A-Mem — Reproducibility Study

**Qi Zheng · 1155268235 · FTEC5660**

**Student GitHub URL: pending provision.** This is the local submission-ready repository;
no student remote is configured and this revision has not been committed or pushed.
The authors' repo below is the source, **not** the student's submission link. The assignment
requires both the student's URL and reviewable commits before submission.

Source: [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem), upstream commit `b31a1dd`.
The complete upstream README is retained below. Report: [`REPORT.md`](REPORT.md).

## Scope and contribution

One LoCoMo conversation (`conv-26`, 419 turns, 199 questions), DeepSeek requested as
`deepseek-chat`, three stochastic QA trials sharing **one memory build**, local seeds 42/43/44.
The seed controls category-5 option order, not API sampling. No matched-condition naive retrieval,
LoCoMo or MemGPT reruns: paper comparisons are descriptive, not proof of relative improvement.

**Formal modification:** my `analyze_c5.py` evaluation-quality workflow, not a memory algorithm
or model-accuracy improvement. The upstream reference fix and refusal predicate are the authors'.

| Before: upstream reporting | After: independent added workflow |
|---|---|
| stored flags / F1 without an independent audit | consistency assertions against re-derived values |
| refusal uncertainty not surfaced | question-cluster bootstrap (47 × 3, 20,000 draws, seed 20261026) |
| no exclusive failure decomposition | failure-mode classification: 43 trap, 2 other, 0 hedged |
| separate F1 / refusal fields | dual-reference cross-tab; 139/141 disagreement at F1 ≥ 0.5 |

The refusal rate is **68.09 ± 2.13%** across fixed-memory QA trials: 96/141 observations;
exploratory 95% question-cluster interval **[56.7, 79.4]**. Naive pooled Wilson [60.0, 75.2]
ignores repeated-question dependence and is not the preferred combined interval. Neither interval
represents full-benchmark generalisation, memory-rebuild variability or between-question
correlation within a conversation. Consistency is not semantic validation of the upstream predicate.

## Reproduce the main tables — offline, one command, no API key

From the repository root with Python 3 (standard library only; no install needed):

```bash
python3 reproduce_tables.py --out-dir reproduced_tables
python3 reproduce_tables.py --selftest
```

Outputs: `reproduced_tables/tables.md` and `tables.json`. The entry validates **597 stored F1
values**, identical question/gold/category sets, aggregate statistics, category-5 flags, evidence
SHA-256 hashes, Bootstrap seed, BERTScore before/after values, meter arithmetic and the paired
ablation outputs. Self-tests reject modified F1, a missing question and hash mismatch.

Original baseline results (F1 %, mean ± sample std of three fixed-memory QA trials):

| Category | n | Paper GPT-4o-mini | Reproduced, paper reference |
|---|---|---|---|
| Multi-Hop | 32 | 27.02 | 27.04 ± 4.40 |
| Temporal | 37 | 45.85 | 21.56 ± 2.44 |
| Open-Domain | 13 | 12.14 | 13.88 ± 1.75 |
| Single-Hop | 70 | 44.65 | 43.08 ± 1.36 |
| Adversarial | 47 | 50.03 | 31.93 ± 2.05 |
| Overall | 199 | not reported | 31.96 ± 1.11 |

Corrected-reference Adversarial: **69.40 ± 1.70**; Overall: **40.81 ± 1.43**. The exact mean
Adversarial scoring shift is 37.46499 → **+37.46**, rounded after aggregation; it changes no
predictions. The paper's trap-overlap F1 is not refusal accuracy.

## Evidence included and provenance

- `evidence/predictions/`: three original result JSON files, copied byte-for-byte. Their metadata
  retains the original local dataset path, intentionally, to preserve raw-file identity.
- `evidence/logs/`: original metering runs and BERTScore before/after logs. Machine absolute paths,
  ANSI sequences and any credential-like strings are removed; timestamps/usage remain.
- `evidence/baseline/`: 8 pairs × 3 BERTScore values; all 24 bit-identical before/after.
- `evidence/temporal_contexts.json`: 37 exact historical answer prompts and context hashes.
- `evidence/temporal_ablation.json` and `temporal_requests.jsonl`: paired predictions, response
  model/usage fields and call records; no request headers or keys.
- `evidence/manifest.json`, `provenance.json`, `sanitization.json`: SHA-256, full upstream and local
  Git commits, real **uncommitted working-tree state** and log-redaction record.

A hash checks file integrity/internal consistency, not independent attestation. The original
three trials retained only the *requested* model. Their API-returned model field is **unknown**.
All 74 new paired-ablation responses retain API `model: deepseek-flash`; that value does not
retroactively identify the earlier trials or independently pin a provider checkpoint.

## Supporting test — actual bounded Temporal prompt ablation

Both arms use the same 37 seed-42 logged contexts; no new memory, retrieval or query generation.
Only the date instruction changes: approximate dates versus absolute dates anchored to the
relevant evidence session (`D Month YYYY`, `Month YYYY`, `YYYY`, supported precision only).
System message, requested model, temperature 0.7, max_tokens 1000 and `utils.calculate_metrics`
are unchanged; no gold dates are placed into prompts. Arms alternate first-arm order.

| Condition | F1 | Same regex candidates | Share |
|---|---|---|---|
| Historical seed 42, context source only | 20.89 | 13/37 | 35.1% |
| Original prompt, fresh paired control | **23.66** | **13/37** | **35.1%** |
| Absolute-date prompt, fresh paired treatment | **57.49** | **2/37** | **5.4%** |
| Treatment − fresh control | **+33.84 points** | **−11** | **−29.7 percentage points** |

One pass per arm, 74 answer calls, no API sampling seed: **prompt sensitivity, not established
stable accuracy or a significance claim**. The candidate share is the unchanged
`analyze_temporal.REL/ABS` screening proxy, not a verified date-error rate. 57.49 is not evidence
of exceeding the paper's 45.85. The old ≈9.28 sensitivity calculation is a different hypothetical
subgroup-score replacement, not a causal decomposition of the ablation.

The new calls used **980,412 input / 307 output tokens**, historical-meter estimate **$0.2746**
(0.28/M input, 0.42/M output, cache discounts ignored; not an invoice).

## Install for new paid pipeline runs

The offline table reproduction above needs no dependencies. To generate new model predictions:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Enter your own key in .env; never commit that file.
```

Tested original environment: macOS Apple Silicon, CPU, Python 3.13.12. If the `rouge-score`
sdist cannot extract, download its official PyPI sdist, extract and install the local directory
with `--no-build-isolation`. For NLTK, install `punkt`, `punkt_tab`, `wordnet`, `omw-1.4` from the
official `nltk_data` source if network restrictions block downloads; restrict write access.
Model weights/corpora are required for full metrics but not for the offline evidence-table entry.

```bash
# API costs money. First run builds memory; cached repeats run QA only.
./run_eval.sh --ratio 0.1 --seed 42 --retrieve_k 10 --output result_seed42.json
./run_trials.sh 3 0.1
python rescore_c5.py result_seed42.json --json
python aggregate_legacy.py result_seed42_rescore.json
```

`run_eval.sh` reads `.env` and maps DeepSeek onto `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
Original first conversation: 1,723 calls (~1,325 memory + 398 QA), ~$1.13 historical meter;
QA repeats: 398 calls, ~$0.73 each. Original three completed trials: **$2.59**; plus ablation,
**~$2.86**. These are no-cache meter estimates, not account bills. Full-benchmark **scenarios**:
~22,600 first-run calls / $12.9; ~3,972 QA-only calls / $7.3 with memory reused. They are not
measured full-benchmark totals; phase dollars use QA-repeat costs as a first-run QA proxy.

## Re-run the paired ablation (74 paid answer calls)

Do not re-run merely to obtain the shipped table: use the offline entry. To generate a new run:

```bash
python ablate_temporal_prompt.py run   --inputs evidence/temporal_contexts.json   --checkpoint temporal_requests_new.jsonl --out temporal_ablation_new.json   --env-file .env --max-meter-usd 1.0
```

An existing checkpoint is refused unless `--resume` is explicit. The budget is checked between
calls, not a guaranteed provider spending cap. Metrics are validated against the same original
F1 implementation. To extract new contexts from a completed original log:

```bash
python ablate_temporal_prompt.py prepare   --baseline result_seed42.json --log path_to_completed_evaluation.log   --out temporal_contexts_new.json
```

`utils.py` changes only BERTScorer reuse. Metrics went **7.83 → 0.12 s/question (~65×)**; BERTScore
was 6.07/7.83 s (~77%), not 99.8%. The report separates this engineering fix from the formal
modification. Original memory/retrieval and prompts are preserved; paired replay is a new driver.

## Secrets and publishing status

`.env`, keys, private memory caches, virtual environments and model weights are not included.
Only the explicit `evidence/` files are intended for public sharing; ordinary new run outputs
remain ignored. Do not upload arbitrary logs or credentials. No new commit or push was made;
`provenance.json` records the honest local HEAD rather than pretending the current files are committed.

## Formal references / data sources

1. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. (2025). *A-Mem: Agentic Memory for
   LLM Agents*. NeurIPS 2025. [arXiv:2502.12110v11](https://arxiv.org/abs/2502.12110v11).
2. Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. (2024).
   *Evaluating Very Long-Term Conversational Memory of LLM Agents*. ACL 2024, 13851–13870.
   [doi:10.18653/v1/2024.acl-long.747](https://aclanthology.org/2024.acl-long.747/).
3. [A-Mem evaluation repository](https://github.com/WujiangXu/A-mem), pinned `b31a1dd`;
   [upstream scoring-fix commit](https://github.com/WujiangXu/A-mem/commit/b31a1dd).
4. [LoCoMo official repository](https://github.com/snap-research/locomo) and
   [project page](https://snap-research.github.io/locomo/). Actual evaluated bytes are the
   in-repo `data/locomo10.json`, fingerprinted in the manifest.

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
