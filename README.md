# A-Mem — Reproducibility Study

**Qi Zheng · 1155268235 · FTEC5660**

**Repository:** https://github.com/77qii/FTEC5660-Reproducibility
**Final commit:** `662402da83edaf57f8b9a56abd1cc4af2cf85d73`

Source reproduced: [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem) at `b31a1dd`.
Full write-up: [`REPORT.md`](REPORT.md). Rendered PDF, slides and the verified evidence archive
are under `outputs/`.

## Scope

One LoCoMo conversation (`conv-26`, 419 turns, 199 questions), DeepSeek requested as
`deepseek-chat`, three stochastic QA trials sharing **one memory build** (local seeds 42/43/44).
The local seed permutes category-5 option order; it is not an API sampling seed. No matched-condition
naive-retrieval, LoCoMo or MemGPT rerun was performed, so every comparison with the paper is
descriptive, not a verification of relative improvement.

## Two modifications, ordered by role

### Primary — Temporal prompt ablation: F1 23.66 → 57.49

A single controlled intervention on the category-2 answer instruction. The 37 Temporal questions
reuse the exact prompts and retrieved contexts recorded in the seed-42 log, so retrieval and
memory are held fixed; only the date instruction changes, from "an approximate date" to an absolute
date anchored to the relevant evidence session (`D Month YYYY` / `Month YYYY` / `YYYY`). No gold
date enters either prompt. Arms are interleaved with alternating order: 37 answers per arm,
**74 calls**, zero failures, no API sampling seed.

| Condition (same 37 questions, same contexts) | Temporal F1 | Format candidates | Share |
|---|---|---|---|
| Historical seed 42 — context source only | 20.89 | 13 / 37 | 35.1% |
| Control: original instruction, fresh run | **23.66** | **13 / 37** | **35.1%** |
| Treatment: absolute-date instruction | **57.49** | **2 / 37** | **5.4%** |
| Treatment − fresh control | **+33.84** | **−11** | **−29.7 pt** |

Re-asking the same questions already moves F1 from 20.89 to 23.66 with no prompt change, which is
provider stochasticity on identical inputs; the prompt effect is therefore measured against the
**fresh control**. One stochastic answer per question per arm means this is a measured paired
difference, not a repeat-run variance estimate, confidence interval or significance test. The
candidate share is the unchanged `analyze_temporal.py` regex screen — a format proxy, not a verified
date-correctness rate. **57.49 is not a comparison with the paper's 45.85** (different model, full
benchmark), and the historical 24.29-point gap is not thereby attributed to format.

Cost of the ablation: 980,412 input / 307 output tokens, **$0.2746** at the original meter's
no-cache rates (0.28/M in, 0.42/M out) — an upper estimate, not an invoice.

### Additional — category-5 diagnostic workflow (`analyze_c5.py`)

Evaluation **quality**, not model accuracy: the workflow reads stored predictions, makes zero API
calls and changes no output. The refusal predicate and reference fix are upstream's.

| Evaluation aspect | Before: upstream reporting | After: this workflow |
|---|---|---|
| Consistency assertions | stored flags / F1, no independent audit | F1 and refusal flags re-derived on 141 observations |
| Uncertainty | refusal interval not surfaced | 47 clusters × 3, 20,000 draws, seed 20261026 → [56.7, 79.4] |
| Failure classification | no exclusive breakdown | 43 trap, 2 other, 0 refusal-plus-trap |
| Cross-tabulation | F1 and flag reported apart | dual-reference F1 × refusal; 139/141 disagree at F1 ≥ 0.5 |

Refusal rate **68.09 ± 2.13%** (96/141 observations). The pooled Wilson interval [60.0, 75.2] is
**naive** — it ignores repeated-question dependence; the cluster bootstrap **[56.7, 79.4]** is the
interval to quote. It is exploratory, conditional on the stored memory and trials, and covers
neither memory rebuilding nor dependence between different questions of one conversation. It is
not a full-benchmark interval, and implementation agreement is not semantic validation.

### Not mine — the upstream scoring fix in `b31a1dd`

The authors corrected how category 5 is scored. Re-scoring identical predictions under both
conventions moves Adversarial F1 by **+37.46** and Overall by **+8.85** with no change in model
output; the exact unrounded shift is 37.46499, rounded after aggregation. I audited and quantified
that change; I did not make it, and whether the paper's baseline rows used the same convention is
not verifiable from inspected public artifacts.

## Reproduce the main tables — offline, one command, no API key

Standard library only; no install, no network, no credentials.

```bash
python3 reproduce_tables.py --selftest
python3 reproduce_tables.py --out-dir reproduced_tables
```

Verified in this repository at commit `662402d` (Python 3.13.12, clean environment, API keys
unset):

```
self-test: PASS (4/4)
  ok   real evidence reproduces
  ok   tampered stored F1 rejected
  ok   missing question rejected
  ok   hash mismatch rejected
```

`--out-dir` writes `tables.md` and `tables.json`. It recomputes, and refuses to print a table when
a stored value does not reproduce: **597 stored F1 values**, identical question/gold/category sets
across trials, aggregate statistics, category-5 flags, evidence SHA-256 hashes, Bootstrap seed,
BERTScore before/after values, meter arithmetic and the paired ablation records. These checks
establish internal consistency and file integrity, not independent attestation.

## Baseline results (F1 %, mean ± sample std of three fixed-memory QA trials)

| Category | n | Paper GPT-4o-mini | Reproduced, paper reference |
|---|---|---|---|
| Multi-Hop | 32 | 27.02 | 27.04 ± 4.40 |
| Temporal | 37 | 45.85 | 21.56 ± 2.44 |
| Open-Domain | 13 | 12.14 | 13.88 ± 1.75 |
| Single-Hop | 70 | 44.65 | 43.08 ± 1.36 |
| Adversarial | 47 | 50.03 | 31.93 ± 2.05 |
| Overall | 199 | not reported | 31.96 ± 1.11 |

Corrected-reference Adversarial: **69.40 ± 1.70**; Overall: **40.81 ± 1.43**.

## Evidence and provenance

- `evidence/predictions/` — three original result files, byte-identical to the run outputs. Their
  metadata keeps the original local dataset path, deliberately, so raw-file identity holds.
- `evidence/logs/` — the three metered runs plus BERTScore before/after logs. Machine absolute
  paths, ANSI escapes and credential-like strings are removed; timestamps and usage are untouched.
- `evidence/baseline/` — 8 pairs × 3 BERTScore values, all 24 bit-identical before/after.
- `evidence/temporal_contexts.json` — the 37 exact historical answer prompts and context hashes.
- `evidence/temporal_ablation.json`, `temporal_requests.jsonl` — paired predictions, response
  `model`/`usage` fields, 74 call records. No headers, no keys.
- `evidence/manifest.json`, `provenance.json`, `sanitization.json` — SHA-256 fingerprints, upstream
  and local commits, the real working-tree state at preparation time, and the redaction record.

The original three trials retained only the *requested* model name; the API-returned `model` field
was not logged and is **unknown** for them. All 74 ablation responses retain `model:
deepseek-flash`, which evidences routing for that run only and cannot be transferred back.

## Install, run and re-run the ablation

The offline table entry above needs nothing installed. To generate new predictions:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env      # put your own key in .env; never commit it
```

Tested on macOS Apple Silicon, CPU only, Python 3.13.12. If the `rouge-score` sdist cannot extract,
download the official PyPI sdist, extract it and install that local directory with
`--no-build-isolation`. If network restrictions block NLTK downloads, fetch `punkt`, `punkt_tab`,
`wordnet` and `omw-1.4` from the official `nltk_data` source and restrict write permission.

```bash
# API calls cost money. The first run builds memory; cached repeats run QA only.
./run_eval.sh --ratio 0.1 --seed 42 --retrieve_k 10 --output result_seed42.json
./run_trials.sh 3 0.1
python rescore_c5.py result_seed42.json --json
python aggregate_legacy.py result_seed42_rescore.json
```

Original first conversation: 1,723 calls (~1,325 memory + 398 QA), ~$1.13 by the historical meter;
each QA repeat: 398 calls, ~$0.73. Original three completed trials **$2.59**; with the ablation
**~$2.86**. These price every input token at list rate and ignore cache discounts, so they are
upper estimates, not bills. Full-benchmark **scenarios** (not measurements): ~22,600 first-run
calls / $12.9, or ~3,972 QA-only calls / $7.3 with memory reused.

```bash
# Re-run the paired ablation: 74 paid answer calls. A new run needs a new checkpoint path.
python ablate_temporal_prompt.py run   --inputs evidence/temporal_contexts.json   --checkpoint temporal_requests_new.jsonl --out temporal_ablation_new.json   --env-file .env --max-meter-usd 1.0
```

An existing checkpoint is refused unless `--resume` is passed; the budget is checked between calls
and is not a provider-side spending cap. To rebuild the inputs from a completed original log:

```bash
python ablate_temporal_prompt.py prepare   --baseline result_seed42.json --log path_to_completed_evaluation.log   --out temporal_contexts_new.json
```

`utils.py` changes only BERTScorer reuse: metrics **7.83 → 0.12 s/question (~65×)**, with BERTScore
6.07 of 7.83 s (~77%). Values are unchanged — 24/24 compared numbers bit-identical, two full
199-question dry runs with zero prediction and zero metric differences. This engineering fix is
kept separate from both modifications.

## Secrets

`.env`, keys, private memory caches, virtual environments and model weights are not in this
repository. Only the explicit `evidence/` files are published; other run outputs stay ignored.

## References and data sources

1. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. (2025). *A-Mem: Agentic Memory for
   LLM Agents*. NeurIPS 2025. [arXiv:2502.12110v11](https://arxiv.org/abs/2502.12110v11).
2. Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. (2024).
   *Evaluating Very Long-Term Conversational Memory of LLM Agents*. ACL 2024, 13851–13870.
   [doi:10.18653/v1/2024.acl-long.747](https://aclanthology.org/2024.acl-long.747/).
3. [A-Mem evaluation repository](https://github.com/WujiangXu/A-mem), pinned `b31a1dd`;
   [upstream scoring-fix commit](https://github.com/WujiangXu/A-mem/commit/b31a1dd).
4. [LoCoMo repository](https://github.com/snap-research/locomo) and
   [project page](https://snap-research.github.io/locomo/). The evaluated bytes are the in-repo
   `data/locomo10.json`, fingerprinted in the manifest.

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
