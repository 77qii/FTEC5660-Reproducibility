"""冒烟测试：不调用任何 LLM，只验证代码路径能走通。

检查项：
  1. 所有模块能 import
  2. LoCoMo 数据集能按仓库自己的 loader 加载
  3. 句向量模型能下载并编码（记忆检索依赖它）
  4. 指标函数能算（F1 / ROUGE / BLEU）
  5. prompt 模板能格式化（不缺占位符）
"""
import os
import sys
import traceback

# 仓库代码是平铺在 A-mem/ 下的，脚本放在父目录，需要把仓库目录塞进 sys.path
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

ok, fail = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            ok.append(name)
            print(f"  [OK]   {name}")
        except Exception as e:
            fail.append((name, e))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        return fn
    return deco


print("== 1. import 检查 ==")
with_llm = True
try:
    import memory_layer_robust  # noqa
    import memory_layer  # noqa
    import llm_text_parsers  # noqa
    import load_dataset  # noqa
    import utils  # noqa
    ok.append("import all modules")
    print("  [OK]   import all modules")
except Exception as e:
    with_llm = False
    fail.append(("import all modules", e))
    print(f"  [FAIL] import all modules: {type(e).__name__}: {e}")
    traceback.print_exc(limit=3)


@check("load_locomo_dataset")
def _():
    from load_dataset import load_locomo_dataset
    samples = load_locomo_dataset("data/locomo10.json")
    assert len(samples) == 10, f"expected 10 conversations, got {len(samples)}"
    s0 = samples[0]
    n_turns = sum(len(t.turns) for t in s0.conversation.sessions.values())
    print(f"         conversations={len(samples)} conv0 turns={n_turns} qa={len(s0.qa)}")
    cats = sorted({q.category for q in s0.qa})
    print(f"         conv0 categories={cats}")


@check("SentenceTransformer(all-MiniLM-L6-v2)")
def _():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    v = m.encode(["hello world", "another sentence"])
    print(f"         embedding shape={v.shape} dtype={v.dtype}")


@check("metrics (F1/ROUGE/BLEU)")
def _():
    from utils import score_prediction, aggregate_metrics
    from load_dataset import QA
    qa = QA(question="When did Caroline go to the support group?", answer="7 May 2023", evidence=[], category=4)
    ref, metrics = score_prediction("7 May 2023", qa)
    print(f"         ref={ref!r} f1={metrics.get('f1')} rouge1={metrics.get('rouge1_f')}")
    # 顺带验证 cat-5 的两套打分口径
    qa5 = QA(question="q", answer=None, evidence=[], category=5, adversarial_answer="wrong thing")
    ref_new, _ = score_prediction("Not mentioned in the conversation", qa5, legacy_c5=False)
    ref_old, _ = score_prediction("Not mentioned in the conversation", qa5, legacy_c5=True)
    print(f"         cat5 reference: 现行口径={ref_new!r}  论文口径={ref_old!r}")


@check("prompt 模板格式化")
def _():
    from llm_text_parsers import (
        ANALYZE_CONTENT_PROMPT, EVOLUTION_DECISION_PROMPT,
        STRENGTHEN_DETAILS_PROMPT, UPDATE_NEIGHBORS_PROMPT,
    )
    ANALYZE_CONTENT_PROMPT.format(content="x")
    EVOLUTION_DECISION_PROMPT.format(context="c", content="x", keywords="k", nearest_neighbors_memories="n")
    STRENGTHEN_DETAILS_PROMPT.format(content="x", keywords="k", nearest_neighbors_memories="n")
    UPDATE_NEIGHBORS_PROMPT.format(content="x", context="c", nearest_neighbors_memories="n",
                                   max_neighbor_idx=0, neighbor_count=1)
    print("         all templates formatted OK")


print(f"\n== 结果: {len(ok)} 通过 / {len(fail)} 失败 ==")
if fail:
    print("失败项:")
    for n, e in fail:
        print(f"  - {n}: {type(e).__name__}: {e}")
    sys.exit(1)
