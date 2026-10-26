"""端到端干跑：用假的 LLM 后端把整条流水线走一遍，不产生任何 API 费用。

目的不是拿分数（假回答的分数没有意义），而是确认这些环节都没问题：
  记忆构建循环 -> 记忆缓存落盘 -> retriever 存取 -> 检索 -> 问答 -> 指标聚合 -> JSON 输出

跑法：
    ./venv/bin/python dry_run.py --ratio 0.1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
sys.path.insert(0, REPO)
os.chdir(REPO)

import memory_layer_robust as MLR  # noqa: E402
import test_advanced_robust as TAR  # noqa: E402

CALLS = {"n": 0}


class StubController:
    """按 prompt 类型返回格式合法的假响应，不联网。"""

    def __init__(self, *args, **kwargs):
        self.llm = self          # 仓库代码通过 .llm.get_completion 调用
        self.model = kwargs.get("model") or (args[0] if args else "stub")

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        CALLS["n"] += 1
        p = prompt.lower()

        if "analyze the following content" in p:
            return ("KEYWORDS: alpha, beta, gamma\n"
                    "CONTEXT: A stub summary of the memory content.\n"
                    "TAGS: domain, format, type")
        if "memory evolution agent" in p:
            return "DECISION: NO_EVOLUTION\nREASON: stub backend"
        if "generate several keywords" in p:
            return "alpha, beta, gamma"
        if "select the correct answer" in p or "short answer" in p:
            return "Not mentioned in the conversation"
        return "stub response"

    def check_connectivity(self):
        return True


# 两处都要替换：记忆层内部建一次，评测脚本里又建一次
MLR.RobustLLMController = StubController
MLR.RobustOpenAIController = StubController
TAR.RobustLLMController = StubController


def main():
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    if "--model" not in sys.argv:
        sys.argv += ["--model", "stub-dry-run"]
    if "--output" not in sys.argv:
        sys.argv += ["--output", "dry_run_result.json"]
    if "--backend" not in sys.argv:
        sys.argv += ["--backend", "openai"]

    print(f"== 干跑：假后端，参数 {sys.argv[1:]} ==")
    TAR.main()
    print(f"\n假 LLM 调用次数: {CALLS['n']:,}（真实运行时这些就是计费调用）")


if __name__ == "__main__":
    main()
