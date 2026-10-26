"""Token / 成本计量器。

做法：monkey-patch OpenAI SDK 的 `Completions.create` 类方法，
把每次响应的 usage 累加起来。这样无论客户端在哪被创建都能抓到，
不需要改动仓库里的任何代码。

价格通过环境变量配置（单位：美元 / 百万 token）：
  PRICE_IN   输入单价，默认 0.28（DeepSeek V3.2 缓存未命中）
  PRICE_OUT  输出单价，默认 0.42
"""
import os
import sys

STATS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "failed_calls": 0,
}


def _install():
    import openai.resources.chat.completions as mod

    original = mod.Completions.create

    def patched(self, *args, **kwargs):
        try:
            resp = original(self, *args, **kwargs)
        except Exception:
            STATS["failed_calls"] += 1
            raise
        usage = getattr(resp, "usage", None)
        if usage is not None:
            STATS["calls"] += 1
            STATS["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            STATS["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                STATS["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0
        return resp

    mod.Completions.create = patched


def cost_usd():
    p_in = float(os.environ.get("PRICE_IN", "0.28"))
    p_out = float(os.environ.get("PRICE_OUT", "0.42"))
    return (STATS["prompt_tokens"] * p_in + STATS["completion_tokens"] * p_out) / 1_000_000


def report(*_):
    p_in = float(os.environ.get("PRICE_IN", "0.28"))
    p_out = float(os.environ.get("PRICE_OUT", "0.42"))
    s = STATS
    print("\n" + "=" * 58)
    print("TOKEN / 成本统计")
    print("=" * 58)
    print(f"  成功调用      : {s['calls']:,}")
    print(f"  失败调用      : {s['failed_calls']:,}")
    print(f"  输入 tokens   : {s['prompt_tokens']:,}")
    print(f"  其中命中缓存  : {s['cached_tokens']:,}")
    print(f"  输出 tokens   : {s['completion_tokens']:,}")
    print(f"  单价(美元/M)  : in={p_in} out={p_out}")
    print(f"  合计成本      : ${cost_usd():.4f}")
    if s["calls"]:
        print(f"  平均每次调用  : {s['prompt_tokens']/s['calls']:.0f} in / "
              f"{s['completion_tokens']/s['calls']:.0f} out")
    print("=" * 58)


_install()
