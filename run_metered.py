"""带计量的评测入口：先装计量器，再原样调用仓库的 main()。

用法与 test_advanced_robust.py 完全一致：
    ./venv/bin/python run_metered.py --ratio 0.1 --backend openai --model deepseek-chat
"""
import atexit
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
sys.path.insert(0, HERE)   # meter.py
sys.path.insert(0, REPO)   # 仓库模块
os.chdir(REPO)

import meter  # noqa: E402  —— 必须先 import，完成 patch

import test_advanced_robust as T  # noqa: E402

atexit.register(meter.report)
T.main()
