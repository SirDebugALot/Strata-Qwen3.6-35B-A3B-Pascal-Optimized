"""engines/qwen35moe/tools/make_profile.py - build the routing profile the VRAM expert cache starts from
(data/expert-profile-qwen36.bin).

Runs a small, varied chat workload (Japanese / English / Chinese; code, math, writing, knowledge, tool calls; thinking
on and off) through the engine with Qwen's recommended sampling, then quits it, which saves its routing counts
(--profile-save).  Generated tokens dominate the counts, which is what matters: the cache only serves decode (prefill
streams experts).

    python engines/qwen35moe/tools/make_profile.py --model <gguf> --tokenizer <dir> [--out FILE] [--max-new 768]
    python engines/qwen35moe/tools/make_profile.py ... --eval --profile data/expert-profile-qwen36.bin   (measure one)

Starts from a uniform cache and an empty profile, so the output reflects this workload only.  A running server keeps
adding its own traffic to its copy of the profile (setup's --profile-save file).
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import StrataEngine  # noqa: E402

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_files", "description": "Search the repository for a regular expression.",
    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                   "required": ["pattern"]}}}

PROMPTS = [
    # Japanese
    "日本の四季について、それぞれの特徴と代表的な行事を説明してください。",
    "量子コンピュータと古典コンピュータの違いを高校生にもわかるように説明して。",
    "新入社員向けに、ビジネスメールの書き方のポイントを5つ挙げてください。",
    "「吾輩は猫である」の作者と、その作品の特徴を簡単に教えてください。",
    "東京から京都へ2泊3日で旅行します。おすすめの旅程を作ってください。",
    "次の文章を敬語に直してください：「明日の会議、ちょっと遅れるかも。資料は先に送っとくね。」",
    "PythonでCSVファイルを読み込み、列ごとの平均値を計算するコードを書いてください。",
    "二次方程式 x^2 - 5x + 6 = 0 を解き、解き方を説明してください。",
    "RustとGoの違いを、並行処理とメモリ管理の観点から比較してください。",
    "短い怪談を一つ書いてください。舞台は古い図書館です。",
    "健康的な朝食のメニューを一週間分考えてください。",
    "機械学習における過学習とは何か、その対策も含めて説明してください。",
    # English
    "Explain how a transformer language model works, from tokenization to sampling.",
    "Write a Python function that returns the longest palindromic substring of a string, with tests.",
    "What caused the fall of the Western Roman Empire? Give a balanced summary.",
    "Prove that the square root of 2 is irrational.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare PostgreSQL and SQLite: when should I use each?",
    "Implement a thread-safe LRU cache in C++17 and explain the design.",
    "A train leaves at 3:15 pm and travels 210 km at 84 km/h. When does it arrive? Show your work.",
    "Summarize the main ideas of stoic philosophy in plain language.",
    "Write a bash script that finds the ten largest files under a directory.",
    "I have a React component that re-renders too often. What are common causes and fixes?",
    "Explain the difference between TCP and UDP with examples of where each is used.",
    "Draft a polite email declining a meeting invitation and proposing another time.",
    "What is the time complexity of quicksort in the best, average and worst case, and why?",
    "Write a SQL query that returns the top 3 customers by total order value per country.",
    "Explain CUDA warps, shared memory and memory coalescing to a new GPU programmer.",
    # Chinese / other
    "请用中文介绍一下长城的历史。",
    "写一首关于秋天的现代诗。",
    "Explique en français la différence entre l'imparfait et le passé composé.",
    "Escribe una receta sencilla de tortilla de patatas.",
    # Math / reasoning
    "If 3 painters paint 3 walls in 3 hours, how long do 9 painters take to paint 9 walls? Explain.",
    "Find all integer solutions of x^2 - y^2 = 45.",
    "1から100までの素数の和を求めてください。計算過程も示してください。",
    "Compute the derivative of f(x) = x^3 * ln(x) and find its critical points.",
    # Code review / debugging
    "What is wrong with this code?\n\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```",
    "Convert this JavaScript to TypeScript with proper types:\n\n```js\nfunction groupBy(arr, key) {\n  return arr.reduce((acc, x) => {\n    (acc[x[key]] = acc[x[key]] || []).push(x);\n    return acc;\n  }, {});\n}\n```",
    "次のエラーの原因と対処法を教えてください：`TypeError: 'NoneType' object is not subscriptable`",
    "Write a Dockerfile for a Python FastAPI app with a multi-stage build.",
]

TOOL_PROMPTS = [
    ("大阪と札幌の今の天気を教えて。", [WEATHER_TOOL]),
    ("What's the weather like in Paris in Fahrenheit?", [WEATHER_TOOL]),
    ("Find where the function parse_config is defined in the src directory.", [SEARCH_TOOL]),
    ("リポジトリ内でTODOコメントを探して、一覧にしてください。", [SEARCH_TOOL, WEATHER_TOOL]),
]

# held out: --eval measures a profile on these instead of building one
EVAL_PROMPTS = [
    "明治維新が日本の社会に与えた影響をまとめてください。",
    "JavaScriptのPromiseとasync/awaitの違いをコード例つきで説明して。",
    "Write a Go HTTP server with graceful shutdown and explain each part.",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a haiku sequence about the ocean, then explain the imagery.",
    "猫を飼い始める人のためのチェックリストを作ってください。",
    "What are the trade-offs between microservices and a monolith?",
    "解释一下什么是区块链，以及它的优缺点。",
]


def load_tokenizer(tdir: Path):
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    return ST.Tokenizer(tokens, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tdir / "token_type.json").read_text()))


def main():
    exe = "strata-qwen35moe.exe" if os.name == "nt" else "strata-qwen35moe"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the GGUF")
    ap.add_argument("--tokenizer", required=True, help="the tokenizer directory setup exported (packs/<tag>/tokenizer)")
    ap.add_argument("--exe", default=str(REPO / "engine-qwen35moe" / exe))
    ap.add_argument("--out", default=str(REPO / "data" / "expert-profile-qwen36.bin"))
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (testing)")
    ap.add_argument("--only", default="", help="comma-separated 1-based job numbers (testing)")
    ap.add_argument("--profile", default="", help="start from this profile (with --eval: measure it)")
    ap.add_argument("--adapt", action="store_true", help="let the cache adapt while running")
    ap.add_argument("--eval", action="store_true", help="run the held-out prompts, adapt on, do not save")
    ap.add_argument("--engine-args", default="", help="extra engine arguments")
    a = ap.parse_args()
    if a.eval:
        a.adapt = True
    tdir = Path(a.tokenizer)
    tok = load_tokenizer(tdir)
    tpl = ChatTemplate(tdir / "chat_template.jinja")
    args = ["-m", a.model, "--max-context", "8192", "--expert-profile", a.profile] + \
        ([] if a.adapt else ["--no-adapt"]) + a.engine_args.split()
    out = Path(a.out)
    if not a.eval:
        out.unlink(missing_ok=True)                 # --profile-save reads an existing file as its starting profile
        args += ["--profile-save", str(out)]
    eng = StrataEngine(a.exe, args, log=str(REPO / "make_profile.log"))
    jobs = [(p, None, i % 3 != 2) for i, p in enumerate(PROMPTS)] + [(p, t, True) for p, t in TOOL_PROMPTS]
    if a.eval:
        jobs = [(p, None, i % 2 == 0) for i, p in enumerate(EVAL_PROMPTS)]
    order = list(range(len(jobs)))
    if a.only:
        order = [int(x) - 1 for x in a.only.split(",")]
    elif a.limit:
        order = order[:a.limit]
    total, t0, never = 0, time.time(), threading.Event()
    for i in order:
        prompt, tools, think = jobs[i]
        text = tpl.render([{"role": "user", "content": prompt}], tools=[t["function"] for t in tools or []] or None,
                          enable_thinking=think)
        ids = tok.encode(text, parse_special=True)
        sp = ({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "presence_penalty": 1.5} if think else
              {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "presence_penalty": 1.5})
        sp["seed"] = 1000 + i
        n = sum(1 for t in eng.generate(ids, a.max_new, sp, never) if t is not None)
        total += n
        dm = float(eng.last.get("decode_ms", 0)) or 1.0
        print(f"[{i + 1:2d}/{len(jobs)}] {len(ids):4d} + {n:4d} tokens  {n * 1000 / dm:6.1f} tok/s  "
              f"{'think' if think else 'plain'}{' tools' if tools else ''}  {prompt[:40]!r}", flush=True)
    print(f"{total} generated tokens in {time.time() - t0:.0f} s")
    eng.close()                                     # QUIT: the engine writes --profile-save
    if not a.eval:
        print(f"profile written to {out}" if out.exists() else f"the engine did not write {out}")


if __name__ == "__main__":
    main()
