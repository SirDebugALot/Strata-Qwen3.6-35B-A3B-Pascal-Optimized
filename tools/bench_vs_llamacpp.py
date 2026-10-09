"""tools/bench_vs_llamacpp.py - the same chat requests to a Strata server and to a llama.cpp server, for docs/QWEN36.md.

Both answer /v1/chat/completions with llama.cpp-style `timings`, so one client measures both:

    python tools/bench_vs_llamacpp.py --url http://127.0.0.1:8081 --label strata --out strata.json
    python tools/bench_vs_llamacpp.py --url http://127.0.0.1:8099 --label llama.cpp --out llama.json

  decode   3 chat prompts (code, Japanese, math), thinking on, greedy, 384 tokens, x reps
  long     a ~30K-token document, then 400 tokens of answer (thinking off, greedy)
  prefill  ~8K- and ~32K-token documents, 8 tokens of answer

Each prompt starts with a fresh random id, so neither server can reuse a cached prefix (llama.cpp also gets
cache_prompt=false).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT = ["Write a Python function that parses a CSV file and returns the average of each numeric column.",
        "富士山について300字程度で説明してください。",
        "Prove that there are infinitely many primes."]


def corpus() -> str:
    text = ""
    for pat in ("docs/*.md", "serve/*.py", "tools/*.py", "src/core/*.cpp"):
        for f in sorted(ROOT.glob(pat)):
            text += f.read_text(encoding="utf-8", errors="replace") + "\n"
    return text


def ask(url, content, max_tokens, think, key=""):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}, "cache_prompt": False}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or 'x'}"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    t = d.get("timings") or {}
    return {"prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
            "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = random.Random()
    tag = lambda: f"[request {rng.getrandbits(48):012x}]\n"   # noqa: E731
    text = corpus()
    ask(a.url, tag() + "Say hi.", 8, False)                   # warm-up
    res = {"label": a.label, "decode": [], "long": [], "prefill": {}}
    for _ in range(a.reps):
        for p in CHAT:
            r = ask(a.url, tag() + p, 384, True)
            res["decode"].append(r)
            print(f"{a.label} decode   {r['completion_tokens']:4d} tok  {r['predicted_per_second']:6.1f} tok/s", flush=True)
    for _ in range(a.reps):
        r = ask(a.url, tag() + text[:96000] + "\n\nSummarize the text above in detail.", 400, False)
        res["long"].append(r)
        print(f"{a.label} long     {r['prompt_tokens']:6d} prompt  {r['predicted_per_second']:6.1f} tok/s out", flush=True)
    for name, chars in (("8k", 25000), ("32k", 100000)):
        res["prefill"][name] = []
        for _ in range(a.reps):
            r = ask(a.url, tag() + text[:chars] + "\n\nWhat is this text about? One word.", 8, False)
            res["prefill"][name].append(r)
            print(f"{a.label} prefill  {r['prompt_tokens']:6d} prompt  {r['prompt_per_second']:7.1f} tok/s", flush=True)
    mean = lambda xs, k: statistics.mean(x[k] for x in xs)   # noqa: E731
    res["summary"] = {"decode_tok_s": mean(res["decode"], "predicted_per_second"),
                      "long_prompt_tokens": mean(res["long"], "prompt_tokens"),
                      "long_decode_tok_s": mean(res["long"], "predicted_per_second"),
                      **{f"prefill_{k}_tokens": mean(v, "prompt_tokens") for k, v in res["prefill"].items()},
                      **{f"prefill_{k}_tok_s": mean(v, "prompt_per_second") for k, v in res["prefill"].items()}}
    print(json.dumps(res["summary"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
