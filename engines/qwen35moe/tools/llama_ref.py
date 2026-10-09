"""engines/qwen35moe/tools/llama_ref.py - capture reference outputs from a running llama-server (the oracle) for
parity tests.

    python engines/qwen35moe/tools/llama_ref.py http://127.0.0.1:8099 ref_llama.json

For each prompt: the token ids, llama.cpp's greedy continuation and the top-10 probabilities of the first
generated token.  tools/parity.py later runs the same ids through the engine and compares.
"""
import json
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "日本で一番高い山は",
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "In 1905, Albert Einstein published four papers that changed physics. The first one explained the photoelectric effect, "
    "showing that light is quantized. The second one explained Brownian motion, providing evidence for atoms. The third "
    "introduced special relativity, and the fourth derived the most famous equation in physics, which states that",
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    base, out = sys.argv[1], sys.argv[2]
    for _ in range(600):
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    break
        except Exception:
            pass
        time.sleep(1)
    res = []
    for p in PROMPTS:
        ids = post(base + "/tokenize", {"content": p, "add_special": False, "parse_special": True})["tokens"]
        c = post(base + "/completion", {"prompt": ids, "n_predict": 48, "temperature": 0, "top_k": 1,
                                        "n_probs": 10, "cache_prompt": False, "return_tokens": True,
                                        "post_sampling_probs": False})
        gen = c.get("tokens") or [t["id"] for t in c["completion_probabilities"]]
        first = c["completion_probabilities"][0]["top_logprobs"]
        res.append({"prompt": p, "ids": ids, "greedy": gen,
                    "first_top": [{"id": t["id"], "logprob": t["logprob"]} for t in first],
                    "text": c["content"]})
        print(len(ids), "->", c["content"][:80].replace("\n", "\n"))
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
