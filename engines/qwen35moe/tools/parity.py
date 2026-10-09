"""engines/qwen35moe/tools/parity.py - compare the engine against the llama.cpp reference captured by llama_ref.py.

    python engines/qwen35moe/tools/parity.py <model.gguf> ref_llama.json [extra engine args]

Per prompt: how many greedy tokens agree with llama.cpp, and the first token's top-10 log-probabilities
(max |difference|).  Exact agreement is not required - the two engines quantize activations differently -
but a real bug shows up as an early divergence and large log-prob differences.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

EXE = os.environ.get("STRATA_QWEN35MOE_EXE") or os.path.join(os.path.dirname(__file__), "..", "..", "..", "engine-qwen35moe",
                                                   "strata-qwen35moe.exe" if os.name == "nt" else "strata-qwen35moe")


def main():
    model, ref_path, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    refs = json.load(open(ref_path, encoding="utf-8"))
    worst = 0.0
    for i, r in enumerate(refs):
        ids = r["ids"]
        n = len(r["greedy"])
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "logits.bin")
            out = subprocess.run([EXE, "-m", model, "--ids", ",".join(map(str, ids)), "-n", str(n), "--dump-logits", lp,
                                  "--ctx", "4096", *extra], capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr)
                sys.exit(1)
            logits = np.fromfile(lp, dtype=np.float32)
        gen = [int(t) for t in out.stdout.split("generated:")[1].split()]
        agree = 0
        while agree < min(len(gen), n) and gen[agree] == r["greedy"][agree]:
            agree += 1
        m = logits.max()
        logp = logits - (m + np.log(np.exp(logits - m).sum()))
        diffs = [abs(logp[t["id"]] - t["logprob"]) for t in r["first_top"] if t["logprob"] > -20]
        d = max(diffs)
        worst = max(worst, d)
        speed = [l for l in out.stderr.splitlines() if "generated" in l]
        print(f"[{i}] {len(ids):3d} prompt tokens: greedy agrees for {agree}/{n} tokens; first-token top-10 max |dlogp| = {d:.4f}")
        if speed:
            print("     " + speed[-1].strip())
    print(f"worst first-token |dlogp| = {worst:.4f}")


if __name__ == "__main__":
    main()
