# Qwen3.6-35B-A3B (Strata-Qwen36)

This fork of [Strata](https://github.com/Niko1221/Strata) runs **Qwen3.6-35B-A3B** from Unsloth's
[MTP GGUFs](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) with Strata's server, web app, APIs and tools.
The first size is **UD-Q4_K_M** (22.7 GB). The model's own MTP layer (in the GGUF) drafts tokens for speculative
decoding.

## How fast (measured)

RTX 5080 (16 GB), Ryzen 7 9800X3D (8 cores, AVX-512), 64 GB DDR5, Windows 11, UD-Q4_K_M, 64K context, 2026-10-01.

| | |
| --- | --- |
| Output, short chat (greedy, thinking on, 384 tokens) | **195-205 tok/s** (3 prompts: code, Japanese, math) |
| Output in the web app (its default sampling: temperature 0.6) | 188 tok/s |
| Output after a long prompt (greedy, 600 tokens) | 1K: 175 · 9K: 177 · 30K: 159 · 54K: 135 tok/s |
| Reading the prompt | 8K: 3,550 · 31K: 2,260 · 61K: 1,520 tok/s |
| Start (load) | 5-7 s; ~20 GB of RAM in use while it runs |

**Against llama.cpp** on the same PC, with the same GGUF and the same requests (greedy, no prefix reuse), llama.cpp
tuned per row (`--n-cpu-moe`, `--spec-type draft-mtp`, `-ub 2048`):

| tokens/s | Strata-Qwen36 | llama.cpp | llama.cpp + MTP | llama.cpp + MTP, `-ub 2048` |
| --- | ---: | ---: | ---: | ---: |
| Output, short chat | **187.6** | 84.1 | 94.7 | 90.8 |
| Output after a 28.7K-token prompt | **157.3** | 77.6 | 98.6 | 87.8 |
| Reading a 7.9K-token prompt | **3,269** | 1,011 | 911 | 2,283 |
| Reading a 29.8K-token prompt | **2,196** | 1,000 | 898 | 2,156 |

How it was measured, every request and the llama.cpp settings tried:
[bench/results/2026-10-01-qwen36-vs-llamacpp](../bench/results/2026-10-01-qwen36-vs-llamacpp/README.md)
(`tools/bench_vs_llamacpp.py`).

The MTP drafts are accepted 75-95% of the time, depending on the text. Needle-in-a-haystack
(`tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 8k,32k,64k`): 9 of 9 found.

## Install

You need an NVIDIA RTX 20 series or newer card (8 GB of VRAM or more; only 16 GB has been measured), **32 GB of RAM**
(all the experts, ~20 GB, are kept in RAM; 64 GB measured), an AVX2 CPU, ~25 GB of free disk, and Windows 10/11 or
Linux.

1. Clone or unzip this repository.
2. Run **`START-HERE.bat`** (Linux: `./setup.sh`). It asks for the size and the context, then:
   - compiles the engine for your GPU (`engines/qwen35moe`; a few minutes). This needs Visual Studio 2022 Build Tools
     and the CUDA Toolkit 13; setup offers to install them (it is upstream's build-tools step). There is no
     ready-made engine yet.
   - downloads `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` from a pinned revision and checks its SHA-256,
   - exports the tokenizer and chat template from the GGUF,
   - writes `strata-qwen36-q4_k_m.json` and `run-qwen36-q4_k_m.bat` / `.sh`, and starts the server.
3. The browser opens the Strata app at **`http://127.0.0.1:8081`**. Next time `START-HERE.bat` starts it right away.

Options: `--context 131072`, `--gpu 1`, `--threads 8`, `--host 0.0.0.0 --api-key KEY`, `--data-dir D:\models`,
`--gguf <file>` (a qwen35moe GGUF you already have), `--setup` (change settings), `--yes`, `--check`. Upstream's setup
for Qwen3.8-Flash-Next is still there: `START-HERE.bat --flash-next ...`.

## Next to an upstream Strata on the same PC

Both can be installed and run at the same time; nothing is shared:

| | upstream Strata | this fork |
| --- | --- | --- |
| Settings | `%APPDATA%\Strata` (`~/.config/strata`) | `%APPDATA%\Strata-Qwen36` (`~/.config/strata-qwen36`) |
| Model files | `Strata-data` next to the folder | `Strata-Qwen36-data` next to the folder |
| Server port | 8080 | 8081 (`chat.py` defaults to it) |
| Engine | `engine/strata(.exe)` | `engine-qwen35moe/strata-qwen35moe(.exe)` |

Upstream's setup moves model files out of other Strata folders it finds next to it; this fork's setup only looks at
other copies of this fork (a folder with `setup_qwen36.py`), so it never takes an upstream install's files. The fork's
own folder keeps no model files, so upstream's setup finds nothing to take from it either.

Both engines fill the free VRAM with experts, so on one GPU the one started second gets what the first left.

## Using it

Everything in the [README](../README.md#using-it) applies with port 8081: the web app (Chat, Monitor), the
OpenAI-compatible API at `http://127.0.0.1:8081/v1`, the Anthropic API at `/v1/messages`, tool calls, MCP servers,
`chat.py`.

**Thinking.** Qwen3.6's chat template thinks or does not (no levels). So that the web app's levels still mean
something, setup writes `"reasoning_budget_by_effort": {"low": 1024, "medium": 4096}` into the config: low and medium
cap the thinking at that many tokens (then it wraps up and answers, upstream's #123 mechanism), high thinks without a
cap, off does not think. An Anthropic request's `thinking.budget_tokens` is used as its budget. Edit or remove the key
in `strata-qwen36-*.json` to change this.

**Sampling.** As upstream: requests choose (the web app sends temperature 0.6, top_p 0.95, top_k 20); a request
without sampling fields is greedy. Qwen recommends temperature 1.0, top_p 0.95, top_k 20 with thinking on; a
`"sampling"` block in the config sets defaults for clients that send none (see `serve/server.py`).

**Prefix reuse.** The engine keeps three checkpoints of the delta-net state: the end of the last prompt, the start of
its last turn (Qwen's template drops earlier turns' reasoning, so the next turn diverges there) and the end of the
system message, which has a slot of its own. A follow-up turn and a new conversation with the same system prompt
read only what is new.

## How it works

`engines/qwen35moe` is its own engine (and its own CMake project): the qwen35moe architecture - 40 layers, 30 Gated
DeltaNet + 10 gated-attention layers, 256 experts (8 routed + 1 shared), the MTP layer - is different enough from
Qwen3.8-Flash-Next (hyper-connections, the PLE table, sparse attention, 512 experts) that upstream's engine, whose
kernels are compiled for Flash-Next's shapes, is left as it is. The new engine speaks upstream's `--serve` protocol,
so `serve/server.py`, the web app and the tools drive it unchanged. Upstream changes to the server, the web app, the
tokenizer and setup's helpers apply here directly.

- **GPU:** every dense weight (Q8_0; other types such as Q4_K_M's Q6_K output head are converted to Q8_0 at load), the
  KV cache (fp16), the delta-net states, and an expert cache that fills the rest of VRAM, starting from a routing
  profile (`data/expert-profile-qwen36.bin`) and adapting to the conversation.
- **RAM:** all 41 x 256 experts (the MTP layer's too) in pinned RAM, as the GGUF stores them (Q4_K / Q5_K / Q6_K). The
  CPU computes the experts that are not in VRAM, at the same time as the GPU computes the cached ones (AVX-512 VNNI or
  AVX2 kernels, picked at start).
- **Decode:** one CUDA graph per step; with MTP, up to 3 drafted tokens are verified in one step of the trunk, and the
  delta-net state is rolled back to the last accepted token. Sampled requests use speculative sampling, which keeps
  the output distribution of plain sampling.
- **Prompts:** batched, cuBLAS tensor-core GEMMs; experts not in VRAM are streamed over PCIe.

Details, the command line and tests: [engines/qwen35moe/README.md](../engines/qwen35moe/README.md).

## Correctness

- Greedy output against llama.cpp (LM Studio's CUDA build of llama.cpp, the same GGUF; `engines/qwen35moe/tools/
  llama_ref.py` + `parity.py`): 48 of 48 tokens identical on 3 of 5 prompts, the other two diverge after 1 and 21
  tokens at near-ties; the first token's top-10 log-probabilities differ by at most 0.35. The engine quantizes
  activations to int8 per 32 (GPU) or 256 (CPU) values; the UD-Q4_K_XL file, whose dense weights are all Q8_0 (no
  conversion), shows the same level of difference against its own llama.cpp reference.
- CPU kernels: `test_cpu_kernels` checks both instruction sets against scalar dequantization (relative error < 1e-7
  per row) and a whole expert layer against float (< 2%).
- `strata-qwen35moe --selfcheck N` compares batched prefill with token-by-token decode, the MTP verification step and
  its rollback; `--spec-check` checks that greedy decoding with drafts equals decoding without.

## Not yet

- Other sizes in setup (the engine runs any qwen35moe GGUF whose experts are Q4_K, Q5_K or Q6_K - UD-Q4_K_XL was
  checked too - and refuses other expert types by name).
- Images (the engine answers `GENI` with an error; the mmproj is in the same Hugging Face repository).
- Several GPUs (setup uses the biggest card, `--gpu N` another), the low-RAM mode, 8-bit KV, `--calibrate`,
  a ready-made engine.
- Linux: the code builds for it (pthread affinity, cpuid, mmap) but has not been compiled or run there yet.
- From the prototype this engine comes from: twice in ~150 runs of 512 tokens it ended with an illegal memory access
  during decoding; not reproduced here so far. The server starts a crashed engine again at the next request.

## Credits

The engine started as the StrataQwen35B prototype by this fork's author, after Strata's design. The model is Qwen's
([Apache-2.0](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)); the quantization is
[Unsloth](https://huggingface.co/unsloth)'s. The K-quant block formats follow ggml (MIT).
