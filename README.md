# Strata Qwen3.6 35B-A3B — Pascal Optimized

An experimental source fork of Strata's Qwen3.6 engine, tuned for the **NVIDIA GeForce GTX 1050 Ti Mobile 4 GB** (`sm61`). It runs the 35B-A3B MoE model with CPU/GPU expert execution, an adaptive GPU expert cache, MTP drafting, and a long-prompt layer-major prefill path.

This repository contains source code and reproducible Windows build/run scripts. It does **not** contain the model, binaries, DLLs, benchmark logs, or release archives.

## Performance on the GTX 1050 Ti Mobile

Same machine, model, prompt token IDs, 25,088-token effective context, single stream, greedy sampling, and 1,000 generated tokens:

| Engine | 8K prefill | 8K decode | 16K prefill | 16K decode |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp b11524 | 113.47 t/s | 7.88 t/s | 105.12 t/s | 7.16 t/s |
| Original Strata Qwen3.6 branch, Pascal compatibility build | 141.75 t/s | 15.00 t/s | 106.76 t/s | 13.63 t/s |
| **Strata Qwen3.6 Pascal Optimized** | **256.80 t/s** | **24.53 t/s** | **210.98 t/s** | **21.40 t/s** |

The optimized build improved:

- 8K: **+81.2% prefill / +63.6% decode** over the original compatibility build, and **+126.3% / +211.3%** over llama.cpp.
- 16K: **+97.6% prefill / +57.1% decode** over the original compatibility build, and **+100.7% / +198.8%** over llama.cpp.

Test system: Intel Core i7-8750H, 48 GB RAM, Windows 11, NVIDIA driver 577.00. The optimized executable used CUDA 12.9; llama.cpp used its official CUDA 12.4 Windows package. The optimized figures average four runs at each prompt size; each comparison row averages two runs.

The untouched original commit does not build for `sm61` or import all expert tensor types in this GGUF. Its measured row therefore uses the smallest required Pascal/import compatibility changes while retaining the original inference path. These are throughput measurements, not model-quality scores; the fixed-length document answers exposed factual or structural review flags across the tested engines.

## Model

Download the exact tested model from Hugging Face:

- Repository: [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- File: [`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)
- Tested SHA-256: `36f9ec0e4c775f6efd3a61c1ea76f0875128469c3d012e3d9e2495a90e7a7150`

The file is about 14.1 GB. Git LFS is not required for this source repository because the GGUF is not included.

## Windows build for Pascal `sm61`

Requirements:

- Windows 10 or 11
- Visual Studio 2022 Build Tools with Desktop C++
- CUDA Toolkit 12.x with Pascal support
- CMake, Ninja, and Git on `PATH`
- An AVX2-capable CPU

Run:

```bat
BUILD-PASCAL-SM61.bat
```

The script checks out the pinned llama.cpp dependency at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, configures the Qwen engine with `SQ_ENABLE_PREFILL_MMQ=ON`, and writes:

```text
build-sm61\strata-qwen35moe.exe
```

No prebuilt executable is committed.

## Run the optimized profile

Create the Python environment once:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the server with the GGUF path:

```bat
START-OPTIMIZED-PASCAL.bat "D:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
```

The launcher extracts the tokenizer from the GGUF on first use, writes only local ignored state under `.local-data` and `runtime`, and serves the OpenAI-compatible API and web UI on [http://127.0.0.1:8005](http://127.0.0.1:8005). Press `Ctrl+C` in its console to stop it.

Optional arguments:

```text
START-OPTIMIZED-PASCAL.bat MODEL.gguf --port 8005 --context 25000 --threads 6
START-OPTIMIZED-PASCAL.bat MODEL.gguf --tokenizer D:\path\to\tokenizer
```

The tested profile assumes one inference stream and a 4 GB Pascal GPU. Display workloads also consume VRAM; start the server after connecting the monitors you intend to use.

## Main optimized paths

- Native mixed-IQ expert import, avoiding startup conversion of expert weights to Q6_K.
- AVX2 CPU expert kernels with asynchronous CPU/GPU MoE execution.
- Adaptive GPU expert caching and prompt-route refill.
- MTP depth 3 with reduced draft work and last-row handling.
- Phase-switched prefill that temporarily loans decode-only VRAM.
- Packed-IQ DP4A MMQ on Pascal.
- Bounded layer-major prefill with 1,024-token tiles and 8,192-token windows.
- Shared attention K/V preparation, AQ6 attention banks, and Pascal dense GEMM selection.

Layer-major windowing bounds temporary activation memory and amortizes expert-weight preparation. Persistent KV memory still grows with context, so it does not make context length unlimited.

## Source lineage

This fork starts from [`ansonsi/Strata-Qwen3.6-35B-A3B`](https://github.com/ansonsi/Strata-Qwen3.6-35B-A3B) commit `f6b748c78885ed19369860441660c9015c1a1c5f`, itself derived from [`Niko1221/Strata`](https://github.com/Niko1221/Strata). The packed-MMQ path builds against pinned llama.cpp/ggml source. See [SOURCE-PROVENANCE.md](SOURCE-PROVENANCE.md), [UPSTREAM-README.md](UPSTREAM-README.md), and [LICENSE](LICENSE).

This is an independent experimental fork and is not an official Qwen, Unsloth, llama.cpp, or upstream Strata release.
