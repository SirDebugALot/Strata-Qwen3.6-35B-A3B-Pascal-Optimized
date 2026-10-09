# engines/qwen35moe

The engine for Qwen3.5/3.6 MoE GGUFs (`general.architecture = qwen35moe`), built for Qwen3.6-35B-A3B. Users get it
through `setup_qwen36.py` (see [docs/QWEN36.md](../../docs/QWEN36.md)); this page is for working on it.

It is a CMake project of its own, so upstream's engine (`src/`, `include/`, the root `CMakeLists.txt`) is untouched
and merges from upstream do not meet it.

## Build

```sh
cmake -G Ninja -S engines/qwen35moe -B build-qwen35moe -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-qwen35moe
```

(On Windows from a "x64 Native Tools" prompt, or `-G "Visual Studio 17 2022" -A x64`.) `CMAKE_CUDA_ARCHITECTURES`
defaults to `native`. CUDA 12+ (13 for RTX 50); the CPU kernels need AVX2, and AVX-512 VNNI is used where the CPU has
it (`STRATA_CPU_ISA=avx2` forces AVX2).

## Files

| | |
| --- | --- |
| `src/model.*` | GGUF -> GPU (dense weights as split Q8_0: int8 + fp16 scales) and pinned RAM (experts as stored) |
| `src/engine.*` | state, the VRAM expert cache and its adaptation, decode steps (CUDA graphs), MTP drafting and verification, checkpoints, sampling |
| `src/kernels.cu` | decode kernels: GEMVs, gated DeltaNet, gated attention (IMROPE), router, expert GEMVs (Q4_K/Q5_K/Q6_K), combine |
| `src/prefill.cu` | batched prompt processing (cuBLAS; experts streamed over PCIe) |
| `src/cpu_moe.*` | the CPU half of the MoE: the worker pool and the GPU <-> CPU hand-off (host-mapped mailboxes) |
| `src/cpu_avx512.cpp`, `src/cpu_avx2.cpp` | the expert dot products per instruction set (each compiled with its flags) |
| `src/main.cpp` | the command line and `--serve` (upstream's engine protocol) |
| `tools/` | `llama_ref.py` + `parity.py` (greedy parity with llama.cpp), `ref_forward.py` (numpy float reference), `make_profile.py` (the routing profile) |

## The command line

```sh
strata-qwen35moe -m model.gguf --serve [--max-context N] [--threads N] [--expert-profile F] [--profile-save F]
strata-qwen35moe -m model.gguf --ids 1,2,3 -n 64            # greedy continuation, with speed
strata-qwen35moe -m model.gguf --bench --bench-prompt 2048 -n 128
strata-qwen35moe -m model.gguf --selfcheck 300              # prefill vs decode, MTP verify + rollback vs decode
strata-qwen35moe -m model.gguf --ids @ids.txt -n 384 --spec-check   # greedy with drafts == without
```

`--help` lists the rest (`--mtp-draft`, `--draft-p`, `--ckpt-slots`, `--cache-mb`, `--vram-reserve-mib`, ...).
Random-token prompts make the MTP look better than it is (it drafts their repetition): measure speculation on text.

`--serve` follows `src/program/generate.cpp`'s protocol: `INFO` and `READY <ctx> stop` at start; `GEN <max_new>
[temperature= top_p= top_k= min_p= penalty_last_n= penalty_repeat= penalty_freq= penalty_present= seed=] <ids>`;
`PP`, `T`, `DONE` (with the conversation and expert-cache fields), `ERR`; `STOP`, `QUIT`. `GENI` (images) answers
with an error for now.

## Tests

```sh
build-qwen35moe/test_cpu_kernels model.gguf 8         # both CPU instruction sets vs scalar dequantization
.venv/Scripts/python engines/qwen35moe/tools/llama_ref.py http://127.0.0.1:8099 ref.json   # a llama-server on the GGUF
.venv/Scripts/python engines/qwen35moe/tools/parity.py model.gguf ref.json
.venv/Scripts/python -m unittest discover -s tools -p test_setup_qwen36.py
.venv/Scripts/python tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 8k,32k,64k
```

## The routing profile

`data/expert-profile-qwen36.bin` (magic `SQEP`, layers, experts, then float counts per layer and expert, the MTP
layer last) ranks the experts for the cache's first fill. It was made with `tools/make_profile.py` on UD-Q4_K_XL; the
routing is the model's, so it serves the other quantizations. The server adds its own traffic to a copy
(`--profile-save`, in the data folder).
