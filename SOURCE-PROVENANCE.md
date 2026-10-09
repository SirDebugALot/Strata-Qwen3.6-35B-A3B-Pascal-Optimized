# Source provenance

This source publication is based on:

- `https://github.com/ansonsi/Strata-Qwen3.6-35B-A3B`
  - branch: `qwen36`
  - commit: `f6b748c78885ed19369860441660c9015c1a1c5f`
- `https://github.com/Niko1221/Strata`
  - upstream project from which the Qwen3.6 fork was derived
- `https://github.com/ggml-org/llama.cpp`
  - pinned build dependency: `3cf03257f219afbe7334045ff7c6a06ac68c627d`
  - only its ggml MMQ and activation-quantization source is linked by the optimized prefill target

The local publication contains no GGUF model, generated expert cache, executable, DLL, benchmark run directory, or release ZIP.

The benchmark comparison used the official llama.cpp `b11524` Windows CUDA 12.4 artifact and the same model bytes as the Strata runs. The exact model SHA-256 is recorded in the main README.

Strata and the copied adapter source are provided under the repository's MIT license. llama.cpp/ggml is MIT-licensed; its checkout remains a separately fetched build dependency under `third_party/llama.cpp`.
