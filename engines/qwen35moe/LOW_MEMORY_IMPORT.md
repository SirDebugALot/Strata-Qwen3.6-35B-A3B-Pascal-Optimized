# Low-memory import of Unsloth UD-IQ3_XXS

The filename does not describe every tensor's encoding. The tested
`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` contains IQ2_S, IQ3_S, IQ4_XS, Q2_K,
and Q3_K routed experts. This local adaptation decodes those formats and
converts experts to Q6_K for the engine's existing CPU and CUDA kernels.
The original GGUF stays read-only. This conversion introduces a small
additional quantization error; it does not restore precision lost in the
original low-bit model.

Three optional environment variables control the low-memory path:

- `STRATA_EXPERT_CACHE_FILE`: absolute path to a derived expert cache. The
  cache uses a pageable Windows file mapping instead of pinning all expert
  weights. Allow approximately 24.61 GiB of disk space for the 40 main
  layers, or 25.22 GiB including MTP. Inference can become disk-bound when
  the active working set does not fit in available RAM.
- `STRATA_EXPERT_CONVERSION_FAST=1`: use ggml's initial weighted scale fit
  instead of the reference quantizer's 19 additional scale-search trials.
  On 81 sampled blocks from this model, both decoders matched independent
  gguf-py output exactly. The added Q6_K relative RMS error was 0.9002% for
  the reference conversion and 0.9847% for the fast conversion. A scalar
  benchmark measured approximately 11.5 versus 99.9 million weights per
  second. These are conversion measurements, not generation speed or
  full-model quality guarantees. The default uses reference conversion.
- `STRATA_CPU_LM_HEAD=1`: keep the Q8 output head in host RAM and project
  logits on the CPU, saving approximately 515 MiB of GPU memory.

The cache records source path, size, modification time, a hash of the
first MiB, model shapes, converted tensor layouts, and conversion mode.
Each completed layer is flushed before its progress marker is saved, so
interrupted conversion resumes at the last complete layer. A matching
finished cache is reused. An existing file with an unrecognized magic
header is never overwritten. Changing conversion mode or the model
invalidates the derived cache and starts conversion again.

Scalar validation source: `src/test_quant_import.cpp`. It consumes sampled
raw GGUF blocks followed by their independent float32 reference values,
compares every decoded element, and measures the additional Q6_K error.
Use the optional `--fast` argument to validate fast conversion.
