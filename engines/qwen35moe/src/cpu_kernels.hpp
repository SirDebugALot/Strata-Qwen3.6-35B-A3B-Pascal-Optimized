// cpu_kernels.hpp - the CPU expert kernels per instruction set.  Each set lives in its own translation unit, compiled
// with that set's flags (cpu_avx512.cpp, cpu_avx2.cpp); cpu_moe.cpp picks one at run time from what the CPU and the
// OS support, so one binary runs on every x86-64 CPU with AVX2 and uses AVX-512 VNNI where it is there.
#pragma once

#include "cpu_moe.hpp"

namespace sq {

/// One instruction set's kernels.  The dot products take `nb` super-blocks of one weight row against `a`.
struct CpuKernels {
    const char* name;
    void (*quantize_q8k)(const float* x, Q8K* y, int n);
    float (*dot_q4k)(const void* w, const Q8K* a, int nb);
    float (*dot_q5k)(const void* w, const Q8K* a, int nb);
    float (*dot_q6k)(const void* w, const Q8K* a, int nb);
};

namespace avx512 {
void quantize_q8k(const float* x, Q8K* y, int n);
float dot_q4k(const void* w, const Q8K* a, int nb);
float dot_q5k(const void* w, const Q8K* a, int nb);
float dot_q6k(const void* w, const Q8K* a, int nb);
}  // namespace avx512

namespace avx2 {
void quantize_q8k(const float* x, Q8K* y, int n);
// Explicit scalar/vector packing choice for exact-parity and bounded performance tests; same quantization.
void quantize_q8k_test_pack(const float* x, Q8K* y, int n, bool vector_pack);
float dot_q4k(const void* w, const Q8K* a, int nb);
float dot_q5k(const void* w, const Q8K* a, int nb);
float dot_q6k(const void* w, const Q8K* a, int nb);
}  // namespace avx2

}  // namespace sq
