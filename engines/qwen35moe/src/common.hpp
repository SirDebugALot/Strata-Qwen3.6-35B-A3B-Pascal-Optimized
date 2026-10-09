// common.hpp - logging, error checks, timing.
#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sq {

inline void log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

[[noreturn]] inline void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "fatal: ");
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::exit(1);
}

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace sq

#define SQ_CHECK(cond, ...) do { if (!(cond)) ::sq::die(__VA_ARGS__); } while (0)

#ifdef __CUDACC__
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    ::sq::die("CUDA %s at %s:%d: %s", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)
#else
#include <cuda_runtime_api.h>
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    ::sq::die("CUDA %s at %s:%d: %s", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)
#endif
