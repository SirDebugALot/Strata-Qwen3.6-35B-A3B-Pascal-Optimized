// Default-off read-only ActQ staging. No quantization or arithmetic changes.
#pragma once
#include "common.hpp"
#include <cstdlib>
#include <cstring>

namespace sq {
inline bool actq_readonly_enabled() {
    static const bool value = [] {
        const char* p = std::getenv("STRATA_ACTQ_READONLY");
        SQ_CHECK(!p || !*p || !std::strcmp(p, "0") || !std::strcmp(p, "1"),
                 "STRATA_ACTQ_READONLY must be 0 or 1");
        return p && !std::strcmp(p, "1");
    }();
    return value;
}

// Only use for an allocation that is not written by any kernel/host operation
// during this consumer kernel's lifetime. The default branch is the old load.
template<bool ReadOnly, class V>
__device__ __forceinline__ V actq_load(const V* p) {
    if constexpr (ReadOnly) return __ldg(p);
    else return *p;
}
} // namespace sq
